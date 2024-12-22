scPalette <- function(n) {
  colorSpace <- c('#E41A1C','#377EB8','#4DAF4A','#984EA3','#F29403','#F781BF','#BC9DCC','#A65628','#54B0E4','#222F75','#1B9E77','#B2DF8A',
                  '#E3BE00','#FB9A99','#E7298A','#910241','#00CDD1','#A6CEE3','#CE1261','#5E4FA2','#8CA77B','#00441B','#DEDC00','#B3DE69',
                  '#8DD3C7','#999999')
  
  if (n <= length(colorSpace)) {
    colors <- colorSpace[1:n]
  } else {
    colors <- grDevices::colorRampPalette(colorSpace)(n)
  }
  return(colors)
}

umap_data <- function(embedding_data, output.path, cate) {
  graph.idx <- which(sapply(strsplit(names(embedding_data), "\\.",), "[[", 3) %in% 
                       c("MIRA", "Seurat_v4", "scMoMaT", "Matilda_test")) # "MIRA", "Seurat_v4", "scMoMaT" are graph-based
  
  if (length(graph.idx) > 0) {
    embedding_data.fil <- embedding_data[-graph.idx]
  } else {
    embedding_data.fil <- embedding_data
  }
  
  embedding_data.fil <- embedding_data[-graph.idx]
  
  umap_data <- list()
  for (i in 1:length(embedding_data.fil)) {
    print(i)
    
    output.file <- paste0("Di/result/umap/cross_umap/", names(embedding_data.fil)[i], ".umap.csv")
    
    if (!file.exists(output.file)) {
      dataset <- sapply(strsplit(names(embedding_data.fil)[i], "\\."), "[[", 2)
      method <- sapply(strsplit(names(embedding_data.fil)[i], "\\."), "[[", 3)
      
      dat <- embedding_data.fil[[i]]$data
      
      # Ensure the data has more columns than rows, otherwise transpose it
      if (ncol(dat) < nrow(dat)) {
        dat <- t(dat)
      }
      
      # Check if column names (cell names) are present in the matrix, if not, assign generic names
      if (is.null(colnames(dat))) {
        colnames(dat) <- paste0("Cell_", 1:ncol(dat))  # Assign generic cell names if missing
      }
      
      # Ensure there are row names for PCA dimensions (e.g., "PC_1", "PC_2", ...)
      if (is.null(rownames(dat))) {
        rownames(dat) <- paste0("PC_", 1:nrow(dat))
      }
      
      # Create a Seurat object using only metadata (based on cell names from the embeddings)
      meta_data <- data.frame(row.names = colnames(dat))  
      seurat.object <- Seurat::CreateSeuratObject(counts = dat, meta.data = meta_data)
      
      # Assign the embedding matrix (dat) as the PCA result in Seurat
      seurat.object[["pca"]] <- CreateDimReducObject(embeddings = t(dat), 
                                                     key = "PC_", 
                                                     assay = DefaultAssay(seurat.object)) 
      
      set.seed(0)
      seurat.object <- RunUMAP(seurat.object, reduction = "pca", dims = 1:nrow(dat))  # Adjust dimensions as needed??
      
      # Extract UMAP embeddings
      seurat_umap_dims <- Embeddings(seurat.object, reduction = "umap")
      colnames(seurat_umap_dims) <- c("UMAP_1", "UMAP_2")
      seurat_umap_dims <- as.data.frame(seurat_umap_dims)
      
      if (cate %in% c("cross")) {
        if (method %in% c("Matilda", "UnitedNet")) {
          seurat_umap_dims$label <- ctys.supervised[[dataset]]$x
          seurat_umap_dims$batch <- batches.supervised[[dataset]]
        } else {
          seurat_umap_dims$label <- ctys[[dataset]]$x
          seurat_umap_dims$batch <- batches[[dataset]]
        }
      } else {
        seurat_umap_dims$label <- ctys[[dataset]]$x
        seurat_umap_dims$batch <- batches[[dataset]]
      }
      write.csv(seurat_umap_dims, output.file)
    }
  }
}

umap_data_graph <- function(files = c("/Seurat_v4/dist.h5", "/Seurat_v4/idx.h5"), 
                            input.path,
                            output.path,
                            method = "Seurat_v4", 
                            indiceplus1 = FALSE) {
  
  path <- "Chunlei/result/embedding/cross integration"
  dataset_dirs <- list.dirs(path = path, full.names = TRUE, recursive = FALSE)
  print(dataset_dirs)
  
  dist_list <- list()
  idx_list <- list()
  
  # Loop through each dataset folder
  for (dir in dataset_dirs) {
    dist_file <- file.path(dir, files[1])
    idx_file <- file.path(dir, files[2])
    
    if (file.exists(dist_file) && file.exists(idx_file)) {
      # Load distance data from dist.h5
      dist_data <- h5read(dist_file, "/")
      dist_list[[dir]] <- dist_data
      
      # Load index data from idx.h5
      idx_data <- h5read(idx_file, "/")
      idx_list[[dir]] <- idx_data
    }
  }
  names(idx_list) <- names(dist_list) <-  gsub(" ", "_", sapply(strsplit(names(dist_list), "/"), function(x) paste0(x[[4]], ".", x[[5]])))
  
  umap_data <- list()
  for (i in 1:length(dist_list)) {
    dataset <- sapply(strsplit(names(dist_list)[i], "\\."), function(x)x[[2]])
    output.file <- paste0("Di/result/umap/cross_umap/cross_integration.", dataset, ".", method, ".umap.csv")
    print(output.file)
    
    if(!file.exists(output.file)) {
      dist <- as.matrix(dist_list[[i]]$data)
      indice <- idx_list[[i]]$data
      if (indiceplus1) {indice <- indice + 1}
      if (dim(dist)[1] < dim(dist)[2]) {
        dist <- t(dist)
        indice <- t(indice)
      }
      rownames(dist) <- rownames(indice) <- paste0("Cell_", 1:nrow(dist))
      neighborlist <- list("idx" = indice,
                           "dist" = dist)
                           print(7)
      seurat_obj <- RunUMAP(object = neighborlist)
      
      seurat_umap_dims <- as.data.frame(seurat_obj@cell.embeddings)
      seurat_umap_dims$label <- ctys[[dataset]]$x
      seurat_umap_dims$batch <- batches[[dataset]]
      write.csv(seurat_umap_dims, file = output.file)
    }
  }
}
