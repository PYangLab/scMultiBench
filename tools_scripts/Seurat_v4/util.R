suppressPackageStartupMessages({
  library(Signac)
  library(Seurat)
  library(dplyr)
  library(rhdf5)
  library(HDF5Array)
})

h5_to_matrix <- function(path){
    h5_data <- h5read(path,"matrix") 
    feature <- h5_data$features
    barcode <- h5_data$barcodes
    data <- t(h5_data$data)
    colnames(data) <- as.character(barcode)
    colnames(data) <- as.character(paste0(c(1:dim(data)[2]),barcode))
    rownames(data) <- as.character(paste0(c(1:dim(data)[1]),feature))
    return (data)
}

write_h5 <- function(exprs_list, h5file_list) {
  for (i in seq_along(exprs_list)) {
    if (file.exists(h5file_list[i])) {
      warning("h5file exists! will rewrite it.")
      system(paste("rm", h5file_list[i]))
    }
    h5createFile(h5file_list[i])
    writeHDF5Array(((exprs_list[[i]])), h5file_list[i], name = "data")
    print(h5ls(h5file_list[i]))
  }
}

run_Seurat_RNA_ADT <- function(rna, adt){
  bm <- CreateSeuratObject(counts = rna)
  adt_assay <- CreateAssayObject(counts = adt)
  bm[["ADT"]] <- adt_assay
  
  DefaultAssay(bm) <- 'RNA'
  bm <- NormalizeData(bm) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()

  DefaultAssay(bm) <- 'ADT'
  # we will use all ADT features for dimensional reduction
  # we set a dimensional reduction name to avoid overwriting the 
  VariableFeatures(bm) <- rownames(bm[["ADT"]])
  bm <- NormalizeData(bm, normalization.method = 'CLR', margin = 2) %>% 
    ScaleData() %>% RunPCA(reduction.name = 'apca')
  
  # Identify multimodal neighbors. These will be stored in the neighbors slot, 
  # and can be accessed using bm[['weighted.nn']]
  # The WNN graph can be accessed at bm[["wknn"]], 
  # and the SNN graph used for clustering at bm[["wsnn"]]
  # Cell-specific modality weights can be accessed at bm$RNA.weight
  bm <- FindMultiModalNeighbors(
    bm, reduction.list = list("pca", "apca"), 
    dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
  )

  #bm <- RunUMAP(bm, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_", return.model = TRUE)
  bm <- FindClusters(bm, graph.name = "wsnn", algorithm = 3, resolution = 2, verbose = FALSE)
  
  return(bm) # the dimension of embedding is 2
}

run_Seurat_RNA_ATAC <- function(rna, atac){
    pbmc <- CreateSeuratObject(counts = rna)
    atac_assay <- CreateAssayObject(counts = atac)
    pbmc[["ATAC"]] <- atac_assay
    
    DefaultAssay(pbmc) <- "RNA"
    pbmc <- SCTransform(pbmc, verbose = FALSE) %>% RunPCA() %>% RunUMAP(dims = 1:50, reduction.name = 'umap.rna', reduction.key = 'rnaUMAP_')
    
    # ATAC analysis
    # We exclude the first dimension as this is typically correlated with sequencing depth
    DefaultAssay(pbmc) <- "ATAC"
    pbmc <- RunTFIDF(pbmc)
    pbmc <- FindTopFeatures(pbmc, min.cutoff = 'q0')
    pbmc <- RunSVD(pbmc)
    pbmc <- RunUMAP(pbmc, reduction = 'lsi', dims = 2:50, reduction.name = "umap.atac", reduction.key = "atacUMAP_")
    
    pbmc <- FindMultiModalNeighbors(pbmc, reduction.list = list("pca", "lsi"), dims.list = list(1:50, 2:50))
    #pbmc <- RunUMAP(pbmc, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_", return.model = TRUE)
    pbmc <- FindClusters(pbmc, graph.name = "wsnn", algorithm = 3, verbose = FALSE)
    
    return(pbmc)
}

LT_RNA_ADT_qRNA <- function(reference, query_path){
  DefaultAssay(reference) <- 'RNA'
  query_data <- h5_to_matrix(query_path)
  query_data <- CreateSeuratObject(counts = query_data)
  query_data <- NormalizeData(query_data, verbose = FALSE)
  anchors <- FindTransferAnchors(
    reference = reference,
    query = query_data,
    normalization.method = "LogNormalize",
    reference.reduction = "spca",
    dims = 1:30 #consistent with the WNN methods
  )
  
  query_data <- MapQuery(
    anchorset = anchors,
    query = query_data,
    reference = reference,
    refdata = list(
      celltype.l1 = "celltype",
      predicted_ADT = "ADT"
    ),
    reference.reduction = "spca",
    reduction.model = "wnn.umap"
  )
  return(query_data)
}

LT_RNA_ATAC_qRNA <- function(reference, query_path){
  query_data <- h5_to_matrix(query_path)
  query_data <- CreateSeuratObject(counts = query_data)
  query_data <- SCTransform(query_data, verbose = FALSE)
  anchors <- FindTransferAnchors(
    reference = reference,
    query = query_data,
    normalization.method = "SCT",
    reference.reduction = "spca",
    dims = 1:50
  )
  
  query_data <- MapQuery(
    anchorset = anchors,
    query = query_data,
    reference = reference,
    refdata = list(
      celltype.l1 = "celltype",
      predicted_ATAC = "ATAC"
    ),
    reference.reduction = "spca",
    reduction.model = "wnn.umap"
  )
  return(query_data)
}
