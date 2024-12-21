
h5_to_matrix <- function(path){
    h5_data <- h5read(path,"matrix") 
    feature <- h5_data$features
    barcode <- h5_data$barcodes
    data <- t(h5_data$data)
    colnames(data) <- as.character(barcode)
    rownames(data) <- as.character(feature)
    return (data)
}

write_h5 <- function(exprs_list, h5file_list) {
  if (length(unique(lapply(exprs_list, rownames))) != 1) {
    stop("rownames of exprs_list are not identical.")
  }
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

process_rna <- function(expr){
  sce <- SingleCellExperiment(list(counts=expr))
  sce <- logNormCounts(sce)
  #decomp <- modelGeneVar(sce)
  #hvgs = getTopHVGs(decomp, n = 1000)
  #hvgs <- rownames(decomp)[decomp$mean>0.01 & decomp$p.value <= 0.05]
  return(list(logcounts = sce@assays@data$logcounts, counts = sce@assays@data$counts))
}

process_atac <- function(expr){
  sce <- SingleCellExperiment(list(counts=as.matrix(expr)))
  sce <- logNormCounts(sce)
  decomp <- modelGeneVar(sce)
  #hvgs <- rownames(decomp)[decomp$mean>0.25& decomp$p.value <= 0.05]
  hvgs = getTopHVGs(decomp, n = 1000)
  return(list(logcounts = sce@assays@data$logcounts, counts = sce@assays@data$counts))
}

process_adt <- function(expr){
  ori_expr <- expr
  expr <- NormalizeData(expr, normalization.method = 'CLR')
  return(list(normalized = expr, original = ori_expr))
}

read_h5_data <- function(rna_path=NULL, adt_path=NULL, atac_path=NULL, list_len = 3) {
  rna_list <- list()
  adt_list <- list()
  atac_list <- list()
  ori_rna_list <- list()
  ori_adt_list <- list()
  ori_atac_list <- list()
  ##### read rna ####
  if (!is.null(rna_path)){
    for (i in seq_along(rna_path)) {
      if (is.null(rna_path[[i]])) {
        rna_list[[i]] <- NULL
        } else {
          processed_rna <- process_rna(h5_to_matrix(rna_path[[i]]))
          rna_list[[i]] <- processed_rna$logcounts
          ori_rna_list[[i]] <- processed_rna$counts
        }
    }
  }
  ##### read adt ####
  if (!is.null(adt_path)){
    for (i in seq_along(adt_path)) {
      if (is.null(adt_path[[i]])) {
        adt_list[[i]] <- NULL
        } else {
          processed_adt <- process_adt(h5_to_matrix(adt_path[[i]]))
          adt_list[[i]] <- processed_adt$normalized
          ori_adt_list[[i]] <- processed_adt$original
        }
    }
  }
  ##### read atac ####
  if (!is.null(atac_path)){
    for (i in seq_along(atac_path)) {
      if (is.null(atac_path[[i]])) {
        atac_list[[i]] <- NULL
        } else {
          processed_atac <- process_atac(h5_to_matrix(atac_path[[i]]))
          atac_list[[i]] <- processed_atac$logcounts
          ori_atac_list[[i]] <- processed_atac$counts
        }
    }
  }

  result <- list(rna = rna_list, adt = adt_list, atac = atac_list, ori_rna = ori_rna_list, ori_adt = ori_adt_list, ori_atac = ori_atac_list)
  return(result)
}


stabmap_input_function <- function(file_paths, a){
  # no null for rna/adt/atac
  data_list = list()
  if (!is.null(file_paths$rna_path) & !is.null(file_paths$adt_path) & !is.null(file_paths$atac_path)){
    for (i in 1:(length(file_paths[[1]]))) {
      if (!is.null(a$rna[[i]]) & is.null(a$adt[[i]]) & is.null(a$atac[[i]])) {
      data <- a$rna[[i]]
      } else if (is.null(a$rna[[i]]) & !is.null(a$adt[[i]]) & is.null(a$atac[[i]])) {
      data <- a$adt[[i]]
      } else if (is.null(a$rna[[i]]) & is.null(a$adt[[i]]) & !is.null(a$atac[[i]])) {
      data <- a$atac[[i]]
      } else if (!is.null(a$rna[[i]]) & !is.null(a$adt[[i]]) & is.null(a$atac[[i]])) {
      data <- rbind(a$rna[[i]], a$adt[[i]])
      } else if (!is.null(a$rna[[i]]) & is.null(a$adt[[i]]) & !is.null(a$atac[[i]])) {
      data <- rbind(a$rna[[i]], a$atac[[i]])
      } else if (is.null(a$rna[[i]]) & !is.null(a$adt[[i]]) & !is.null(a$atac[[i]])) {
      data <- rbind(a$adt[[i]], a$atac[[i]])
      } else {
      data <- rbind(a$adt[[i]], a$atac[[i]])
      }
      data_list[[i]] <- data
    }
  }
  
  ## only has rna and adt data
  if (!is.null(file_paths$rna_path) & !is.null(file_paths$adt_path) & is.null(file_paths$atac_path)){
    for (i in 1:(length(file_paths[[1]]))) {
      if (!is.null(a$rna[[i]]) & is.null(a$adt[[i]])) {
      data <- a$rna[[i]]
      } else if (is.null(a$rna[[i]]) & !is.null(a$adt[[i]])) {
      data <- a$adt[[i]]
      } else if (!is.null(a$rna[[i]]) & !is.null(a$adt[[i]])) {
      data <- rbind(a$rna[[i]], a$adt[[i]])
      } 
      data_list[[i]] <- data
    }
  }
  
  ## only has rna and atac data
  if (!is.null(file_paths$rna_path) & is.null(file_paths$adt_path) & !is.null(file_paths$atac_path)){
    for (i in 1:(length(file_paths[[1]]))) {
      if (!is.null(a$rna[[i]]) & is.null(a$atac[[i]])) {
      data <- a$rna[[i]]
      } else if (is.null(a$rna[[i]]) & !is.null(a$atac[[i]])) {
      data <- a$atac[[i]]
      } else if (!is.null(a$rna[[i]]) & !is.null(a$atac[[i]])) {
      data <- rbind(a$rna[[i]], a$atac[[i]])
      } 
      data_list[[i]] <- data
    }
  }
  
  ## only has adt and atac data
  if (is.null(file_paths$rna_path) & !is.null(file_paths$adt_path) & !is.null(file_paths$atac_path)){
    for (i in 1:(length(file_paths[[1]]))) {
      if (!is.null(a$adt[[i]]) & is.null(a$atac[[i]])) {
      data <- a$adt[[i]]
      } else if (is.null(a$adt[[i]]) & !is.null(a$atac[[i]])) {
      data <- a$atac[[i]]
      } else if (!is.null(a$adt[[i]]) & !is.null(a$atac[[i]])) {
      data <- rbind(a$adt[[i]], a$atac[[i]])
      } 
      data_list[[i]] <- data
    }
  }
  
  ## only has rna data
  if (!is.null(file_paths$rna_path) & is.null(file_paths$adt_path) & is.null(file_paths$atac_path)){
    for (i in 1:(length(file_paths[[1]]))) {
      data_list[[i]] <- a$rna[[i]]
    }
  }
  
  ## only has adt data
  if (is.null(file_paths$rna_path) & !is.null(file_paths$adt_path) & is.null(file_paths$atac_path)){
    for (i in 1:(length(file_paths[[1]]))) {
      data_list[[i]] <- a$adt[[i]]
    }
  }
  
  ## only has atac data
  if (is.null(file_paths$rna_path) & is.null(file_paths$adt_path) & !is.null(file_paths$atac_path)){
    for (i in 1:(length(file_paths[[1]]))) {
      data_list[[i]] <- a$atac[[i]]
    }
  }
  names(data_list) <- paste0("data", 1:length(data_list))
  for (i in seq_along(data_list)) {
    colnames(data_list[[i]]) <- paste0("data", i, "_", 1:ncol(data_list[[i]]))
  }
  
  return(data_list)
}


