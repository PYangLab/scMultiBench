
library(rhdf5)
library(reshape2)
library(ggplot2)
library(MOFA2)
library(HDF5Array)

# useful functions
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

read_h5_data <- function(rna_path_list=NULL, adt_path_list=NULL, atac_path_list=NULL) {
  ##### read rna #### 
  rna <- list()
  adt <- list()
  atac <- list()
  if (!is.null(rna_path_list)){
    batch <- c()
    for (rna_path in rna_path_list){
      base <- basename(rna_path)
      rna[[base]] <- h5_to_matrix(rna_path)
      print(dim(rna[[base]]))
    }
    for (i in seq_along(rna)) {
      num_columns <- ncol(rna[[i]])
      batch <- c(batch, rep(i, num_columns))
    }
    rna <- do.call(cbind, rna)
  } else {rna <- NULL}
  ##### read adt ####
  if (!is.null(adt_path_list)){
    batch <- c()
    for (adt_path in adt_path_list){
      base <- basename(adt_path)
      adt[[base]] <- h5_to_matrix(adt_path)
      print(dim(adt[[base]]))
    }
    for (i in seq_along(adt)) {
      num_columns <- ncol(adt[[i]])
      batch <- c(batch, rep(i, num_columns))
    }
    adt <- do.call(cbind, adt)
  } else {adt <- NULL}
  ##### read atac ####
  if (!is.null(atac_path_list)) {
    batch <- c()
    for (atac_path in atac_path_list){
      base <- basename(atac_path)
      atac[[base]] <- h5_to_matrix(atac_path)
    } 
    for (i in seq_along(atac)) {
      num_columns <- ncol(atac[[i]])
      batch <- c(batch, rep(i, num_columns))
    }
    atac <- do.call(cbind, atac)
  }else {atac <- NULL}
  result <- list(rna = rna, adt = adt, atac = atac, batch=batch)
  return(result)
}

make_unique_colnames <- function(colnames) {
  new_colnames <- colnames
  dupes <- colnames[duplicated(colnames)]
  for (dupe in unique(dupes)) {
    idx <- which(colnames == dupe)
    new_colnames[idx] <- paste0(dupe, "_", seq_along(idx))
  }
  return(new_colnames)
}
