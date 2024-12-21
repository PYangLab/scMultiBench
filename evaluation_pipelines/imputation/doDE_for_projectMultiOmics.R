#suppressPackageStartupMessages(library(MAST))
suppressPackageStartupMessages(library(edgeR))
suppressPackageStartupMessages(library(limma))
#suppressPackageStartupMessages(library(DESeq2))
suppressPackageStartupMessages(library(data.table))

library(limma)


doLimma_singlecore <- function(exprsMat, cellTypes, exprs_pct = 0.05){
  cellTypes <- droplevels(as.factor(cellTypes))
  tt <- list()
  for (i in 1:nlevels(cellTypes)) {
    tmp_celltype <- (ifelse(cellTypes == levels(cellTypes)[i], 1, 0))
    design <- stats::model.matrix(~tmp_celltype)
    meanExprs <- do.call(cbind, lapply(c(0,1), function(i){
      Matrix::rowMeans(exprsMat[, tmp_celltype == i, drop = FALSE])
    }))
    meanPct <- do.call(cbind, lapply(c(0,1), function(i){
      Matrix::rowSums(exprsMat[, tmp_celltype == i, drop = FALSE] > 0)/sum(tmp_celltype == i)
    }))
    #keep <- meanPct[,2] > exprs_pct
    y <- methods::new("EList")
    #y$E <- exprsMat[keep, ]
    y$E <- exprsMat
    fit <- limma::lmFit(y, design = design)
    fit <- limma::eBayes(fit, trend = TRUE, robust = TRUE)
    tt[[i]] <- limma::topTable(fit, n = Inf, adjust.method = "BH", coef = 2)
    if (!is.null(tt[[i]]$ID)) {
      tt[[i]] <- tt[[i]][!duplicated(tt[[i]]$ID),]
      rownames(tt[[i]]) <- tt[[i]]$ID
    }
    tt[[i]]$meanExprs.1 <- meanExprs[rownames(tt[[i]]), 1]
    tt[[i]]$meanExprs.2 <- meanExprs[rownames(tt[[i]]), 2]
    tt[[i]]$meanPct.1 <- meanPct[rownames(tt[[i]]), 1]
    tt[[i]]$meanPct.2 <- meanPct[rownames(tt[[i]]), 2]
  }
  names(tt) <- levels(cellTypes)
  return(tt)
}










doLimma <- function(exprsMat, cellTypes, exprs_pct = 0.05){
  message("Limma multicore")
  cellTypes <- droplevels(as.factor(cellTypes))
  
  tt <- mclapply(c(1:nlevels(cellTypes)), mc.cores =1, function(i){
    tmp_celltype <- (ifelse(cellTypes == levels(cellTypes)[i], 1, 0))
    design <- stats::model.matrix(~tmp_celltype)
    meanExprs <- do.call(cbind, lapply(c(0,1), function(i){
      Matrix::rowMeans(exprsMat[, tmp_celltype == i, drop = FALSE])
    }))
    meanPct <- do.call(cbind, lapply(c(0,1), function(i){
      Matrix::rowSums(exprsMat[, tmp_celltype == i, drop = FALSE] > 0)/sum(tmp_celltype == i)
    }))
    #keep <- meanPct[,2] > exprs_pct
    y <- methods::new("EList")
    #y$E <- exprsMat[keep, ]
    y$E <- exprsMat
    fit <- limma::lmFit(y, design = design)
    fit <- limma::eBayes(fit, trend = TRUE, robust = TRUE)
    temp <- limma::topTable(fit, n = Inf, adjust.method = "BH", coef = 2)
    if (!is.null(temp$ID)) {
      temp <- temp[!duplicated(temp$ID),]
      rownames(temp) <- temp$ID
    }
    temp$meanExprs.1 <- meanExprs[rownames(temp), 1]
    temp$meanExprs.2 <- meanExprs[rownames(temp), 2]
    temp$meanPct.1 <- meanPct[rownames(temp), 1]
    temp$meanPct.2 <- meanPct[rownames(temp), 2]
    return(temp)
  })
  names(tt) <- levels(cellTypes)
  return(tt)
}


doTest <- function(exprsMat, cellTypes) {
  # input must be normalised, log-transformed data
  message("t-test multicore")
  cty <- droplevels(as.factor(cellTypes))
  names(cty) <- colnames(exprsMat)
  
  tt <- mclapply(c(1:nlevels(cty)), mc.cores =3, function(i){
  
    tmp_celltype <- (ifelse(cty == levels(cty)[i], 1, 0))
    
    temp <- t(apply(exprsMat, 1, function(x) {
      x1 <- x[tmp_celltype == 0]
      x2 <- x[tmp_celltype == 1]
      
      res <- stats::t.test(x2, y=x1)
      return(c(stats=res$statistic,
               pvalue=res$p.value))
    }))
    temp <- as.data.frame(temp)
    temp$adj.pvalue <- stats::p.adjust(temp$pvalue, method = "BH")
    return(temp)
  })
  names(tt) <- levels(cty)
  return(tt)
}
