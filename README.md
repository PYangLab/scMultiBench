# scMultiBench
Multi-task benchmarking of single-cell multimodal omics integration methods

Single-cell multimodal omics technologies have empowered the profiling of complex biological systems at a resolution and scale that were previously unattainable. These biotechnologies have propelled the fast-paced innovation and development of data integration methods, leading to a critical need for their systematic categorisation, evaluation, and benchmark. Navigating and selecting the most pertinent integration approach poses a significant challenge, contingent upon the tasks relevant to the study goals and the combination of modalities and batches present in the data at hand. Understanding how well each method performs multiple tasks, including dimension reduction, batch correction, cell type classification and clustering, imputation, feature selection, and spatial registration, and at which combinations will help guide this decision. This study aims to develop a much-needed guideline on choosing the most appropriate method for single-cell multimodal omics data analysis through a systematic categorisation and comprehensive benchmarking of current methods.

<img width=100% src="https://github.com/PYangLab/scMultiBench/blob/main/figure/main.png"/>

## Integration Tools

In this benchmark, we evaluated 34 integration methods across the four data integration categories on 53 datasets. In particular, we include 17 vertical integration methods, 11 diagonal integration tools, 7 mosaic integration tools, and 14 cross integration tools. Tools that are compared include:

Vertical Integration:
- [Matilda](https://github.com/PYangLab/Matilda) Github Version: 7d71480
- [SnapCCESS](https://github.com/PYangLab/SnapCCESS) v0.2.1
- [Concerto](https://github.com/melobio/Concerto-reproducibility) Github Version: ab1fc7f
- [totalVI](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/totalVI.html) v0.20.3
- [Seurat_WNN](https://satijalab.org/seurat/articles/weighted_nearest_neighbor_analysis) v4.1.1.9001 
- [scMSI](https://github.com/ChengmingZhang-CAS/scMSI-master) Github Version: dffcbb2
- [moETM](https://github.com/manqizhou/moETM) Github Version: ad89fe2
- [MOFA+](https://biofam.github.io/MOFA2/) v1.6.0
- [Multigrate](https://multigrate.readthedocs.io/en/latest/index.html) v0.0.2
- [scMoMaT](https://github.com/PeterZZQ/scMoMaT) v0.2.2
- [UINMF](http://htmlpreview.github.io/?https://github.com/welch-lab/liger/blob/master/vignettes/UINMF_vignette.html) v2.0.1
- [sciPENN](https://github.com/jlakkis/sciPENN) v1.0.0
- [MIRA](https://github.com/cistrome/MIRA) v2.1.0
- [iPOLNG](https://github.com/cuhklinlab/iPoLNG) v0.0.2
- [UnitedNet](https://github.com/LiuLab-Bioelectronics-Harvard/UnitedNet) Github Version: 3689da8
- [scMDC](https://github.com/xianglin226/scMDC/tree/v1.0.0) Github Version:  43b0c3a
- [scMM](https://github.com/kodaim1115/scMM) Github Version: c5c8579

Diagonal Integration:




Mosaic Integration:

Cross Integration:

## Metrics
