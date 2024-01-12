# scMultiBench
Multi-task benchmarking of single-cell multimodal omics integration methods

Single-cell multimodal omics technologies have empowered the profiling of complex biological systems at a resolution and scale that were previously unattainable. These biotechnologies have propelled the fast-paced innovation and development of data integration methods, leading to a critical need for their systematic categorisation, evaluation, and benchmark. Navigating and selecting the most pertinent integration approach poses a significant challenge, contingent upon the tasks relevant to the study goals and the combination of modalities and batches present in the data at hand. Understanding how well each method performs multiple tasks, including dimension reduction, batch correction, cell type classification and clustering, imputation, feature selection, and spatial registration, and at which combinations will help guide this decision. This study aims to develop a much-needed guideline on choosing the most appropriate method for single-cell multimodal omics data analysis through a systematic categorisation and comprehensive benchmarking of current methods.

<img width=100% src="https://github.com/PYangLab/scMultiBench/blob/main/figure/main.png"/>

## Integration Tools

In this benchmark, we evaluated 34 integration methods across the four data integration categories on 53 datasets. In particular, we include 17 vertical integration methods, 11 diagonal integration tools, 7 mosaic integration tools, and 14 cross integration tools. The installation environment is set up according to the respective tutorials. Tools that are compared include:

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
- [scBridge](https://github.com/XLearning-SCU/scBridge) Github Version: ff17561
- [Portal](https://github.com/YangLabHKUST/Portal) v1.0.2
- [VIPCCA](https://github.com/jhu99/vipcca) v0.2.7
- [scJoint](https://github.com/SydneyBioX/scJoint)  Github Version: cbbfa5d
- [SCALEX](https://github.com/jsxlei/SCALEX) v1.0.2
- [Conos](https://github.com/kharchenkolab/conos) v1.4.6
- [LIGER](https://github.com/welch-lab/liger) v2.0.1
- [iNMF](http://htmlpreview.github.io/?https://github.com/welch-lab/liger/blob/master/vignettes/UINMF_vignette.html) v2.0.1
- [GLUE](https://github.com/tanlabcode/GLUER) Github Version: 192bb6e
- [Seurat v3](https://satijalab.org/seurat/articles/seurat5_atacseq_integration_vignette) v4.1.1.9001 
- [MultiMAP](https://github.com/Teichlab/MultiMAP) v0.0.1

Mosaic Integration:
- [MultiVI](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/MultiVI_tutorial.html) v0.20.3
- [StabMap](https://github.com/MarioniLab/StabMap) v0.1.8
- [Cobolt](https://github.com/epurdom/cobolt) v1.0.1
- [Seurat v5](https://satijalab.org/seurat/articles/seurat5_integration_bridge) v4.1.1.9001 
- [UINMF](http://htmlpreview.github.io/?https://github.com/welch-lab/liger/blob/master/vignettes/UINMF_vignette.html) v2.0.1
- [scMoMaT](https://github.com/PeterZZQ/scMoMaT) v0.2.2
- [Multigrate](https://multigrate.readthedocs.io/en/latest/index.html) v0.0.2

Cross Integration:
- [totalVI](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/totalVI.html) v0.20.3
- [scMoMaT](https://github.com/PeterZZQ/scMoMaT) v0.2.2
- [UnitedNet](https://github.com/LiuLab-Bioelectronics-Harvard/UnitedNet) Github Version: 3689da8
- [sciPENN](https://github.com/jlakkis/sciPENN) v1.0.0
- [Concerto](https://github.com/melobio/Concerto-reproducibility) Github Version: ab1fc7f
- [scMDC](https://github.com/xianglin226/scMDC/tree/v1.0.0) Github Version:  43b0c3a
- [StabMap](https://github.com/MarioniLab/StabMap) v0.1.8
- [UINMF](http://htmlpreview.github.io/?https://github.com/welch-lab/liger/blob/master/vignettes/UINMF_vignette.html) v2.0.1
- [scMM](https://github.com/kodaim1115/scMM) Github Version: c5c8579
- [MOFA+](https://biofam.github.io/MOFA2/) v1.6.0
- [Multigrate](https://multigrate.readthedocs.io/en/latest/index.html) v0.0.2
- [PASTE](https://github.com/raphael-group/paste) v1.4.0
- [eggplant](https://github.com/almaan/eggplant) v0.2.3
- [GPSA](https://github.com/andrewcharlesjones/spatial-alignment) v0.8


## Evaluation Pipeline



## License

This project is covered under the Apache 2.0 License.
