# scMultiBench
Multi-task benchmarking of single-cell multimodal omics integration methods

Single-cell multimodal omics technologies have empowered the profiling of complex biological systems at a resolution and scale that were previously unattainable. These biotechnologies have propelled the fast-paced innovation and development of data integration methods, leading to a critical need for their systematic categorisation, evaluation, and benchmark. Navigating and selecting the most pertinent integration approach poses a significant challenge, contingent upon the tasks relevant to the study goals and the combination of modalities and batches present in the data at hand. Understanding how well each method performs multiple tasks, including dimension reduction, batch correction, cell type classification and clustering, imputation, feature selection, and spatial registration, and at which combinations will help guide this decision. This study aims to develop a much-needed guideline on choosing the most appropriate method for single-cell multimodal omics data analysis through a systematic categorisation and comprehensive benchmarking of current methods.

<img width=100% src="https://github.com/PYangLab/scMultiBench/blob/main/figure/main.png"/>

## Installation
Matilda is developed using PyTorch 1.9.1 and requires >=1 GPU to run. We recommend using conda enviroment to install and run Matilda. We assume conda is installed. You can use the provided environment or install the environment by yourself accoring to your hardware settings. Note the following installation code snippets were tested on a Ubuntu system (v20.04) with NVIDIA GeForce 3090 GPU. The installation process needs about 15 minutes.
