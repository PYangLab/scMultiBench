## Evaluation Pipeline

# Dimension Reduction, Batch Correction, Clustering
This page provides guidance on evaluating performance after obtaining results from specific methods. Since the metrics for dimension reduction, clustering, and batch correction follows the scib package pipeline, we will discuss them collectively.
The followings figures show the pipelines for dimension reduction, batch correction, and clustering, respectively.
<img width=65% src="https://github.com/PYangLab/scMultiBench/blob/main/figure/DR.png"/>
<img width=65% src="https://github.com/PYangLab/scMultiBench/blob/main/figure/BC.png"/>
<img width=80% src="https://github.com/PYangLab/scMultiBench/blob/main/figure/CLU.png"/>


To execute the metrics for dimension reduction, clustering, and batch correction, the user can follow these steps:
```
cd evaluation_pipelines/scib_metrics
python scib_metrics.py --data_path "../../data/dr&bc/embedding/embedding.h5" --cty_path "../../data/dr&bc/embedding/cty1.csv" "../../data/dr&bc/embedding/cty2.csv" "../../data/dr&bc/embedding/cty3.csv" --cluster_path "../../data/clustering/embedding/sinfonia_clustering.h5" --batch_cluster_path "../../data/clustering/embedding/sinfonia_clustering_batch.h5" --save_path "../../results/scib_metrics/"
```
This command includes various parameters:

--data_path: Specifies the path to the file containing embedding results obtained from various integration methods. In this example, the data is located at "../../data/dr&bc/embedding/embedding.h5".

--cty_path: Indicates the paths to the real label information files from the original datasets. Multiple cty files can be provided, with each representing a different batch. In this command, three cty files (3 batches) are specified.

--cluster_path: Specifies the clustering result used for cell-type clustering metrics.

--batch_cluster_path: Specifies the clustering result used for batch-mixing metrics.

--save_path: Determines the directory where the results of the script will be saved. In this case, the results will be saved to "../../results/scib_metrics/".

After running the above commands, the results is 
```
NMI_cluster/label           0.72662257
ARI_cluster/label           0.614753361
ASW_label                   0.569600554
ASW_label/batch             0.918011439
isolated_label_F1           0.974358974
isolated_label_silhouette   0.865638424
graph_conn                  0.910545501
kBET                        0.664840253
iLISI                       0.703958323
cLISI                       0.986387374
```

The primary package used in `scib_metrics.py` is scIB. To tailor the evaluation metrics to your specific needs, enable, disable, or replace the individual metric calls in that script.

# Classification
For the classification task, a linear classifier is employed. The following figure shows the pipeline of the classification task.
<img width=75% src="https://github.com/PYangLab/scMultiBench/blob/main/figure/CLA.png"/>

To execute the classification and obtain the results, the following command should be run:
```
cd evaluation_pipelines/classification/MLP_classification
python main.py --data_path1 "../../../data/classification/demo_data/data1.h5" --data_path2 "../../../data/classification/demo_data/data2.h5" --cty_path1 "../../../data/classification/demo_data/cty1.csv" --cty_path2 "../../../data/classification/demo_data/cty2.csv" --save_path "../../../results/classification/demo/"
``` 
This command includes various parameters:

--data_path1: Specifies the reference dataset for classification, sourced from a specific batch of integrated embeddings.

--data_path2: Specifies the query dataset for classification. It is another batch of the same integrated embeddings.

--cty_path1: Indicates the cell type information for the reference dataset.

--cty_path2: Indicates the cell type information for the query dataset.

--save_path: Determines the location where the results, including the ground truth labels and predicted labels for the query data, will be stored.

Then, the results 'predict.csv' and groundtruth 'query.csv' will be saved in the directory specified by the 'save_path'.

Upon obtaining the classification results, 4 classification metrics can be employed to evaluate the performance using `classification_metrics.Rmd` file. To apply this file, it's necessary for users to modify the file path using their own file path. After running the complete script, the results for the example data are displayed as follows:

```
"Overall Accuracy: 0.78511354079058"
"Average Accuracy: 0.705289835367602"
"F1 Score: 0.697196367552268"
"Sensitivity: 0.705289835367602"
"Specificity: 0.733608424476035"
```

# Feature Selection 
In the feature selection process, we first obtain the importance score by running each method which applicable for feature selection. Following that, we compute the specificity and reproducibility for each respective method. Additionally, we apply clustering and classification techniques using the top features identified. The process for clustering and classification is similar to the one previously described, hence it will not be demonstrated here. Our primary focus will be on illustrating how to calculate specificity and reproducibility. The following figure shows the pipeline of the feature selection task.

<img width=60% src="https://github.com/PYangLab/scMultiBench/blob/main/figure/FS.png"/>

To calculate specificity, users can execute the `evaluation_pipelines/fs/marker_intersection.Rmd` file. It's important for users to modify the file path in the script to their own path. Upon running this script, it will output the pairwise top marker intersections across different cell types. A smaller intersection indicates higher specificity, signifying the distinctiveness of the selected markers.

To calculate reproducibility, use `evaluation_pipelines/fs/marker_correlation.Rmd` after running each feature-selection method on downsampled subsets. The reproducibility of the results under different data percentages is evaluated using the Pearson correlation coefficient. Higher Pearson coefficient indicates better reproducibility.

# Imputation

In the imputation process, we initially employ various imputation methods to create the imputed results. Subsequently, we utilize both the imputed data and the original dataset (maybe normalised, depends on different methods) to calculate metrics such as Mean Squared Error (MSE), pFCS, and pDES. The following figure shows the pipeline of the imputation task.

<img width=65% src="https://github.com/PYangLab/scMultiBench/blob/main/figure/IMP.png"/>

We provide imputation metric scripts in `evaluation_pipelines/imputation/`. To illustrate these metrics, we have provided an example dataset in the `data/imputation` directory.

The `sMSE.Rmd` script focuses on calculating the Mean Squared Error (MSE) between the imputed data and the ground truth data. This provides a quantitative measure of the imputation accuracy.

The second part of the script involves employing 'modelGeneVar' and 'getTopHVGs' functions to identify the top 100 highly variable genes (HVGs) in the ground truth data. After selecting these HVGs, we then apply them to both the imputed and ground truth data to evaluate their correlation.

The third part of the script operates similarly to the second. The key distinction lies in the selection of markers: instead of using the top 100 HVGs, we utilize the union of the top 5 markers as identified by the Limma method. This approach allows for an alternative perspective on the correlation and imputation quality, based on different gene selection criteria.

# Spatial Registration

For spatial registration, metric scripts can be found in `evaluation_pipelines/spatial_registration/`. Three specific metrics are highlighted: Pairwise Alignment Accuracy (PAA), Spatial Coherence Score (SCS), and Label Transfer ARI (LTARI). The following figure shows the pipeline of the spatial registration task.

<img width=50% src="https://github.com/PYangLab/scMultiBench/blob/main/figure/SR.png"/>
