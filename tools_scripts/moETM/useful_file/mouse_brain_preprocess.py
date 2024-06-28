import anndata as ad
from scipy.sparse import csc_matrix, coo_matrix
import numpy as np
import pandas as pd

###### preprcoess rna data

adata = ad.read_text("./GSM4156610_brain.rna.counts.txt")
adata = adata.transpose()
X = csc_matrix(adata.X)
obs = list(adata.obs.index)
obs_new = []
for n in obs:
    n = n.replace(',','.')
    obs_new.append(n)

adata = ad.AnnData(X=X, obs=pd.DataFrame(index=obs_new), var=adata.var)
adata.write("./rna_count.h5ad")

del adata, X

##### preprcoess atac data

f = open('./GSM4156599_brain.barcodes.txt')
obs_bc = []
line = f.readline()
obs_bc.append(line[:-1])
while line:
    line = f.readline()
    if line=='':
        break
    obs_bc.append(line[:-1])

f.close()

f = open('./GSM4156599_brain.peaks.bed')
var_name = []
line = f.readline()
var_name.append(('-').join(line.rstrip().split('\t')))
while line:
    line = f.readline()
    if line=='':
        break
    var_name.append(('-').join(line.rstrip().split('\t')))

f.close()

f = open('./GSM4156599_brain.counts.txt')
line = f.readline()
line = f.readline()
num_obs = int(line.split(' ')[1])
num_var = int(line.split(' ')[0])
row = []
col = []
value = []
while line:
    line = f.readline()
    if line == '':
        break
    row.append(int(line.split(' ')[1]))
    col.append(int(line.split(' ')[0]))
    value.append(int(line.split(' ')[-1][:-1]))

row = np.array(row)
col = np.array(col)
value = np.array(value)

X = coo_matrix((value, (row-1,col-1)), shape=(num_obs, num_var))
adata = ad.AnnData(X=X.tocsc(), obs=pd.DataFrame(index=obs_bc), var=pd.DataFrame(index=var_name))
adata.write("./atac_count.h5ad")
del adata, X

#############################################################################################
f = open('./GSM4156599_brain.barcodes.txt')
obs_bc = []
line = f.readline()
obs_bc.append(line[:-1])
while line:
    line = f.readline()
    if line=='':
        break
    obs_bc.append(line[:-1])

f.close()


f = open('./GSM4156599_brain_celltype.txt')
line = f.readline()
atac_bc = []
rna_bc = []
cell_type = []
while line:
    line = f.readline()
    if line=='':
        break
    List = line.split('\t')
    atac_bc.append(List[0])
    rna_bc.append(List[1])
    cell_type.append(List[2][:-1])

f.close()
rna2atac_bc = dict(zip(rna_bc, atac_bc))
rna2celltype = dict(zip(rna_bc, cell_type))

adata_gex = ad.read_h5ad("./rna_count.h5ad")
adata_atac = ad.read_h5ad("./atac_count.h5ad")
barcode_gex = list(adata_gex.obs.index)

FLAG = []
ATAC_barcode_new = []
GEX_barcode_new = []
Cell_type_new = []
for sample in obs_bc:
    FLAG.append(barcode_gex.index(sample))
    ATAC_barcode_new.append(rna2atac_bc[sample])
    GEX_barcode_new.append(sample)
    Cell_type_new.append(rna2celltype[sample])

X = adata_gex.X[np.array(FLAG)]
var_name = list(adata_gex.var.index)
adata = ad.AnnData(X=X, obs=pd.DataFrame(index=GEX_barcode_new), var=pd.DataFrame(index=var_name))
adata.obs['cell_type'] = Cell_type_new
adata.write("./rna_count_new.h5ad")

X = adata_atac.X
var_name = list(adata_atac.var.index)
adata = ad.AnnData(X=X, obs=pd.DataFrame(index=ATAC_barcode_new), var=pd.DataFrame(index=var_name))
adata.obs['cell_type'] = Cell_type_new
adata.write("./atac_count_new.h5ad")

############################################################################################
adata_gex = ad.read_h5ad("./rna_count_new.h5ad")
adata_atac = ad.read_h5ad("./atac_count_new.h5ad")


