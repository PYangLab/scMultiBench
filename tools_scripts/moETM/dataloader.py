
import numpy as np
import torch
import anndata as ad
import scanpy as sc


import gc


def load_nips_rna_atac_dataset(mod_file_path, gene_encoding):
    adata = ad.read_h5ad(mod_file_path)

    feature_gex_index = np.array(adata.var.feature_types) == 'GEX'
    feature_adt_index = np.array(adata.var.feature_types) == 'ATAC'

    gex = adata[:, feature_gex_index].copy()
    atac = adata[:, feature_adt_index].copy()
    del adata

    gc.collect()

    index = []
    for i in range(gex.shape[1]):
        if gex.var['gene_id'][i] != gene_encoding['gene_id'][i]:
            print('Warning')
        else:
            A = bool(gene_encoding['is_gene_coding'][i])
            index.append(A)

    gex = gex[:, index].copy()

    # gex.var.to_csv('./gex_name.csv')
    # atac.var.to_csv('./atac_name.csv')

    adata_mod1 = gex.copy()
    adata_mod1.X = adata_mod1.layers['counts']
    del gex

    adata_mod2 = atac.copy()
    adata_mod2.X = adata_mod2.layers['counts']
    del atac

    gc.collect()

    # obs = adata.obs
    # adata_mod1 = ad.AnnData(X=adata.layers['counts'][:, feature_gex_index], obs=obs)
    # adata_mod2 = ad.AnnData(X=adata.layers['counts'][:, feature_adt_index], obs=obs)

    adata_mod1_original = ad.AnnData.copy(adata_mod1)
    adata_mod2_original = ad.AnnData.copy(adata_mod2)

    sc.pp.normalize_total(adata_mod1, target_sum=1e4)
    sc.pp.log1p(adata_mod1)
    sc.pp.highly_variable_genes(adata_mod1)
    index = adata_mod1.var['highly_variable'].values

    adata_mod1 = ad.AnnData.copy(adata_mod1_original)
    adata_mod1 = adata_mod1[:, index].copy()

    del adata_mod1_original
    gc.collect()

    sc.pp.normalize_total(adata_mod2, target_sum=1e4)
    sc.pp.log1p(adata_mod2)
    sc.pp.highly_variable_genes(adata_mod2)
    index = adata_mod2.var['highly_variable'].values

    adata_mod2 = ad.AnnData.copy(adata_mod2_original)
    del adata_mod2_original
    gc.collect()

    adata_mod2 = adata_mod2[:, index].copy()

    return adata_mod1, adata_mod2

def prepare_nips_dataset(adata_gex, adata_mod2,
                         batch_col = 'batch',
                        ):

    batch_index = np.array(adata_gex.obs[batch_col].values)
    unique_batch = list(np.unique(batch_index))
    batch_index = np.array([unique_batch.index(xs) for xs in batch_index])

    obs = adata_gex.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)
    adata_gex = ad.AnnData(X=adata_gex.X, obs=obs)

    obs = adata_mod2.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)

    X = adata_mod2.X
    adata_mod2 = ad.AnnData(X=X, obs=obs)

    Index = np.array(X.sum(1)>0).squeeze()

    adata_gex = adata_gex[Index]
    obs = adata_gex.obs
    adata_gex = ad.AnnData(X=adata_gex.X, obs=obs)

    adata_mod2 = adata_mod2[Index]
    obs = adata_mod2.obs
    adata_mod2 = ad.AnnData(X=adata_mod2.X, obs=obs)

    return adata_gex, adata_mod2

def data_process_moETM(adata_mod1, adata_mod2):

    train_adata_mod1 = adata_mod1
    train_adata_mod2 = adata_mod2

    ########################################################
    # Training dataset
    X_mod1 = np.array(train_adata_mod1.X.todense())
    X_mod2 = np.array(train_adata_mod2.X.todense())
    batch_index = np.array(train_adata_mod1.obs['batch_indices'])

    X_mod1 = X_mod1 / X_mod1.sum(1)[:, np.newaxis]
    X_mod2 = X_mod2 / X_mod2.sum(1)[:, np.newaxis]

    X_mod1_train_T = torch.from_numpy(X_mod1).float()
    X_mod2_train_T = torch.from_numpy(X_mod2).float()
    batch_index_train_T = torch.from_numpy(batch_index).to(torch.int64)

    del X_mod1, X_mod2, batch_index

    return X_mod1_train_T, X_mod2_train_T, batch_index_train_T, train_adata_mod1

def data_process_moETM_cross_prediction(adata_mod1, adata_mod2, n_sample):
    from sklearn.utils import resample

    Index = np.arange(0, n_sample)
    train_index = resample(Index, n_samples=n_sample)
    test_index = np.array(list(set(range(n_sample)).difference(train_index)))

    train_adata_mod1 = adata_mod1[train_index]
    obs = train_adata_mod1.obs
    X = train_adata_mod1.X
    train_adata_mod1 = ad.AnnData(X=X, obs=obs)

    train_adata_mod2 = adata_mod2[train_index]
    obs = train_adata_mod2.obs
    X = train_adata_mod2.X
    train_adata_mod2 = ad.AnnData(X=X, obs=obs)

    test_adata_mod1 = adata_mod1[test_index]
    obs = test_adata_mod1.obs
    X = test_adata_mod1.X
    test_adata_mod1 = ad.AnnData(X=X, obs=obs)

    test_adata_mod2 = adata_mod2[test_index]
    obs = test_adata_mod2.obs
    X = test_adata_mod2.X
    test_adata_mod2 = ad.AnnData(X=X, obs=obs)

    ########################################################
    # Training dataset
    X_mod1 = np.array(train_adata_mod1.X.todense())
    X_mod2 = np.array(train_adata_mod2.X.todense())
    batch_index = np.array(train_adata_mod1.obs['batch_indices'])

    X_mod1 = X_mod1 / X_mod1.sum(1)[:, np.newaxis]
    X_mod2 = X_mod2 / X_mod2.sum(1)[:, np.newaxis]

    X_mod1_train_T = torch.from_numpy(X_mod1).float()
    X_mod2_train_T = torch.from_numpy(X_mod2).float()
    batch_index_train_T = torch.from_numpy(batch_index).to(torch.int64).cuda()

    # Testing dataset
    X_mod1 = np.array(test_adata_mod1.X.todense())
    X_mod2 = np.array(test_adata_mod2.X.todense())
    batch_index = np.array(test_adata_mod1.obs['batch_indices'])

    sum1 = X_mod1.sum(1)
    sum2 = X_mod2.sum(1)

    X_mod1 = X_mod1 / X_mod1.sum(1)[:, np.newaxis]
    X_mod2 = X_mod2 / X_mod2.sum(1)[:, np.newaxis]

    X_mod1_test_T = torch.from_numpy(X_mod1).float()
    X_mod2_test_T = torch.from_numpy(X_mod2).float()
    batch_index_test_T = torch.from_numpy(batch_index).to(torch.int64)


    del X_mod1, X_mod2, batch_index

    return X_mod1_train_T, X_mod2_train_T, batch_index_train_T, X_mod1_test_T, X_mod2_test_T, batch_index_test_T, test_adata_mod1, train_adata_mod1, sum1, sum2


def load_nips_dataset_rna_protein_dataset(mod_file_path, gene_encoding, protein_encoding):

    adata = ad.read_h5ad(mod_file_path)

    feature_gex_index = np.array(adata.var.feature_types) == 'GEX'
    feature_adt_index = np.array(adata.var.feature_types) == 'ADT'

    adata_mod1 = adata[:, feature_gex_index].copy()
    adata_mod2 = adata[:, feature_adt_index].copy()

    adata_mod1.X = adata_mod1.layers['counts']
    adata_mod2.X = adata_mod2.layers['counts']

    index = []
    for i in range(adata_mod1.shape[1]):
        if adata_mod1.var.index[i] != gene_encoding['X'][i]:
            print('Warning')
        else:
            index.append(bool(gene_encoding['is_gene_coding'][i]))

    adata_mod1_original = adata_mod1[:, index].copy()
    adata_mod1 = adata_mod1[:, index].copy()

    sc.pp.normalize_total(adata_mod1, target_sum=1e4)
    sc.pp.log1p(adata_mod1)
    sc.pp.highly_variable_genes(adata_mod1)  # n_top_genes
    index = adata_mod1.var['highly_variable'].values

    adata_mod1_original = adata_mod1_original[:, index].copy()

    index = []
    for i in range(adata_mod2.shape[1]):
        if adata_mod2.var.index[i] != protein_encoding['X'][i]:
            print('Warning')
        else:
            index.append(bool(protein_encoding['is_protein_coding'][i]))

    adata_mod2 = adata_mod2[:, index].copy()

    return adata_mod1_original, adata_mod2

def load_nips_rna_atac_dataset_with_pathway(mod_file_path, gene_encoding, gene_pathway):
    adata = ad.read_h5ad(mod_file_path)

    feature_gex_index = np.array(adata.var.feature_types) == 'GEX'
    feature_adt_index = np.array(adata.var.feature_types) == 'ATAC'

    gex = adata[:, feature_gex_index].copy()
    atac = adata[:, feature_adt_index].copy()
    del adata

    gc.collect()

    gene_pathway_sum = gene_pathway.sum(0)
    index = []
    for i in range(gex.shape[1]):
        if gex.var['gene_id'][i] != gene_encoding['gene_id'][i]:
            print('Warning')
        else:
            A = bool(gene_encoding['is_gene_coding'][i])
            B = bool(gene_pathway_sum[i])
            index.append(A & B)

    gex = gex[:, index].copy()
    gene_pathway = gene_pathway[:, index].copy()

    adata_mod1 = gex.copy()
    adata_mod1.X = adata_mod1.layers['counts']
    del gex

    adata_mod2 = atac.copy()
    adata_mod2.X = adata_mod2.layers['counts']
    del atac

    gc.collect()

    adata_mod1_original = ad.AnnData.copy(adata_mod1)
    adata_mod2_original = ad.AnnData.copy(adata_mod2)

    sc.pp.normalize_total(adata_mod1, target_sum=1e4)
    sc.pp.log1p(adata_mod1)
    sc.pp.highly_variable_genes(adata_mod1)
    index = adata_mod1.var['highly_variable'].values

    adata_mod1 = ad.AnnData.copy(adata_mod1_original)
    adata_mod1 = adata_mod1[:, index].copy()
    gene_pathway = gene_pathway[:, index].copy()

    del adata_mod1_original
    gc.collect()

    sc.pp.normalize_total(adata_mod2, target_sum=1e4)
    sc.pp.log1p(adata_mod2)
    sc.pp.highly_variable_genes(adata_mod2)
    index = adata_mod2.var['highly_variable'].values

    adata_mod2 = ad.AnnData.copy(adata_mod2_original)
    del adata_mod2_original
    gc.collect()

    adata_mod2 = adata_mod2[:, index].copy()

    return adata_mod1, adata_mod2, gene_pathway