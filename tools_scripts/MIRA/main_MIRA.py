import mira
import os, sys
import anndata
import scanpy as sc
import numpy as np
import h5py
from scipy.sparse import csr_matrix
import argparse
import time
from logger import *

parser = argparse.ArgumentParser("MIRA")
parser.add_argument('--rna', metavar='rna_batch', nargs='+', default=[], help='path to rna train data')
parser.add_argument('--atac', metavar='adt_batch', nargs='+', default=[], help='path to atac train data')
parser.add_argument('--save_path', metavar='save_path', default='NULL', help='path to save the output data')
parser.add_argument('--latent_dim', type=int, default=20, help="latent embedding dimension, default 20")
parser.add_argument('--epoch', type=int, default=24, help="max epoch for model training, default 24")
parser.add_argument('--load_model', action='store_true', help="If to use the existing topic models")
parser.add_argument('--filter_atac_cells', action='store_true', help="If to filter atac cells")
parser.add_argument('--rna_topic_modal_path', default='NULL', help='path to load the RNA topic model')
parser.add_argument('--atac_topic_modal_path', default='NULL', help='path to load the ATAC topic model')
parser.add_argument('--acc', type=str, default='gpu', choices=["cpu", "gpu", "tpu", "ipu", "hpu","mps","auto"], help="accelerator for training, default cpu")
parser.add_argument('--save_name', type=str, default='mira', help='filename for saving the outputs (without extension)')


args = parser.parse_args()

begin_time = time.time()
def data_loader(path):
    with h5py.File(path, "r") as f:
        X = np.array(f['matrix/data']).astype(np.float32).transpose()
        counts = csr_matrix(X)
        X = anndata.AnnData(counts)
    return X


def train_topic_model(data, categorical_cov=None, mod='expression', args=None):
    end_key = 'endogenous_peaks' if mod=='accessibility' else None
    hv_key = 'highly_variable' if mod=='expression' else None
    ct_layer = 'counts' if mod=='expression' else None
    atc_enc = "light" if args.acc=='cpu' else "skipDAN"
    
    model = mira.topics.make_model(
    data.n_obs, data.n_vars, # helps MIRA choose reasonable values for some hyperparameters which are not tuned.
    feature_type = mod,
    highly_variable_key=hv_key,
    counts_layer=ct_layer,
    categorical_covariates=categorical_cov,
    endogenous_key=end_key,
    atac_encoder=atc_enc
    )
    
    print("-----> Set learning rate bound as default (1e-3, 0.1) ")
    model.set_learning_rates(1e-3, 0.1) # for larger datasets, the default of 1e-3, 0.1 usually works well.
    #model.plot_learning_rate_bounds(figsize=(7,3))  

    print("-----> Hyperparameter tunning via Gradient-based method  ")
    topic_contributions = mira.topics.gradient_tune(model, data)
    NUM_TOPICS = list(np.array(topic_contributions)<0.05).index(True)
    mira.pl.plot_topic_contributions(topic_contributions, NUM_TOPICS)
    
    print("-----> Training the topic model")
    model = model.set_params(num_topics = NUM_TOPICS).fit(data)

    sv_nm = os.path.join(args.save_path, args.save_name+'-'+mod+'_topic_model.pth')
    print("-----> Saving the topic model to "+sv_nm)
    model.save(sv_nm)
    return sv_nm

    

def run_mira(file_paths, args):
    np.random.seed(0)
    
    rna_path=file_paths['rna_path']
    atac_path=file_paths['atac_path']

    
    assert len(rna_path) == len(atac_path)
    

    if len(rna_path)>1:
        #lgr.info("----> Concatenating RNA data batches....  ("+ str(len(rna_path)) + " batches )")
        USE_BATCH=True
        rna_data = anndata.concat({'batch'+str(i+1): data_loader(rna_path[i]) for i in range(len(rna_path))}, label='batch', index_unique=':')
    else:
        USE_BATCH=False
        rna_data = data_loader(rna_path[0])
    
#    lgr.info("-----> Preprocessing RNA data ")
        
    sc.pp.filter_genes(rna_data, min_cells=15)
    rawdata = rna_data.X.copy()
    sc.pp.normalize_total(rna_data, target_sum=1e4)
    sc.pp.log1p(rna_data)
    sc.pp.highly_variable_genes(rna_data, min_disp = 0.5)
    rna_data.layers['counts'] = rawdata

    sc.tl.pca(rna_data)
    sc.pp.neighbors(rna_data, n_pcs=6)
    sc.tl.umap(rna_data, min_dist = 0.2, negative_sample_rate=0.2)
    
    
    cat_key = 'batch' if USE_BATCH else None
    #--> train RNA topic model if required, otherwise load the model based on the provided path
    if not args.load_model:
        #lgr.info("-----> Initializing RNA topic model ")
        rna_topic_model_pth = train_topic_model(data=rna_data, categorical_cov=cat_key, mod='expression', args=args)
        
    else:    
        assert args.rna_topic_modal_path is not None
        rna_topic_model_pth =  args.rna_topic_modal_path
        
    #lgr.info("-----> Loading RNA topic model from "+ rna_topic_model_pth)
    rna_topic_model = mira.topic_model.load_model(rna_topic_model_pth)
    
    

    if len(atac_path)>1:
        #lgr.info("----> Concatenating ATAC data batches....  ("+ str(len(atac_path)) + " batches )")
        atac_data = anndata.concat({'batch'+str(i+1): data_loader(atac_path[i]) for i in range(len(atac_path))}, label='batch', index_unique=':')
    else:
        atac_data = data_loader(atac_path[0])
    
    #lgr.info("-----> Preprocessing ATAC data ")
    sc.pp.filter_genes(atac_data, min_cells = 30)
    sc.pp.calculate_qc_metrics(atac_data, inplace=True, log1p=False)
    if args.filter_atac_cells:
        #lgr.info("------> Filtering the ATAC cells based on min_genes=1000")
        sc.pp.filter_cells(atac_data, min_genes=1000)
    
    atac_data.var['endogenous_peaks'] = np.random.rand(atac_data.shape[1]) <= min(1e5/atac_data.shape[1], 1)
    ###
    
    #--> train RNA topic model if required, otherwise load the model based on the provided path
    if not args.load_model:
        #lgr.info("-----> Initializing ATAC topic model ")
        print(cat_key)
        print(args)
        atac_topic_model_pth = train_topic_model(data=atac_data, categorical_cov=cat_key, mod='accessibility', args=args)
        
    else:    
        assert args.atac_topic_modal_path is not None
        atac_topic_model_pth =  args.atac_topic_modal_path

    atac_topic_model = mira.topic_model.load_model(atac_topic_model_pth)
    atac_topic_model.predict(atac_data)
    rna_topic_model.predict(rna_data)
    
    rna_topic_model.get_umap_features(rna_data, box_cox=0.25)
    atac_topic_model.get_umap_features(atac_data, box_cox=0.25)
    
    print(rna_data.shape, "rna_data.shape")
    print(atac_data.shape, "atac_data.shape")
    
    rna_data, atac_data = mira.utils.make_joint_representation(rna_data, atac_data)
    sc.pp.neighbors(rna_data, use_rep = 'X_joint_umap_features', metric = 'manhattan',
               n_neighbors = 20)
    print(rna_data)
               
    distances = rna_data.obsp['distances']
    connectivities = rna_data.obsp['connectivities']
    print(distances.shape, connectivities.shape)
    return distances, connectivities


file_paths = {
    "rna_path": args.rna,
    "atac_path": args.atac
}


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
    print("create path for saving")
else:
    print("the saving path exits")
distances, connectivities=run_mira(file_paths, args)
end_time = time.time()
all_time = end_time - begin_time

file = h5py.File(args.save_path+"/distancess.h5", 'w')
file.create_dataset('data', data=distances.toarray())
file.close()

file = h5py.File(args.save_path+"/connectivities.h5", 'w')
file.create_dataset('data', data=connectivities.toarray())
file.close()
