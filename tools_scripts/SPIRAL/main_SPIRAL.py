
import os
import time
import glob
import torch
import scipy
import warnings
import anndata
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.neighbors
import umap.umap_ as umap
from spiral.layers import *
from spiral.utils import *
from sklearn.cluster import KMeans
from spiral.main import SPIRAL_integration
from Process.Spatial_Net import Cal_Spatial_Net
from spiral.CoordAlignment import CoordAlignment
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# %%
#robjects.r('install.packages("mclust", repos="http://cran.r-project.org")')

# %%
parser = argparse.ArgumentParser('spiral')
parser.add_argument('--data_dir', default='../unified_data/SCC/patient_2/', help='path to the data directory')
parser.add_argument('--save_dir', default='./aligned_slices/', help='path to save the output data')
args = parser.parse_args()

# %%
def load_slices_h5ad(data_dir):
    slices = []
    file_paths = glob.glob(data_dir + "*.h5ad")
    for file_path in file_paths:
        slice_i = sc.read_h5ad(file_path)
        
        if scipy.sparse.issparse(slice_i.X):
            slice_i.X = slice_i.X.toarray()
        
        n_counts = slice_i.obs['n_genes']
        ground_truth = slice_i.obs['Ground_Truth']
        slice_i.obs = pd.DataFrame({'n_counts': n_counts, 'Ground Truth': ground_truth})
        slice_i.var = pd.DataFrame({'n_counts': slice_i.var['n_cells']})
        #slice_i.var = pd.DataFrame({'n_counts': slice_i.var['n_counts']})
        slices.append(slice_i)
    
    return slices



# %%
def create_flags(num_slices,file_names):
    extracted_numbers = [name.split('_')[0] for name in file_names]
    sample_name = extracted_numbers
    IDX=np.arange(0,num_slices)
    flags=str(sample_name[IDX[0]])
    for i in np.arange(1,len(IDX)):
        flags=flags+'-'+str(sample_name[IDX[i]])
    flags=flags+"_"
    return sample_name, flags

# %%
# https://github.com/guott15/SPIRAL/blob/main/Demo/GenerateEdges.ipynb
def Generate_Edges(dirs,slices,sample_name,flags):
    knn=6
    processed_slices = [slice.copy() for slice in slices]

    for j , slice in enumerate(processed_slices):
        sample1=sample_name[j]

        # the following three rows are for matching the names of obs
        cells=[str(sample_name[j])+'-'+i for i in slice.obs_names]
        prefix = str(sample_name[j]) + '-'
        slice.obs_names = [prefix + obs_name for obs_name in slice.obs_names]
        Cal_Spatial_Net(slice, rad_cutoff=None, k_cutoff=knn, model='KNN', verbose=True)
        if 'highly_variable' in slice.var.columns:
            adata_Vars =  slice[:, slice.var['highly_variable']]
        else:
            adata_Vars = slice
        features = pd.DataFrame(adata_Vars.X[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)
        cells = np.array(features.index)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        if 'Spatial_Net' not in slice.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

        Spatial_Net = slice.uns['Spatial_Net']
        G_df = Spatial_Net.copy()
        if os.path.exists(dirs+"gtt_input_scanpy/"):
            print("path exit")
        else:
            os.makedirs(dirs+"gtt_input_scanpy/")
        np.savetxt(dirs+"gtt_input_scanpy/"+flags+str(sample1)+"_edge_KNN_"+str(knn)+".csv",G_df.values[:,:2],fmt='%s')
        

# %%
# https://github.com/guott15/SPIRAL/blob/main/Demo/preprocess.ipynb

def label_position_features(dirs,slices,file_names,flags):
    extracted_numbers = [name.split('_')[0] for name in file_names]
    sample_name = extracted_numbers
    IDX=np.arange(0,len(slices))
    VF=[]
    MAT=[]
    for k in np.arange(len(IDX)):
        adata = slices[k]
        adata.var_names_make_unique()
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.obs['batch']=str(sample_name[IDX[k]])
        cells=[str(sample_name[IDX[k]])+'-'+i for i in adata.obs_names]
        mat1=pd.DataFrame(adata.X,columns=adata.var_names,index=cells)
        coord1=pd.DataFrame(adata.obsm['spatial'],columns=['x','y'],index=cells)
        meta1=adata.obs[['Ground Truth', 'batch']]
        meta1.columns=['celltype','batch']
        meta1.index=cells
        meta1.to_csv(dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[k]])+"_label-1.txt")
        coord1.to_csv(dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[k]])+"_positions-1.txt")
        MAT.append(mat1)
        VF=np.union1d(VF,adata.var_names[adata.var['highly_variable']])
    for i in np.arange(len(IDX)):
        mat = MAT[i]
        new_mat = pd.DataFrame(0.0, index=mat.index, columns=VF)
        new_mat.update(mat)
        new_mat.to_csv(dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[i]])+"_features-1.txt")

# %%
# https://github.com/guott15/SPIRAL/blob/main/Demo/run_spiral_DLPFC.ipynb
def process_1(data_dir,dirs,file_names,num_slices):
    sample_name, flags = create_flags(num_slices,file_names)


    slices = load_slices_h5ad(data_dir)

    
    unique_layers = set()
    for slice in slices:
        unique_layers.update(slice.obs['Ground Truth'].unique())

    n_clust = len(unique_layers)

    Generate_Edges(dirs,slices,sample_name, flags)
    label_position_features(dirs,slices,file_names,flags)
    extracted_numbers = [name.split('_')[0] for name in file_names]
    SEP=','
    net_cate='_KNN_'
    rad=150
    knn=6

    N_WALKS=knn
    WALK_LEN=1
    N_WALK_LEN=knn
    NUM_NEG=knn

    feat_file=[]
    edge_file=[]
    meta_file=[]
    coord_file=[]
    flags=''
    flags1=str(sample_name[0])
    for i in range(1,len(sample_name)):
        flags1=flags1+'-'+str(sample_name[i])
    for i in range(len(sample_name)):
        feat_file.append(dirs+"gtt_input_scanpy/"+flags1+'_'+str(sample_name[i])+"_features-1.txt")
        edge_file.append(dirs+"gtt_input_scanpy/"+flags1+'_'+str(sample_name[i])+"_edge_KNN_"+str(knn)+".csv")
        meta_file.append(dirs+"gtt_input_scanpy/"+flags1+'_'+str(sample_name[i])+"_label-1.txt")
        coord_file.append(dirs+"gtt_input_scanpy/"+flags1+'_'+str(sample_name[i])+"_positions-1.txt")
        flags=flags+'_'+str(sample_name[i])
    N=pd.read_csv(feat_file[0],header=0,index_col=0).shape[1]
    if (len(sample_name)==2):
        M=1
    else:
        M=len(sample_name)
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='The seed of initialization.')
    parser.add_argument('--AEdims', type=list, default=[N,[512],32], help='Dim of encoder.')
    parser.add_argument('--AEdimsR', type=list, default=[32,[512],N], help='Dim of decoder.')
    parser.add_argument('--GSdims', type=list, default=[512,32], help='Dim of GraphSAGE.')
    parser.add_argument('--zdim', type=int, default=32, help='Dim of embedding.')
    parser.add_argument('--znoise_dim', type=int, default=4, help='Dim of noise embedding.')
    parser.add_argument('--CLdims', type=list, default=[4,[],M], help='Dim of classifier.')
    parser.add_argument('--DIdims', type=list, default=[28,[32,16],M], help='Dim of discriminator.')
    parser.add_argument('--beta', type=float, default=1.0, help='weight of GraphSAGE.')
    parser.add_argument('--agg_class', type=str, default=MeanAggregator, help='Function of aggregator.')
    parser.add_argument('--num_samples', type=str, default=knn, help='number of neighbors to sample.')

    parser.add_argument('--N_WALKS', type=int, default=N_WALKS, help='number of walks of random work for postive pairs.')
    parser.add_argument('--WALK_LEN', type=int, default=WALK_LEN, help='walk length of random work for postive pairs.')
    parser.add_argument('--N_WALK_LEN', type=int, default=N_WALK_LEN, help='number of walks of random work for negative pairs.')
    parser.add_argument('--NUM_NEG', type=int, default=NUM_NEG, help='number of negative pairs.')

    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Size of batches to train.') ####512 for withon donor;1024 for across donor###
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
    parser.add_argument('--alpha1', type=float, default=N, help='Weight of decoder loss.')
    parser.add_argument('--alpha2', type=float, default=1, help='Weight of GraphSAGE loss.')
    parser.add_argument('--alpha3', type=float, default=1, help='Weight of classifier loss.')
    parser.add_argument('--alpha4', type=float, default=1, help='Weight of discriminator loss.')
    parser.add_argument('--lamda', type=float, default=1, help='Weight of GRL.')
    parser.add_argument('--Q', type=float, default=10, help='Weight negative loss for sage losss.')

    params,unknown=parser.parse_known_args()

    return params,feat_file,edge_file,meta_file,coord_file,sample_name,flags,n_clust

# %%
def process_2(params,feat_file,edge_file,meta_file,coord_file,sample_name,flags,n_clust,spot_size=20):
    dirs = './txt_file/'
    SPII=SPIRAL_integration(params,feat_file,edge_file,meta_file)
    SPII.train()
    # if not os.path.exists(dirs+"model/"):
    #     os.makedirs(dirs+"model/")
    # model_file=dirs+"model/SPIRAL"+flags+"_model_"+str(SPII.params.batch_size)+".pt"
    # torch.save(SPII.model.state_dict(),model_file)
    SPII.model.eval()
    all_idx=np.arange(SPII.feat.shape[0])
    all_layer,all_mapping=layer_map(all_idx.tolist(),SPII.adj,len(SPII.params.GSdims))
    all_rows=SPII.adj.tolil().rows[all_layer[0]]
    all_feature=torch.Tensor(SPII.feat.iloc[all_layer[0],:].values).float().cuda()
    all_embed,ae_out,clas_out,disc_out=SPII.model(all_feature,all_layer,all_mapping,all_rows,SPII.params.lamda,SPII.de_act,SPII.cl_act)
    [ae_embed,gs_embed,embed]=all_embed
    [x_bar,x]=ae_out
    embed=embed.cpu().detach()
    names=['GTT_'+str(i) for i in range(embed.shape[1])]
    embed1=pd.DataFrame(np.array(embed),index=SPII.feat.index,columns=names)
    if not os.path.exists(dirs+"gtt_output/"):
        os.makedirs(dirs+"gtt_output/")
        
    embed_file=dirs+"gtt_output/SPIRAL"+flags+"_embed_"+str(SPII.params.batch_size)+".csv"
    embed1.to_csv(embed_file)
    meta=SPII.meta.values

    # embed_new=torch.cat((torch.zeros((embed.shape[0],SPII.params.znoise_dim)),embed.iloc[:,SPII.params.znoise_dim:]),dim=1)
    embed_new = torch.cat((torch.zeros((embed.shape[0], SPII.params.znoise_dim)), embed[:, SPII.params.znoise_dim:]), dim=1)

    xbar_new=np.array(SPII.model.agc.ae.de(embed_new.cuda(),nn.Sigmoid())[1].cpu().detach())
    xbar_new1=pd.DataFrame(xbar_new,index=SPII.feat.index,columns=SPII.feat.columns)
    xbar_new1.to_csv(dirs+"gtt_output/SPIRAL"+flags+"_correct_"+str(SPII.params.batch_size)+".csv")
    
    meta=SPII.meta.values




    ann=anndata.AnnData(SPII.feat)
    ann.obsm['spiral']=embed1.iloc[:,SPII.params.znoise_dim:].values
    sc.pp.neighbors(ann,use_rep='spiral')

    
    # res1=0.5 ####adjust to make sure 7 clusters
    # res2=0.5
    # sc.tl.leiden(ann,resolution=res1)
    # sc.tl.louvain(ann,resolution=res2)
    ann = mclust_R(ann, used_obsm='spiral', num_cluster=n_clust)

    ann.obs['batch']=SPII.meta.loc[:,'batch'].values
    ub=np.unique(ann.obs['batch'])
    sc.tl.umap(ann)
    coord=pd.read_csv(coord_file[0],header=0,index_col=0)
    for i in np.arange(1,len(sample_name)):
        coord=pd.concat((coord,pd.read_csv(coord_file[i],header=0,index_col=0)))

    coord.columns=['y','x']
    ann.obsm['spatial']=coord.loc[ann.obs_names,:].values
    cluster_file=dirs+"gtt_output/SPIRAL"+flags+"_mclust.csv"
    pd.DataFrame(ann.obs['mclust']).to_csv(cluster_file)
    knn=6

    ann.obs['SPIRAL']=ann.obs['mclust']
    ann.obs['SPIRAL_refine']=ann.obs['SPIRAL']
    ub=np.unique(ann.obs['batch'])
    for i in range(len(ub)):
        idx=np.where(ann.obs['batch']==ub[i])[0]
        ann1=ann[idx,:]
        sample_id=ann1.obs_names
        pred=ann1.obs['SPIRAL']
        dis=euclidean_distances(ann1.obsm['spatial'],ann1.obsm['spatial'])
        refined_pred=refine(sample_id, pred, dis, num_nbs=knn)
        ann.obs['SPIRAL_refine'][idx]=refined_pred
    
    cluster_file_save=dirs+"metrics/spiral"+flags+"_mclust_modify.csv"
    if not os.path.exists(dirs+"metrics"):
        os.makedirs(dirs+"metrics")
    pd.DataFrame(ann.obs['SPIRAL_refine']).to_csv(cluster_file_save)



    return ann,embed_file,cluster_file

# %%
#https://github.com/guott15/SPIRAL/blob/main/Demo/run_spiral_DLPFC.ipynb

def coord_align(ann,flags,meta_file,coord_file,embed_file,cluster_file,sample_name,data_dir,file_names):
    knn=6
    dirs = './txt_file/'
    ann.obs['SPIRAL']=ann.obs['mclust']
    ann.obs['SPIRAL_refine']=ann.obs['SPIRAL']
    ub=np.unique(ann.obs['batch'])
    for i in range(len(ub)):
        idx=np.where(ann.obs['batch']==ub[i])[0]
        ann1=ann[idx,:]
        sample_id=ann1.obs_names
        pred=ann1.obs['SPIRAL']
        dis=euclidean_distances(ann1.obsm['spatial'],ann1.obsm['spatial'])
        refined_pred=refine(sample_id, pred, dis, num_nbs=knn)
        ann.obs['SPIRAL_refine'][idx]=refined_pred

        
    cluster_file_save=dirs+"metrics/spiral"+flags+"_mclust_modify.csv"
    pd.DataFrame(ann.obs['SPIRAL_refine']).to_csv(cluster_file_save)
    clust_cate='louvain'
    input_file=[meta_file,coord_file,embed_file,cluster_file]
    output_dirs=dirs+"gtt_output/SPIRAL_alignment/"
    if not os.path.exists(output_dirs):
        os.makedirs(output_dirs)
    ub=np.unique(ann.obs['batch'])

  

    
    alpha=0.5
    types="weighted_mean"
    R_dirs="/opt/miniforge3/bin/R"

    
    CA=CoordAlignment(input_file=input_file,output_dirs=output_dirs,ub=ub,flags=flags,clust_cate=clust_cate,R_dirs=R_dirs,alpha=alpha,types=types)

    

    
    New_Coord=CA.New_Coord
    New_Coord.to_csv(output_dirs+"new_coord"+flags+"_modify.csv")
    ann.obsm['aligned_spatial']=New_Coord.loc[ann.obs_names,:].values
    
    # Store the aligned coordinates into the orginal slices
    
    aligned_coords = ann.obsm['aligned_spatial'][:, :2]  

    
    
    obs_names = ann.obs.index.to_numpy()  
 

    aligned_df = pd.DataFrame(aligned_coords, columns=['x_aligned', 'y_aligned'])
    aligned_df['obs_name'] = obs_names

    
    print(aligned_df)

    
    aligned_df['obs_name'] = aligned_df['obs_name'].apply(lambda x: '-'.join(x.split('-')[1:]))


    slices = load_slices_h5ad(data_dir)
        
    start_index = 0  

    for slice_data in slices:
        slice_data.obs.index.name = 'obs_name'
        
        slice_size = len(slice_data.obs)
        
        aligned_subset = aligned_df.iloc[start_index:start_index + slice_size].copy()
        
        slice_data.obs.reset_index(inplace=True)
        
        slice_data.obs = pd.merge(slice_data.obs, aligned_subset, on='obs_name', how='left')
        
        slice_data.obsm['spatial'] = slice_data.obs[['x_aligned', 'y_aligned']].values
        
        start_index += slice_size
    return slices



# %%
def whole_process(data_dir,dirs,file_names,num_slices,n_category):
    dirs = './txt_file/'
    params,feat_file,edge_file,meta_file,coord_file,sample_name,flags,n_clust = process_1(data_dir,dirs,file_names,num_slices)
    ann,embed_file,cluster_file = process_2(params,feat_file,edge_file,meta_file,coord_file,sample_name,flags,n_clust)
    print("DONE")
    slices_coordinated = coord_align(ann,flags,meta_file,coord_file,embed_file,cluster_file,sample_name,data_dir,file_names)
    

    return slices_coordinated

# %%
def combine(data_dir, save_dir):
    dirs = './txt_file/'
    file_names = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    num_slices = len(file_names)
    slices = load_slices_h5ad(data_dir)
    
    unique_layers = set()
    for slice in slices:
        unique_layers.update(slice.obs['Ground Truth'].unique())
    n_category = len(unique_layers)
    
    slices_coordinated = whole_process(data_dir, dirs, file_names, num_slices, n_category)
    
  
    os.makedirs(save_dir, exist_ok=True)
    

    save_subdir = os.path.join(save_dir, "spiral_aligned_slices")
    os.makedirs(save_subdir, exist_ok=True)

    for i, slice in enumerate(slices_coordinated):
        save_path = os.path.join(save_subdir, f"aligned_slice_{i}.h5ad")
        sc.write(save_path, slice)
    
    return slices_coordinated

# %%
combine(args.data_dir,args.save_dir)


#python spiral.py --data_dir ../../../../dataset/benchmark_dataset/cross\ integration/SCC/patient_2 --save_dir './aligned_slices'
