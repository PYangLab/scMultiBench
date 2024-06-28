import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp_sparse
import torch
import anndata
import warnings
import sys
from typing import Iterable
import os
os.environ["OMP_NUM_THREADS"] = '1'

from rich.console import Console
from rich.progress import track as track_base
from tqdm import tqdm as tqdm_base
from scipy.linalg import norm
from math import ceil, floor

# from scvi import settings
# from scvi._compat import Literal
from collections.abc import Iterable as IterableClass
from typing import Dict, List, Optional, Sequence, Tuple, Union
from anndata._core.sparse_dataset import SparseDataset
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

Number = Union[int, float]


# ======================================== data.utils ========================================
def _check_nonnegative_integers(
        data: Union[pd.DataFrame, np.ndarray, sp_sparse.spmatrix, h5py.Dataset]
):
    """Approximately checks values of data to ensure it is count data."""

    # for backed anndata
    if isinstance(data, h5py.Dataset) or isinstance(data, SparseDataset):
        data = data[:100]

    if isinstance(data, np.ndarray):
        data = data
    elif issubclass(type(data), sp_sparse.spmatrix):
        data = data.data
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    else:
        raise TypeError("data type not understood")

    n = len(data)
    inds = np.random.permutation(n)[:20]
    check = data.flat[inds]
    return ~np.any(_is_not_count(check))


def _is_not_count(d):
    return d < 0 or d % 1 != 0


def read_txt(filename, dtypefloat=True):
    data = {}
    f = open(filename, "r")
    line = f.readlines()
    flag = 0
    for l in line:
        flag += 1
        t = l.split()
        if flag == 1:
            title = [eval(t[k]) for k in range(len(t))]
        else:
            if dtypefloat:
                data[eval(t[0])] = [float(t[k]) for k in range(1, len(t))]
            else:
                data[eval(t[0])] = [eval(t[k]) for k in range(1, len(t))]
    f.close()
    df = pd.DataFrame(data, index=title)
    return df.T


# ======================================== module.utils ========================================
def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c) + (1.0 - lmbd) / 2.0 * torch.pow(c, 2)


def regularizer_l12(c, lmbd=1.0):
    return torch.norm(torch.norm(c, p=1, dim=1), p=2)


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


def get_anchor_index(data, n_pcs=50, n_clusters=1000):
    """get the anchor sample index."""
    n_pcs = min(data.shape[1], n_pcs)
    pca = PCA(n_components=n_pcs, svd_solver="arpack", random_state=0)
    z = pca.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters).fit(z)
    dist_mat = cdist(z, kmeans.cluster_centers_)
    index = dist_mat.argmin(axis=0)
    return index


def get_sample_index(data, train_size=0.9, validation_size=0.1, n_pcs=50, n_sample_ref=1000):
    """get the splited sample index."""
    n_samples = data.shape[0]
    n_train = ceil(train_size * n_samples)
    n_val = floor(n_samples * validation_size)
    random_state = np.random.RandomState(seed=42)
    permutation = random_state.permutation(data.shape[0])
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val: (n_val + n_train)]
    test_idx = permutation[(n_val + n_train):]
    train_data = data[train_idx]
    n_pcs = min(train_data.shape[1], n_pcs)
    pca = PCA(n_components=n_pcs, svd_solver="arpack", random_state=0)
    z = pca.fit_transform(train_data)
    kmeans = KMeans(n_clusters=n_sample_ref).fit(z)
    dist_mat = cdist(z, kmeans.cluster_centers_)
    anchor_idx = dist_mat.argmin(axis=0)
    return train_idx, val_idx, test_idx, anchor_idx


def get_top_coeff(coeff, non_zeros=1000, ord=1):
    coeff_top = torch.tensor(coeff)
    N, M = coeff_top.shape
    non_zeros = min(M, non_zeros)
    values, indices = torch.topk(torch.abs(coeff_top), dim=1, k=non_zeros)
    coeff_top[coeff_top < values[:, -1].reshape(-1, 1)] = 0
    coeff_top = coeff_top.data.numpy()
    if ord is not None:
        coeff_top = coeff_top / norm(coeff_top, ord=ord, axis=1, keepdims=True)
    return coeff_top


def get_sparse_rep(coeff, non_zeros=1000):
    N, M = coeff.shape
    non_zeros = min(M, non_zeros)
    _, index = torch.topk(torch.abs(coeff), dim=1, k=non_zeros)

    val = coeff.gather(1, index).reshape([-1]).cpu().data.numpy()
    indicies = index.reshape([-1]).cpu().data.numpy()
    indptr = [non_zeros * i for i in range(N + 1)]

    C_sparse = sp_sparse.csr_matrix((val, indicies, indptr), shape=coeff.shape)
    return C_sparse


def get_knn_Aff(C_sparse_normalized, k=3, mode='symmetric'):
    C_knn = kneighbors_graph(C_sparse_normalized, k, mode='connectivity', include_self=False, n_jobs=10)
    if mode == 'symmetric':
        Aff_knn = 0.5 * (C_knn + C_knn.T)
    elif mode == 'reciprocal':
        Aff_knn = C_knn.multiply(C_knn.T)
    else:
        raise Exception("Mode must be 'symmetric' or 'reciprocal'")
    return Aff_knn


# ======================================== model.utils ========================================
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def _get_var_names_from_setup_anndata(adata):
    """Gets var names by checking if using raw."""
    var_names = adata.var_names
    return var_names


def _get_batch_code_from_category(
        adata: anndata.AnnData, category: Sequence[Union[Number, str]]
):
    if not isinstance(category, IterableClass) or isinstance(category, str):
        category = [category]

    categorical_mappings = adata.uns["_scvi"]["categorical_mappings"]
    batch_mappings = categorical_mappings["_scvi_batch"]["mapping"]
    batch_code = []
    for cat in category:
        if cat is None:
            batch_code.append(None)
        elif cat not in batch_mappings:
            raise ValueError('"{}" not a valid batch category.'.format(cat))
        else:
            batch_loc = np.where(batch_mappings == cat)[0][0]
            batch_code.append(batch_loc)
    return batch_code


def init_library_size(
        adata: anndata.AnnData, n_batch: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes and returns library size.

    Parameters
    ----------
    adata
        AnnData object setup with `scvi`.
    n_batch
        Number of batches.

    Returns
    -------
    type
        Tuple of two 1 x n_batch ``np.ndarray`` containing the means and variances
        of library size in each batch in adata.

        If a certain batch is not present in the adata, the mean defaults to 0,
        and the variance defaults to 1. These defaults are arbitrary placeholders which
        should not be used in any downstream computation.
    """
    # data = adata.X
    data = adata.layers["rna_expression"]
    batch_indices = adata.obs["batch"].to_numpy().reshape(-1, 1)

    library_log_means = np.zeros(n_batch)
    library_log_vars = np.ones(n_batch)

    for i_batch in np.unique(batch_indices):
        idx_batch = np.squeeze(batch_indices == i_batch)
        batch_data = data[
            idx_batch.nonzero()[0]
        ]  # h5ad requires integer indexing arrays.
        sum_counts = batch_data.sum(axis=1)
        masked_log_sum = np.ma.log(sum_counts)
        if np.ma.is_masked(masked_log_sum):
            warnings.warn(
                "This dataset has some empty cells, this might fail inference."
                "Data should be filtered with `scanpy.pp.filter_cells()`"
            )

        log_counts = masked_log_sum.filled(0)
        library_log_means[int(i_batch)] = np.mean(log_counts).astype(np.float32)
        library_log_vars[int(i_batch)] = np.var(log_counts).astype(np.float32)

    return library_log_means.reshape(1, -1), library_log_vars.reshape(1, -1)


def get_protein_priors(adata_raw, batch_mask, n_cells=100):
    """Compute an empirical prior for protein background."""

    print("Computing empirical prior initialization for protein background.")
    # cats = adata.uns["_scvi"]["categorical_mappings"]["_scvi_batch"]["mapping"]
    # codes = np.arange(len(cats))
    adata = adata_raw[adata_raw.obsm["protein_expression"].sum(1) > 0].copy()
    batch = adata.obs["batch"]
    batch_avg_mus, batch_avg_scales = [], []
    for b in np.unique(batch):
        # can happen during online updates
        # the values of these batches will not be used
        num_in_batch = np.sum(batch == b)
        if num_in_batch == 0:
            batch_avg_mus.append(0)
            batch_avg_scales.append(1)
            continue
        pro_exp = adata.obsm["protein_expression"]
        if isinstance(pro_exp, pd.DataFrame):
            pro_exp = np.asarray(pro_exp)
        pro_exp = pro_exp[batch == b]
        # non missing
        if batch_mask is not None:
            pro_exp = pro_exp[:, batch_mask[b]]
            if pro_exp.shape[1] < 5:
                print(
                    f"Batch {b} has too few proteins to set prior, setting randomly."
                )
                batch_avg_mus.append(0.0)
                batch_avg_scales.append(0.05)
                continue

        # a batch is missing because it's in the reference but not query data
        # for scarches case, these values will be replaced by original state dict
        if pro_exp.shape[0] == 0:
            batch_avg_mus.append(0.0)
            batch_avg_scales.append(0.05)
            continue

        cells = np.random.choice(np.arange(pro_exp.shape[0]), size=n_cells)
        if isinstance(pro_exp, pd.DataFrame):
            pro_exp = pro_exp.to_numpy()
        pro_exp = pro_exp[cells]
        gmm = GaussianMixture(n_components=2)
        mus, scales = [], []
        # fit per cell GMM
        for c in pro_exp:
            try:
                gmm.fit(np.log1p(c.reshape(-1, 1)))
            # when cell is all 0
            except ConvergenceWarning:
                mus.append(0)
                scales.append(0.5)
                continue

            means = gmm.means_.ravel()
            sorted_fg_bg = np.argsort(means)
            mu = means[sorted_fg_bg].ravel()[0]
            covariances = gmm.covariances_[sorted_fg_bg].ravel()[0]
            scale = np.sqrt(covariances)
            mus.append(mu)
            scales.append(scale)

        # average distribution over cells
        batch_avg_mu = np.mean(mus)
        batch_avg_scale = np.sqrt(np.sum(np.square(scales)) / (n_cells ** 2))

        batch_avg_mus.append(batch_avg_mu)
        batch_avg_scales.append(batch_avg_scale)

    # repeat prior for each protein
    n_proteins = adata.obsm["protein_expression"].shape[1]
    batch_avg_mus = np.array(batch_avg_mus, dtype=np.float32).reshape(1, -1)
    batch_avg_scales = np.array(batch_avg_scales, dtype=np.float32).reshape(1, -1)
    batch_avg_mus = np.tile(batch_avg_mus, (n_proteins, 1))
    batch_avg_scales = np.tile(batch_avg_scales, (n_proteins, 1))

    warnings.resetwarnings()

    return batch_avg_mus, batch_avg_scales


def parse_use_gpu_arg(
        use_gpu: Optional[Union[str, int, bool]] = None,
        return_device=True,
):
    """
    Parses the use_gpu arg in codebase.

    Returned gpus are is compatible with PytorchLightning's gpus arg.
    If return_device is True, will also return the device.

    Parameters
    ----------
    use_gpu
        Use default GPU if available (if None or True), or index of GPU to use (if int),
        or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
    return_device
        If True, will return the torch.device of use_gpu.
    """
    gpu_available = torch.cuda.is_available()
    if (use_gpu is None and not gpu_available) or (use_gpu is False):
        gpus = 0
        device = torch.device("cpu")
    elif (use_gpu is None and gpu_available) or (use_gpu is True):
        current = torch.cuda.current_device()
        device = torch.device(current)
        gpus = [current]
    elif isinstance(use_gpu, int):
        device = torch.device(use_gpu)
        gpus = [use_gpu]
    elif isinstance(use_gpu, str):
        device = torch.device(use_gpu)
        # changes "cuda:0" to "0,"
        gpus = use_gpu.split(":")[-1] + ","
    else:
        raise ValueError("use_gpu argument not understood.")

    if return_device:
        return gpus, device
    else:
        return gpus

#
# def track(
#     sequence: Iterable,
#     description: str = "Working...",
#     disable: bool = False,
#     style: Literal["rich", "tqdm"] = None,
#     **kwargs
# ):
#     """
#     Progress bar with `'rich'` and `'tqdm'` styles.
#
#     Parameters
#     ----------
#     sequence
#         Iterable sequence.
#     description
#         First text shown to left of progress bar.
#     disable
#         Switch to turn off progress bar.
#     style
#         One of ["rich", "tqdm"]. "rich" is interactive
#         and is not persistent after close.
#     **kwargs
#         Keyword args to tqdm or rich.
#
#     Examples
#     --------
#     >>> from scvi.utils import track
#     >>> my_list = [1, 2, 3]
#     >>> for i in track(my_list): print(i)
#     """
#     if style is None:
#         style = settings.progress_bar_style
#     if style not in ["rich", "tqdm"]:
#         raise ValueError("style must be one of ['rich', 'tqdm']")
#     if disable:
#         return sequence
#     if style == "tqdm":
#         # fixes repeated pbar in jupyter
#         # see https://github.com/tqdm/tqdm/issues/375
#         if hasattr(tqdm_base, "_instances"):
#             for instance in list(tqdm_base._instances):
#                 tqdm_base._decr_instances(instance)
#         return tqdm_base(sequence, desc=description, file=sys.stdout, **kwargs)
#     else:
#         in_colab = "google.colab" in sys.modules
#         force_jupyter = None if not in_colab else True
#         console = Console(force_jupyter=force_jupyter)
#         return track_base(sequence, description=description, console=console, **kwargs)