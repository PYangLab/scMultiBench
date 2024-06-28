import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csgraph import connected_components


def get_root(adata_pre, adata_post, ct_key, pseudotime_key="dpt_pseudotime", dpt_dim=3):
    """Determine root cell for integrated adata based on unintegrated adata
    :param adata_pre: unintegrated adata
    :param adata_post: integrated adata
    :param label_key: column in ``adata_pre.obs`` of the groups used to precompute the trajectory
    :param pseudotime_key: column in ``adata_pre.obs`` in which the pseudotime is saved in.
        Column can contain empty entries, the dataset will be subset to the cells with scores.
    :param dpt_dim: number of diffmap dimensions used to determine root
    """
    n_components, adata_post.obs["neighborhood"] = connected_components(
        csgraph=adata_post.obsp["connectivities"], directed=False, return_labels=True
    )

    start_clust = adata_pre.obs.groupby([ct_key]).mean()[pseudotime_key].idxmin()
    min_dpt = adata_pre.obs[adata_pre.obs[ct_key] == start_clust].index
    which_max_neigh = (
        adata_post.obs["neighborhood"]
        == adata_post.obs["neighborhood"].value_counts().idxmax()
    )
    min_dpt = [
        value for value in min_dpt if value in adata_post.obs[which_max_neigh].index
    ]

    adata_post_ti = adata_post[which_max_neigh]

    min_dpt = [adata_post_ti.obs_names.get_loc(i) for i in min_dpt]

    # compute Diffmap for adata_post
    sc.tl.diffmap(adata_post_ti)

    # determine most extreme cell in adata_post Diffmap
    min_dpt_cell = np.zeros(len(min_dpt))
    for dim in np.arange(dpt_dim):

        diffmap_mean = adata_post_ti.obsm["X_diffmap"][:, dim].mean()
        diffmap_min_dpt = adata_post_ti.obsm["X_diffmap"][min_dpt, dim]

        # count opt cell
        # if len(diffmap_min_dpt) == 0:
        #     raise RootCellError("No root cell in largest component")

        # choose optimum function
        if len(diffmap_min_dpt) > 0 and diffmap_min_dpt.mean() < diffmap_mean:
            opt = np.argmin
        else:
            opt = np.argmax

        min_dpt_cell[opt(diffmap_min_dpt)] += 1

    # root cell is cell with max vote
    return min_dpt[np.argmax(min_dpt_cell)], adata_post_ti