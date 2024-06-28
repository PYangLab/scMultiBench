from math import ceil, floor
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import anndata
from anndata import AnnData


def generate_synthetic(
        batch_size: int = 500,
        n_genes: int = 2000,
        n_proteins: int = 20,
        n_batches: int = 2,
        n_labels: int = 3,
) -> AnnData:
    #  Here samples are drawn from a negative binomial distribution with specified parameters,
    # `n` failures and `p` probability of failure where `n` is > 0 and `p` is in the interval
    #  [0, 1], `n` is equal to diverse dispersion parameter.
    data = np.zeros(shape=(batch_size * n_batches, n_genes))
    mu = np.random.randint(low=1, high=20, size=n_labels)
    p = mu / (mu + 5)
    for i in range(n_batches):
        data[batch_size * i: batch_size * (i + 1), :] = np.random.negative_binomial(5, 1 - p[i],
                                                                                    size=(batch_size, n_genes))
    data = np.random.negative_binomial(5, 0.3, size=(batch_size * n_batches, n_genes))
    mask = np.random.binomial(n=1, p=0.7, size=(batch_size * n_batches, n_genes))
    data = data * mask  # We put the batch index first
    labels = np.random.randint(0, n_labels, size=(batch_size * n_batches,))
    labels = np.array(["label_%d" % i for i in labels])

    batch = []
    for i in range(n_batches):
        batch += ["batch_{}".format(i)] * batch_size

    adata = AnnData(data)
    batch = np.random.randint(high=n_batches, low=0, size=(batch_size * n_batches, 1)).astype(np.float32)
    # adata.obs["batch"] = pd.Categorical(batch)
    adata.obs["batch"] = batch
    adata.obs["labels"] = pd.Categorical(labels)
    adata.uns['n_batch'] = n_batches

    # Protein measurements
    p_data = np.zeros(shape=(adata.shape[0], n_proteins))
    mu = np.random.randint(low=1, high=20, size=n_labels)
    p = mu / (mu + 5)
    for i in range(n_batches):
        p_data[batch_size * i: batch_size * (i + 1), :] = np.random.negative_binomial(5, 1 - p[i],
                                                                                      size=(batch_size, n_proteins))
    p_data = np.random.negative_binomial(5, 0.3, size=(adata.shape[0], n_proteins))
    adata.obsm["protein_expression"] = p_data
    adata.uns["protein_names"] = np.arange(n_proteins).astype(str)

    return adata


def data_splitter(
        adata: AnnData,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
):
    """Split indices in train/test/val sets."""
    n_train, n_val = validate_data_split(adata.n_obs, train_size, validation_size)
    random_state = np.random.RandomState(seed=42)
    permutation = random_state.permutation(adata.n_obs)
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val: (n_val + n_train)]
    test_idx = permutation[(n_val + n_train):]

    train_adata = adata[train_idx]
    val_adata = adata[val_idx]
    if test_idx.shape[0] == 0:
        return train_adata, val_adata
    else:
        test_adata = adata[test_idx]
        return train_adata, val_adata, test_adata


def validate_data_split(
        n_samples: int, train_size: float, validation_size: Optional[float] = None
):
    """
    Check data splitting parameters and return n_train and n_val.

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, train_size={} and validation_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, train_size, validation_size)
        )

    return n_train, n_val


def batch_sampler(
        adata: AnnData,
        batch_size: int,
        shuffle: bool = False,
        drop_last: Union[bool, int] = False, ):
    """
    Custom torch Sampler that returns a list of indices of size batch_size.

    Parameters
    ----------
    adata
        adata to sample from
    batch_size
        batch size of each iteration
    shuffle
        if ``True``, shuffles indices before sampling
    drop_last
        if int, drops the last batch if its length is less than drop_last.
        if drop_last == True, drops last non-full batch.
        if drop_last == False, iterate over all batches.
    """
    if drop_last > batch_size:
        raise ValueError(
            "drop_last can't be greater than batch_size. "
            + "drop_last is {} but batch_size is {}.".format(drop_last, batch_size)
        )

    last_batch_len = adata.n_obs % batch_size
    if (drop_last is True) or (last_batch_len < drop_last):
        drop_last_n = last_batch_len
    elif (drop_last is False) or (last_batch_len >= drop_last):
        drop_last_n = 0
    else:
        raise ValueError("Invalid input for drop_last param. Must be bool or int.")

    if shuffle is True:
        idx = torch.randperm(adata.n_obs).tolist()
    else:
        idx = torch.arange(adata.n_obs).tolist()

    if drop_last_n != 0:
        idx = idx[: -drop_last_n]

    adata_iter = [
        adata[idx[i: i + batch_size]]
        for i in range(0, len(idx), batch_size)
    ]
    return adata_iter
