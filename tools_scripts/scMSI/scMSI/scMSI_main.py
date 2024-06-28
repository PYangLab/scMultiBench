import logging
import warnings
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from anndata import AnnData
from tqdm import tqdm
from torch.distributions import Normal, Poisson
from scipy.linalg import norm
from scipy.sparse import csr_matrix, vstack

from scMSI.utils import get_anchor_index, set_seed
from scMSI.dataloaders import data_splitter, batch_sampler
from scMSI.utils import (
    # _get_batch_code_from_category,
    # _get_var_names_from_setup_anndata,
    init_library_size,
    # cite_seq_raw_counts_properties,
    get_protein_priors
)
from scMSI.scMSIVAE import RNAProteinSVAE, CiteRnaSVAE, RNAPeakSVAE, RNASVAE, ProteinSVAE, PeakSVAE, ImageSVAE, scVIVAE
from scMSI.scMSIVAE import RNAVAE, ProteinVAE, RNAProteinSE
import torch.utils.data as data_utils

Number = TypeVar("Number", int, float)
set_seed(0)


class SCMSIRNAProtein(nn.Module):
    """
    **Acknowledgments**
    This code is inspired by the python library of scvi-tools.
    Gayoso, A., Lopez, R., Xing, G. et al. A Python library for probabilistic analysis of single-cell omics data.
    Nat Biotechnol 40, 163–166 (2022). https://doi.org/10.1038/s41587-021-01206-w.

    Parameters
    ----------
    adata
        AnnData object.
    n_latent
        Dimensionality of the latent space.
    gene_dispersion
        One of the following:

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    protein_dispersion
        One of the following:

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    protein_likelihood
        One of:

        * ``'nbm'`` - Negative binomial mixture distribution
        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    empirical_protein_background_prior
        Set the initialization of protein background prior empirically. This option fits a GMM for each of
        100 cells per batch and averages the distributions. Note that even with this option set to `True`,
        this only initializes a parameter that is learned during inference. If `False`, randomly initializes.
        The default (`None`), sets this to `True` if greater than 10 proteins are used.
    override_missing_proteins
        If `True`, will not treat proteins with all 0 expression in a particular batch as missing.
    **model_kwargs
        Keyword args for :class:`~scMSI.scMSI_main.scMSIVAE`
    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            gene_dispersion: Literal[
                "gene", "gene-batch", "gene-label", "gene-cell"
            ] = "gene",
            protein_dispersion: Literal[
                "protein", "protein-batch", "protein-label"
            ] = "protein",
            gene_likelihood: Literal["zinb", "nb"] = "nb",
            protein_likelihood: Literal["nbm", "nb", "poisson"] = "nbm",
            latent_distribution: Literal["normal", "ln"] = "normal",
            empirical_protein_background_prior: Optional[bool] = None,
            n_sample_ref: Optional[int] = 100,
            batch_size: Optional[int] = 128,
            **model_kwargs,
    ):
        super(SCMSIRNAProtein, self).__init__()
        self.adata = adata
        if "batch" not in adata.obs.keys():
            adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
        if "n_batch" not in adata.uns.keys():
            adata.uns["n_batch"] = 1

        n_cats_per_cov = (
            adata.uns["n_cats_per_key"]
            if "n_cats_per_key" in adata.uns.keys()
            else None
        )
        n_batch = self.adata.uns["n_batch"]
        library_log_means, library_log_vars = init_library_size(adata, n_batch)
        emp_prior = (
            empirical_protein_background_prior
            if empirical_protein_background_prior is not None
            else (adata.obsm["protein_expression"].shape[1] >= 10)
        )
        if emp_prior:
            batch_mask = None
            prior_mean, prior_scale = get_protein_priors(adata, batch_mask)
        else:
            prior_mean, prior_scale = None, None
        # prior_mean[1] = 5
        # prior_mean[3] = 5
        # prior_mean[7] = 6
        # prior_mean[8] = 3.2

        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=0.9,
            validation_size=0.1,
            use_gpu=False,
        )
        self.train_adata = train_adata
        self.val_adata = val_adata
        self.batch_size = batch_size
        if n_sample_ref > train_adata.shape[0]:
            ref_idx = np.arange(train_adata.shape[0])
        else:
            ref_idx = get_anchor_index(train_adata.X, n_pcs=20, n_clusters=n_sample_ref)
            print(f"Choose {n_sample_ref} anchor samples for self-expressive learning")
        # ref_idx = np.random.permutation(train_adata.shape[1])[0:500]
        self.ref_adata = train_adata[ref_idx, :]

        self.module = RNAProteinSVAE(
            n_input_genes=adata.X.shape[1],
            n_input_proteins=adata.obsm['protein_expression'].shape[1],
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=0,
            n_cats_per_cov=n_cats_per_cov,
            gene_dispersion=gene_dispersion,
            protein_dispersion=protein_dispersion,
            gene_likelihood=gene_likelihood,
            protein_likelihood=protein_likelihood,
            latent_distribution=latent_distribution,
            protein_background_prior_mean=prior_mean,
            protein_background_prior_scale=prior_scale,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            use_observed_lib_size=True,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 4e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 256,
            early_stopping: bool = False,
            check_val_every_n_epoch: Optional[int] = None,
            reduce_lr_on_plateau: bool = True,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            adversarial_classifier: Optional[bool] = None,
            plan_kwargs: Optional[dict] = None,
            weight_decay=1e-6,
            record_loss=True,
            **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        weight_decay
            weight decay
        record_loss
            whether save loss
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.module.cuda(device=device)
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        train_adata = self.train_adata
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_loss = []
        epoch_rec_loss = {}
        epoch_kl_loss = {}
        epoch_se_loss = {}
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            batch_loss = []
            batch_rec_loss = {}
            batch_kl_loss = {}
            batch_se_loss = {}
            create_dict = True
            for train_batch in train_adata_batch:
                input_genes = torch.tensor(train_batch.layers['rna_expression'], dtype=torch.float32)
                input_proteins = torch.tensor(train_batch.obsm['protein_expression'], dtype=torch.float32)
                batch_index = torch.tensor(train_batch.obs['batch'], dtype=torch.float32).reshape(-1, 1)
                if "cat_covs" in train_batch.obsm.keys():
                    cat_covs = torch.tensor(train_batch.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
                else:
                    cat_covs = None
                # batch_index = (batch_index, batch_index)
                # print(len(batch_index))
                _, _, _, all_loss = self.module(input_genes, input_proteins, ref_adata, batch_index=batch_index,
                                                cat_covs=cat_covs)
                loss, reconst_loss, kl_local, se_losses = all_loss
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), 0.001)
                optimizer.step()
                # batch_loss.append(loss.data.numpy())
                if create_dict:
                    for k in reconst_loss.keys():
                        batch_rec_loss[k] = []
                    for k in kl_local.keys():
                        batch_kl_loss[k] = []
                    for k in se_losses.keys():
                        batch_se_loss[k] = []
                    create_dict = False
                batch_loss.append(loss.cpu().data.numpy())
                if record_loss:
                    for k in batch_rec_loss.keys():
                        batch_rec_loss[k].append(torch.mean(reconst_loss[k]).cpu().data.numpy())
                    for k in batch_kl_loss.keys():
                        batch_kl_loss[k].append(torch.mean(kl_local[k]).cpu().data.numpy())
                    for k in batch_se_loss.keys():
                        batch_se_loss[k].append(torch.mean(se_losses[k]).cpu().data.numpy())

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                print("epoch: {}; loss: {} ".format(epoch, np.mean(batch_loss)))
            scheduler.step()
            # self.module.eval()
            if epoch == 0:
                for k in batch_rec_loss.keys():
                    epoch_rec_loss[k] = []
                for k in batch_kl_loss.keys():
                    epoch_kl_loss[k] = []
                for k in batch_se_loss.keys():
                    epoch_se_loss[k] = []

            epoch_loss.append(np.mean(batch_loss))
            if record_loss:
                for k in epoch_rec_loss.keys():
                    epoch_rec_loss[k].append(np.mean(batch_rec_loss[k]))
                for k in epoch_kl_loss.keys():
                    epoch_kl_loss[k].append(np.mean(batch_kl_loss[k]))
                for k in epoch_se_loss.keys():
                    epoch_se_loss[k].append(np.mean(batch_se_loss[k]))
        self.history = dict(
            epoch_loss=epoch_loss,
            epoch_rec_loss=epoch_rec_loss,
            epoch_kl_loss=epoch_kl_loss,
            epoch_se_loss=epoch_se_loss,
        )

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent_gene = []
        latent_pro = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            enc_gene_outputs, enc_pro_outputs = self.module.inference(input_genes, input_proteins,
                                                                      batch_index=batch_index, cat_covs=cat_covs)
            qz_m_gene = enc_gene_outputs["qz_m"]
            qz_v_gene = enc_gene_outputs["qz_v"]
            z_gene = enc_gene_outputs["z"]
            qz_m_pro = enc_pro_outputs["qz_m"]
            qz_v_pro = enc_pro_outputs["qz_v"]
            z_pro = enc_pro_outputs["z"]
            if give_mean:
                samples = Normal(qz_m_gene, qz_v_gene.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z_gene = self.module.encoder_genes.z_transformation(samples)
                z_gene = z_gene.mean(dim=0)
                samples = Normal(qz_m_pro, qz_v_pro.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z_pro = self.module.encoder_proteins.z_transformation(samples)
                z_pro = z_pro.mean(dim=0)
            else:
                z_gene = qz_m_gene
                z_pro = qz_m_pro

            latent_gene += [z_gene.cpu()]
            latent_pro += [z_pro.cpu()]
        lat_gene = torch.cat(latent_gene).numpy()
        lat_pro = torch.cat(latent_pro).numpy()

        return lat_gene, lat_pro

    @torch.no_grad()
    def get_reconstruct_latent(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        ref_z_gene_list = []
        ref_z_pro_list = []
        rec_z_gene_list = []
        rec_z_pro_list = []
        for data in ref_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_proteins = torch.tensor(data.obsm["protein_expression"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                refer_cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                refer_cat_covs = None
            ref_outputs_gene, ref_outputs_pro = self.module.inference(refer_genes, refer_proteins,
                                                                      batch_index=refer_index, cat_covs=refer_cat_covs)
            ref_z_gene_list.append(ref_outputs_gene['z'])
            ref_z_pro_list.append(ref_outputs_pro['z'])
        ref_z_genes = torch.cat(ref_z_gene_list, dim=0)
        ref_z_pros = torch.cat(ref_z_pro_list, dim=0)
        if not give_mean:
            mc_samples = 1
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm["protein_expression"], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            inference_genes, inference_pros = self.module.inference(input_genes, input_proteins, batch_index,
                                                                    n_samples=mc_samples, cat_covs=cat_covs)
            z_genes = inference_genes['z'].mean(dim=0)
            z_pros = inference_pros['z'].mean(dim=0)
            se_gene_outputs, se_pro_outputs = self.module.self_expressiveness(z_genes, z_pros, ref_z_genes, ref_z_pros)
            rec_z_genes = se_gene_outputs['rec_queries']
            rec_z_pros = se_pro_outputs['rec_queries']
            rec_z_gene_list.append(rec_z_genes.cpu())
            rec_z_pro_list.append(rec_z_pros.cpu())
        rec_z_gs = torch.cat(rec_z_gene_list, dim=0).numpy()
        rec_z_ps = torch.cat(rec_z_pro_list, dim=0).numpy()
        return rec_z_gs, rec_z_ps

    @torch.no_grad()
    def get_coeff(
            self,
            adata: Optional[AnnData] = None,
            batch_size: int = 128,
            pro_w=1,
            ord=1,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        coeff_genes_list = []
        coeff_pros_list = []
        ref_adata = self.ref_adata
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            _, senet_outputs, _ = self.module(input_genes, input_proteins, ref_adata, batch_index=batch_index,
                                              cat_covs=cat_covs, compute_loss=False)
            se_gene_outputs, se_pro_outputs = senet_outputs
            coeff_genes_list.append(se_gene_outputs['coeff'])
            coeff_pros_list.append(se_pro_outputs['coeff'])
        coeff_genes = torch.cat(coeff_genes_list, dim=0)
        coeff_proteins = torch.cat(coeff_pros_list, dim=0)
        coeff = coeff_genes + pro_w * coeff_proteins
        coeff_genes = coeff_genes.cpu().numpy()
        coeff_proteins = coeff_proteins.cpu().numpy()
        coeff = coeff.cpu().numpy()
        if ord is not None:
            coeff_genes = coeff_genes / norm(coeff_genes, ord=ord, axis=1, keepdims=True)
            coeff_proteins = coeff_proteins / norm(coeff_proteins, ord=ord, axis=1, keepdims=True)
            coeff = coeff / norm(coeff, ord=ord, axis=1, keepdims=True)
        return coeff_genes, coeff_proteins, coeff

    @torch.no_grad()
    def get_latent_library_size(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.module.training:
            self.module.eval()
        libraries = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            outputs, _ = self.module.inference(input_genes, input_proteins, batch_index, cat_covs=cat_covs)
            ql_m = outputs["ql_m"]
            ql_v = outputs["ql_v"]
            library_gene = outputs["library_gene"]
            if give_mean and (not self.module.use_observed_lib_size):
                untran_l = Normal(ql_m, ql_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                library = torch.exp(untran_l)
                library = library.mean(dim=0)
            else:
                library = library_gene

            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()

    @torch.no_grad()
    def get_normalized_expression(
            self,
            adata=None,
            indices=None,
            n_samples_overall: Optional[int] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            gene_list: Optional[Sequence[str]] = None,
            protein_list: Optional[Sequence[str]] = None,
            library_size: Optional[Union[float, Literal["latent"]]] = 1,
            n_samples: int = 1,
            sample_protein_mixing: bool = False,
            scale_protein: bool = False,
            include_protein_background: bool = False,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        r"""
        Returns the normalized gene expression and protein expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to use in total
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude.
        n_samples
            Get sample scale from multiple samples.
        sample_protein_mixing
            Sample mixing bernoulli, setting background to zero
        scale_protein
            Make protein expression sum to 1
        include_protein_background
            Include background component for protein expression
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
            names as columns. If either n_samples=1 or return_mean=True, defaults to False.
            Otherwise, it defaults to True.

        Returns
        -------
        - **gene_normalized_expression** - normalized expression for RNA
        - **protein_normalized_expression** - normalized expression for proteins

        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is ``(samples, cells, genes)``.
        Otherwise, shape is ``(cells, genes)``. Return type is ``pd.DataFrame`` unless ``return_numpy`` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]
        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = adata.uns['protein_names']
            protein_mask = [True if p in protein_list else False for p in all_proteins]
        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        scale_list_gene = []
        scale_list_pro = []

        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            px_scale = torch.zeros_like(input_genes)
            py_scale = torch.zeros_like(input_proteins)
            if n_samples > 1:
                px_scale = torch.stack(n_samples * [px_scale])
                py_scale = torch.stack(n_samples * [py_scale])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_genes, input_proteins, ref_adata, batch_index,
                                                       cat_covs=cat_covs, compute_loss=False, transform_batch=b)
                px_, py_ = generative_outputs
                if library_size == "latent":
                    px_scale += px_["rate"].cpu()
                else:
                    px_scale += px_["scale"].cpu()
                px_scale = px_scale[..., gene_mask]

                # probability of background
                protein_mixing = 1 / (1 + torch.exp(-py_["mixing"].cpu()))
                if sample_protein_mixing is True:
                    protein_mixing = torch.distributions.Bernoulli(
                        protein_mixing
                    ).sample()
                protein_val = py_["fore_rate"].cpu() * (1 - protein_mixing)
                if include_protein_background is True:
                    protein_val += py_["back_rate"].cpu() * protein_mixing

                if scale_protein is True:
                    protein_val = torch.nn.functional.normalize(
                        protein_val, p=1, dim=-1
                    )
                protein_val = protein_val[..., protein_mask]
                py_scale += protein_val
            px_scale /= len(transform_batch)
            py_scale /= len(transform_batch)
            scale_list_gene.append(px_scale)
            scale_list_pro.append(py_scale)

        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            scale_list_gene = torch.cat(scale_list_gene, dim=1)
            scale_list_pro = torch.cat(scale_list_pro, dim=1)
            # (cells, features, samples)
            scale_list_gene = scale_list_gene.permute(1, 2, 0)
            scale_list_pro = scale_list_pro.permute(1, 2, 0)
        else:
            scale_list_gene = torch.cat(scale_list_gene, dim=0)
            scale_list_pro = torch.cat(scale_list_pro, dim=0)

        if return_mean is True and n_samples > 1:
            scale_list_gene = torch.mean(scale_list_gene, dim=-1)
            scale_list_pro = torch.mean(scale_list_pro, dim=-1)

        scale_list_gene = scale_list_gene.cpu().numpy()
        scale_list_pro = scale_list_pro.cpu().numpy()
        if return_numpy is None or return_numpy is False:
            gene_df = pd.DataFrame(
                scale_list_gene,
                columns=self.adata.var_names[gene_mask],
                index=self.adata.obs_names[indices],
            )
            pro_df = pd.DataFrame(
                scale_list_pro,
                columns=self.adata.uns["protein_names"][protein_mask],
                index=self.adata.obs_names[indices],
            )

            return gene_df, pro_df
        else:
            return scale_list_gene, scale_list_pro

    @torch.no_grad()
    def get_protein_foreground_probability(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            protein_list: Optional[Sequence[str]] = None,
            n_samples: int = 1,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ):
        r"""
        Returns the foreground probability for proteins.

        This is denoted as :math:`(1 - \pi_{nt})` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        - **foreground_probability** - probability foreground for each protein

        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)

        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = adata.uns['protein_names']
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        py_mixings = []
        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            py_mixing = torch.zeros_like(input_pros[..., protein_mask])
            if n_samples > 1:
                py_mixing = torch.stack(n_samples * [py_mixing])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_genes, input_pros, ref_adata, batch_index,
                                                       cat_covs=cat_covs, compute_loss=False, transform_batch=b)
                _, py_ = generative_outputs
                # probability of background
                py_mixing += torch.sigmoid(py_["mixing"])[..., protein_mask].cpu()
            py_mixing /= len(transform_batch)
            py_mixings += [py_mixing]
        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            py_mixings = torch.cat(py_mixings, dim=1)
            # (cells, features, samples)
            py_mixings = py_mixings.permute(1, 2, 0)
        else:
            py_mixings = torch.cat(py_mixings, dim=0)

        if return_mean is True and n_samples > 1:
            py_mixings = torch.mean(py_mixings, dim=-1)

        py_mixings = py_mixings.cpu().numpy()

        if return_numpy is True:
            return 1 - py_mixings
        else:
            pro_names = self.adata.uns["protein_names"]
            foreground_prob = pd.DataFrame(
                1 - py_mixings,
                columns=pro_names[protein_mask],
                index=adata.obs_names[indices],
            )
            return foreground_prob

    @torch.no_grad()
    def get_protein_likelihood_parameters(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        back_rate_list = []
        fore_rate_list = []
        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            _, _, generative_outputs = self.module(input_genes, input_pros, ref_adata, batch_index, cat_covs=cat_covs,
                                                   compute_loss=False)
            _, py_ = generative_outputs
            back_rate_list += [py_['back_rate']]
            fore_rate_list += [py_['fore_rate']]
        back_rate = torch.cat(back_rate_list, dim=0).cpu().numpy()
        fore_rate = torch.cat(fore_rate_list, dim=0).cpu().numpy()
        return back_rate, fore_rate


class SCMSICiteRna(nn.Module):
    """
    Integration of paired (cite-seq) and unpaired data (rna) with scMSI
    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            gene_dispersion: Literal[
                "gene", "gene-batch", "gene-label", "gene-cell"
            ] = "gene",
            protein_dispersion: Literal[
                "protein", "protein-batch", "protein-label"
            ] = "protein",
            gene_likelihood: Literal["zinb", "nb"] = "nb",
            protein_likelihood: Literal["nbm", "nb", "poisson"] = "nbm",
            latent_distribution: Literal["normal", "ln"] = "normal",
            empirical_protein_background_prior: Optional[bool] = None,
            n_sample_ref: Optional[int] = 100,
            batch_size: Optional[int] = None,
            **model_kwargs,
    ):
        super(SCMSICiteRna, self).__init__()
        self.adata = adata
        if "batch" not in adata.obs.keys():
            adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
        if "n_batch" not in adata.uns.keys():
            adata.uns["n_batch"] = 1

        n_cats_per_cov = (
            adata.uns["n_cats_per_key"]
            if "n_cats_per_key" in adata.uns.keys()
            else None
        )
        n_batch = self.adata.uns["n_batch"]
        library_log_means, library_log_vars = init_library_size(adata, n_batch)
        emp_prior = (
            empirical_protein_background_prior
            if empirical_protein_background_prior is not None
            else (adata.obsm["protein_expression"].shape[1] >= 10)
        )
        if emp_prior:
            batch_mask = None
            prior_mean, prior_scale = get_protein_priors(adata, batch_mask)
        else:
            prior_mean, prior_scale = None, None
        # prior_mean[1] = 5
        # prior_mean[3] = 5
        # prior_mean[7] = 6
        # prior_mean[8] = 3.2

        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=0.9,
            validation_size=0.1,
            use_gpu=False,
        )
        self.train_adata = train_adata
        self.val_adata = val_adata
        self.batch_size = batch_size
        paired_mask = train_adata.obsm['protein_expression'].sum(axis=1) != 0
        if n_sample_ref > sum(paired_mask):
            ref_idx = np.arange(train_adata.shape[0])[paired_mask]
        else:
            ref_idx_mask = get_anchor_index(train_adata[paired_mask].X, n_pcs=20, n_clusters=n_sample_ref)
            ref_idx = np.nonzero(paired_mask)[0][ref_idx_mask]
            print(f"Choose {n_sample_ref} anchor samples for self-expressive learning")
        # ref_idx = np.random.permutation(train_adata.shape[1])[0:500]
        self.ref_adata = train_adata[ref_idx, :]

        self.module = CiteRnaSVAE(
            n_input_genes=adata.X.shape[1],
            n_input_proteins=adata.obsm['protein_expression'].shape[1],
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=0,
            n_cats_per_cov=n_cats_per_cov,
            gene_dispersion=gene_dispersion,
            protein_dispersion=protein_dispersion,
            gene_likelihood=gene_likelihood,
            protein_likelihood=protein_likelihood,
            latent_distribution=latent_distribution,
            protein_background_prior_mean=prior_mean,
            protein_background_prior_scale=prior_scale,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            use_observed_lib_size=True,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 4e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 256,
            early_stopping: bool = False,
            check_val_every_n_epoch: Optional[int] = None,
            reduce_lr_on_plateau: bool = True,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            adversarial_classifier: Optional[bool] = None,
            plan_kwargs: Optional[dict] = None,
            weight_decay=1e-6,
            record_loss=True,
            **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        weight_decay
            weight decay
        record_loss
            whether save loss
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.module.cuda(device=device)
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        self.batch_size = batch_size
        train_adata = self.train_adata
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_loss = []
        epoch_rec_loss = {}
        epoch_kl_loss = {}
        epoch_se_loss = {}
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            batch_loss = []
            batch_rec_loss = {}
            batch_kl_loss = {}
            batch_se_loss = {}
            create_dict = True
            for train_batch in train_adata_batch:
                input_genes = torch.tensor(train_batch.layers['rna_expression'], dtype=torch.float32)
                input_proteins = torch.tensor(train_batch.obsm['protein_expression'], dtype=torch.float32)
                batch_index = torch.tensor(train_batch.obs['batch'], dtype=torch.float32).reshape(-1, 1)
                if "cat_covs" in train_batch.obsm.keys():
                    cat_covs = torch.tensor(train_batch.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
                else:
                    cat_covs = None
                # batch_index = (batch_index, batch_index)
                # print(len(batch_index))
                _, _, _, all_loss = self.module(input_genes, input_proteins, ref_adata, batch_index=batch_index,
                                                cat_covs=cat_covs)
                loss, reconst_loss, kl_local, se_losses = all_loss
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), 0.001)
                optimizer.step()
                # batch_loss.append(loss.data.numpy())
                if create_dict:
                    for k in reconst_loss.keys():
                        batch_rec_loss[k] = []
                    for k in kl_local.keys():
                        batch_kl_loss[k] = []
                    for k in se_losses.keys():
                        batch_se_loss[k] = []
                    create_dict = False
                batch_loss.append(loss.cpu().data.numpy())
                if record_loss:
                    for k in batch_rec_loss.keys():
                        batch_rec_loss[k].append(torch.mean(reconst_loss[k]).cpu().data.numpy())
                    for k in batch_kl_loss.keys():
                        batch_kl_loss[k].append(torch.mean(kl_local[k]).cpu().data.numpy())
                    for k in batch_se_loss.keys():
                        batch_se_loss[k].append(torch.mean(se_losses[k]).cpu().data.numpy())

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                print("epoch: {}; loss: {} ".format(epoch, np.mean(batch_loss)))
            scheduler.step()
            # self.module.eval()
            if epoch == 0:
                for k in batch_rec_loss.keys():
                    epoch_rec_loss[k] = []
                for k in batch_kl_loss.keys():
                    epoch_kl_loss[k] = []
                for k in batch_se_loss.keys():
                    epoch_se_loss[k] = []

            epoch_loss.append(np.mean(batch_loss))
            if record_loss:
                for k in epoch_rec_loss.keys():
                    epoch_rec_loss[k].append(np.mean(batch_rec_loss[k]))
                for k in epoch_kl_loss.keys():
                    epoch_kl_loss[k].append(np.mean(batch_kl_loss[k]))
                for k in epoch_se_loss.keys():
                    epoch_se_loss[k].append(np.mean(batch_se_loss[k]))
        self.history = dict(
            epoch_loss=epoch_loss,
            epoch_rec_loss=epoch_rec_loss,
            epoch_kl_loss=epoch_kl_loss,
            epoch_se_loss=epoch_se_loss,
        )

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent_gene = []
        latent_pro = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            enc_gene_outputs, enc_pro_outputs = self.module.inference(input_genes, input_proteins,
                                                                      batch_index=batch_index, cat_covs=cat_covs)
            qz_m_gene = enc_gene_outputs["qz_m"]
            qz_v_gene = enc_gene_outputs["qz_v"]
            z_gene = enc_gene_outputs["z"]
            qz_m_pro = enc_pro_outputs["qz_m"]
            qz_v_pro = enc_pro_outputs["qz_v"]
            z_pro = enc_pro_outputs["z"]
            if give_mean:
                samples = Normal(qz_m_gene, qz_v_gene.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z_gene = self.module.encoder_genes.z_transformation(samples)
                z_gene = z_gene.mean(dim=0)
                samples = Normal(qz_m_pro, qz_v_pro.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z_pro = self.module.encoder_proteins.z_transformation(samples)
                z_pro = z_pro.mean(dim=0)
            else:
                z_gene = qz_m_gene
                z_pro = qz_m_pro

            latent_gene += [z_gene.cpu()]
            latent_pro += [z_pro.cpu()]
        lat_gene = torch.cat(latent_gene).numpy()
        lat_pro = torch.cat(latent_pro).numpy()

        return lat_gene, lat_pro

    @torch.no_grad()
    def get_reconstruct_latent(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        ref_z_gene_list = []
        ref_z_pro_list = []
        rec_z_gene_list = []
        rec_z_pro_list = []
        for data in ref_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_proteins = torch.tensor(data.obsm["protein_expression"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                refer_cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                refer_cat_covs = None
            ref_outputs_gene, ref_outputs_pro = self.module.inference(refer_genes, refer_proteins,
                                                                      batch_index=refer_index, cat_covs=refer_cat_covs)
            ref_z_gene_list.append(ref_outputs_gene['z'])
            ref_z_pro_list.append(ref_outputs_pro['z'])
        ref_z_genes = torch.cat(ref_z_gene_list, dim=0)
        ref_z_pros = torch.cat(ref_z_pro_list, dim=0)
        if not give_mean:
            mc_samples = 1
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm["protein_expression"], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            inference_genes, inference_pros = self.module.inference(input_genes, input_proteins, batch_index,
                                                                    n_samples=mc_samples, cat_covs=cat_covs)
            z_genes = inference_genes['z'].mean(dim=0)
            z_pros = inference_pros['z'].mean(dim=0)
            se_gene_outputs, se_pro_outputs = self.module.self_expressiveness(z_genes, z_pros, ref_z_genes, ref_z_pros)
            rec_z_genes = se_gene_outputs['rec_queries']
            rec_z_pros = se_pro_outputs['rec_queries']
            rec_z_gene_list.append(rec_z_genes.cpu())
            rec_z_pro_list.append(rec_z_pros.cpu())
        rec_z_gs = torch.cat(rec_z_gene_list, dim=0).numpy()
        rec_z_ps = torch.cat(rec_z_pro_list, dim=0).numpy()
        return rec_z_gs, rec_z_ps

    @torch.no_grad()
    def get_coeff(
            self,
            adata: Optional[AnnData] = None,
            batch_size: int = 128,
            pro_w=1,
            ord=1,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        coeff_genes_list = []
        coeff_pros_list = []
        ref_adata = self.ref_adata
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            _, senet_outputs, _ = self.module(input_genes, input_proteins, ref_adata, batch_index=batch_index,
                                              cat_covs=cat_covs, compute_loss=False)
            se_gene_outputs, se_pro_outputs = senet_outputs
            coeff_genes_list.append(se_gene_outputs['coeff'])

            # only use the protein coefficients of cells with protein data
            pro_mask = input_proteins.sum(dim=1) > 0
            device = se_pro_outputs['coeff'].device
            pro_mask_expanded = pro_mask.to(device).unsqueeze(1).expand_as(se_pro_outputs['coeff'])
            pro_coeff = torch.where(pro_mask_expanded, se_pro_outputs['coeff'],
                                    torch.zeros_like(se_pro_outputs['coeff']))
            # pro_coeff = se_pro_outputs['coeff']
            coeff_pros_list.append(pro_coeff)
        coeff_genes = torch.cat(coeff_genes_list, dim=0)
        coeff_proteins = torch.cat(coeff_pros_list, dim=0)

        # if no protein data, use the mean of all protein coefficients
        pro_mask = adata.obsm['protein_expression'].sum(axis=1) > 0
        pro_coef_mean = coeff_proteins[pro_mask].mean(dim=0)
        coeff_proteins[~pro_mask] = pro_coef_mean

        coeff = coeff_genes + pro_w * coeff_proteins
        coeff_genes = coeff_genes.cpu().numpy()
        coeff_proteins = coeff_proteins.cpu().numpy()
        coeff = coeff.cpu().numpy()
        if ord is not None:
            coeff_genes = coeff_genes / norm(coeff_genes, ord=ord, axis=1, keepdims=True)
            coeff_proteins = coeff_proteins / norm(coeff_proteins, ord=ord, axis=1, keepdims=True)
            coeff = coeff / norm(coeff, ord=ord, axis=1, keepdims=True)
        return coeff_genes, coeff_proteins, coeff

    @torch.no_grad()
    def get_latent_library_size(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.module.training:
            self.module.eval()
        libraries = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            outputs, _ = self.module.inference(input_genes, input_proteins, batch_index, cat_covs=cat_covs)
            ql_m = outputs["ql_m"]
            ql_v = outputs["ql_v"]
            library_gene = outputs["library_gene"]
            if give_mean and (not self.module.use_observed_lib_size):
                untran_l = Normal(ql_m, ql_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                library = torch.exp(untran_l)
                library = library.mean(dim=0)
            else:
                library = library_gene

            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()

    @torch.no_grad()
    def get_normalized_expression(
            self,
            adata=None,
            indices=None,
            n_samples_overall: Optional[int] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            gene_list: Optional[Sequence[str]] = None,
            protein_list: Optional[Sequence[str]] = None,
            library_size: Optional[Union[float, Literal["latent"]]] = 1,
            n_samples: int = 1,
            sample_protein_mixing: bool = False,
            scale_protein: bool = False,
            include_protein_background: bool = False,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        r"""
        Returns the normalized gene expression and protein expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to use in total
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude.
        n_samples
            Get sample scale from multiple samples.
        sample_protein_mixing
            Sample mixing bernoulli, setting background to zero
        scale_protein
            Make protein expression sum to 1
        include_protein_background
            Include background component for protein expression
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
            names as columns. If either n_samples=1 or return_mean=True, defaults to False.
            Otherwise, it defaults to True.

        Returns
        -------
        - **gene_normalized_expression** - normalized expression for RNA
        - **protein_normalized_expression** - normalized expression for proteins

        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is ``(samples, cells, genes)``.
        Otherwise, shape is ``(cells, genes)``. Return type is ``pd.DataFrame`` unless ``return_numpy`` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]
        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = adata.uns['protein_names']
            protein_mask = [True if p in protein_list else False for p in all_proteins]
        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        scale_list_gene = []
        scale_list_pro = []

        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            px_scale = torch.zeros_like(input_genes)
            py_scale = torch.zeros_like(input_proteins)
            if n_samples > 1:
                px_scale = torch.stack(n_samples * [px_scale])
                py_scale = torch.stack(n_samples * [py_scale])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_genes, input_proteins, ref_adata, batch_index,
                                                       cat_covs=cat_covs, compute_loss=False, transform_batch=b)
                px_, py_ = generative_outputs
                if library_size == "latent":
                    px_scale += px_["rate"].cpu()
                else:
                    px_scale += px_["scale"].cpu()
                px_scale = px_scale[..., gene_mask]

                # probability of background
                protein_mixing = 1 / (1 + torch.exp(-py_["mixing"].cpu()))
                if sample_protein_mixing is True:
                    protein_mixing = torch.distributions.Bernoulli(
                        protein_mixing
                    ).sample()
                protein_val = py_["fore_rate"].cpu() * (1 - protein_mixing)
                if include_protein_background is True:
                    protein_val += py_["back_rate"].cpu() * protein_mixing

                if scale_protein is True:
                    protein_val = torch.nn.functional.normalize(
                        protein_val, p=1, dim=-1
                    )
                protein_val = protein_val[..., protein_mask]
                py_scale += protein_val
            px_scale /= len(transform_batch)
            py_scale /= len(transform_batch)
            scale_list_gene.append(px_scale)
            scale_list_pro.append(py_scale)

        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            scale_list_gene = torch.cat(scale_list_gene, dim=1)
            scale_list_pro = torch.cat(scale_list_pro, dim=1)
            # (cells, features, samples)
            scale_list_gene = scale_list_gene.permute(1, 2, 0)
            scale_list_pro = scale_list_pro.permute(1, 2, 0)
        else:
            scale_list_gene = torch.cat(scale_list_gene, dim=0)
            scale_list_pro = torch.cat(scale_list_pro, dim=0)

        if return_mean is True and n_samples > 1:
            scale_list_gene = torch.mean(scale_list_gene, dim=-1)
            scale_list_pro = torch.mean(scale_list_pro, dim=-1)

        scale_list_gene = scale_list_gene.cpu().numpy()
        scale_list_pro = scale_list_pro.cpu().numpy()
        if return_numpy is None or return_numpy is False:
            gene_df = pd.DataFrame(
                scale_list_gene,
                columns=self.adata.var_names[gene_mask],
                index=self.adata.obs_names[indices],
            )
            pro_df = pd.DataFrame(
                scale_list_pro,
                columns=self.adata.uns["protein_names"][protein_mask],
                index=self.adata.obs_names[indices],
            )

            return gene_df, pro_df
        else:
            return scale_list_gene, scale_list_pro

    @torch.no_grad()
    def get_protein_foreground_probability(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            protein_list: Optional[Sequence[str]] = None,
            n_samples: int = 1,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ):
        r"""
        Returns the foreground probability for proteins.

        This is denoted as :math:`(1 - \pi_{nt})` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        - **foreground_probability** - probability foreground for each protein

        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)

        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = adata.uns['protein_names']
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        py_mixings = []
        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            py_mixing = torch.zeros_like(input_pros[..., protein_mask])
            if n_samples > 1:
                py_mixing = torch.stack(n_samples * [py_mixing])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_genes, input_pros, ref_adata, batch_index,
                                                       cat_covs=cat_covs, compute_loss=False, transform_batch=b)
                _, py_ = generative_outputs
                # probability of background
                py_mixing += torch.sigmoid(py_["mixing"])[..., protein_mask].cpu()
            py_mixing /= len(transform_batch)
            py_mixings += [py_mixing]
        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            py_mixings = torch.cat(py_mixings, dim=1)
            # (cells, features, samples)
            py_mixings = py_mixings.permute(1, 2, 0)
        else:
            py_mixings = torch.cat(py_mixings, dim=0)

        if return_mean is True and n_samples > 1:
            py_mixings = torch.mean(py_mixings, dim=-1)

        py_mixings = py_mixings.cpu().numpy()

        if return_numpy is True:
            return 1 - py_mixings
        else:
            pro_names = self.adata.uns["protein_names"]
            foreground_prob = pd.DataFrame(
                1 - py_mixings,
                columns=pro_names[protein_mask],
                index=adata.obs_names[indices],
            )
            return foreground_prob

    @torch.no_grad()
    def get_protein_likelihood_parameters(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        back_rate_list = []
        fore_rate_list = []
        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            _, _, generative_outputs = self.module(input_genes, input_pros, ref_adata, batch_index, cat_covs=cat_covs,
                                                   compute_loss=False)
            _, py_ = generative_outputs
            back_rate_list += [py_['back_rate']]
            fore_rate_list += [py_['fore_rate']]
        back_rate = torch.cat(back_rate_list, dim=0).cpu().numpy()
        fore_rate = torch.cat(fore_rate_list, dim=0).cpu().numpy()
        return back_rate, fore_rate


class SCMSIRNAPeak(nn.Module):
    """
    Parameters
    ----------
    adata
        AnnData object.
    n_latent
        Dimensionality of the latent space.
    gene_dispersion
        One of the following:

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    empirical_protein_background_prior
        Set the initialization of protein background prior empirically. This option fits a GMM for each of
        100 cells per batch and averages the distributions. Note that even with this option set to `True`,
        this only initializes a parameter that is learned during inference. If `False`, randomly initializes.
        The default (`None`), sets this to `True` if greater than 10 proteins are used.
    **model_kwargs
        Keyword args for :class:`~scMSI.scMSI_main.scMSIVAE`
    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            gene_dispersion: Literal[
                "gene", "gene-batch", "gene-label", "gene-cell"
            ] = "gene",
            gene_likelihood: Literal["zinb", "nb"] = "nb",
            latent_distribution: Literal["normal", "ln"] = "normal",
            n_sample_ref: int = 50,
            batch_size: int = 128,
            cc_weight: float = 1e-4,
            **model_kwargs,
    ):
        super(SCMSIRNAPeak, self).__init__()
        self.adata = adata
        if "batch" not in adata.obs.keys():
            adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
        if "n_batch" not in adata.uns.keys():
            adata.uns["n_batch"] = 1

        n_cats_per_cov = (
            adata.uns["extra_categoricals"]["n_cats_per_key"]
            if "extra_categoricals" in adata.uns.keys()
            else None
        )
        n_batch = self.adata.uns["n_batch"]
        library_log_means, library_log_vars = init_library_size(adata, n_batch)
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=0.9,
            validation_size=0.1,
            use_gpu=False,
        )
        self.train_adata = train_adata
        self.val_adata = val_adata
        self.batch_size = batch_size

        if n_sample_ref > train_adata.shape[0]:
            ref_idx = np.arange(train_adata.shape[0])
        else:
            ref_idx = get_anchor_index(train_adata.X, n_pcs=20, n_clusters=n_sample_ref)
            print(f"Choose {n_sample_ref} anchor samples for self-expressive learning")
        # ref_idx = np.random.permutation(train_adata.shape[1])[0:500]
        self.ref_adata = train_adata[ref_idx, :]

        self.module = RNAPeakSVAE(
            n_input_genes=adata.X.shape[1],
            n_input_peaks=adata.obsm['peak_counts'].shape[1],
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=0,
            n_cats_per_cov=n_cats_per_cov,
            gene_dispersion=gene_dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            use_observed_lib_size=True,
            cc_weight=cc_weight,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 1e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 256,
            early_stopping: bool = False,
            check_val_every_n_epoch: Optional[int] = None,
            reduce_lr_on_plateau: bool = True,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            adversarial_classifier: Optional[bool] = None,
            plan_kwargs: Optional[dict] = None,
            weight_decay=1e-3,
            record_loss=True,
            **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        weight_decay
            weight decay
        record_loss
            whether save loss
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.module.cuda(device=device)
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])
        train_adata = self.train_adata
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_loss = []
        epoch_rec_loss = {}
        epoch_kl_loss = {}
        epoch_se_loss = {}
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            batch_loss = []
            batch_rec_loss = {}
            batch_kl_loss = {}
            batch_se_loss = {}
            create_dict = True
            for train_batch in train_adata_batch:
                input_genes = torch.tensor(train_batch.layers['rna_expression'], dtype=torch.float32)
                input_proteins = torch.tensor(train_batch.obsm['peak_counts'], dtype=torch.float32)
                if "batch" in train_batch.obs.keys():
                    batch_index = torch.tensor(train_batch.obs['batch'], dtype=torch.float32).reshape(-1, 1)
                else:
                    batch_index = None
                _, _, _, all_loss = self.module(input_genes, input_proteins, ref_adata, batch_index)
                loss, reconst_loss, kl_local, se_losses = all_loss
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), 0.001)
                optimizer.step()
                # batch_loss.append(loss.data.numpy())
                if create_dict:
                    for k in reconst_loss.keys():
                        batch_rec_loss[k] = []
                    for k in kl_local.keys():
                        batch_kl_loss[k] = []
                    for k in se_losses.keys():
                        batch_se_loss[k] = []
                    create_dict = False
                batch_loss.append(loss.cpu().data.numpy())
                if record_loss:
                    for k in batch_rec_loss.keys():
                        batch_rec_loss[k].append(torch.mean(reconst_loss[k]).cpu().data.numpy())
                    for k in batch_kl_loss.keys():
                        batch_kl_loss[k].append(torch.mean(kl_local[k]).cpu().data.numpy())
                    for k in batch_se_loss.keys():
                        batch_se_loss[k].append(torch.mean(se_losses[k]).cpu().data.numpy())

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                print("epoch: {}; loss: {} ".format(epoch, np.mean(batch_loss)))
            # scheduler.step()
            # self.module.eval()
            if epoch == 0:
                for k in batch_rec_loss.keys():
                    epoch_rec_loss[k] = []
                for k in batch_kl_loss.keys():
                    epoch_kl_loss[k] = []
                for k in batch_se_loss.keys():
                    epoch_se_loss[k] = []

            epoch_loss.append(np.mean(batch_loss))
            if record_loss:
                for k in epoch_rec_loss.keys():
                    epoch_rec_loss[k].append(np.mean(batch_rec_loss[k]))
                for k in epoch_kl_loss.keys():
                    epoch_kl_loss[k].append(np.mean(batch_kl_loss[k]))
                for k in epoch_se_loss.keys():
                    epoch_se_loss[k].append(np.mean(batch_se_loss[k]))

        self.history = dict(
            epoch_loss=epoch_loss,
            epoch_rec_loss=epoch_rec_loss,
            epoch_kl_loss=epoch_kl_loss,
            epoch_se_loss=epoch_se_loss,
        )

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent_gene = []
        latent_peak = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_peaks = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            enc_gene_outputs, enc_peak_outputs = self.module.inference(input_genes, input_peaks, batch_index)
            qz_m_gene = enc_gene_outputs["qz_m"]
            qz_v_gene = enc_gene_outputs["qz_v"]
            z_gene = enc_gene_outputs["z"]
            qz_m_peak = enc_peak_outputs["qz_m"]
            qz_v_peak = enc_peak_outputs["qz_v"]
            z_peak = enc_peak_outputs["z"]
            if give_mean:
                samples = Normal(qz_m_gene, qz_v_gene.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z_gene = self.module.encoder_genes.z_transformation(samples)
                z_gene = z_gene.mean(dim=0)
                samples = Normal(qz_m_peak, qz_v_peak.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z_peak = self.module.encoder_peaks.z_transformation(samples)
                z_peak = z_peak.mean(dim=0)
            else:
                z_gene = qz_m_gene
                z_peak = qz_m_peak

            latent_gene += [z_gene.cpu()]
            latent_peak += [z_peak.cpu()]
        lat_gene = torch.cat(latent_gene).numpy()
        lat_peak = torch.cat(latent_peak).numpy()

        return lat_gene, lat_peak

    @torch.no_grad()
    def get_reconstruct_latent(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ):
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        ref_z_gene_list = []
        ref_z_peak_list = []
        rec_z_gene_list = []
        rec_z_peak_list = []
        for data in ref_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_peaks = torch.tensor(data.obsm["peak_counts"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            ref_outputs_gene, ref_outputs_peak = self.module.inference(refer_genes, refer_peaks, refer_index)
            ref_z_gene_list.append(ref_outputs_gene['z'])
            ref_z_peak_list.append(ref_outputs_peak['z'])
        ref_z_genes = torch.cat(ref_z_gene_list, dim=0)
        ref_z_peaks = torch.cat(ref_z_peak_list, dim=0)
        if not give_mean:
            mc_samples = 1
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_peaks = torch.tensor(data.obsm["peak_counts"], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            inference_genes, inference_peaks = self.module.inference(input_genes, input_peaks, batch_index,
                                                                     n_samples=mc_samples)
            z_genes = inference_genes['z'].mean(dim=0)
            z_peaks = inference_peaks['z'].mean(dim=0)
            se_gene_outputs, se_peak_outputs = self.module.self_expressiveness(z_genes, z_peaks, ref_z_genes,
                                                                               ref_z_peaks)
            rec_z_genes = se_gene_outputs['rec_queries']
            rec_z_peaks = se_peak_outputs['rec_queries']
            rec_z_gene_list.append(rec_z_genes.cpu())
            rec_z_peak_list.append(rec_z_peaks.cpu())
        rec_z_gs = torch.cat(rec_z_gene_list, dim=0).numpy()
        rec_z_ps = torch.cat(rec_z_peak_list, dim=0).numpy()
        return rec_z_gs, rec_z_ps

    @torch.no_grad()
    def get_coeff(
            self,
            adata: Optional[AnnData] = None,
            batch_size: int = 128,
            peak_w=1,
            ord=1,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        coeff_genes_list = []
        coeff_pros_list = []
        ref_adata = self.ref_adata
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            _, senet_outputs, _ = self.module(input_genes, input_proteins, ref_adata, batch_index, compute_loss=False)
            se_gene_outputs, se_pro_outputs = senet_outputs
            coeff_genes_list.append(se_gene_outputs['coeff'])
            coeff_pros_list.append(se_pro_outputs['coeff'])
        coeff_genes = torch.cat(coeff_genes_list, dim=0)
        coeff_proteins = torch.cat(coeff_pros_list, dim=0)
        coeff = coeff_genes + peak_w * coeff_proteins
        coeff_genes = coeff_genes.cpu().numpy()
        coeff_proteins = coeff_proteins.cpu().numpy()
        coeff = coeff.cpu().numpy()
        if ord is not None:
            coeff_genes = coeff_genes / norm(coeff_genes, ord=ord, axis=1, keepdims=True)
            coeff_proteins = coeff_proteins / norm(coeff_proteins, ord=ord, axis=1, keepdims=True)
            coeff = coeff / norm(coeff, ord=ord, axis=1, keepdims=True)
        return coeff_genes, coeff_proteins, coeff

    @torch.no_grad()
    def get_latent_library_size(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.module.training:
            self.module.eval()
        libraries = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs, _ = self.module.inference(input_genes, input_proteins, batch_index)
            ql_m = outputs["ql_m"]
            ql_v = outputs["ql_v"]
            library_gene = outputs["library_gene"]
            if give_mean and (not self.module.use_observed_lib_size):
                untran_l = Normal(ql_m, ql_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                library = torch.exp(untran_l)
                library = library.mean(dim=0)
            else:
                library = library_gene

            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()

    @torch.no_grad()
    def get_normalized_expression(
            self,
            adata=None,
            indices=None,
            n_samples_overall: Optional[int] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            gene_list: Optional[Sequence[str]] = None,
            library_size: Optional[Union[float, Literal["latent"]]] = 1,
            n_samples: int = 1,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ):
        """
        Returns the normalized gene expression and protein expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to use in total
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude.
        n_samples
            Get sample scale from multiple samples.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
            names as columns. If either n_samples=1 or return_mean=True, defaults to False.
            Otherwise, it defaults to True.

        Returns
        -------
        - **gene_normalized_expression** - normalized expression for RNA
        - **protein_normalized_expression** - normalized expression for proteins

        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is ``(samples, cells, genes)``.
        Otherwise, shape is ``(cells, genes)``. Return type is ``pd.DataFrame`` unless ``return_numpy`` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]
        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        scale_list_gene = []
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            px_scale = torch.zeros_like(input_genes)
            py_scale = torch.zeros_like(input_proteins)
            if n_samples > 1:
                px_scale = torch.stack(n_samples * [px_scale])
                py_scale = torch.stack(n_samples * [py_scale])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_genes, input_proteins, ref_adata, batch_index,
                                                       compute_loss=False, transform_batch=b)
                px_, py_ = generative_outputs
                if library_size == "latent":
                    px_scale += px_["rate"].cpu()
                else:
                    px_scale += px_["scale"].cpu()
                px_scale = px_scale[..., gene_mask]
            px_scale /= len(transform_batch)
            scale_list_gene.append(px_scale)

        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            scale_list_gene = torch.cat(scale_list_gene, dim=1)
            # (cells, features, samples)
            scale_list_gene = scale_list_gene.permute(1, 2, 0)
        else:
            scale_list_gene = torch.cat(scale_list_gene, dim=0)

        if return_mean is True and n_samples > 1:
            scale_list_gene = torch.mean(scale_list_gene, dim=-1)

        scale_list_gene = scale_list_gene.cpu().numpy()
        if return_numpy is None or return_numpy is False:
            gene_df = pd.DataFrame(
                scale_list_gene,
                columns=self.adata.var_names[gene_mask],
                index=self.adata.obs_names[indices],
            )

            return gene_df
        else:
            return scale_list_gene

    @torch.no_grad()
    def get_accessibility_estimates(
            self,
            adata: Optional[AnnData] = None,
            indices: Sequence[int] = None,
            n_samples_overall: Optional[int] = None,
            region_list: Optional[Sequence[str]] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            use_z_mean: bool = True,
            threshold: Optional[float] = None,
            normalize_cells: bool = False,
            normalize_regions: bool = False,
            batch_size: Optional[int] = None,
            return_numpy: Optional[bool] = None,
    ) -> Union[pd.DataFrame, np.ndarray, csr_matrix]:
        """
        Impute the full accessibility matrix.

        Returns a matrix of accessibility probabilities for each cell and genomic region in the input
        (for return matrix A, A[i,j] is the probability that region j is accessible in cell i).

        Parameters
        ----------
        adata
            AnnData object that has been registered with scvi. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to return in total
        region_list
            Return accessibility estimates for this subset of regions. if `None`, all regions are used.
            This can save memory when dealing with large datasets.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
        use_z_mean
            If True (default), use the distribution mean. Otherwise, sample from the distribution.
        threshold
            If provided, values below the threshold are replaced with 0 and a sparse matrix
            is returned instead. This is recommended for very large matrices. Must be between 0 and 1.
        normalize_cells
            Whether to reintroduce library size factors to scale the normalized probabilities.
            This makes the estimates closer to the input, but removes the library size correction.
            False by default.
        normalize_regions
            Whether to reintroduce region factors to scale the normalized probabilities. This makes
            the estimates closer to the input, but removes the region-level bias correction. False by
            default.
        batch_size
            Minibatch size for data loading into model
        return_numpy
            If `True` and `threshold=None`, return :class:`~numpy.ndarray`. If `True` and `threshold` is
            given, return :class:`~scipy.sparse.csr_matrix`. If `False`, return :class:`~pandas.DataFrame`.
            DataFrame includes regions names as columns.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)

        if region_list is None:
            region_mask = slice(None)
        else:
            all_regions = adata.var_names
            region_mask = [region in region_list for region in all_regions]
        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]
        if threshold is not None and (threshold < 0 or threshold > 1):
            raise ValueError("the provided threshold must be between 0 and 1")

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        imputed_list = []
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            imputed = torch.zeros_like(input_genes)
            for b in transform_batch:
                inference_outputs, _, generative_outputs = self.module(input_genes, input_proteins, ref_adata,
                                                                       batch_index, compute_loss=False,
                                                                       transform_batch=b)
                generative_outputs = generative_outputs[0]
                p = generative_outputs["prob"].cpu()
                if normalize_cells:
                    p *= inference_outputs["lib"].cpu()
                if normalize_regions:
                    p *= generative_outputs["rf"].cpu()
                if threshold:
                    p[p < threshold] = 0
                    p = csr_matrix(p.numpy())
                if region_list is not None:
                    p = p[:, region_mask]
                imputed += p

            imputed /= len(transform_batch)
            imputed_list.append(imputed)

        if threshold:  # imputed is a list of csr_matrix objects
            imputed_peak = vstack(imputed_list, format="csr")
        else:  # imputed is a list of tensors
            imputed_peak = torch.cat(imputed_list).numpy()

        if return_numpy:
            return imputed_peak
        elif threshold:
            return pd.DataFrame.sparse.from_spmatrix(
                imputed_peak,
                index=adata.obs_names[indices],
                columns=adata.var_names[region_mask],
            )
        else:
            return pd.DataFrame(
                imputed_peak,
                index=adata.obs_names[indices],
                columns=adata.uns['peak_names'][region_mask],
            )


class SCMSIRNA(nn.Module):
    """
    Parameters
    ----------
    adata
        AnnData object.
    n_latent
        Dimensionality of the latent space.
    gene_dispersion
        One of the following:

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    empirical_protein_background_prior
        Set the initialization of protein background prior empirically. This option fits a GMM for each of
        100 cells per batch and averages the distributions. Note that even with this option set to `True`,
        this only initializes a parameter that is learned during inference. If `False`, randomly initializes.
        The default (`None`), sets this to `True` if greater than 10 proteins are used.
    override_missing_proteins
        If `True`, will not treat proteins with all 0 expression in a particular batch as missing.
    **model_kwargs
        Keyword args for :class:`~scMSI.scMSI_main.scMSIVAE`

    Notes
    -----
    See further usage examples in the following tutorials:

    to be updated

    """

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        gene_dispersion: Literal[
            "gene", "gene-batch", "gene-label", "gene-cell"
        ] = "gene",
        gene_likelihood: Literal["zinb", "nb"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs,
    ):
        super(SCMSIRNA, self).__init__()
        self.adata = adata
        if "batch" not in adata.obs.keys():
            adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
        if "n_batch" not in adata.uns.keys():
            adata.uns["n_batch"] = 1

        n_cats_per_cov = (
            adata.uns["extra_categoricals"]["n_cats_per_key"]
            if "extra_categoricals" in adata.uns.keys()
            else None
        )
        n_batch = self.adata.uns["n_batch"]
        library_log_means, library_log_vars = init_library_size(adata, n_batch)

        self.module = RNASVAE(
            n_input_genes=adata.X.shape[1],
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=0,
            n_cats_per_cov=n_cats_per_cov,
            gene_dispersion=gene_dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            use_observed_lib_size=True,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 4e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            n_sample_ref: int = 50,
            split_index: Optional[float] = None,
            early_stopping: bool = False,
            check_val_every_n_epoch: Optional[int] = None,
            reduce_lr_on_plateau: bool = True,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            adversarial_classifier: Optional[bool] = None,
            plan_kwargs: Optional[dict] = None,
            weight_decay=1e-6,
            record_loss=True,
            **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        n_sample_ref
            reference samples number.
        split_index
            (train_idx, val_idx, test_idx, anchor_idx)
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        weight_decay
            weight decay
        record_loss
            whether save loss
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=0.9,
            validation_size=0.1,
            use_gpu=True,
        )
        if split_index is not None:
            train_adata = self.adata[split_index[0]]
            val_adata = self.adata[split_index[1]]

        self.train_adata = train_adata
        self.val_adata = val_adata
        self.batch_size = batch_size

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        if n_sample_ref > train_adata.shape[0]:
            ref_idx = np.arange(train_adata.shape[0])
        else:
            ref_idx = get_anchor_index(train_adata.X, n_pcs=50, n_clusters=n_sample_ref)
            if split_index is not None:
                ref_idx = split_index[3]
        # ref_idx = np.random.permutation(train_adata.shape[1])[0:500]
        self.ref_adata = train_adata[ref_idx, :]
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_loss = []
        epoch_rec_loss = []
        epoch_kl_z_loss = []
        epoch_kl_l_loss = []
        epoch_se_loss = []
        epoch_coef_reg_loss = []
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            batch_loss = []
            batch_rec_loss = []
            batch_kl_z_loss = []
            batch_kl_l_loss = []
            batch_se_loss = []
            batch_coef_reg_loss = []
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            for train_batch in train_adata_batch:
                input_genes = torch.tensor(train_batch.layers['rna_expression'], dtype=torch.float32)
                if "batch" in train_batch.obs.keys():
                    batch_index = torch.tensor(train_batch.obs['batch'], dtype=torch.float32).reshape(-1, 1)
                else:
                    batch_index = None
                _, _, _, all_loss = self.module(input_genes, ref_adata, batch_index)
                loss, reconst_loss, kl_local, se_losses = all_loss
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), 0.001)
                optimizer.step()
                if record_loss:
                    batch_loss.append(loss.data.numpy())
                    batch_rec_loss.append(torch.mean(reconst_loss).data.numpy())
                    batch_kl_z_loss.append(torch.mean(kl_local['kl_div_z']).data.numpy())
                    if not self.module.use_observed_lib_size:
                        batch_kl_l_loss.append(torch.mean(kl_local['kl_div_l']).data.numpy())
                    else:
                        batch_kl_l_loss.append(kl_local['kl_div_l'])
                    batch_se_loss.append(torch.mean(se_losses['se_loss']).data.numpy())
                    batch_coef_reg_loss.append(torch.mean(se_losses['coeff_reg']).data.numpy())

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                print("epoch: {}; loss: {} ".format(epoch, np.mean(batch_loss)))
            scheduler.step()
            # self.module.eval()
            epoch_loss.append(np.mean(batch_loss))
            epoch_rec_loss.append(np.mean(batch_rec_loss))
            epoch_kl_z_loss.append(np.mean(batch_kl_z_loss))
            epoch_kl_l_loss.append(np.mean(batch_kl_l_loss))
            epoch_se_loss.append(np.mean(batch_se_loss))
            epoch_coef_reg_loss.append(np.mean(batch_rec_loss))
        self.history = dict(
            epoch_loss=epoch_loss,
            epoch_rec_loss=epoch_rec_loss,
            epoch_kl_z_loss=epoch_kl_z_loss,
            epoch_kl_l_loss=epoch_kl_l_loss,
            epoch_se_loss=epoch_se_loss,
            epoch_coef_reg_loss=epoch_coef_reg_loss
        )

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            qz_m = outputs["qz_m"]
            qz_v = outputs["qz_v"]
            z = outputs["z"]
            if give_mean:
                samples = Normal(qz_m, qz_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z = self.module.encoder_genes.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m

            latent += [z.cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_reconstruct_latent(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        ref_z_list = []
        rec_z_list = []
        for data in ref_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.module.inference(refer_genes, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_genes = torch.cat(ref_z_list, dim=0)
        if not give_mean:
            mc_samples = 1
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            inference_outputs = self.module.inference(input_genes, batch_index, n_samples=mc_samples)
            z_genes = inference_outputs['z'].mean(dim=0)
            senet_outputs = self.module.self_expressiveness(z_genes, ref_z_genes)
            rec_z = senet_outputs['rec_queries']
            rec_z_list.append(rec_z.cpu())
        return torch.cat(rec_z_list, dim=0).numpy()

    @torch.no_grad()
    def get_coeff(
            self,
            adata: Optional[AnnData] = None,
            batch_size: int = 128,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        coeff_genes_list = []
        ref_adata = self.ref_adata
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            _, senet_outputs, _ = self.module(input_genes, ref_adata, batch_index, compute_loss=False)
            coeff_genes_list.append(senet_outputs['coeff'])
        coeff_genes = torch.cat(coeff_genes_list, dim=0)
        return coeff_genes.cpu().numpy()

    @torch.no_grad()
    def get_latent_library_size(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.module.training:
            self.module.eval()
        libraries = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            ql_m = outputs["ql_m"]
            ql_v = outputs["ql_v"]
            library_gene = outputs["library_gene"]
            if give_mean and (not self.module.use_observed_lib_size):
                untran_l = Normal(ql_m, ql_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                library = torch.exp(untran_l)
                library = library.mean(dim=0)
            else:
                library = library_gene

            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()

    @torch.no_grad()
    def get_normalized_expression(
            self,
            adata=None,
            indices=None,
            n_samples_overall: Optional[int] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            gene_list: Optional[Sequence[str]] = None,
            library_size: Optional[Union[float, Literal["latent"]]] = 1,
            n_samples: int = 1,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        r"""
        Returns the normalized gene expression and protein expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to use in total
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude.
        n_samples
            Get sample scale from multiple samples.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
            names as columns. If either n_samples=1 or return_mean=True, defaults to False.
            Otherwise, it defaults to True.

        Returns
        -------
        - **gene_normalized_expression** - normalized expression for RNA

        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is ``(samples, cells, genes)``.
        Otherwise, shape is ``(cells, genes)``. Return type is ``pd.DataFrame`` unless ``return_numpy`` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        scale_list_gene = []

        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            px_scale = torch.zeros_like(input_genes)
            if n_samples > 1:
                px_scale = torch.stack(n_samples * [px_scale])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_genes, ref_adata, batch_index, compute_loss=False,
                                                       transform_batch=b)
                if library_size == "latent":
                    px_scale += generative_outputs["rate"].cpu()
                else:
                    px_scale += generative_outputs["scale"].cpu()
                px_scale = px_scale[..., gene_mask]  # this doesn't change px_scale?

            px_scale /= len(transform_batch)
            scale_list_gene.append(px_scale)

        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            scale_list_gene = torch.cat(scale_list_gene, dim=1)
            # (cells, features, samples)
            scale_list_gene = scale_list_gene.permute(1, 2, 0)
        else:
            scale_list_gene = torch.cat(scale_list_gene, dim=0)

        if return_mean is True and n_samples > 1:
            scale_list_gene = torch.mean(scale_list_gene, dim=-1)

        scale_list_gene = scale_list_gene.cpu().numpy()
        if return_numpy is None or return_numpy is False:
            gene_df = pd.DataFrame(
                scale_list_gene,
                columns=self.adata.var_names[gene_mask],
                index=self.adata.obs_names[indices],
            )
            return gene_df
        else:
            return scale_list_gene

    # @torch.no_grad()
    # def get_protein_foreground_probability(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     protein_list: Optional[Sequence[str]] = None,
    #     n_samples: int = 1,
    #     batch_size: Optional[int] = None,
    #     return_mean: bool = True,
    #     return_numpy: Optional[bool] = None,
    # ):
    #     r"""
    #     Returns the foreground probability for proteins.
    #
    #     This is denoted as :math:`(1 - \pi_{nt})` in the totalVI paper.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     transform_batch
    #         Batch to condition on.
    #         If transform_batch is:
    #
    #         - None, then real observed batch is used
    #         - int, then batch transform_batch is used
    #         - List[int], then average over batches in list
    #     protein_list
    #         Return protein expression for a subset of genes.
    #         This can save memory when working with large datasets and few genes are
    #         of interest.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     return_mean
    #         Whether to return the mean of the samples.
    #     return_numpy
    #         Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
    #         gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
    #         Otherwise, it defaults to `True`.
    #
    #     Returns
    #     -------
    #     - **foreground_probability** - probability foreground for each protein
    #
    #     If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
    #     Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
    #     """
    #     adata = self._validate_anndata(adata)
    #     post = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     if protein_list is None:
    #         protein_mask = slice(None)
    #     else:
    #         all_proteins = self.scvi_setup_dict_["protein_names"]
    #         protein_mask = [True if p in protein_list else False for p in all_proteins]
    #
    #     if n_samples > 1 and return_mean is False:
    #         if return_numpy is False:
    #             warnings.warn(
    #                 "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
    #             )
    #         return_numpy = True
    #     if indices is None:
    #         indices = np.arange(adata.n_obs)
    #
    #     py_mixings = []
    #     if not isinstance(transform_batch, IterableClass):
    #         transform_batch = [transform_batch]
    #
    #     transform_batch = _get_batch_code_from_category(adata, transform_batch)
    #     for tensors in post:
    #         y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
    #         py_mixing = torch.zeros_like(y[..., protein_mask])
    #         if n_samples > 1:
    #             py_mixing = torch.stack(n_samples * [py_mixing])
    #         for b in transform_batch:
    #             generative_kwargs = dict(transform_batch=b)
    #             inference_kwargs = dict(n_samples=n_samples)
    #             _, generative_outputs = self.module.forward(
    #                 tensors=tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )
    #             py_mixing += torch.sigmoid(generative_outputs["py_"]["mixing"])[
    #                 ..., protein_mask
    #             ].cpu()
    #         py_mixing /= len(transform_batch)
    #         py_mixings += [py_mixing]
    #     if n_samples > 1:
    #         # concatenate along batch dimension -> result shape = (samples, cells, features)
    #         py_mixings = torch.cat(py_mixings, dim=1)
    #         # (cells, features, samples)
    #         py_mixings = py_mixings.permute(1, 2, 0)
    #     else:
    #         py_mixings = torch.cat(py_mixings, dim=0)
    #
    #     if return_mean is True and n_samples > 1:
    #         py_mixings = torch.mean(py_mixings, dim=-1)
    #
    #     py_mixings = py_mixings.cpu().numpy()
    #
    #     if return_numpy is True:
    #         return 1 - py_mixings
    #     else:
    #         pro_names = self.scvi_setup_dict_["protein_names"]
    #         foreground_prob = pd.DataFrame(
    #             1 - py_mixings,
    #             columns=pro_names[protein_mask],
    #             index=adata.obs_names[indices],
    #         )
    #         return foreground_prob
    #
    # def _expression_for_de(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples_overall=None,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     scale_protein=False,
    #     batch_size: Optional[int] = None,
    #     sample_protein_mixing=False,
    #     include_protein_background=False,
    #     protein_prior_count=0.5,
    # ):
    #     rna, protein = self.get_normalized_expression(
    #         adata=adata,
    #         indices=indices,
    #         n_samples_overall=n_samples_overall,
    #         transform_batch=transform_batch,
    #         return_numpy=True,
    #         n_samples=1,
    #         batch_size=batch_size,
    #         scale_protein=scale_protein,
    #         sample_protein_mixing=sample_protein_mixing,
    #         include_protein_background=include_protein_background,
    #     )
    #     protein += protein_prior_count
    #
    #     joint = np.concatenate([rna, protein], axis=1)
    #     return joint
    #
    #
    # def differential_expression(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     groupby: Optional[str] = None,
    #     group1: Optional[Iterable[str]] = None,
    #     group2: Optional[str] = None,
    #     idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     mode: Literal["vanilla", "change"] = "change",
    #     delta: float = 0.25,
    #     batch_size: Optional[int] = None,
    #     all_stats: bool = True,
    #     batch_correction: bool = False,
    #     batchid1: Optional[Iterable[str]] = None,
    #     batchid2: Optional[Iterable[str]] = None,
    #     fdr_target: float = 0.05,
    #     silent: bool = False,
    #     protein_prior_count: float = 0.1,
    #     scale_protein: bool = False,
    #     sample_protein_mixing: bool = False,
    #     include_protein_background: bool = False,
    #     **kwargs,
    # ) -> pd.DataFrame:
    #     r"""
    #     A unified method for differential expression analysis.
    #
    #     Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.
    #
    #     Parameters
    #     ----------
    #     {doc_differential_expression}
    #     protein_prior_count
    #         Prior count added to protein expression before LFC computation
    #     scale_protein
    #         Force protein values to sum to one in every single cell (post-hoc normalization)
    #     sample_protein_mixing
    #         Sample the protein mixture component, i.e., use the parameter to sample a Bernoulli
    #         that determines if expression is from foreground/background.
    #     include_protein_background
    #         Include the protein background component as part of the protein expression
    #     **kwargs
    #         Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`
    #
    #     Returns
    #     -------
    #     Differential expression DataFrame.
    #     """
    #     adata = self._validate_anndata(adata)
    #     model_fn = partial(
    #         self._expression_for_de,
    #         scale_protein=scale_protein,
    #         sample_protein_mixing=sample_protein_mixing,
    #         include_protein_background=include_protein_background,
    #         protein_prior_count=protein_prior_count,
    #         batch_size=batch_size,
    #     )
    #     col_names = np.concatenate(
    #         [
    #             np.asarray(_get_var_names_from_setup_anndata(adata)),
    #             self.scvi_setup_dict_["protein_names"],
    #         ]
    #     )
    #     result = _de_core(
    #         adata,
    #         model_fn,
    #         groupby,
    #         group1,
    #         group2,
    #         idx1,
    #         idx2,
    #         all_stats,
    #         cite_seq_raw_counts_properties,
    #         col_names,
    #         mode,
    #         batchid1,
    #         batchid2,
    #         delta,
    #         batch_correction,
    #         fdr_target,
    #         silent,
    #         **kwargs,
    #     )
    #
    #     return result
    #
    # @torch.no_grad()
    # def posterior_predictive_sample(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: int = 1,
    #     batch_size: Optional[int] = None,
    #     gene_list: Optional[Sequence[str]] = None,
    #     protein_list: Optional[Sequence[str]] = None,
    # ) -> np.ndarray:
    #     r"""
    #     Generate observation samples from the posterior predictive distribution.
    #
    #     The posterior predictive distribution is written as :math:`p(\hat{x}, \hat{y} \mid x, y)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of required samples for each cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     gene_list
    #         Names of genes of interest
    #     protein_list
    #         Names of proteins of interest
    #
    #     Returns
    #     -------
    #     x_new : :class:`~numpy.ndarray`
    #         tensor with shape (n_cells, n_genes, n_samples)
    #     """
    #     if self.module.gene_likelihood not in ["nb"]:
    #         raise ValueError("Invalid gene_likelihood")
    #
    #     adata = self._validate_anndata(adata)
    #     if gene_list is None:
    #         gene_mask = slice(None)
    #     else:
    #         all_genes = _get_var_names_from_setup_anndata(adata)
    #         gene_mask = [True if gene in gene_list else False for gene in all_genes]
    #     if protein_list is None:
    #         protein_mask = slice(None)
    #     else:
    #         all_proteins = self.scvi_setup_dict_["protein_names"]
    #         protein_mask = [True if p in protein_list else False for p in all_proteins]
    #
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         rna_sample, protein_sample = self.module.sample(
    #             tensors, n_samples=n_samples
    #         )
    #         rna_sample = rna_sample[..., gene_mask]
    #         protein_sample = protein_sample[..., protein_mask]
    #         data = torch.cat([rna_sample, protein_sample], dim=-1).numpy()
    #
    #         scdl_list += [data]
    #         if n_samples > 1:
    #             scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #     scdl_list = np.concatenate(scdl_list, axis=0)
    #
    #     return scdl_list
    #
    # @torch.no_grad()
    # def _get_denoised_samples(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 25,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[int] = None,
    # ) -> np.ndarray:
    #     """
    #     Return samples from an adjusted posterior predictive.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         indices of `adata` to use
    #     n_samples
    #         How may samples per cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         int of which batch to condition on for all cells
    #     """
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         x = tensors[_CONSTANTS.X_KEY]
    #         y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
    #
    #         generative_kwargs = dict(transform_batch=transform_batch)
    #         inference_kwargs = dict(n_samples=n_samples)
    #         with torch.no_grad():
    #             inference_outputs, generative_outputs, = self.module.forward(
    #                 tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )
    #         px_ = generative_outputs["px_"]
    #         py_ = generative_outputs["py_"]
    #         device = px_["r"].device
    #
    #         pi = 1 / (1 + torch.exp(-py_["mixing"]))
    #         mixing_sample = torch.distributions.Bernoulli(pi).sample()
    #         protein_rate = py_["rate_fore"]
    #         rate = torch.cat((rna_size_factor * px_["scale"], protein_rate), dim=-1)
    #         if len(px_["r"].size()) == 2:
    #             px_dispersion = px_["r"]
    #         else:
    #             px_dispersion = torch.ones_like(x).to(device) * px_["r"]
    #         if len(py_["r"].size()) == 2:
    #             py_dispersion = py_["r"]
    #         else:
    #             py_dispersion = torch.ones_like(y).to(device) * py_["r"]
    #
    #         dispersion = torch.cat((px_dispersion, py_dispersion), dim=-1)
    #
    #         # This gamma is really l*w using scVI manuscript notation
    #         p = rate / (rate + dispersion)
    #         r = dispersion
    #         l_train = torch.distributions.Gamma(r, (1 - p) / p).sample()
    #         data = l_train.cpu().numpy()
    #         # make background 0
    #         data[:, :, self.adata.shape[1] :] = (
    #             data[:, :, self.adata.shape[1] :] * (1 - mixing_sample).cpu().numpy()
    #         )
    #         scdl_list += [data]
    #
    #         scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #
    #     return np.concatenate(scdl_list, axis=0)
    #
    # @torch.no_grad()
    # def get_feature_correlation_matrix(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 10,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     correlation_type: Literal["spearman", "pearson"] = "spearman",
    #     log_transform: bool = False,
    # ) -> pd.DataFrame:
    #     """
    #     Generate gene-gene correlation matrix using scvi uncertainty and expression.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         Batches to condition on.
    #         If transform_batch is:
    #
    #         - None, then real observed batch is used
    #         - int, then batch transform_batch is used
    #         - list of int, then values are averaged over provided batches.
    #     correlation_type
    #         One of "pearson", "spearman".
    #     log_transform
    #         Whether to log transform denoised values prior to correlation calculation.
    #
    #     Returns
    #     -------
    #     Gene-protein-gene-protein correlation matrix
    #     """
    #     from scipy.stats import spearmanr
    #
    #     adata = self._validate_anndata(adata)
    #
    #     if not isinstance(transform_batch, IterableClass):
    #         transform_batch = [transform_batch]
    #
    #     transform_batch = _get_batch_code_from_category(adata, transform_batch)
    #
    #     corr_mats = []
    #     for b in transform_batch:
    #         denoised_data = self._get_denoised_samples(
    #             n_samples=n_samples,
    #             batch_size=batch_size,
    #             rna_size_factor=rna_size_factor,
    #             transform_batch=b,
    #         )
    #         flattened = np.zeros(
    #             (denoised_data.shape[0] * n_samples, denoised_data.shape[1])
    #         )
    #         for i in range(n_samples):
    #             flattened[
    #                 denoised_data.shape[0] * (i) : denoised_data.shape[0] * (i + 1)
    #             ] = denoised_data[:, :, i]
    #         if log_transform is True:
    #             flattened[:, : self.n_genes] = np.log(
    #                 flattened[:, : self.n_genes] + 1e-8
    #             )
    #             flattened[:, self.n_genes :] = np.log1p(flattened[:, self.n_genes :])
    #         if correlation_type == "pearson":
    #             corr_matrix = np.corrcoef(flattened, rowvar=False)
    #         else:
    #             corr_matrix, _ = spearmanr(flattened, axis=0)
    #         corr_mats.append(corr_matrix)
    #
    #     corr_matrix = np.mean(np.stack(corr_mats), axis=0)
    #     var_names = _get_var_names_from_setup_anndata(adata)
    #     names = np.concatenate(
    #         [np.asarray(var_names), self.scvi_setup_dict_["protein_names"]]
    #     )
    #     return pd.DataFrame(corr_matrix, index=names, columns=names)
    #
    # @torch.no_grad()
    # def get_likelihood_parameters(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: Optional[int] = 1,
    #     give_mean: Optional[bool] = False,
    #     batch_size: Optional[int] = None,
    # ) -> Dict[str, np.ndarray]:
    #     r"""
    #     Estimates for the parameters of the likelihood :math:`p(x, y \mid z)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     give_mean
    #         Return expected value of parameters or a samples
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     """
    #     raise NotImplementedError
    #
    # def _validate_anndata(
    #     self, adata: Optional[AnnData] = None, copy_if_view: bool = True
    # ):
    #     adata = super()._validate_anndata(adata, copy_if_view)
    #     error_msg = "Number of {} in anndata different from when setup_anndata was run. Please rerun setup_anndata."
    #     if _CONSTANTS.PROTEIN_EXP_KEY in adata.uns["_scvi"]["data_registry"].keys():
    #         pro_exp = get_from_registry(adata, _CONSTANTS.PROTEIN_EXP_KEY)
    #         if self.summary_stats["n_proteins"] != pro_exp.shape[1]:
    #             raise ValueError(error_msg.format("proteins"))
    #         is_nonneg_int = _check_nonnegative_integers(pro_exp)
    #         if not is_nonneg_int:
    #             warnings.warn(
    #                 "Make sure the registered protein expression in anndata contains unnormalized count data."
    #             )
    #     else:
    #         raise ValueError("No protein data found, please setup or transfer anndata")
    #
    #     return adata
    #
    # @torch.no_grad()
    # def get_protein_background_mean(self, adata, indices, batch_size):
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #     background_mean = []
    #     for tensors in scdl:
    #         _, inference_outputs, _ = self.module.forward(tensors)
    #         b_mean = inference_outputs["py_"]["rate_back"]
    #         background_mean += [b_mean.cpu().numpy()]
    #     return np.concatenate(background_mean)
    #
    # @staticmethod
    # @setup_anndata_dsp.dedent
    # def setup_anndata(
    #     adata: AnnData,
    #     protein_expression_obsm_key: str,
    #     protein_names_uns_key: Optional[str] = None,
    #     batch_key: Optional[str] = None,
    #     layer: Optional[str] = None,
    #     categorical_covariate_keys: Optional[List[str]] = None,
    #     continuous_covariate_keys: Optional[List[str]] = None,
    #     copy: bool = False,
    # ) -> Optional[AnnData]:
    #     """
    #     %(summary)s.
    #
    #     Parameters
    #     ----------
    #     %(param_adata)s
    #     protein_expression_obsm_key
    #         key in `adata.obsm` for protein expression data.
    #     protein_names_uns_key
    #         key in `adata.uns` for protein names. If None, will use the column names of `adata.obsm[protein_expression_obsm_key]`
    #         if it is a DataFrame, else will assign sequential names to proteins.
    #     %(param_batch_key)s
    #     %(param_layer)s
    #     %(param_cat_cov_keys)s
    #     %(param_cont_cov_keys)s
    #     %(param_copy)s
    #
    #     Returns
    #     -------
    #     %(returns)s
    #     """
    #     return _setup_anndata(
    #         adata,
    #         batch_key=batch_key,
    #         layer=layer,
    #         protein_expression_obsm_key=protein_expression_obsm_key,
    #         protein_names_uns_key=protein_names_uns_key,
    #         categorical_covariate_keys=categorical_covariate_keys,
    #         continuous_covariate_keys=continuous_covariate_keys,
    #         copy=copy,
    #     )
    #
    #
    #


class SCMSIProtein(nn.Module):
    """
    Parameters
    ----------
    adata
        AnnData object.
    n_latent
        Dimensionality of the latent space.
    protein_dispersion
        One of the following:

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED
    protein_likelihood
        One of:

        * ``'nbm'`` - Negative binomial mixture distribution
        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    empirical_protein_background_prior
        Set the initialization of protein background prior empirically. This option fits a GMM for each of
        100 cells per batch and averages the distributions. Note that even with this option set to `True`,
        this only initializes a parameter that is learned during inference. If `False`, randomly initializes.
        The default (`None`), sets this to `True` if greater than 10 proteins are used.

    **model_kwargs
        Keyword args for :class:`~scMSI.scMSI_main.scMSIVAE`

    Notes
    -----
    See further usage examples in the following tutorials:

    to be updated

    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            protein_dispersion: Literal[
                "protein", "protein-batch", "protein-label"
            ] = "protein",
            protein_likelihood: Literal["nbm", "nb", "poisson"] = "nbm",
            latent_distribution: Literal["normal", "ln"] = "normal",
            empirical_protein_background_prior: Optional[bool] = None,
            **model_kwargs,
    ):
        super(SCMSIProtein, self).__init__()
        self.adata = adata
        emp_prior = (
            empirical_protein_background_prior
            if empirical_protein_background_prior is not None
            else (adata.obsm["protein_expression"].shape[1] > 10)
        )
        if "batch" not in adata.obs.keys():
            adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
        if "n_batch" not in adata.uns.keys():
            adata.uns["n_batch"] = 1
        if emp_prior:
            batch_mask = None
            prior_mean, prior_scale = get_protein_priors(adata, batch_mask)
        else:
            prior_mean, prior_scale = None, None

        n_cats_per_cov = (
            adata.uns["extra_categoricals"]["n_cats_per_key"]
            if "extra_categoricals" in adata.uns.keys()
            else None
        )
        self.adata = adata
        n_batch = self.adata.uns["n_batch"]
        # train_adata, val_adata = data_splitter(
        #     self.adata,
        #     train_size=0.9,
        #     validation_size=0.1,
        #     use_gpu=False,
        # )
        # self.train_adata = train_adata
        # self.val_adata = val_adata
        prior_mean[4] = 5.0
        self.module = ProteinSVAE(
            n_input_proteins=adata.obsm['protein_expression'].shape[1],
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=0,
            n_cats_per_cov=n_cats_per_cov,
            protein_dispersion=protein_dispersion,
            protein_likelihood=protein_likelihood,
            latent_distribution=latent_distribution,
            protein_background_prior_mean=prior_mean,
            protein_background_prior_scale=prior_scale,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 4e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            n_batch_ref: int = 4,
            early_stopping: bool = False,
            check_val_every_n_epoch: Optional[int] = None,
            reduce_lr_on_plateau: bool = True,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            adversarial_classifier: Optional[bool] = None,
            plan_kwargs: Optional[dict] = None,
            weight_decay=1e-6,
            record_loss=True,
            **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        n_batch_ref
            reference samples number.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        weight_decay
            weight decay
        record_loss
            whether save loss
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=0.9,
            validation_size=0.1,
            use_gpu=False,
        )
        self.train_adata = train_adata
        self.val_adata = val_adata
        self.batch_size = batch_size

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        if n_batch_ref * batch_size > train_adata.shape[0]:
            ref_idx = np.arange(train_adata.shape[0])
        else:
            ref_idx = get_anchor_index(train_adata.X, n_pcs=50, n_clusters=n_batch_ref * batch_size)
        # ref_idx = np.random.permutation(train_adata.shape[1])[0:500]
        self.ref_adata = train_adata[ref_idx, :]
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_loss = []
        epoch_rec_loss = []
        epoch_kl_z_loss = []
        epoch_kl_back_loss = []
        epoch_se_loss = []
        epoch_coef_reg_loss = []
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            batch_loss = []
            batch_rec_loss = []
            batch_kl_z_loss = []
            batch_kl_back_loss = []
            batch_se_loss = []
            batch_coef_reg_loss = []
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            for train_batch in train_adata_batch:
                input_proteins = torch.tensor(train_batch.obsm['protein_expression'], dtype=torch.float32)
                if "batch" in train_batch.obs.keys():
                    batch_index = torch.tensor(train_batch.obs['batch'], dtype=torch.float32).reshape(-1, 1)
                else:
                    batch_index = None
                _, _, _, all_loss = self.module(input_proteins, ref_adata, batch_index)
                loss, reconst_loss, kl_local, se_losses = all_loss
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), 0.001)
                optimizer.step()
                if record_loss:
                    batch_loss.append(loss.data.numpy())
                    batch_rec_loss.append(torch.mean(reconst_loss).data.numpy())
                    batch_kl_z_loss.append(torch.mean(kl_local['kl_div_z']).data.numpy())
                    if self.module.protein_likelihood == 'nbm':
                        batch_kl_back_loss.append(torch.mean(kl_local['kl_div_back']).data.numpy())
                    else:
                        batch_kl_back_loss.append(kl_local['kl_div_back'])
                    batch_se_loss.append(torch.mean(se_losses['se_loss']).data.numpy())
                    batch_coef_reg_loss.append(torch.mean(se_losses['coeff_reg']).data.numpy())

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                print("epoch: {}; loss: {} ".format(epoch, np.mean(batch_loss)))
            scheduler.step()
            # self.module.eval()
            epoch_loss.append(np.mean(batch_loss))
            epoch_rec_loss.append(np.mean(batch_rec_loss))
            epoch_kl_z_loss.append(np.mean(batch_kl_z_loss))
            epoch_kl_back_loss.append(np.mean(batch_kl_back_loss))
            epoch_se_loss.append(np.mean(batch_se_loss))
            epoch_coef_reg_loss.append(np.mean(batch_rec_loss))
        self.history = dict(
            epoch_loss=epoch_loss,
            epoch_rec_loss=epoch_rec_loss,
            epoch_kl_z_loss=epoch_kl_z_loss,
            epoch_kl_back_loss=epoch_kl_back_loss,
            epoch_se_loss=epoch_se_loss,
            epoch_coef_reg_loss=epoch_coef_reg_loss
        )

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            qz_m = outputs["qz_m"]
            qz_v = outputs["qz_v"]
            z = outputs["z"]
            if give_mean:
                samples = Normal(qz_m, qz_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z = self.module.encoder_proteins.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m

            latent += [z.cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_reconstruct_latent(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        ref_z_list = []
        rec_z_list = []
        for data in ref_adata:
            refer_genes = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.module.inference(refer_genes, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_genes = torch.cat(ref_z_list, dim=0)
        if not give_mean:
            mc_samples = 1
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            inference_outputs = self.module.inference(input_genes, batch_index, n_samples=mc_samples)
            z_genes = inference_outputs['z'].mean(dim=0)
            senet_outputs = self.module.self_expressiveness(z_genes, ref_z_genes)
            rec_z = senet_outputs['rec_queries']
            rec_z_list.append(rec_z.cpu())
        return torch.cat(rec_z_list, dim=0).numpy()

    @torch.no_grad()
    def get_coeff(
            self,
            adata: Optional[AnnData] = None,
            batch_size: int = 128,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        coeff_genes_list = []
        ref_adata = self.ref_adata
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            _, senet_outputs, _ = self.module(input_genes, ref_adata, batch_index, compute_loss=False)
            coeff_genes_list.append(senet_outputs['coeff'])
        coeff_genes = torch.cat(coeff_genes_list, dim=0)
        return coeff_genes.cpu().numpy()

    @torch.no_grad()
    def get_normalized_expression(
            self,
            adata=None,
            indices=None,
            n_samples_overall: Optional[int] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            protein_list: Optional[Sequence[str]] = None,
            n_samples: int = 1,
            sample_protein_mixing: bool = False,
            scale_protein: bool = False,
            include_protein_background: bool = False,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        r"""
        Returns the normalized gene expression and protein expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to use in total
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Get sample scale from multiple samples.
        sample_protein_mixing
            Sample mixing bernoulli, setting background to zero
        scale_protein
            Make protein expression sum to 1
        include_protein_background
            Include background component for protein expression
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
            names as columns. If either n_samples=1 or return_mean=True, defaults to False.
            Otherwise, it defaults to True.

        Returns
        -------
        - **protein_normalized_expression** - normalized expression for proteins

        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is ``(samples, cells, genes)``.
        Otherwise, shape is ``(cells, genes)``. Return type is ``pd.DataFrame`` unless ``return_numpy`` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = adata.uns['protein_names']
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        scale_list_pro = []
        for data in adata_batch:
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            py_scale = torch.zeros_like(input_pros)
            if n_samples > 1:
                py_scale = torch.stack(n_samples * [py_scale])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_pros, ref_adata, batch_index, compute_loss=False,
                                                       transform_batch=b)
                py_ = generative_outputs
                # probability of background
                protein_mixing = 1 / (1 + torch.exp(-py_["mixing"].cpu()))
                if sample_protein_mixing is True:
                    protein_mixing = torch.distributions.Bernoulli(
                        protein_mixing
                    ).sample()
                protein_val = py_["fore_rate"].cpu() * (1 - protein_mixing)
                if include_protein_background is True:
                    protein_val += py_["back_rate"].cpu() * protein_mixing

                if scale_protein is True:
                    protein_val = torch.nn.functional.normalize(
                        protein_val, p=1, dim=-1
                    )
                protein_val = protein_val[..., protein_mask]
                py_scale += protein_val
            py_scale /= len(transform_batch)
            scale_list_pro.append(py_scale)

        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            scale_list_pro = torch.cat(scale_list_pro, dim=1)
            # (cells, features, samples)
            scale_list_pro = scale_list_pro.permute(1, 2, 0)
        else:
            scale_list_pro = torch.cat(scale_list_pro, dim=0)

        if return_mean is True and n_samples > 1:
            scale_list_pro = torch.mean(scale_list_pro, dim=-1)

        scale_list_pro = scale_list_pro.cpu().numpy()
        if return_numpy is None or return_numpy is False:
            pro_df = pd.DataFrame(
                scale_list_pro,
                columns=self.adata.uns["protein_names"][protein_mask],
                index=self.adata.obs_names[indices],
            )

            return pro_df
        else:
            return scale_list_pro

    @torch.no_grad()
    def get_protein_foreground_probability(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            protein_list: Optional[Sequence[str]] = None,
            n_samples: int = 1,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ):
        r"""
        Returns the foreground probability for proteins.

        This is denoted as :math:`(1 - \pi_{nt})` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        - **foreground_probability** - probability foreground for each protein

        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)

        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = adata.uns['protein_names']
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        py_mixings = []
        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        for data in adata_batch:
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            py_mixing = torch.zeros_like(input_pros[..., protein_mask])
            if n_samples > 1:
                py_mixing = torch.stack(n_samples * [py_mixing])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_pros, ref_adata, batch_index, compute_loss=False,
                                                       transform_batch=b)
                py_ = generative_outputs
                # probability of background
                py_mixing += torch.sigmoid(py_["mixing"])[..., protein_mask].cpu()
            py_mixing /= len(transform_batch)
            py_mixings += [py_mixing]
        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            py_mixings = torch.cat(py_mixings, dim=1)
            # (cells, features, samples)
            py_mixings = py_mixings.permute(1, 2, 0)
        else:
            py_mixings = torch.cat(py_mixings, dim=0)

        if return_mean is True and n_samples > 1:
            py_mixings = torch.mean(py_mixings, dim=-1)

        py_mixings = py_mixings.cpu().numpy()

        if return_numpy is True:
            return 1 - py_mixings
        else:
            pro_names = self.adata.uns["protein_names"]
            foreground_prob = pd.DataFrame(
                1 - py_mixings,
                columns=pro_names[protein_mask],
                index=adata.obs_names[indices],
            )
            return foreground_prob

    @torch.no_grad()
    def get_protein_likelihood_parameters(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        back_rate_list = []
        fore_rate_list = []
        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        for data in adata_batch:
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            _, _, generative_outputs = self.module(input_pros, ref_adata, batch_index, compute_loss=False)
            py_ = generative_outputs
            back_rate_list += [py_['back_rate']]
            fore_rate_list += [py_['fore_rate']]
        back_rate = torch.cat(back_rate_list, dim=0).cpu().numpy()
        fore_rate = torch.cat(fore_rate_list, dim=0).cpu().numpy()
        return back_rate, fore_rate

    def _expression_for_de(
            self,
            adata=None,
            indices=None,
            n_samples_overall=None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            scale_protein=False,
            batch_size: Optional[int] = None,
            sample_protein_mixing=False,
            include_protein_background=False,
            protein_prior_count=0.5,
    ):
        rna, protein = self.get_normalized_expression(
            adata=adata,
            indices=indices,
            n_samples_overall=n_samples_overall,
            transform_batch=transform_batch,
            return_numpy=True,
            n_samples=1,
            batch_size=batch_size,
            scale_protein=scale_protein,
            sample_protein_mixing=sample_protein_mixing,
            include_protein_background=include_protein_background,
        )
        protein += protein_prior_count

        joint = np.concatenate([rna, protein], axis=1)
        return joint

    # def differential_expression(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     groupby: Optional[str] = None,
    #     group1: Optional[Iterable[str]] = None,
    #     group2: Optional[str] = None,
    #     idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     mode: Literal["vanilla", "change"] = "change",
    #     delta: float = 0.25,
    #     batch_size: Optional[int] = None,
    #     all_stats: bool = True,
    #     batch_correction: bool = False,
    #     batchid1: Optional[Iterable[str]] = None,
    #     batchid2: Optional[Iterable[str]] = None,
    #     fdr_target: float = 0.05,
    #     silent: bool = False,
    #     protein_prior_count: float = 0.1,
    #     scale_protein: bool = False,
    #     sample_protein_mixing: bool = False,
    #     include_protein_background: bool = False,
    #     **kwargs,
    # ) -> pd.DataFrame:
    #     r"""
    #     A unified method for differential expression analysis.
    #
    #     Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.
    #
    #     Parameters
    #     ----------
    #     {doc_differential_expression}
    #     protein_prior_count
    #         Prior count added to protein expression before LFC computation
    #     scale_protein
    #         Force protein values to sum to one in every single cell (post-hoc normalization)
    #     sample_protein_mixing
    #         Sample the protein mixture component, i.e., use the parameter to sample a Bernoulli
    #         that determines if expression is from foreground/background.
    #     include_protein_background
    #         Include the protein background component as part of the protein expression
    #     **kwargs
    #         Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`
    #
    #     Returns
    #     -------
    #     Differential expression DataFrame.
    #     """
    #     adata = self._validate_anndata(adata)
    #     model_fn = partial(
    #         self._expression_for_de,
    #         scale_protein=scale_protein,
    #         sample_protein_mixing=sample_protein_mixing,
    #         include_protein_background=include_protein_background,
    #         protein_prior_count=protein_prior_count,
    #         batch_size=batch_size,
    #     )
    #     col_names = np.concatenate(
    #         [
    #             np.asarray(_get_var_names_from_setup_anndata(adata)),
    #             self.scvi_setup_dict_["protein_names"],
    #         ]
    #     )
    #     result = _de_core(
    #         adata,
    #         model_fn,
    #         groupby,
    #         group1,
    #         group2,
    #         idx1,
    #         idx2,
    #         all_stats,
    #         cite_seq_raw_counts_properties,
    #         col_names,
    #         mode,
    #         batchid1,
    #         batchid2,
    #         delta,
    #         batch_correction,
    #         fdr_target,
    #         silent,
    #         **kwargs,
    #     )
    #
    #     return result

    # @torch.no_grad()
    # def posterior_predictive_sample(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: int = 1,
    #     batch_size: Optional[int] = None,
    #     gene_list: Optional[Sequence[str]] = None,
    #     protein_list: Optional[Sequence[str]] = None,
    # ) -> np.ndarray:
    #     r"""
    #     Generate observation samples from the posterior predictive distribution.
    #
    #     The posterior predictive distribution is written as :math:`p(\hat{x}, \hat{y} \mid x, y)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of required samples for each cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     gene_list
    #         Names of genes of interest
    #     protein_list
    #         Names of proteins of interest
    #
    #     Returns
    #     -------
    #     x_new : :class:`~numpy.ndarray`
    #         tensor with shape (n_cells, n_genes, n_samples)
    #     """
    #     if self.module.gene_likelihood not in ["nb"]:
    #         raise ValueError("Invalid gene_likelihood")
    #
    #     adata = self._validate_anndata(adata)
    #     if gene_list is None:
    #         gene_mask = slice(None)
    #     else:
    #         all_genes = _get_var_names_from_setup_anndata(adata)
    #         gene_mask = [True if gene in gene_list else False for gene in all_genes]
    #     if protein_list is None:
    #         protein_mask = slice(None)
    #     else:
    #         all_proteins = self.scvi_setup_dict_["protein_names"]
    #         protein_mask = [True if p in protein_list else False for p in all_proteins]
    #
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         rna_sample, protein_sample = self.module.sample(
    #             tensors, n_samples=n_samples
    #         )
    #         rna_sample = rna_sample[..., gene_mask]
    #         protein_sample = protein_sample[..., protein_mask]
    #         data = torch.cat([rna_sample, protein_sample], dim=-1).numpy()
    #
    #         scdl_list += [data]
    #         if n_samples > 1:
    #             scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #     scdl_list = np.concatenate(scdl_list, axis=0)
    #
    #     return scdl_list
    #
    # @torch.no_grad()
    # def _get_denoised_samples(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 25,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[int] = None,
    # ) -> np.ndarray:
    #     """
    #     Return samples from an adjusted posterior predictive.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         indices of `adata` to use
    #     n_samples
    #         How may samples per cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         int of which batch to condition on for all cells
    #     """
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         x = tensors[_CONSTANTS.X_KEY]
    #         y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
    #
    #         generative_kwargs = dict(transform_batch=transform_batch)
    #         inference_kwargs = dict(n_samples=n_samples)
    #         with torch.no_grad():
    #             inference_outputs, generative_outputs, = self.module.forward(
    #                 tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )
    #         px_ = generative_outputs["px_"]
    #         py_ = generative_outputs["py_"]
    #         device = px_["r"].device
    #
    #         pi = 1 / (1 + torch.exp(-py_["mixing"]))
    #         mixing_sample = torch.distributions.Bernoulli(pi).sample()
    #         protein_rate = py_["rate_fore"]
    #         rate = torch.cat((rna_size_factor * px_["scale"], protein_rate), dim=-1)
    #         if len(px_["r"].size()) == 2:
    #             px_dispersion = px_["r"]
    #         else:
    #             px_dispersion = torch.ones_like(x).to(device) * px_["r"]
    #         if len(py_["r"].size()) == 2:
    #             py_dispersion = py_["r"]
    #         else:
    #             py_dispersion = torch.ones_like(y).to(device) * py_["r"]
    #
    #         dispersion = torch.cat((px_dispersion, py_dispersion), dim=-1)
    #
    #         # This gamma is really l*w using scVI manuscript notation
    #         p = rate / (rate + dispersion)
    #         r = dispersion
    #         l_train = torch.distributions.Gamma(r, (1 - p) / p).sample()
    #         data = l_train.cpu().numpy()
    #         # make background 0
    #         data[:, :, self.adata.shape[1] :] = (
    #             data[:, :, self.adata.shape[1] :] * (1 - mixing_sample).cpu().numpy()
    #         )
    #         scdl_list += [data]
    #
    #         scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #
    #     return np.concatenate(scdl_list, axis=0)
    #
    # @torch.no_grad()
    # def get_feature_correlation_matrix(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 10,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     correlation_type: Literal["spearman", "pearson"] = "spearman",
    #     log_transform: bool = False,
    # ) -> pd.DataFrame:
    #     """
    #     Generate gene-gene correlation matrix using scvi uncertainty and expression.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         Batches to condition on.
    #         If transform_batch is:
    #
    #         - None, then real observed batch is used
    #         - int, then batch transform_batch is used
    #         - list of int, then values are averaged over provided batches.
    #     correlation_type
    #         One of "pearson", "spearman".
    #     log_transform
    #         Whether to log transform denoised values prior to correlation calculation.
    #
    #     Returns
    #     -------
    #     Gene-protein-gene-protein correlation matrix
    #     """
    #     from scipy.stats import spearmanr
    #
    #     adata = self._validate_anndata(adata)
    #
    #     if not isinstance(transform_batch, IterableClass):
    #         transform_batch = [transform_batch]
    #
    #     transform_batch = _get_batch_code_from_category(adata, transform_batch)
    #
    #     corr_mats = []
    #     for b in transform_batch:
    #         denoised_data = self._get_denoised_samples(
    #             n_samples=n_samples,
    #             batch_size=batch_size,
    #             rna_size_factor=rna_size_factor,
    #             transform_batch=b,
    #         )
    #         flattened = np.zeros(
    #             (denoised_data.shape[0] * n_samples, denoised_data.shape[1])
    #         )
    #         for i in range(n_samples):
    #             flattened[
    #                 denoised_data.shape[0] * (i) : denoised_data.shape[0] * (i + 1)
    #             ] = denoised_data[:, :, i]
    #         if log_transform is True:
    #             flattened[:, : self.n_genes] = np.log(
    #                 flattened[:, : self.n_genes] + 1e-8
    #             )
    #             flattened[:, self.n_genes :] = np.log1p(flattened[:, self.n_genes :])
    #         if correlation_type == "pearson":
    #             corr_matrix = np.corrcoef(flattened, rowvar=False)
    #         else:
    #             corr_matrix, _ = spearmanr(flattened, axis=0)
    #         corr_mats.append(corr_matrix)
    #
    #     corr_matrix = np.mean(np.stack(corr_mats), axis=0)
    #     var_names = _get_var_names_from_setup_anndata(adata)
    #     names = np.concatenate(
    #         [np.asarray(var_names), self.scvi_setup_dict_["protein_names"]]
    #     )
    #     return pd.DataFrame(corr_matrix, index=names, columns=names)
    #
    # @torch.no_grad()
    # def get_likelihood_parameters(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: Optional[int] = 1,
    #     give_mean: Optional[bool] = False,
    #     batch_size: Optional[int] = None,
    # ) -> Dict[str, np.ndarray]:
    #     r"""
    #     Estimates for the parameters of the likelihood :math:`p(x, y \mid z)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     give_mean
    #         Return expected value of parameters or a samples
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     """
    #     raise NotImplementedError
    #
    # def _validate_anndata(
    #     self, adata: Optional[AnnData] = None, copy_if_view: bool = True
    # ):
    #     adata = super()._validate_anndata(adata, copy_if_view)
    #     error_msg = "Number of {} in anndata different from when setup_anndata was run. Please rerun setup_anndata."
    #     if _CONSTANTS.PROTEIN_EXP_KEY in adata.uns["_scvi"]["data_registry"].keys():
    #         pro_exp = get_from_registry(adata, _CONSTANTS.PROTEIN_EXP_KEY)
    #         if self.summary_stats["n_proteins"] != pro_exp.shape[1]:
    #             raise ValueError(error_msg.format("proteins"))
    #         is_nonneg_int = _check_nonnegative_integers(pro_exp)
    #         if not is_nonneg_int:
    #             warnings.warn(
    #                 "Make sure the registered protein expression in anndata contains unnormalized count data."
    #             )
    #     else:
    #         raise ValueError("No protein data found, please setup or transfer anndata")
    #
    #     return adata

    # @torch.no_grad()
    # def get_protein_background_mean(self, adata, indices, batch_size):
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #     background_mean = []
    #     for tensors in scdl:
    #         _, inference_outputs, _ = self.module.forward(tensors)
    #         b_mean = inference_outputs["py_"]["rate_back"]
    #         background_mean += [b_mean.cpu().numpy()]
    #     return np.concatenate(background_mean)

    # @staticmethod
    # @setup_anndata_dsp.dedent
    # def setup_anndata(
    #     adata: AnnData,
    #     protein_expression_obsm_key: str,
    #     protein_names_uns_key: Optional[str] = None,
    #     batch_key: Optional[str] = None,
    #     layer: Optional[str] = None,
    #     categorical_covariate_keys: Optional[List[str]] = None,
    #     continuous_covariate_keys: Optional[List[str]] = None,
    #     copy: bool = False,
    # ) -> Optional[AnnData]:
    #     """
    #     %(summary)s.
    #
    #     Parameters
    #     ----------
    #     %(param_adata)s
    #     protein_expression_obsm_key
    #         key in `adata.obsm` for protein expression data.
    #     protein_names_uns_key
    #         key in `adata.uns` for protein names. If None, will use the column names of `adata.obsm[protein_expression_obsm_key]`
    #         if it is a DataFrame, else will assign sequential names to proteins.
    #     %(param_batch_key)s
    #     %(param_layer)s
    #     %(param_cat_cov_keys)s
    #     %(param_cont_cov_keys)s
    #     %(param_copy)s
    #
    #     Returns
    #     -------
    #     %(returns)s
    #     """
    #     return _setup_anndata(
    #         adata,
    #         batch_key=batch_key,
    #         layer=layer,
    #         protein_expression_obsm_key=protein_expression_obsm_key,
    #         protein_names_uns_key=protein_names_uns_key,
    #         categorical_covariate_keys=categorical_covariate_keys,
    #         continuous_covariate_keys=continuous_covariate_keys,
    #         copy=copy,
    #     )
    #
    #
    #


class SCMSIPeak(nn.Module):
    """
    Parameters
    ----------
    adata
        AnnData object.
    n_latent
        Dimensionality of the latent space.
    gene_dispersion
        One of the following:

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    empirical_protein_background_prior
        Set the initialization of protein background prior empirically. This option fits a GMM for each of
        100 cells per batch and averages the distributions. Note that even with this option set to `True`,
        this only initializes a parameter that is learned during inference. If `False`, randomly initializes.
        The default (`None`), sets this to `True` if greater than 10 proteins are used.
    override_missing_proteins
        If `True`, will not treat proteins with all 0 expression in a particular batch as missing.
    **model_kwargs
        Keyword args for :class:`~scMSI.scMSI_main.scMSIVAE`

    Notes
    -----
    See further usage examples in the following tutorials:

    to be updated

    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            latent_distribution: Literal["normal", "ln"] = "normal",
            **model_kwargs,
    ):
        super(SCMSIPeak, self).__init__()
        self.adata = adata
        if "batch" not in adata.obs.keys():
            adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
        if "n_batch" not in adata.uns.keys():
            adata.uns["n_batch"] = 1

        n_cats_per_cov = (
            adata.uns["extra_categoricals"]["n_cats_per_key"]
            if "extra_categoricals" in adata.uns.keys()
            else None
        )
        n_batch = self.adata.uns["n_batch"]

        self.module = PeakSVAE(
            n_input_peaks=adata.obsm['peak_counts'].shape[1],
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=0,
            n_cats_per_cov=n_cats_per_cov,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 1e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            n_sample_ref: int = 50,
            early_stopping: bool = False,
            check_val_every_n_epoch: Optional[int] = None,
            reduce_lr_on_plateau: bool = True,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            adversarial_classifier: Optional[bool] = None,
            plan_kwargs: Optional[dict] = None,
            weight_decay=1e-3,
            record_loss=True,
            **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        n_sample_ref
            reference samples number.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        weight_decay
            weight decay
        record_loss
            whether save loss
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=0.9,
            validation_size=0.1,
            use_gpu=False,
        )
        self.train_adata = train_adata
        self.val_adata = val_adata
        self.batch_size = batch_size

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        if n_sample_ref > train_adata.shape[0]:
            ref_idx = np.arange(train_adata.shape[0])
        else:
            ref_idx = get_anchor_index(train_adata.X, n_pcs=50, n_clusters=n_sample_ref)
        # ref_idx = np.random.permutation(train_adata.shape[1])[0:500]
        self.ref_adata = train_adata[ref_idx, :]
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_loss = []
        epoch_rec_loss = []
        epoch_kl_z_loss = []
        epoch_kl_l_loss = []
        epoch_se_loss = []
        epoch_coef_reg_loss = []
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            batch_loss = []
            batch_rec_loss = []
            batch_kl_z_loss = []
            batch_kl_l_loss = []
            batch_se_loss = []
            batch_coef_reg_loss = []
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            for train_batch in train_adata_batch:
                input_peaks = torch.tensor(train_batch.obsm['peak_counts'], dtype=torch.float32)
                if "batch" in train_batch.obs.keys():
                    batch_index = torch.tensor(train_batch.obs['batch'], dtype=torch.float32).reshape(-1, 1)
                else:
                    batch_index = None
                _, _, _, all_loss = self.module(input_peaks, ref_adata, batch_index)
                loss, reconst_loss, kl_local, se_losses = all_loss
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), 0.001)
                optimizer.step()
                if record_loss:
                    batch_loss.append(loss.data.numpy())
                    batch_rec_loss.append(torch.mean(reconst_loss).data.numpy())
                    batch_kl_z_loss.append(torch.mean(kl_local['kl_div_z']).data.numpy())
                    batch_se_loss.append(torch.mean(se_losses['se_loss']).data.numpy())
                    batch_coef_reg_loss.append(torch.mean(se_losses['coeff_reg']).data.numpy())

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                print("epoch: {}; loss: {} ".format(epoch, np.mean(batch_loss)))
            # scheduler.step()
            # self.module.eval()
            epoch_loss.append(np.mean(batch_loss))
            epoch_rec_loss.append(np.mean(batch_rec_loss))
            epoch_kl_z_loss.append(np.mean(batch_kl_z_loss))
            epoch_se_loss.append(np.mean(batch_se_loss))
            epoch_coef_reg_loss.append(np.mean(batch_rec_loss))
        self.history = dict(
            epoch_loss=epoch_loss,
            epoch_rec_loss=epoch_rec_loss,
            epoch_kl_z_loss=epoch_kl_z_loss,
            epoch_se_loss=epoch_se_loss,
            epoch_coef_reg_loss=epoch_coef_reg_loss
        )

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            qz_m = outputs["qz_m"]
            qz_v = outputs["qz_v"]
            z = outputs["z"]
            if give_mean:
                samples = Normal(qz_m, qz_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z = self.module.encoder_peaks.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m

            latent += [z.cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_reconstruct_latent(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        ref_z_list = []
        rec_z_list = []
        for data in ref_adata:
            refer_peaks = torch.tensor(data.obsm["peak_counts"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.module.inference(refer_peaks, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_peaks = torch.cat(ref_z_list, dim=0)
        if not give_mean:
            mc_samples = 1
        for data in adata_batch:
            input_peaks = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            inference_outputs = self.module.inference(input_peaks, batch_index, n_samples=mc_samples)
            z_peaks = inference_outputs['z'].mean(dim=0)
            senet_outputs = self.module.self_expressiveness(z_peaks, ref_z_peaks)
            rec_z = senet_outputs['rec_queries']
            rec_z_list.append(rec_z.cpu())
        return torch.cat(rec_z_list, dim=0).numpy()

    @torch.no_grad()
    def get_coeff(
            self,
            adata: Optional[AnnData] = None,
            batch_size: int = 128,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        coeff_genes_list = []
        ref_adata = self.ref_adata
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            _, senet_outputs, _ = self.module(input_genes, ref_adata, batch_index, compute_loss=False)
            coeff_genes_list.append(senet_outputs['coeff'])
        coeff_genes = torch.cat(coeff_genes_list, dim=0)
        return coeff_genes.cpu().numpy()

    @torch.no_grad()
    def get_cell_factors(
            self,
            adata: Optional[AnnData] = None,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.module.training:
            self.module.eval()
        cell_factors = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            cell_factor = outputs["lib"]
            cell_factors += [cell_factor.cpu()]
        return torch.cat(cell_factors).numpy()

    @torch.no_grad()
    def get_region_factors(self):
        """Return region-specific factors."""
        if self.module.region_factors is None:
            raise RuntimeError("region factors were not included in this model")
        return torch.sigmoid(self.module.region_factors).cpu().numpy()

    @torch.no_grad()
    def get_accessibility_estimates(
            self,
            adata: Optional[AnnData] = None,
            indices: Sequence[int] = None,
            n_samples_overall: Optional[int] = None,
            region_list: Optional[Sequence[str]] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            use_z_mean: bool = True,
            threshold: Optional[float] = None,
            normalize_cells: bool = False,
            normalize_regions: bool = False,
            batch_size: Optional[int] = None,
            return_numpy: Optional[bool] = None,
    ) -> Union[pd.DataFrame, np.ndarray, csr_matrix]:
        """
        Impute the full accessibility matrix.

        Returns a matrix of accessibility probabilities for each cell and genomic region in the input
        (for return matrix A, A[i,j] is the probability that region j is accessible in cell i).

        Parameters
        ----------
        adata
            AnnData object that has been registered with scvi. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to return in total
        region_list
            Return accessibility estimates for this subset of regions. if `None`, all regions are used.
            This can save memory when dealing with large datasets.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
        use_z_mean
            If True (default), use the distribution mean. Otherwise, sample from the distribution.
        threshold
            If provided, values below the threshold are replaced with 0 and a sparse matrix
            is returned instead. This is recommended for very large matrices. Must be between 0 and 1.
        normalize_cells
            Whether to reintroduce library size factors to scale the normalized probabilities.
            This makes the estimates closer to the input, but removes the library size correction.
            False by default.
        normalize_regions
            Whether to reintroduce region factors to scale the normalized probabilities. This makes
            the estimates closer to the input, but removes the region-level bias correction. False by
            default.
        batch_size
            Minibatch size for data loading into model
        return_numpy
            If `True` and `threshold=None`, return :class:`~numpy.ndarray`. If `True` and `threshold` is
            given, return :class:`~scipy.sparse.csr_matrix`. If `False`, return :class:`~pandas.DataFrame`.
            DataFrame includes regions names as columns.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)

        if region_list is None:
            region_mask = slice(None)
        else:
            all_regions = adata.var_names
            region_mask = [region in region_list for region in all_regions]
        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]
        if threshold is not None and (threshold < 0 or threshold > 1):
            raise ValueError("the provided threshold must be between 0 and 1")

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        imputed_list = []
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            imputed = torch.zeros_like(input_genes)
            for b in transform_batch:
                inference_outputs, _, generative_outputs = self.module(input_genes, ref_adata, batch_index,
                                                                       compute_loss=False, transform_batch=b)
                p = generative_outputs["prob"].cpu()
                if normalize_cells:
                    p *= inference_outputs["lib"].cpu()
                if normalize_regions:
                    p *= generative_outputs["rf"].cpu()
                if threshold:
                    p[p < threshold] = 0
                    p = csr_matrix(p.numpy())
                if region_list is not None:
                    p = p[:, region_mask]
                imputed += p

            imputed /= len(transform_batch)
            imputed_list.append(imputed)

        if threshold:  # imputed is a list of csr_matrix objects
            imputed_peak = vstack(imputed_list, format="csr")
        else:  # imputed is a list of tensors
            imputed_peak = torch.cat(imputed_list).numpy()

        if return_numpy:
            return imputed_peak
        elif threshold:
            return pd.DataFrame.sparse.from_spmatrix(
                imputed_peak,
                index=adata.obs_names[indices],
                columns=adata.var_names[region_mask],
            )
        else:
            return pd.DataFrame(
                imputed_peak,
                index=adata.obs_names[indices],
                columns=adata.uns['peak_names'][region_mask],
            )

    # @torch.no_grad()
    # def get_protein_foreground_probability(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     protein_list: Optional[Sequence[str]] = None,
    #     n_samples: int = 1,
    #     batch_size: Optional[int] = None,
    #     return_mean: bool = True,
    #     return_numpy: Optional[bool] = None,
    # ):
    #     r"""
    #     Returns the foreground probability for proteins.
    #
    #     This is denoted as :math:`(1 - \pi_{nt})` in the totalVI paper.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     transform_batch
    #         Batch to condition on.
    #         If transform_batch is:
    #
    #         - None, then real observed batch is used
    #         - int, then batch transform_batch is used
    #         - List[int], then average over batches in list
    #     protein_list
    #         Return protein expression for a subset of genes.
    #         This can save memory when working with large datasets and few genes are
    #         of interest.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     return_mean
    #         Whether to return the mean of the samples.
    #     return_numpy
    #         Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
    #         gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
    #         Otherwise, it defaults to `True`.
    #
    #     Returns
    #     -------
    #     - **foreground_probability** - probability foreground for each protein
    #
    #     If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
    #     Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
    #     """
    #     adata = self._validate_anndata(adata)
    #     post = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     if protein_list is None:
    #         protein_mask = slice(None)
    #     else:
    #         all_proteins = self.scvi_setup_dict_["protein_names"]
    #         protein_mask = [True if p in protein_list else False for p in all_proteins]
    #
    #     if n_samples > 1 and return_mean is False:
    #         if return_numpy is False:
    #             warnings.warn(
    #                 "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
    #             )
    #         return_numpy = True
    #     if indices is None:
    #         indices = np.arange(adata.n_obs)
    #
    #     py_mixings = []
    #     if not isinstance(transform_batch, IterableClass):
    #         transform_batch = [transform_batch]
    #
    #     transform_batch = _get_batch_code_from_category(adata, transform_batch)
    #     for tensors in post:
    #         y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
    #         py_mixing = torch.zeros_like(y[..., protein_mask])
    #         if n_samples > 1:
    #             py_mixing = torch.stack(n_samples * [py_mixing])
    #         for b in transform_batch:
    #             generative_kwargs = dict(transform_batch=b)
    #             inference_kwargs = dict(n_samples=n_samples)
    #             _, generative_outputs = self.module.forward(
    #                 tensors=tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )
    #             py_mixing += torch.sigmoid(generative_outputs["py_"]["mixing"])[
    #                 ..., protein_mask
    #             ].cpu()
    #         py_mixing /= len(transform_batch)
    #         py_mixings += [py_mixing]
    #     if n_samples > 1:
    #         # concatenate along batch dimension -> result shape = (samples, cells, features)
    #         py_mixings = torch.cat(py_mixings, dim=1)
    #         # (cells, features, samples)
    #         py_mixings = py_mixings.permute(1, 2, 0)
    #     else:
    #         py_mixings = torch.cat(py_mixings, dim=0)
    #
    #     if return_mean is True and n_samples > 1:
    #         py_mixings = torch.mean(py_mixings, dim=-1)
    #
    #     py_mixings = py_mixings.cpu().numpy()
    #
    #     if return_numpy is True:
    #         return 1 - py_mixings
    #     else:
    #         pro_names = self.scvi_setup_dict_["protein_names"]
    #         foreground_prob = pd.DataFrame(
    #             1 - py_mixings,
    #             columns=pro_names[protein_mask],
    #             index=adata.obs_names[indices],
    #         )
    #         return foreground_prob
    #
    # def _expression_for_de(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples_overall=None,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     scale_protein=False,
    #     batch_size: Optional[int] = None,
    #     sample_protein_mixing=False,
    #     include_protein_background=False,
    #     protein_prior_count=0.5,
    # ):
    #     rna, protein = self.get_normalized_expression(
    #         adata=adata,
    #         indices=indices,
    #         n_samples_overall=n_samples_overall,
    #         transform_batch=transform_batch,
    #         return_numpy=True,
    #         n_samples=1,
    #         batch_size=batch_size,
    #         scale_protein=scale_protein,
    #         sample_protein_mixing=sample_protein_mixing,
    #         include_protein_background=include_protein_background,
    #     )
    #     protein += protein_prior_count
    #
    #     joint = np.concatenate([rna, protein], axis=1)
    #     return joint
    #
    #
    # def differential_expression(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     groupby: Optional[str] = None,
    #     group1: Optional[Iterable[str]] = None,
    #     group2: Optional[str] = None,
    #     idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     mode: Literal["vanilla", "change"] = "change",
    #     delta: float = 0.25,
    #     batch_size: Optional[int] = None,
    #     all_stats: bool = True,
    #     batch_correction: bool = False,
    #     batchid1: Optional[Iterable[str]] = None,
    #     batchid2: Optional[Iterable[str]] = None,
    #     fdr_target: float = 0.05,
    #     silent: bool = False,
    #     protein_prior_count: float = 0.1,
    #     scale_protein: bool = False,
    #     sample_protein_mixing: bool = False,
    #     include_protein_background: bool = False,
    #     **kwargs,
    # ) -> pd.DataFrame:
    #     r"""
    #     A unified method for differential expression analysis.
    #
    #     Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.
    #
    #     Parameters
    #     ----------
    #     {doc_differential_expression}
    #     protein_prior_count
    #         Prior count added to protein expression before LFC computation
    #     scale_protein
    #         Force protein values to sum to one in every single cell (post-hoc normalization)
    #     sample_protein_mixing
    #         Sample the protein mixture component, i.e., use the parameter to sample a Bernoulli
    #         that determines if expression is from foreground/background.
    #     include_protein_background
    #         Include the protein background component as part of the protein expression
    #     **kwargs
    #         Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`
    #
    #     Returns
    #     -------
    #     Differential expression DataFrame.
    #     """
    #     adata = self._validate_anndata(adata)
    #     model_fn = partial(
    #         self._expression_for_de,
    #         scale_protein=scale_protein,
    #         sample_protein_mixing=sample_protein_mixing,
    #         include_protein_background=include_protein_background,
    #         protein_prior_count=protein_prior_count,
    #         batch_size=batch_size,
    #     )
    #     col_names = np.concatenate(
    #         [
    #             np.asarray(_get_var_names_from_setup_anndata(adata)),
    #             self.scvi_setup_dict_["protein_names"],
    #         ]
    #     )
    #     result = _de_core(
    #         adata,
    #         model_fn,
    #         groupby,
    #         group1,
    #         group2,
    #         idx1,
    #         idx2,
    #         all_stats,
    #         cite_seq_raw_counts_properties,
    #         col_names,
    #         mode,
    #         batchid1,
    #         batchid2,
    #         delta,
    #         batch_correction,
    #         fdr_target,
    #         silent,
    #         **kwargs,
    #     )
    #
    #     return result
    #
    # @torch.no_grad()
    # def posterior_predictive_sample(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: int = 1,
    #     batch_size: Optional[int] = None,
    #     gene_list: Optional[Sequence[str]] = None,
    #     protein_list: Optional[Sequence[str]] = None,
    # ) -> np.ndarray:
    #     r"""
    #     Generate observation samples from the posterior predictive distribution.
    #
    #     The posterior predictive distribution is written as :math:`p(\hat{x}, \hat{y} \mid x, y)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of required samples for each cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     gene_list
    #         Names of genes of interest
    #     protein_list
    #         Names of proteins of interest
    #
    #     Returns
    #     -------
    #     x_new : :class:`~numpy.ndarray`
    #         tensor with shape (n_cells, n_genes, n_samples)
    #     """
    #     if self.module.gene_likelihood not in ["nb"]:
    #         raise ValueError("Invalid gene_likelihood")
    #
    #     adata = self._validate_anndata(adata)
    #     if gene_list is None:
    #         gene_mask = slice(None)
    #     else:
    #         all_genes = _get_var_names_from_setup_anndata(adata)
    #         gene_mask = [True if gene in gene_list else False for gene in all_genes]
    #     if protein_list is None:
    #         protein_mask = slice(None)
    #     else:
    #         all_proteins = self.scvi_setup_dict_["protein_names"]
    #         protein_mask = [True if p in protein_list else False for p in all_proteins]
    #
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         rna_sample, protein_sample = self.module.sample(
    #             tensors, n_samples=n_samples
    #         )
    #         rna_sample = rna_sample[..., gene_mask]
    #         protein_sample = protein_sample[..., protein_mask]
    #         data = torch.cat([rna_sample, protein_sample], dim=-1).numpy()
    #
    #         scdl_list += [data]
    #         if n_samples > 1:
    #             scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #     scdl_list = np.concatenate(scdl_list, axis=0)
    #
    #     return scdl_list
    #
    # @torch.no_grad()
    # def _get_denoised_samples(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 25,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[int] = None,
    # ) -> np.ndarray:
    #     """
    #     Return samples from an adjusted posterior predictive.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         indices of `adata` to use
    #     n_samples
    #         How may samples per cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         int of which batch to condition on for all cells
    #     """
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         x = tensors[_CONSTANTS.X_KEY]
    #         y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
    #
    #         generative_kwargs = dict(transform_batch=transform_batch)
    #         inference_kwargs = dict(n_samples=n_samples)
    #         with torch.no_grad():
    #             inference_outputs, generative_outputs, = self.module.forward(
    #                 tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )
    #         px_ = generative_outputs["px_"]
    #         py_ = generative_outputs["py_"]
    #         device = px_["r"].device
    #
    #         pi = 1 / (1 + torch.exp(-py_["mixing"]))
    #         mixing_sample = torch.distributions.Bernoulli(pi).sample()
    #         protein_rate = py_["rate_fore"]
    #         rate = torch.cat((rna_size_factor * px_["scale"], protein_rate), dim=-1)
    #         if len(px_["r"].size()) == 2:
    #             px_dispersion = px_["r"]
    #         else:
    #             px_dispersion = torch.ones_like(x).to(device) * px_["r"]
    #         if len(py_["r"].size()) == 2:
    #             py_dispersion = py_["r"]
    #         else:
    #             py_dispersion = torch.ones_like(y).to(device) * py_["r"]
    #
    #         dispersion = torch.cat((px_dispersion, py_dispersion), dim=-1)
    #
    #         # This gamma is really l*w using scVI manuscript notation
    #         p = rate / (rate + dispersion)
    #         r = dispersion
    #         l_train = torch.distributions.Gamma(r, (1 - p) / p).sample()
    #         data = l_train.cpu().numpy()
    #         # make background 0
    #         data[:, :, self.adata.shape[1] :] = (
    #             data[:, :, self.adata.shape[1] :] * (1 - mixing_sample).cpu().numpy()
    #         )
    #         scdl_list += [data]
    #
    #         scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #
    #     return np.concatenate(scdl_list, axis=0)
    #
    # @torch.no_grad()
    # def get_feature_correlation_matrix(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 10,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     correlation_type: Literal["spearman", "pearson"] = "spearman",
    #     log_transform: bool = False,
    # ) -> pd.DataFrame:
    #     """
    #     Generate gene-gene correlation matrix using scvi uncertainty and expression.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         Batches to condition on.
    #         If transform_batch is:
    #
    #         - None, then real observed batch is used
    #         - int, then batch transform_batch is used
    #         - list of int, then values are averaged over provided batches.
    #     correlation_type
    #         One of "pearson", "spearman".
    #     log_transform
    #         Whether to log transform denoised values prior to correlation calculation.
    #
    #     Returns
    #     -------
    #     Gene-protein-gene-protein correlation matrix
    #     """
    #     from scipy.stats import spearmanr
    #
    #     adata = self._validate_anndata(adata)
    #
    #     if not isinstance(transform_batch, IterableClass):
    #         transform_batch = [transform_batch]
    #
    #     transform_batch = _get_batch_code_from_category(adata, transform_batch)
    #
    #     corr_mats = []
    #     for b in transform_batch:
    #         denoised_data = self._get_denoised_samples(
    #             n_samples=n_samples,
    #             batch_size=batch_size,
    #             rna_size_factor=rna_size_factor,
    #             transform_batch=b,
    #         )
    #         flattened = np.zeros(
    #             (denoised_data.shape[0] * n_samples, denoised_data.shape[1])
    #         )
    #         for i in range(n_samples):
    #             flattened[
    #                 denoised_data.shape[0] * (i) : denoised_data.shape[0] * (i + 1)
    #             ] = denoised_data[:, :, i]
    #         if log_transform is True:
    #             flattened[:, : self.n_genes] = np.log(
    #                 flattened[:, : self.n_genes] + 1e-8
    #             )
    #             flattened[:, self.n_genes :] = np.log1p(flattened[:, self.n_genes :])
    #         if correlation_type == "pearson":
    #             corr_matrix = np.corrcoef(flattened, rowvar=False)
    #         else:
    #             corr_matrix, _ = spearmanr(flattened, axis=0)
    #         corr_mats.append(corr_matrix)
    #
    #     corr_matrix = np.mean(np.stack(corr_mats), axis=0)
    #     var_names = _get_var_names_from_setup_anndata(adata)
    #     names = np.concatenate(
    #         [np.asarray(var_names), self.scvi_setup_dict_["protein_names"]]
    #     )
    #     return pd.DataFrame(corr_matrix, index=names, columns=names)
    #
    # @torch.no_grad()
    # def get_likelihood_parameters(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: Optional[int] = 1,
    #     give_mean: Optional[bool] = False,
    #     batch_size: Optional[int] = None,
    # ) -> Dict[str, np.ndarray]:
    #     r"""
    #     Estimates for the parameters of the likelihood :math:`p(x, y \mid z)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     give_mean
    #         Return expected value of parameters or a samples
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     """
    #     raise NotImplementedError
    #
    # def _validate_anndata(
    #     self, adata: Optional[AnnData] = None, copy_if_view: bool = True
    # ):
    #     adata = super()._validate_anndata(adata, copy_if_view)
    #     error_msg = "Number of {} in anndata different from when setup_anndata was run. Please rerun setup_anndata."
    #     if _CONSTANTS.PROTEIN_EXP_KEY in adata.uns["_scvi"]["data_registry"].keys():
    #         pro_exp = get_from_registry(adata, _CONSTANTS.PROTEIN_EXP_KEY)
    #         if self.summary_stats["n_proteins"] != pro_exp.shape[1]:
    #             raise ValueError(error_msg.format("proteins"))
    #         is_nonneg_int = _check_nonnegative_integers(pro_exp)
    #         if not is_nonneg_int:
    #             warnings.warn(
    #                 "Make sure the registered protein expression in anndata contains unnormalized count data."
    #             )
    #     else:
    #         raise ValueError("No protein data found, please setup or transfer anndata")
    #
    #     return adata
    #
    # @torch.no_grad()
    # def get_protein_background_mean(self, adata, indices, batch_size):
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #     background_mean = []
    #     for tensors in scdl:
    #         _, inference_outputs, _ = self.module.forward(tensors)
    #         b_mean = inference_outputs["py_"]["rate_back"]
    #         background_mean += [b_mean.cpu().numpy()]
    #     return np.concatenate(background_mean)
    #
    # @staticmethod
    # @setup_anndata_dsp.dedent
    # def setup_anndata(
    #     adata: AnnData,
    #     protein_expression_obsm_key: str,
    #     protein_names_uns_key: Optional[str] = None,
    #     batch_key: Optional[str] = None,
    #     layer: Optional[str] = None,
    #     categorical_covariate_keys: Optional[List[str]] = None,
    #     continuous_covariate_keys: Optional[List[str]] = None,
    #     copy: bool = False,
    # ) -> Optional[AnnData]:
    #     """
    #     %(summary)s.
    #
    #     Parameters
    #     ----------
    #     %(param_adata)s
    #     protein_expression_obsm_key
    #         key in `adata.obsm` for protein expression data.
    #     protein_names_uns_key
    #         key in `adata.uns` for protein names. If None, will use the column names of `adata.obsm[protein_expression_obsm_key]`
    #         if it is a DataFrame, else will assign sequential names to proteins.
    #     %(param_batch_key)s
    #     %(param_layer)s
    #     %(param_cat_cov_keys)s
    #     %(param_cont_cov_keys)s
    #     %(param_copy)s
    #
    #     Returns
    #     -------
    #     %(returns)s
    #     """
    #     return _setup_anndata(
    #         adata,
    #         batch_key=batch_key,
    #         layer=layer,
    #         protein_expression_obsm_key=protein_expression_obsm_key,
    #         protein_names_uns_key=protein_names_uns_key,
    #         categorical_covariate_keys=categorical_covariate_keys,
    #         continuous_covariate_keys=continuous_covariate_keys,
    #         copy=copy,
    #     )
    #
    #
    #


class SCMSIIMAGE(nn.Module):
    """
    Parameters
    ----------
    adata
        AnnData object.
    n_latent
        Dimensionality of the latent space.
    image_likelihood
        One of:

        * ``'gaussian'`` - Guassian distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scMSI.scMSI_main.scMSIVAE`

    Notes
    -----
    See further usage examples in the following tutorials:

    to be updated

    """

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        image_likelihood: str = "gaussian",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs,
    ):
        super(SCMSIIMAGE, self).__init__()
        self.adata = adata
        if "batch" not in adata.obs.keys():
            adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
        if "n_batch" not in adata.uns.keys():
            adata.uns["n_batch"] = 1

        n_cats_per_cov = (
            adata.uns["extra_categoricals"]["n_cats_per_key"]
            if "extra_categoricals" in adata.uns.keys()
            else None
        )
        n_batch = self.adata.uns["n_batch"]
        library_log_means, library_log_vars = init_library_size(adata, n_batch)

        self.module = ImageSVAE(
            n_input_genes=adata.obsm['X_morphology'].shape[1],
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=0,
            n_cats_per_cov=n_cats_per_cov,
            gene_likelihood=image_likelihood,
            latent_distribution=latent_distribution,
            **model_kwargs,
        )

    def train(
        self,
        max_epochs: Optional[int] = 400,
        lr: float = 4e-4,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        n_sample_ref: int = 50,
        split_index: Optional[float] = None,
        image_key: str = "X_morphology",
        early_stopping: bool = False,
        check_val_every_n_epoch: Optional[int] = None,
        reduce_lr_on_plateau: bool = True,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        adversarial_classifier: Optional[bool] = None,
        plan_kwargs: Optional[dict] = None,
        weight_decay=1e-6,
        record_loss=True,
        **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        n_sample_ref
            reference samples number.
        split_index
            (train_idx, val_idx, test_idx, anchor_idx)
        image_key
            optional: X_morphology, X_tile_feature
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        weight_decay
            weight decay
        record_loss
            whether save loss
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=0.9,
            validation_size=0.1,
            use_gpu=True,
        )
        if split_index is not None:
            train_adata = self.adata[split_index[0]]
            val_adata = self.adata[split_index[1]]

        self.train_adata = train_adata
        self.val_adata = val_adata
        self.batch_size = batch_size

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        if n_sample_ref > train_adata.shape[0]:
            ref_idx = np.arange(train_adata.shape[0])
        else:
            ref_idx = get_anchor_index(train_adata.X, n_pcs=50, n_clusters=n_sample_ref)
            if split_index is not None:
                ref_idx = split_index[3]
        # ref_idx = np.random.permutation(train_adata.shape[1])[0:500]
        self.ref_adata = train_adata[ref_idx, :]
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_loss = []
        epoch_rec_loss = []
        epoch_kl_z_loss = []
        epoch_kl_l_loss = []
        epoch_se_loss = []
        epoch_coef_reg_loss = []
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            batch_loss = []
            batch_rec_loss = []
            batch_kl_z_loss = []
            batch_kl_l_loss = []
            batch_se_loss = []
            batch_coef_reg_loss = []
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            for train_batch in train_adata_batch:
                input_genes = torch.tensor(train_batch.obsm['X_morphology'], dtype=torch.float32)  #
                if "batch" in train_batch.obs.keys():
                    batch_index = torch.tensor(train_batch.obs['batch'], dtype=torch.float32).reshape(-1, 1)
                else:
                    batch_index = None
                _, _, _, all_loss = self.module(input_genes, ref_adata, batch_index)
                loss, reconst_loss, kl_local, se_losses = all_loss
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), 0.001)
                optimizer.step()
                if record_loss:
                    batch_loss.append(loss.data.numpy())
                    batch_rec_loss.append(torch.mean(reconst_loss).data.numpy())
                    batch_kl_z_loss.append(torch.mean(kl_local['kl_div_z']).data.numpy())
                    batch_se_loss.append(torch.mean(se_losses['se_loss']).data.numpy())
                    batch_coef_reg_loss.append(torch.mean(se_losses['coeff_reg']).data.numpy())

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                print("epoch: {}; loss: {} ".format(epoch, np.mean(batch_loss)))
            scheduler.step()
            # self.module.eval()
            epoch_loss.append(np.mean(batch_loss))
            epoch_rec_loss.append(np.mean(batch_rec_loss))
            epoch_kl_z_loss.append(np.mean(batch_kl_z_loss))
            epoch_kl_l_loss.append(np.mean(batch_kl_l_loss))
            epoch_se_loss.append(np.mean(batch_se_loss))
            epoch_coef_reg_loss.append(np.mean(batch_rec_loss))
        self.history = dict(
            epoch_loss=epoch_loss,
            epoch_rec_loss=epoch_rec_loss,
            epoch_kl_z_loss=epoch_kl_z_loss,
            epoch_kl_l_loss=epoch_kl_l_loss,
            epoch_se_loss=epoch_se_loss,
            epoch_coef_reg_loss=epoch_coef_reg_loss
        )

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['X_morphology'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            qz_m = outputs["qz_m"]
            qz_v = outputs["qz_v"]
            z = outputs["z"]
            if give_mean:
                samples = Normal(qz_m, qz_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z = self.module.encoder_genes.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m

            latent += [z.cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_reconstruct_latent(
        self,
        adata: Optional[AnnData] = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        ref_z_list = []
        rec_z_list = []
        for data in ref_adata:
            refer_genes = torch.tensor(data.obsm["X_morphology"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.module.inference(refer_genes, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_genes = torch.cat(ref_z_list, dim=0)
        if not give_mean:
            mc_samples = 1
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['X_morphology'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            inference_outputs = self.module.inference(input_genes, batch_index, n_samples=mc_samples)
            z_genes = inference_outputs['z'].mean(dim=0)
            senet_outputs = self.module.self_expressiveness(z_genes, ref_z_genes)
            rec_z = senet_outputs['rec_queries']
            rec_z_list.append(rec_z.cpu())
        return torch.cat(rec_z_list, dim=0).numpy()

    @torch.no_grad()
    def get_coeff(
        self,
        adata: Optional[AnnData] = None,
        batch_size: int = 128,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        coeff_genes_list = []
        ref_adata = self.ref_adata
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['X_morphology'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            _, senet_outputs, _ = self.module(input_genes, ref_adata, batch_index, compute_loss=False)
            coeff_genes_list.append(senet_outputs['coeff'])
        coeff_genes = torch.cat(coeff_genes_list, dim=0)
        return coeff_genes.cpu().numpy()

    @torch.no_grad()
    def get_latent_library_size(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.module.training:
            self.module.eval()
        libraries = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            ql_m = outputs["ql_m"]
            ql_v = outputs["ql_v"]
            library_gene = outputs["library_gene"]
            if give_mean and (not self.module.use_observed_lib_size):
                untran_l = Normal(ql_m, ql_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                library = torch.exp(untran_l)
                library = library.mean(dim=0)
            else:
                library = library_gene

            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()

    @torch.no_grad()
    def get_normalized_expression(
            self,
            adata=None,
            indices=None,
            n_samples_overall: Optional[int] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            gene_list: Optional[Sequence[str]] = None,
            library_size: Optional[Union[float, Literal["latent"]]] = 1,
            n_samples: int = 1,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        r"""
        Returns the normalized gene expression and protein expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to use in total
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude.
        n_samples
            Get sample scale from multiple samples.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
            names as columns. If either n_samples=1 or return_mean=True, defaults to False.
            Otherwise, it defaults to True.

        Returns
        -------
        - **gene_normalized_expression** - normalized expression for RNA

        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is ``(samples, cells, genes)``.
        Otherwise, shape is ``(cells, genes)``. Return type is ``pd.DataFrame`` unless ``return_numpy`` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        scale_list_gene = []

        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            px_scale = torch.zeros_like(input_genes)
            if n_samples > 1:
                px_scale = torch.stack(n_samples * [px_scale])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_genes, ref_adata, batch_index, compute_loss=False,
                                                       transform_batch=b)
                if library_size == "latent":
                    px_scale += generative_outputs["rate"].cpu()
                else:
                    px_scale += generative_outputs["scale"].cpu()
                px_scale = px_scale[..., gene_mask]  # this doesn't change px_scale?

            px_scale /= len(transform_batch)
            scale_list_gene.append(px_scale)

        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            scale_list_gene = torch.cat(scale_list_gene, dim=1)
            # (cells, features, samples)
            scale_list_gene = scale_list_gene.permute(1, 2, 0)
        else:
            scale_list_gene = torch.cat(scale_list_gene, dim=0)

        if return_mean is True and n_samples > 1:
            scale_list_gene = torch.mean(scale_list_gene, dim=-1)

        scale_list_gene = scale_list_gene.cpu().numpy()
        if return_numpy is None or return_numpy is False:
            gene_df = pd.DataFrame(
                scale_list_gene,
                columns=self.adata.var_names[gene_mask],
                index=self.adata.obs_names[indices],
            )
            return gene_df
        else:
            return scale_list_gene

    # @torch.no_grad()
    # def get_protein_foreground_probability(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     protein_list: Optional[Sequence[str]] = None,
    #     n_samples: int = 1,
    #     batch_size: Optional[int] = None,
    #     return_mean: bool = True,
    #     return_numpy: Optional[bool] = None,
    # ):
    #     r"""
    #     Returns the foreground probability for proteins.
    #
    #     This is denoted as :math:`(1 - \pi_{nt})` in the totalVI paper.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     transform_batch
    #         Batch to condition on.
    #         If transform_batch is:
    #
    #         - None, then real observed batch is used
    #         - int, then batch transform_batch is used
    #         - List[int], then average over batches in list
    #     protein_list
    #         Return protein expression for a subset of genes.
    #         This can save memory when working with large datasets and few genes are
    #         of interest.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     return_mean
    #         Whether to return the mean of the samples.
    #     return_numpy
    #         Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
    #         gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
    #         Otherwise, it defaults to `True`.
    #
    #     Returns
    #     -------
    #     - **foreground_probability** - probability foreground for each protein
    #
    #     If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
    #     Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
    #     """
    #     adata = self._validate_anndata(adata)
    #     post = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     if protein_list is None:
    #         protein_mask = slice(None)
    #     else:
    #         all_proteins = self.scvi_setup_dict_["protein_names"]
    #         protein_mask = [True if p in protein_list else False for p in all_proteins]
    #
    #     if n_samples > 1 and return_mean is False:
    #         if return_numpy is False:
    #             warnings.warn(
    #                 "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
    #             )
    #         return_numpy = True
    #     if indices is None:
    #         indices = np.arange(adata.n_obs)
    #
    #     py_mixings = []
    #     if not isinstance(transform_batch, IterableClass):
    #         transform_batch = [transform_batch]
    #
    #     transform_batch = _get_batch_code_from_category(adata, transform_batch)
    #     for tensors in post:
    #         y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
    #         py_mixing = torch.zeros_like(y[..., protein_mask])
    #         if n_samples > 1:
    #             py_mixing = torch.stack(n_samples * [py_mixing])
    #         for b in transform_batch:
    #             generative_kwargs = dict(transform_batch=b)
    #             inference_kwargs = dict(n_samples=n_samples)
    #             _, generative_outputs = self.module.forward(
    #                 tensors=tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )
    #             py_mixing += torch.sigmoid(generative_outputs["py_"]["mixing"])[
    #                 ..., protein_mask
    #             ].cpu()
    #         py_mixing /= len(transform_batch)
    #         py_mixings += [py_mixing]
    #     if n_samples > 1:
    #         # concatenate along batch dimension -> result shape = (samples, cells, features)
    #         py_mixings = torch.cat(py_mixings, dim=1)
    #         # (cells, features, samples)
    #         py_mixings = py_mixings.permute(1, 2, 0)
    #     else:
    #         py_mixings = torch.cat(py_mixings, dim=0)
    #
    #     if return_mean is True and n_samples > 1:
    #         py_mixings = torch.mean(py_mixings, dim=-1)
    #
    #     py_mixings = py_mixings.cpu().numpy()
    #
    #     if return_numpy is True:
    #         return 1 - py_mixings
    #     else:
    #         pro_names = self.scvi_setup_dict_["protein_names"]
    #         foreground_prob = pd.DataFrame(
    #             1 - py_mixings,
    #             columns=pro_names[protein_mask],
    #             index=adata.obs_names[indices],
    #         )
    #         return foreground_prob
    #
    # def _expression_for_de(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples_overall=None,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     scale_protein=False,
    #     batch_size: Optional[int] = None,
    #     sample_protein_mixing=False,
    #     include_protein_background=False,
    #     protein_prior_count=0.5,
    # ):
    #     rna, protein = self.get_normalized_expression(
    #         adata=adata,
    #         indices=indices,
    #         n_samples_overall=n_samples_overall,
    #         transform_batch=transform_batch,
    #         return_numpy=True,
    #         n_samples=1,
    #         batch_size=batch_size,
    #         scale_protein=scale_protein,
    #         sample_protein_mixing=sample_protein_mixing,
    #         include_protein_background=include_protein_background,
    #     )
    #     protein += protein_prior_count
    #
    #     joint = np.concatenate([rna, protein], axis=1)
    #     return joint
    #
    #
    # def differential_expression(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     groupby: Optional[str] = None,
    #     group1: Optional[Iterable[str]] = None,
    #     group2: Optional[str] = None,
    #     idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     mode: Literal["vanilla", "change"] = "change",
    #     delta: float = 0.25,
    #     batch_size: Optional[int] = None,
    #     all_stats: bool = True,
    #     batch_correction: bool = False,
    #     batchid1: Optional[Iterable[str]] = None,
    #     batchid2: Optional[Iterable[str]] = None,
    #     fdr_target: float = 0.05,
    #     silent: bool = False,
    #     protein_prior_count: float = 0.1,
    #     scale_protein: bool = False,
    #     sample_protein_mixing: bool = False,
    #     include_protein_background: bool = False,
    #     **kwargs,
    # ) -> pd.DataFrame:
    #     r"""
    #     A unified method for differential expression analysis.
    #
    #     Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.
    #
    #     Parameters
    #     ----------
    #     {doc_differential_expression}
    #     protein_prior_count
    #         Prior count added to protein expression before LFC computation
    #     scale_protein
    #         Force protein values to sum to one in every single cell (post-hoc normalization)
    #     sample_protein_mixing
    #         Sample the protein mixture component, i.e., use the parameter to sample a Bernoulli
    #         that determines if expression is from foreground/background.
    #     include_protein_background
    #         Include the protein background component as part of the protein expression
    #     **kwargs
    #         Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`
    #
    #     Returns
    #     -------
    #     Differential expression DataFrame.
    #     """
    #     adata = self._validate_anndata(adata)
    #     model_fn = partial(
    #         self._expression_for_de,
    #         scale_protein=scale_protein,
    #         sample_protein_mixing=sample_protein_mixing,
    #         include_protein_background=include_protein_background,
    #         protein_prior_count=protein_prior_count,
    #         batch_size=batch_size,
    #     )
    #     col_names = np.concatenate(
    #         [
    #             np.asarray(_get_var_names_from_setup_anndata(adata)),
    #             self.scvi_setup_dict_["protein_names"],
    #         ]
    #     )
    #     result = _de_core(
    #         adata,
    #         model_fn,
    #         groupby,
    #         group1,
    #         group2,
    #         idx1,
    #         idx2,
    #         all_stats,
    #         cite_seq_raw_counts_properties,
    #         col_names,
    #         mode,
    #         batchid1,
    #         batchid2,
    #         delta,
    #         batch_correction,
    #         fdr_target,
    #         silent,
    #         **kwargs,
    #     )
    #
    #     return result
    #
    # @torch.no_grad()
    # def posterior_predictive_sample(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: int = 1,
    #     batch_size: Optional[int] = None,
    #     gene_list: Optional[Sequence[str]] = None,
    #     protein_list: Optional[Sequence[str]] = None,
    # ) -> np.ndarray:
    #     r"""
    #     Generate observation samples from the posterior predictive distribution.
    #
    #     The posterior predictive distribution is written as :math:`p(\hat{x}, \hat{y} \mid x, y)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of required samples for each cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     gene_list
    #         Names of genes of interest
    #     protein_list
    #         Names of proteins of interest
    #
    #     Returns
    #     -------
    #     x_new : :class:`~numpy.ndarray`
    #         tensor with shape (n_cells, n_genes, n_samples)
    #     """
    #     if self.module.gene_likelihood not in ["nb"]:
    #         raise ValueError("Invalid gene_likelihood")
    #
    #     adata = self._validate_anndata(adata)
    #     if gene_list is None:
    #         gene_mask = slice(None)
    #     else:
    #         all_genes = _get_var_names_from_setup_anndata(adata)
    #         gene_mask = [True if gene in gene_list else False for gene in all_genes]
    #     if protein_list is None:
    #         protein_mask = slice(None)
    #     else:
    #         all_proteins = self.scvi_setup_dict_["protein_names"]
    #         protein_mask = [True if p in protein_list else False for p in all_proteins]
    #
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         rna_sample, protein_sample = self.module.sample(
    #             tensors, n_samples=n_samples
    #         )
    #         rna_sample = rna_sample[..., gene_mask]
    #         protein_sample = protein_sample[..., protein_mask]
    #         data = torch.cat([rna_sample, protein_sample], dim=-1).numpy()
    #
    #         scdl_list += [data]
    #         if n_samples > 1:
    #             scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #     scdl_list = np.concatenate(scdl_list, axis=0)
    #
    #     return scdl_list
    #
    # @torch.no_grad()
    # def _get_denoised_samples(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 25,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[int] = None,
    # ) -> np.ndarray:
    #     """
    #     Return samples from an adjusted posterior predictive.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         indices of `adata` to use
    #     n_samples
    #         How may samples per cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         int of which batch to condition on for all cells
    #     """
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         x = tensors[_CONSTANTS.X_KEY]
    #         y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
    #
    #         generative_kwargs = dict(transform_batch=transform_batch)
    #         inference_kwargs = dict(n_samples=n_samples)
    #         with torch.no_grad():
    #             inference_outputs, generative_outputs, = self.module.forward(
    #                 tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )
    #         px_ = generative_outputs["px_"]
    #         py_ = generative_outputs["py_"]
    #         device = px_["r"].device
    #
    #         pi = 1 / (1 + torch.exp(-py_["mixing"]))
    #         mixing_sample = torch.distributions.Bernoulli(pi).sample()
    #         protein_rate = py_["rate_fore"]
    #         rate = torch.cat((rna_size_factor * px_["scale"], protein_rate), dim=-1)
    #         if len(px_["r"].size()) == 2:
    #             px_dispersion = px_["r"]
    #         else:
    #             px_dispersion = torch.ones_like(x).to(device) * px_["r"]
    #         if len(py_["r"].size()) == 2:
    #             py_dispersion = py_["r"]
    #         else:
    #             py_dispersion = torch.ones_like(y).to(device) * py_["r"]
    #
    #         dispersion = torch.cat((px_dispersion, py_dispersion), dim=-1)
    #
    #         # This gamma is really l*w using scVI manuscript notation
    #         p = rate / (rate + dispersion)
    #         r = dispersion
    #         l_train = torch.distributions.Gamma(r, (1 - p) / p).sample()
    #         data = l_train.cpu().numpy()
    #         # make background 0
    #         data[:, :, self.adata.shape[1] :] = (
    #             data[:, :, self.adata.shape[1] :] * (1 - mixing_sample).cpu().numpy()
    #         )
    #         scdl_list += [data]
    #
    #         scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #
    #     return np.concatenate(scdl_list, axis=0)
    #
    # @torch.no_grad()
    # def get_feature_correlation_matrix(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 10,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     correlation_type: Literal["spearman", "pearson"] = "spearman",
    #     log_transform: bool = False,
    # ) -> pd.DataFrame:
    #     """
    #     Generate gene-gene correlation matrix using scvi uncertainty and expression.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         Batches to condition on.
    #         If transform_batch is:
    #
    #         - None, then real observed batch is used
    #         - int, then batch transform_batch is used
    #         - list of int, then values are averaged over provided batches.
    #     correlation_type
    #         One of "pearson", "spearman".
    #     log_transform
    #         Whether to log transform denoised values prior to correlation calculation.
    #
    #     Returns
    #     -------
    #     Gene-protein-gene-protein correlation matrix
    #     """
    #     from scipy.stats import spearmanr
    #
    #     adata = self._validate_anndata(adata)
    #
    #     if not isinstance(transform_batch, IterableClass):
    #         transform_batch = [transform_batch]
    #
    #     transform_batch = _get_batch_code_from_category(adata, transform_batch)
    #
    #     corr_mats = []
    #     for b in transform_batch:
    #         denoised_data = self._get_denoised_samples(
    #             n_samples=n_samples,
    #             batch_size=batch_size,
    #             rna_size_factor=rna_size_factor,
    #             transform_batch=b,
    #         )
    #         flattened = np.zeros(
    #             (denoised_data.shape[0] * n_samples, denoised_data.shape[1])
    #         )
    #         for i in range(n_samples):
    #             flattened[
    #                 denoised_data.shape[0] * (i) : denoised_data.shape[0] * (i + 1)
    #             ] = denoised_data[:, :, i]
    #         if log_transform is True:
    #             flattened[:, : self.n_genes] = np.log(
    #                 flattened[:, : self.n_genes] + 1e-8
    #             )
    #             flattened[:, self.n_genes :] = np.log1p(flattened[:, self.n_genes :])
    #         if correlation_type == "pearson":
    #             corr_matrix = np.corrcoef(flattened, rowvar=False)
    #         else:
    #             corr_matrix, _ = spearmanr(flattened, axis=0)
    #         corr_mats.append(corr_matrix)
    #
    #     corr_matrix = np.mean(np.stack(corr_mats), axis=0)
    #     var_names = _get_var_names_from_setup_anndata(adata)
    #     names = np.concatenate(
    #         [np.asarray(var_names), self.scvi_setup_dict_["protein_names"]]
    #     )
    #     return pd.DataFrame(corr_matrix, index=names, columns=names)
    #
    # @torch.no_grad()
    # def get_likelihood_parameters(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: Optional[int] = 1,
    #     give_mean: Optional[bool] = False,
    #     batch_size: Optional[int] = None,
    # ) -> Dict[str, np.ndarray]:
    #     r"""
    #     Estimates for the parameters of the likelihood :math:`p(x, y \mid z)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     give_mean
    #         Return expected value of parameters or a samples
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     """
    #     raise NotImplementedError
    #
    # def _validate_anndata(
    #     self, adata: Optional[AnnData] = None, copy_if_view: bool = True
    # ):
    #     adata = super()._validate_anndata(adata, copy_if_view)
    #     error_msg = "Number of {} in anndata different from when setup_anndata was run. Please rerun setup_anndata."
    #     if _CONSTANTS.PROTEIN_EXP_KEY in adata.uns["_scvi"]["data_registry"].keys():
    #         pro_exp = get_from_registry(adata, _CONSTANTS.PROTEIN_EXP_KEY)
    #         if self.summary_stats["n_proteins"] != pro_exp.shape[1]:
    #             raise ValueError(error_msg.format("proteins"))
    #         is_nonneg_int = _check_nonnegative_integers(pro_exp)
    #         if not is_nonneg_int:
    #             warnings.warn(
    #                 "Make sure the registered protein expression in anndata contains unnormalized count data."
    #             )
    #     else:
    #         raise ValueError("No protein data found, please setup or transfer anndata")
    #
    #     return adata
    #
    # @torch.no_grad()
    # def get_protein_background_mean(self, adata, indices, batch_size):
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #     background_mean = []
    #     for tensors in scdl:
    #         _, inference_outputs, _ = self.module.forward(tensors)
    #         b_mean = inference_outputs["py_"]["rate_back"]
    #         background_mean += [b_mean.cpu().numpy()]
    #     return np.concatenate(background_mean)
    #
    # @staticmethod
    # @setup_anndata_dsp.dedent
    # def setup_anndata(
    #     adata: AnnData,
    #     protein_expression_obsm_key: str,
    #     protein_names_uns_key: Optional[str] = None,
    #     batch_key: Optional[str] = None,
    #     layer: Optional[str] = None,
    #     categorical_covariate_keys: Optional[List[str]] = None,
    #     continuous_covariate_keys: Optional[List[str]] = None,
    #     copy: bool = False,
    # ) -> Optional[AnnData]:
    #     """
    #     %(summary)s.
    #
    #     Parameters
    #     ----------
    #     %(param_adata)s
    #     protein_expression_obsm_key
    #         key in `adata.obsm` for protein expression data.
    #     protein_names_uns_key
    #         key in `adata.uns` for protein names. If None, will use the column names of `adata.obsm[protein_expression_obsm_key]`
    #         if it is a DataFrame, else will assign sequential names to proteins.
    #     %(param_batch_key)s
    #     %(param_layer)s
    #     %(param_cat_cov_keys)s
    #     %(param_cont_cov_keys)s
    #     %(param_copy)s
    #
    #     Returns
    #     -------
    #     %(returns)s
    #     """
    #     return _setup_anndata(
    #         adata,
    #         batch_key=batch_key,
    #         layer=layer,
    #         protein_expression_obsm_key=protein_expression_obsm_key,
    #         protein_names_uns_key=protein_names_uns_key,
    #         categorical_covariate_keys=categorical_covariate_keys,
    #         continuous_covariate_keys=continuous_covariate_keys,
    #         copy=copy,
    #     )
    #
    #
    #


class SCVIRNA(nn.Module):
    """
    Parameters
    ----------
    adata
        AnnData object`.
    n_latent
        Dimensionality of the latent space.
    gene_dispersion
        One of the following:

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    empirical_protein_background_prior
        Set the initialization of protein background prior empirically. This option fits a GMM for each of
        100 cells per batch and averages the distributions. Note that even with this option set to `True`,
        this only initializes a parameter that is learned during inference. If `False`, randomly initializes.
        The default (`None`), sets this to `True` if greater than 10 proteins are used.
    override_missing_proteins
        If `True`, will not treat proteins with all 0 expression in a particular batch as missing.
    **model_kwargs
        Keyword args for :class:`~scvi.module.TOTALVAE`

    Notes
    -----
    See further usage examples in the following tutorials:

    to be updated

    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            gene_dispersion: Literal[
                "gene", "gene-batch", "gene-label", "gene-cell"
            ] = "gene",
            gene_likelihood: Literal["zinb", "nb"] = "nb",
            latent_distribution: Literal["normal", "ln"] = "normal",
            **model_kwargs,
    ):
        super(SCVIRNA, self).__init__()
        self.adata = adata
        if "batch" not in adata.obs.keys():
            adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
        if "n_batch" not in adata.uns.keys():
            adata.uns["n_batch"] = 1

        n_cats_per_cov = (
            adata.uns["extra_categoricals"]["n_cats_per_key"]
            if "extra_categoricals" in adata.uns.keys()
            else None
        )
        n_batch = self.adata.uns["n_batch"]
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=0.9,
            validation_size=0.1,
            use_gpu=False,
        )
        self.train_adata = train_adata
        self.val_adata = val_adata

        library_log_means, library_log_vars = init_library_size(adata, n_batch)

        self.module = scVIVAE(
            n_input_genes=adata.X.shape[1],
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=0,
            n_cats_per_cov=n_cats_per_cov,
            gene_dispersion=gene_dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            use_observed_lib_size=False,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 4e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            early_stopping: bool = True,
            check_val_every_n_epoch: Optional[int] = None,
            reduce_lr_on_plateau: bool = True,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            adversarial_classifier: Optional[bool] = None,
            plan_kwargs: Optional[dict] = None,
            weight_decay=1e-6,
            **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        weight_decay
            weight decay
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        # if adversarial_classifier is None:
        #     adversarial_classifier = self._use_adversarial_classifier
        n_steps_kl_warmup = (
            n_steps_kl_warmup
            if n_steps_kl_warmup is not None
            else int(0.75 * self.adata.n_obs)
        )
        if reduce_lr_on_plateau:
            check_val_every_n_epoch = 1
        self.batch_size = batch_size
        update_dict = {
            "lr": lr,
            "adversarial_classifier": adversarial_classifier,
            "reduce_lr_on_plateau": reduce_lr_on_plateau,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            "check_val_every_n_epoch": check_val_every_n_epoch,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        # plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        # train_adata, val_adata = data_splitter(
        #     self.adata,
        #     train_size=train_size,
        #     validation_size=validation_size,
        #     use_gpu=use_gpu,
        # )
        train_adata = self.train_adata
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        loss_log = []
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            self.module.train()
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            for train_batch in train_adata_batch:
                input_genes = torch.tensor(train_batch.layers['rna_expression'], dtype=torch.float32)
                if "batch" in train_batch.obs.keys():
                    batch_index = torch.tensor(train_batch.obs['batch'], dtype=torch.float32).reshape(-1, 1)
                else:
                    batch_index = None
                _, _, loss = self.module(input_genes, batch_index)
                loss_log.append(loss)
                optimizer.zero_grad()
                loss[0].backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), 0.001)
                optimizer.step()
            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                print("epoch: {}; loss: {} ".format(epoch, loss[0]))
            scheduler.step()
            # self.module.eval()
        return loss_log

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.X, dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            qz_m = outputs["qz_m"]
            qz_v = outputs["qz_v"]
            z = outputs["z"]
            if give_mean:
                samples = Normal(qz_m, qz_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z = self.module.encoder_genes.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = self.module.encoder_genes.z_transformation(qz_m)

            latent += [z.cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_latent_library_size(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.module.training:
            self.module.eval()
        libraries = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.X, dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            ql_m = outputs["ql_m"]
            ql_v = outputs["ql_v"]
            library_gene = outputs["library_gene"]
            if give_mean and (not self.module.use_observed_lib_size):
                untran_l = Normal(ql_m, ql_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                library = torch.exp(untran_l)
                library = library.mean(dim=0)
            else:
                library = library_gene

            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()

    # @torch.no_grad()
    # def get_normalized_expression(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples_overall: Optional[int] = None,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     gene_list: Optional[Sequence[str]] = None,
    #     protein_list: Optional[Sequence[str]] = None,
    #     library_size: Optional[Union[float, Literal["latent"]]] = 1,
    #     n_samples: int = 1,
    #     sample_protein_mixing: bool = False,
    #     scale_protein: bool = False,
    #     include_protein_background: bool = False,
    #     batch_size: Optional[int] = None,
    #     return_mean: bool = True,
    #     return_numpy: Optional[bool] = None,
    # ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
    #     r"""
    #     Returns the normalized gene expression and protein expression.
    #
    #     This is denoted as :math:`\rho_n` in the totalVI paper for genes, and TODO
    #     for proteins, :math:`(1-\pi_{nt})\alpha_{nt}\beta_{nt}`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples_overall
    #         Number of samples to use in total
    #     transform_batch
    #         Batch to condition on.
    #         If transform_batch is:
    #
    #         - None, then real observed batch is used
    #         - int, then batch transform_batch is used
    #         - List[int], then average over batches in list
    #     gene_list
    #         Return frequencies of expression for a subset of genes.
    #         This can save memory when working with large datasets and few genes are
    #         of interest.
    #     protein_list
    #         Return protein expression for a subset of genes.
    #         This can save memory when working with large datasets and few genes are
    #         of interest.
    #     library_size
    #         Scale the expression frequencies to a common library size.
    #         This allows gene expression levels to be interpreted on a common scale of relevant
    #         magnitude.
    #     n_samples
    #         Get sample scale from multiple samples.
    #     sample_protein_mixing
    #         Sample mixing bernoulli, setting background to zero
    #     scale_protein
    #         Make protein expression sum to 1
    #     include_protein_background
    #         Include background component for protein expression
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     return_mean
    #         Whether to return the mean of the samples.
    #     return_numpy
    #         Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
    #         names as columns. If either n_samples=1 or return_mean=True, defaults to False.
    #         Otherwise, it defaults to True.
    #
    #     Returns
    #     -------
    #     - **gene_normalized_expression** - normalized expression for RNA
    #     - **protein_normalized_expression** - normalized expression for proteins
    #
    #     If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is ``(samples, cells, genes)``.
    #     Otherwise, shape is ``(cells, genes)``. Return type is ``pd.DataFrame`` unless ``return_numpy`` is True.
    #     """
    #     adata = self._validate_anndata(adata)
    #     if indices is None:
    #         indices = np.arange(adata.n_obs)
    #     if n_samples_overall is not None:
    #         indices = np.random.choice(indices, n_samples_overall)
    #     post = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     if gene_list is None:
    #         gene_mask = slice(None)
    #     else:
    #         all_genes = _get_var_names_from_setup_anndata(adata)
    #         gene_mask = [True if gene in gene_list else False for gene in all_genes]
    #     if protein_list is None:
    #         protein_mask = slice(None)
    #     else:
    #         all_proteins = self.scvi_setup_dict_["protein_names"]
    #         protein_mask = [True if p in protein_list else False for p in all_proteins]
    #     if indices is None:
    #         indices = np.arange(adata.n_obs)
    #
    #     if n_samples > 1 and return_mean is False:
    #         if return_numpy is False:
    #             warnings.warn(
    #                 "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
    #             )
    #         return_numpy = True
    #
    #     if not isinstance(transform_batch, IterableClass):
    #         transform_batch = [transform_batch]
    #
    #     transform_batch = _get_batch_code_from_category(adata, transform_batch)
    #
    #     scale_list_gene = []
    #     scale_list_pro = []
    #
    #     for tensors in post:
    #         x = tensors[_CONSTANTS.X_KEY]
    #         y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
    #         px_scale = torch.zeros_like(x)
    #         py_scale = torch.zeros_like(y)
    #         if n_samples > 1:
    #             px_scale = torch.stack(n_samples * [px_scale])
    #             py_scale = torch.stack(n_samples * [py_scale])
    #         for b in transform_batch:
    #             generative_kwargs = dict(transform_batch=b)
    #             inference_kwargs = dict(n_samples=n_samples)
    #             _, generative_outputs = self.module.forward(
    #                 tensors=tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )
    #             if library_size == "latent":
    #                 px_scale += generative_outputs["px_"]["rate"].cpu()
    #             else:
    #                 px_scale += generative_outputs["px_"]["scale"].cpu()
    #             px_scale = px_scale[..., gene_mask]
    #
    #             py_ = generative_outputs["py_"]
    #             # probability of background
    #             protein_mixing = 1 / (1 + torch.exp(-py_["mixing"].cpu()))
    #             if sample_protein_mixing is True:
    #                 protein_mixing = torch.distributions.Bernoulli(
    #                     protein_mixing
    #                 ).sample()
    #             protein_val = py_["rate_fore"].cpu() * (1 - protein_mixing)
    #             if include_protein_background is True:
    #                 protein_val += py_["rate_back"].cpu() * protein_mixing
    #
    #             if scale_protein is True:
    #                 protein_val = torch.nn.functional.normalize(
    #                     protein_val, p=1, dim=-1
    #                 )
    #             protein_val = protein_val[..., protein_mask]
    #             py_scale += protein_val
    #         px_scale /= len(transform_batch)
    #         py_scale /= len(transform_batch)
    #         scale_list_gene.append(px_scale)
    #         scale_list_pro.append(py_scale)
    #
    #     if n_samples > 1:
    #         # concatenate along batch dimension -> result shape = (samples, cells, features)
    #         scale_list_gene = torch.cat(scale_list_gene, dim=1)
    #         scale_list_pro = torch.cat(scale_list_pro, dim=1)
    #         # (cells, features, samples)
    #         scale_list_gene = scale_list_gene.permute(1, 2, 0)
    #         scale_list_pro = scale_list_pro.permute(1, 2, 0)
    #     else:
    #         scale_list_gene = torch.cat(scale_list_gene, dim=0)
    #         scale_list_pro = torch.cat(scale_list_pro, dim=0)
    #
    #     if return_mean is True and n_samples > 1:
    #         scale_list_gene = torch.mean(scale_list_gene, dim=-1)
    #         scale_list_pro = torch.mean(scale_list_pro, dim=-1)
    #
    #     scale_list_gene = scale_list_gene.cpu().numpy()
    #     scale_list_pro = scale_list_pro.cpu().numpy()
    #     if return_numpy is None or return_numpy is False:
    #         gene_df = pd.DataFrame(
    #             scale_list_gene,
    #             columns=adata.var_names[gene_mask],
    #             index=adata.obs_names[indices],
    #         )
    #         pro_df = pd.DataFrame(
    #             scale_list_pro,
    #             columns=self.scvi_setup_dict_["protein_names"][protein_mask],
    #             index=adata.obs_names[indices],
    #         )
    #
    #         return gene_df, pro_df
    #     else:
    #         return scale_list_gene, scale_list_pro
    #
    # @torch.no_grad()
    # def get_protein_foreground_probability(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     protein_list: Optional[Sequence[str]] = None,
    #     n_samples: int = 1,
    #     batch_size: Optional[int] = None,
    #     return_mean: bool = True,
    #     return_numpy: Optional[bool] = None,
    # ):
    #     r"""
    #     Returns the foreground probability for proteins.
    #
    #     This is denoted as :math:`(1 - \pi_{nt})` in the totalVI paper.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     transform_batch
    #         Batch to condition on.
    #         If transform_batch is:
    #
    #         - None, then real observed batch is used
    #         - int, then batch transform_batch is used
    #         - List[int], then average over batches in list
    #     protein_list
    #         Return protein expression for a subset of genes.
    #         This can save memory when working with large datasets and few genes are
    #         of interest.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     return_mean
    #         Whether to return the mean of the samples.
    #     return_numpy
    #         Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
    #         gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
    #         Otherwise, it defaults to `True`.
    #
    #     Returns
    #     -------
    #     - **foreground_probability** - probability foreground for each protein
    #
    #     If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
    #     Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
    #     """
    #     adata = self._validate_anndata(adata)
    #     post = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     if protein_list is None:
    #         protein_mask = slice(None)
    #     else:
    #         all_proteins = self.scvi_setup_dict_["protein_names"]
    #         protein_mask = [True if p in protein_list else False for p in all_proteins]
    #
    #     if n_samples > 1 and return_mean is False:
    #         if return_numpy is False:
    #             warnings.warn(
    #                 "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
    #             )
    #         return_numpy = True
    #     if indices is None:
    #         indices = np.arange(adata.n_obs)
    #
    #     py_mixings = []
    #     if not isinstance(transform_batch, IterableClass):
    #         transform_batch = [transform_batch]
    #
    #     transform_batch = _get_batch_code_from_category(adata, transform_batch)
    #     for tensors in post:
    #         y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
    #         py_mixing = torch.zeros_like(y[..., protein_mask])
    #         if n_samples > 1:
    #             py_mixing = torch.stack(n_samples * [py_mixing])
    #         for b in transform_batch:
    #             generative_kwargs = dict(transform_batch=b)
    #             inference_kwargs = dict(n_samples=n_samples)
    #             _, generative_outputs = self.module.forward(
    #                 tensors=tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )
    #             py_mixing += torch.sigmoid(generative_outputs["py_"]["mixing"])[
    #                 ..., protein_mask
    #             ].cpu()
    #         py_mixing /= len(transform_batch)
    #         py_mixings += [py_mixing]
    #     if n_samples > 1:
    #         # concatenate along batch dimension -> result shape = (samples, cells, features)
    #         py_mixings = torch.cat(py_mixings, dim=1)
    #         # (cells, features, samples)
    #         py_mixings = py_mixings.permute(1, 2, 0)
    #     else:
    #         py_mixings = torch.cat(py_mixings, dim=0)
    #
    #     if return_mean is True and n_samples > 1:
    #         py_mixings = torch.mean(py_mixings, dim=-1)
    #
    #     py_mixings = py_mixings.cpu().numpy()
    #
    #     if return_numpy is True:
    #         return 1 - py_mixings
    #     else:
    #         pro_names = self.scvi_setup_dict_["protein_names"]
    #         foreground_prob = pd.DataFrame(
    #             1 - py_mixings,
    #             columns=pro_names[protein_mask],
    #             index=adata.obs_names[indices],
    #         )
    #         return foreground_prob
    #
    # def _expression_for_de(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples_overall=None,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     scale_protein=False,
    #     batch_size: Optional[int] = None,
    #     sample_protein_mixing=False,
    #     include_protein_background=False,
    #     protein_prior_count=0.5,
    # ):
    #     rna, protein = self.get_normalized_expression(
    #         adata=adata,
    #         indices=indices,
    #         n_samples_overall=n_samples_overall,
    #         transform_batch=transform_batch,
    #         return_numpy=True,
    #         n_samples=1,
    #         batch_size=batch_size,
    #         scale_protein=scale_protein,
    #         sample_protein_mixing=sample_protein_mixing,
    #         include_protein_background=include_protein_background,
    #     )
    #     protein += protein_prior_count
    #
    #     joint = np.concatenate([rna, protein], axis=1)
    #     return joint
    #
    #
    # def differential_expression(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     groupby: Optional[str] = None,
    #     group1: Optional[Iterable[str]] = None,
    #     group2: Optional[str] = None,
    #     idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     mode: Literal["vanilla", "change"] = "change",
    #     delta: float = 0.25,
    #     batch_size: Optional[int] = None,
    #     all_stats: bool = True,
    #     batch_correction: bool = False,
    #     batchid1: Optional[Iterable[str]] = None,
    #     batchid2: Optional[Iterable[str]] = None,
    #     fdr_target: float = 0.05,
    #     silent: bool = False,
    #     protein_prior_count: float = 0.1,
    #     scale_protein: bool = False,
    #     sample_protein_mixing: bool = False,
    #     include_protein_background: bool = False,
    #     **kwargs,
    # ) -> pd.DataFrame:
    #     r"""
    #     A unified method for differential expression analysis.
    #
    #     Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.
    #
    #     Parameters
    #     ----------
    #     {doc_differential_expression}
    #     protein_prior_count
    #         Prior count added to protein expression before LFC computation
    #     scale_protein
    #         Force protein values to sum to one in every single cell (post-hoc normalization)
    #     sample_protein_mixing
    #         Sample the protein mixture component, i.e., use the parameter to sample a Bernoulli
    #         that determines if expression is from foreground/background.
    #     include_protein_background
    #         Include the protein background component as part of the protein expression
    #     **kwargs
    #         Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`
    #
    #     Returns
    #     -------
    #     Differential expression DataFrame.
    #     """
    #     adata = self._validate_anndata(adata)
    #     model_fn = partial(
    #         self._expression_for_de,
    #         scale_protein=scale_protein,
    #         sample_protein_mixing=sample_protein_mixing,
    #         include_protein_background=include_protein_background,
    #         protein_prior_count=protein_prior_count,
    #         batch_size=batch_size,
    #     )
    #     col_names = np.concatenate(
    #         [
    #             np.asarray(_get_var_names_from_setup_anndata(adata)),
    #             self.scvi_setup_dict_["protein_names"],
    #         ]
    #     )
    #     result = _de_core(
    #         adata,
    #         model_fn,
    #         groupby,
    #         group1,
    #         group2,
    #         idx1,
    #         idx2,
    #         all_stats,
    #         cite_seq_raw_counts_properties,
    #         col_names,
    #         mode,
    #         batchid1,
    #         batchid2,
    #         delta,
    #         batch_correction,
    #         fdr_target,
    #         silent,
    #         **kwargs,
    #     )
    #
    #     return result
    #
    # @torch.no_grad()
    # def posterior_predictive_sample(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: int = 1,
    #     batch_size: Optional[int] = None,
    #     gene_list: Optional[Sequence[str]] = None,
    #     protein_list: Optional[Sequence[str]] = None,
    # ) -> np.ndarray:
    #     r"""
    #     Generate observation samples from the posterior predictive distribution.
    #
    #     The posterior predictive distribution is written as :math:`p(\hat{x}, \hat{y} \mid x, y)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of required samples for each cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     gene_list
    #         Names of genes of interest
    #     protein_list
    #         Names of proteins of interest
    #
    #     Returns
    #     -------
    #     x_new : :class:`~numpy.ndarray`
    #         tensor with shape (n_cells, n_genes, n_samples)
    #     """
    #     if self.module.gene_likelihood not in ["nb"]:
    #         raise ValueError("Invalid gene_likelihood")
    #
    #     adata = self._validate_anndata(adata)
    #     if gene_list is None:
    #         gene_mask = slice(None)
    #     else:
    #         all_genes = _get_var_names_from_setup_anndata(adata)
    #         gene_mask = [True if gene in gene_list else False for gene in all_genes]
    #     if protein_list is None:
    #         protein_mask = slice(None)
    #     else:
    #         all_proteins = self.scvi_setup_dict_["protein_names"]
    #         protein_mask = [True if p in protein_list else False for p in all_proteins]
    #
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         rna_sample, protein_sample = self.module.sample(
    #             tensors, n_samples=n_samples
    #         )
    #         rna_sample = rna_sample[..., gene_mask]
    #         protein_sample = protein_sample[..., protein_mask]
    #         data = torch.cat([rna_sample, protein_sample], dim=-1).numpy()
    #
    #         scdl_list += [data]
    #         if n_samples > 1:
    #             scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #     scdl_list = np.concatenate(scdl_list, axis=0)
    #
    #     return scdl_list
    #
    # @torch.no_grad()
    # def _get_denoised_samples(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 25,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[int] = None,
    # ) -> np.ndarray:
    #     """
    #     Return samples from an adjusted posterior predictive.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         indices of `adata` to use
    #     n_samples
    #         How may samples per cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         int of which batch to condition on for all cells
    #     """
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         x = tensors[_CONSTANTS.X_KEY]
    #         y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
    #
    #         generative_kwargs = dict(transform_batch=transform_batch)
    #         inference_kwargs = dict(n_samples=n_samples)
    #         with torch.no_grad():
    #             inference_outputs, generative_outputs, = self.module.forward(
    #                 tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )
    #         px_ = generative_outputs["px_"]
    #         py_ = generative_outputs["py_"]
    #         device = px_["r"].device
    #
    #         pi = 1 / (1 + torch.exp(-py_["mixing"]))
    #         mixing_sample = torch.distributions.Bernoulli(pi).sample()
    #         protein_rate = py_["rate_fore"]
    #         rate = torch.cat((rna_size_factor * px_["scale"], protein_rate), dim=-1)
    #         if len(px_["r"].size()) == 2:
    #             px_dispersion = px_["r"]
    #         else:
    #             px_dispersion = torch.ones_like(x).to(device) * px_["r"]
    #         if len(py_["r"].size()) == 2:
    #             py_dispersion = py_["r"]
    #         else:
    #             py_dispersion = torch.ones_like(y).to(device) * py_["r"]
    #
    #         dispersion = torch.cat((px_dispersion, py_dispersion), dim=-1)
    #
    #         # This gamma is really l*w using scVI manuscript notation
    #         p = rate / (rate + dispersion)
    #         r = dispersion
    #         l_train = torch.distributions.Gamma(r, (1 - p) / p).sample()
    #         data = l_train.cpu().numpy()
    #         # make background 0
    #         data[:, :, self.adata.shape[1] :] = (
    #             data[:, :, self.adata.shape[1] :] * (1 - mixing_sample).cpu().numpy()
    #         )
    #         scdl_list += [data]
    #
    #         scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #
    #     return np.concatenate(scdl_list, axis=0)
    #
    # @torch.no_grad()
    # def get_feature_correlation_matrix(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 10,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     correlation_type: Literal["spearman", "pearson"] = "spearman",
    #     log_transform: bool = False,
    # ) -> pd.DataFrame:
    #     """
    #     Generate gene-gene correlation matrix using scvi uncertainty and expression.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         Batches to condition on.
    #         If transform_batch is:
    #
    #         - None, then real observed batch is used
    #         - int, then batch transform_batch is used
    #         - list of int, then values are averaged over provided batches.
    #     correlation_type
    #         One of "pearson", "spearman".
    #     log_transform
    #         Whether to log transform denoised values prior to correlation calculation.
    #
    #     Returns
    #     -------
    #     Gene-protein-gene-protein correlation matrix
    #     """
    #     from scipy.stats import spearmanr
    #
    #     adata = self._validate_anndata(adata)
    #
    #     if not isinstance(transform_batch, IterableClass):
    #         transform_batch = [transform_batch]
    #
    #     transform_batch = _get_batch_code_from_category(adata, transform_batch)
    #
    #     corr_mats = []
    #     for b in transform_batch:
    #         denoised_data = self._get_denoised_samples(
    #             n_samples=n_samples,
    #             batch_size=batch_size,
    #             rna_size_factor=rna_size_factor,
    #             transform_batch=b,
    #         )
    #         flattened = np.zeros(
    #             (denoised_data.shape[0] * n_samples, denoised_data.shape[1])
    #         )
    #         for i in range(n_samples):
    #             flattened[
    #                 denoised_data.shape[0] * (i) : denoised_data.shape[0] * (i + 1)
    #             ] = denoised_data[:, :, i]
    #         if log_transform is True:
    #             flattened[:, : self.n_genes] = np.log(
    #                 flattened[:, : self.n_genes] + 1e-8
    #             )
    #             flattened[:, self.n_genes :] = np.log1p(flattened[:, self.n_genes :])
    #         if correlation_type == "pearson":
    #             corr_matrix = np.corrcoef(flattened, rowvar=False)
    #         else:
    #             corr_matrix, _ = spearmanr(flattened, axis=0)
    #         corr_mats.append(corr_matrix)
    #
    #     corr_matrix = np.mean(np.stack(corr_mats), axis=0)
    #     var_names = _get_var_names_from_setup_anndata(adata)
    #     names = np.concatenate(
    #         [np.asarray(var_names), self.scvi_setup_dict_["protein_names"]]
    #     )
    #     return pd.DataFrame(corr_matrix, index=names, columns=names)
    #
    # @torch.no_grad()
    # def get_likelihood_parameters(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: Optional[int] = 1,
    #     give_mean: Optional[bool] = False,
    #     batch_size: Optional[int] = None,
    # ) -> Dict[str, np.ndarray]:
    #     r"""
    #     Estimates for the parameters of the likelihood :math:`p(x, y \mid z)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     give_mean
    #         Return expected value of parameters or a samples
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     """
    #     raise NotImplementedError
    #
    # def _validate_anndata(
    #     self, adata: Optional[AnnData] = None, copy_if_view: bool = True
    # ):
    #     adata = super()._validate_anndata(adata, copy_if_view)
    #     error_msg = "Number of {} in anndata different from when setup_anndata was run. Please rerun setup_anndata."
    #     if _CONSTANTS.PROTEIN_EXP_KEY in adata.uns["_scvi"]["data_registry"].keys():
    #         pro_exp = get_from_registry(adata, _CONSTANTS.PROTEIN_EXP_KEY)
    #         if self.summary_stats["n_proteins"] != pro_exp.shape[1]:
    #             raise ValueError(error_msg.format("proteins"))
    #         is_nonneg_int = _check_nonnegative_integers(pro_exp)
    #         if not is_nonneg_int:
    #             warnings.warn(
    #                 "Make sure the registered protein expression in anndata contains unnormalized count data."
    #             )
    #     else:
    #         raise ValueError("No protein data found, please setup or transfer anndata")
    #
    #     return adata
    #
    # @torch.no_grad()
    # def get_protein_background_mean(self, adata, indices, batch_size):
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #     background_mean = []
    #     for tensors in scdl:
    #         _, inference_outputs, _ = self.module.forward(tensors)
    #         b_mean = inference_outputs["py_"]["rate_back"]
    #         background_mean += [b_mean.cpu().numpy()]
    #     return np.concatenate(background_mean)
    #
    # @staticmethod
    # @setup_anndata_dsp.dedent
    # def setup_anndata(
    #     adata: AnnData,
    #     protein_expression_obsm_key: str,
    #     protein_names_uns_key: Optional[str] = None,
    #     batch_key: Optional[str] = None,
    #     layer: Optional[str] = None,
    #     categorical_covariate_keys: Optional[List[str]] = None,
    #     continuous_covariate_keys: Optional[List[str]] = None,
    #     copy: bool = False,
    # ) -> Optional[AnnData]:
    #     """
    #     %(summary)s.
    #
    #     Parameters
    #     ----------
    #     %(param_adata)s
    #     protein_expression_obsm_key
    #         key in `adata.obsm` for protein expression data.
    #     protein_names_uns_key
    #         key in `adata.uns` for protein names. If None, will use the column names of `adata.obsm[protein_expression_obsm_key]`
    #         if it is a DataFrame, else will assign sequential names to proteins.
    #     %(param_batch_key)s
    #     %(param_layer)s
    #     %(param_cat_cov_keys)s
    #     %(param_cont_cov_keys)s
    #     %(param_copy)s
    #
    #     Returns
    #     -------
    #     %(returns)s
    #     """
    #     return _setup_anndata(
    #         adata,
    #         batch_key=batch_key,
    #         layer=layer,
    #         protein_expression_obsm_key=protein_expression_obsm_key,
    #         protein_names_uns_key=protein_names_uns_key,
    #         categorical_covariate_keys=categorical_covariate_keys,
    #         continuous_covariate_keys=continuous_covariate_keys,
    #         copy=copy,
    #     )
    #
    #
    #


class SCRNA(nn.Module):
    """
    Single-cell RNA VAE model
    """

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        gene_dispersion: Literal[
            "gene", "gene-batch", "gene-label", "gene-cell"
        ] = "gene",
        gene_likelihood: Literal["zinb", "nb"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs,
    ):
        super(SCRNA, self).__init__()
        self.adata = adata
        if "batch" not in adata.obs.keys():
            adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
        if "n_batch" not in adata.uns.keys():
            adata.uns["n_batch"] = 1

        n_cats_per_cov = (
            adata.uns["extra_categoricals"]["n_cats_per_key"]
            if "extra_categoricals" in adata.uns.keys()
            else None
        )
        n_batch = self.adata.uns["n_batch"]
        library_log_means, library_log_vars = init_library_size(adata, n_batch)

        self.module = RNAVAE(
            n_input_genes=adata.X.shape[1],
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=0,
            n_cats_per_cov=n_cats_per_cov,
            gene_dispersion=gene_dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            use_observed_lib_size=True,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 4e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            n_sample_ref: int = 50,
            split_index: Optional[float] = None,
            early_stopping: bool = False,
            check_val_every_n_epoch: Optional[int] = None,
            reduce_lr_on_plateau: bool = True,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            adversarial_classifier: Optional[bool] = None,
            plan_kwargs: Optional[dict] = None,
            weight_decay=1e-6,
            record_loss=True,
            **kwargs,
    ):
        """
        Trains the model using amortized variational inference.
        """
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=0.9,
            validation_size=0.1,
            use_gpu=True,
        )
        if split_index is not None:
            train_adata = self.adata[split_index[0]]
            val_adata = self.adata[split_index[1]]

        self.train_adata = train_adata
        self.val_adata = val_adata
        self.batch_size = batch_size

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        if n_sample_ref > train_adata.shape[0]:
            ref_idx = np.arange(train_adata.shape[0])
        else:
            ref_idx = get_anchor_index(train_adata.X, n_pcs=50, n_clusters=n_sample_ref)
            if split_index is not None:
                ref_idx = split_index[3]
        # ref_idx = np.random.permutation(train_adata.shape[1])[0:500]
        self.ref_adata = train_adata[ref_idx, :]
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_loss = []
        epoch_rec_loss = []
        epoch_kl_z_loss = []
        epoch_kl_l_loss = []
        epoch_se_loss = []
        epoch_coef_reg_loss = []
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            batch_loss = []
            batch_rec_loss = []
            batch_kl_z_loss = []
            batch_kl_l_loss = []
            batch_se_loss = []
            batch_coef_reg_loss = []
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            for train_batch in train_adata_batch:
                input_genes = torch.tensor(train_batch.layers['rna_expression'], dtype=torch.float32)
                if "batch" in train_batch.obs.keys():
                    batch_index = torch.tensor(train_batch.obs['batch'], dtype=torch.float32).reshape(-1, 1)
                else:
                    batch_index = None
                _, _, _, all_loss = self.module(input_genes, ref_adata, batch_index)
                loss, reconst_loss, kl_local, se_losses = all_loss
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), 0.001)
                optimizer.step()
                if record_loss:
                    batch_loss.append(loss.data.numpy())
                    batch_rec_loss.append(torch.mean(reconst_loss).data.numpy())
                    batch_kl_z_loss.append(torch.mean(kl_local['kl_div_z']).data.numpy())
                    if not self.module.use_observed_lib_size:
                        batch_kl_l_loss.append(torch.mean(kl_local['kl_div_l']).data.numpy())
                    else:
                        batch_kl_l_loss.append(kl_local['kl_div_l'])
                    batch_se_loss.append(torch.mean(se_losses['se_loss']).data.numpy())
                    batch_coef_reg_loss.append(torch.mean(se_losses['coeff_reg']).data.numpy())

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                print("epoch: {}; loss: {} ".format(epoch, np.mean(batch_loss)))
            scheduler.step()
            # self.module.eval()
            epoch_loss.append(np.mean(batch_loss))
            epoch_rec_loss.append(np.mean(batch_rec_loss))
            epoch_kl_z_loss.append(np.mean(batch_kl_z_loss))
            epoch_kl_l_loss.append(np.mean(batch_kl_l_loss))
            epoch_se_loss.append(np.mean(batch_se_loss))
            epoch_coef_reg_loss.append(np.mean(batch_rec_loss))
        self.history = dict(
            epoch_loss=epoch_loss,
            epoch_rec_loss=epoch_rec_loss,
            epoch_kl_z_loss=epoch_kl_z_loss,
            epoch_kl_l_loss=epoch_kl_l_loss,
            epoch_se_loss=epoch_se_loss,
            epoch_coef_reg_loss=epoch_coef_reg_loss
        )

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            qz_m = outputs["qz_m"]
            qz_v = outputs["qz_v"]
            z = outputs["z"]
            if give_mean:
                samples = Normal(qz_m, qz_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z = self.module.encoder_genes.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m

            latent += [z.cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_reconstruct_latent(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        ref_z_list = []
        rec_z_list = []
        for data in ref_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.module.inference(refer_genes, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_genes = torch.cat(ref_z_list, dim=0)
        if not give_mean:
            mc_samples = 1
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            inference_outputs = self.module.inference(input_genes, batch_index, n_samples=mc_samples)
            z_genes = inference_outputs['z'].mean(dim=0)
            senet_outputs = self.module.self_expressiveness(z_genes, ref_z_genes)
            rec_z = senet_outputs['rec_queries']
            rec_z_list.append(rec_z.cpu())
        return torch.cat(rec_z_list, dim=0).numpy()

    @torch.no_grad()
    def get_coeff(
            self,
            adata: Optional[AnnData] = None,
            batch_size: int = 128,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        coeff_genes_list = []
        ref_adata = self.ref_adata
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            _, senet_outputs, _ = self.module(input_genes, ref_adata, batch_index, compute_loss=False)
            coeff_genes_list.append(senet_outputs['coeff'])
        coeff_genes = torch.cat(coeff_genes_list, dim=0)
        return coeff_genes.cpu().numpy()

    @torch.no_grad()
    def get_latent_library_size(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.module.training:
            self.module.eval()
        libraries = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            ql_m = outputs["ql_m"]
            ql_v = outputs["ql_v"]
            library_gene = outputs["library_gene"]
            if give_mean and (not self.module.use_observed_lib_size):
                untran_l = Normal(ql_m, ql_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                library = torch.exp(untran_l)
                library = library.mean(dim=0)
            else:
                library = library_gene

            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()

    @torch.no_grad()
    def get_normalized_expression(
            self,
            adata=None,
            indices=None,
            n_samples_overall: Optional[int] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            gene_list: Optional[Sequence[str]] = None,
            library_size: Optional[Union[float, Literal["latent"]]] = 1,
            n_samples: int = 1,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        r"""
        Returns the normalized gene expression and protein expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to use in total
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude.
        n_samples
            Get sample scale from multiple samples.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
            names as columns. If either n_samples=1 or return_mean=True, defaults to False.
            Otherwise, it defaults to True.

        Returns
        -------
        - **gene_normalized_expression** - normalized expression for RNA

        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is ``(samples, cells, genes)``.
        Otherwise, shape is ``(cells, genes)``. Return type is ``pd.DataFrame`` unless ``return_numpy`` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        scale_list_gene = []

        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            px_scale = torch.zeros_like(input_genes)
            if n_samples > 1:
                px_scale = torch.stack(n_samples * [px_scale])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_genes, ref_adata, batch_index, compute_loss=False,
                                                       transform_batch=b)
                if library_size == "latent":
                    px_scale += generative_outputs["rate"].cpu()
                else:
                    px_scale += generative_outputs["scale"].cpu()
                px_scale = px_scale[..., gene_mask]  # this doesn't change px_scale?

            px_scale /= len(transform_batch)
            scale_list_gene.append(px_scale)

        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            scale_list_gene = torch.cat(scale_list_gene, dim=1)
            # (cells, features, samples)
            scale_list_gene = scale_list_gene.permute(1, 2, 0)
        else:
            scale_list_gene = torch.cat(scale_list_gene, dim=0)

        if return_mean is True and n_samples > 1:
            scale_list_gene = torch.mean(scale_list_gene, dim=-1)

        scale_list_gene = scale_list_gene.cpu().numpy()
        if return_numpy is None or return_numpy is False:
            gene_df = pd.DataFrame(
                scale_list_gene,
                columns=self.adata.var_names[gene_mask],
                index=self.adata.obs_names[indices],
            )
            return gene_df
        else:
            return scale_list_gene


class SCProtein(nn.Module):
    """
    Parameters
    ----------
    adata
        AnnData object.
    n_latent
        Dimensionality of the latent space.
    protein_dispersion
        One of the following:

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED
    protein_likelihood
        One of:

        * ``'nbm'`` - Negative binomial mixture distribution
        * ``'nb'`` - Negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    empirical_protein_background_prior
        Set the initialization of protein background prior empirically. This option fits a GMM for each of
        100 cells per batch and averages the distributions. Note that even with this option set to `True`,
        this only initializes a parameter that is learned during inference. If `False`, randomly initializes.
        The default (`None`), sets this to `True` if greater than 10 proteins are used.

    **model_kwargs
        Keyword args for :class:`~scMSI.scMSI_main.scMSIVAE`

    Notes
    -----
    See further usage examples in the following tutorials:

    to be updated

    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            protein_dispersion: Literal[
                "protein", "protein-batch", "protein-label"
            ] = "protein",
            protein_likelihood: Literal["nbm", "nb", "poisson"] = "nbm",
            latent_distribution: Literal["normal", "ln"] = "normal",
            empirical_protein_background_prior: Optional[bool] = None,
            **model_kwargs,
    ):
        super(SCProtein, self).__init__()
        self.adata = adata
        emp_prior = (
            empirical_protein_background_prior
            if empirical_protein_background_prior is not None
            else (adata.obsm["protein_expression"].shape[1] > 10)
        )
        if "batch" not in adata.obs.keys():
            adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
        if "n_batch" not in adata.uns.keys():
            adata.uns["n_batch"] = 1
        if emp_prior:
            batch_mask = None
            prior_mean, prior_scale = get_protein_priors(adata, batch_mask)
        else:
            prior_mean, prior_scale = None, None

        n_cats_per_cov = (
            adata.uns["extra_categoricals"]["n_cats_per_key"]
            if "extra_categoricals" in adata.uns.keys()
            else None
        )
        self.adata = adata
        n_batch = self.adata.uns["n_batch"]
        # train_adata, val_adata = data_splitter(
        #     self.adata,
        #     train_size=0.9,
        #     validation_size=0.1,
        #     use_gpu=False,
        # )
        # self.train_adata = train_adata
        # self.val_adata = val_adata
        # prior_mean[4] = 5.0
        self.module = ProteinSVAE(
            n_input_proteins=adata.obsm['protein_expression'].shape[1],
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=0,
            n_cats_per_cov=n_cats_per_cov,
            protein_dispersion=protein_dispersion,
            protein_likelihood=protein_likelihood,
            latent_distribution=latent_distribution,
            protein_background_prior_mean=prior_mean,
            protein_background_prior_scale=prior_scale,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 4e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 128,
            n_batch_ref: int = 4,
            early_stopping: bool = False,
            check_val_every_n_epoch: Optional[int] = None,
            reduce_lr_on_plateau: bool = True,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            adversarial_classifier: Optional[bool] = None,
            plan_kwargs: Optional[dict] = None,
            weight_decay=1e-6,
            record_loss=True,
            **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        n_batch_ref
            reference samples number.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        weight_decay
            weight decay
        record_loss
            whether save loss
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=0.9,
            validation_size=0.1,
            use_gpu=False,
        )
        self.train_adata = train_adata
        self.val_adata = val_adata
        self.batch_size = batch_size

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        if n_batch_ref * batch_size > train_adata.shape[0]:
            ref_idx = np.arange(train_adata.shape[0])
        else:
            ref_idx = get_anchor_index(train_adata.X, n_pcs=50, n_clusters=n_batch_ref * batch_size)
        # ref_idx = np.random.permutation(train_adata.shape[1])[0:500]
        self.ref_adata = train_adata[ref_idx, :]
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_loss = []
        epoch_rec_loss = []
        epoch_kl_z_loss = []
        epoch_kl_back_loss = []
        epoch_se_loss = []
        epoch_coef_reg_loss = []
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            batch_loss = []
            batch_rec_loss = []
            batch_kl_z_loss = []
            batch_kl_back_loss = []
            batch_se_loss = []
            batch_coef_reg_loss = []
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            for train_batch in train_adata_batch:
                input_proteins = torch.tensor(train_batch.obsm['protein_expression'], dtype=torch.float32)
                if "batch" in train_batch.obs.keys():
                    batch_index = torch.tensor(train_batch.obs['batch'], dtype=torch.float32).reshape(-1, 1)
                else:
                    batch_index = None
                _, _, _, all_loss = self.module(input_proteins, ref_adata, batch_index)
                loss, reconst_loss, kl_local, se_losses = all_loss
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), 0.001)
                optimizer.step()
                if record_loss:
                    batch_loss.append(loss.data.numpy())
                    batch_rec_loss.append(torch.mean(reconst_loss).data.numpy())
                    batch_kl_z_loss.append(torch.mean(kl_local['kl_div_z']).data.numpy())
                    if self.module.protein_likelihood == 'nbm':
                        batch_kl_back_loss.append(torch.mean(kl_local['kl_div_back']).data.numpy())
                    else:
                        batch_kl_back_loss.append(kl_local['kl_div_back'])
                    batch_se_loss.append(torch.mean(se_losses['se_loss']).data.numpy())
                    batch_coef_reg_loss.append(torch.mean(se_losses['coeff_reg']).data.numpy())

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                print("epoch: {}; loss: {} ".format(epoch, np.mean(batch_loss)))
            scheduler.step()
            # self.module.eval()
            epoch_loss.append(np.mean(batch_loss))
            epoch_rec_loss.append(np.mean(batch_rec_loss))
            epoch_kl_z_loss.append(np.mean(batch_kl_z_loss))
            epoch_kl_back_loss.append(np.mean(batch_kl_back_loss))
            epoch_se_loss.append(np.mean(batch_se_loss))
            epoch_coef_reg_loss.append(np.mean(batch_rec_loss))
        self.history = dict(
            epoch_loss=epoch_loss,
            epoch_rec_loss=epoch_rec_loss,
            epoch_kl_z_loss=epoch_kl_z_loss,
            epoch_kl_back_loss=epoch_kl_back_loss,
            epoch_se_loss=epoch_se_loss,
            epoch_coef_reg_loss=epoch_coef_reg_loss
        )

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            outputs = self.module.inference(input_genes, batch_index)
            qz_m = outputs["qz_m"]
            qz_v = outputs["qz_v"]
            z = outputs["z"]
            if give_mean:
                samples = Normal(qz_m, qz_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z = self.module.encoder_proteins.z_transformation(samples)
                z = z.mean(dim=0)
            else:
                z = qz_m

            latent += [z.cpu()]
        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_reconstruct_latent(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        ref_z_list = []
        rec_z_list = []
        for data in ref_adata:
            refer_genes = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.module.inference(refer_genes, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_genes = torch.cat(ref_z_list, dim=0)
        if not give_mean:
            mc_samples = 1
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            inference_outputs = self.module.inference(input_genes, batch_index, n_samples=mc_samples)
            z_genes = inference_outputs['z'].mean(dim=0)
            senet_outputs = self.module.self_expressiveness(z_genes, ref_z_genes)
            rec_z = senet_outputs['rec_queries']
            rec_z_list.append(rec_z.cpu())
        return torch.cat(rec_z_list, dim=0).numpy()

    @torch.no_grad()
    def get_coeff(
            self,
            adata: Optional[AnnData] = None,
            batch_size: int = 128,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        coeff_genes_list = []
        ref_adata = self.ref_adata
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            if "batch" in data.obs.keys():
                batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            else:
                batch_index = None
            _, senet_outputs, _ = self.module(input_genes, ref_adata, batch_index, compute_loss=False)
            coeff_genes_list.append(senet_outputs['coeff'])
        coeff_genes = torch.cat(coeff_genes_list, dim=0)
        return coeff_genes.cpu().numpy()

    @torch.no_grad()
    def get_normalized_expression(
            self,
            adata=None,
            indices=None,
            n_samples_overall: Optional[int] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            protein_list: Optional[Sequence[str]] = None,
            n_samples: int = 1,
            sample_protein_mixing: bool = False,
            scale_protein: bool = False,
            include_protein_background: bool = False,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        r"""
        Returns the normalized gene expression and protein expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to use in total
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Get sample scale from multiple samples.
        sample_protein_mixing
            Sample mixing bernoulli, setting background to zero
        scale_protein
            Make protein expression sum to 1
        include_protein_background
            Include background component for protein expression
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
            names as columns. If either n_samples=1 or return_mean=True, defaults to False.
            Otherwise, it defaults to True.

        Returns
        -------
        - **protein_normalized_expression** - normalized expression for proteins

        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is ``(samples, cells, genes)``.
        Otherwise, shape is ``(cells, genes)``. Return type is ``pd.DataFrame`` unless ``return_numpy`` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = adata.uns['protein_names']
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        scale_list_pro = []
        for data in adata_batch:
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            py_scale = torch.zeros_like(input_pros)
            if n_samples > 1:
                py_scale = torch.stack(n_samples * [py_scale])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_pros, ref_adata, batch_index, compute_loss=False,
                                                       transform_batch=b)
                py_ = generative_outputs
                # probability of background
                protein_mixing = 1 / (1 + torch.exp(-py_["mixing"].cpu()))
                if sample_protein_mixing is True:
                    protein_mixing = torch.distributions.Bernoulli(
                        protein_mixing
                    ).sample()
                protein_val = py_["fore_rate"].cpu() * (1 - protein_mixing)
                if include_protein_background is True:
                    protein_val += py_["back_rate"].cpu() * protein_mixing

                if scale_protein is True:
                    protein_val = torch.nn.functional.normalize(
                        protein_val, p=1, dim=-1
                    )
                protein_val = protein_val[..., protein_mask]
                py_scale += protein_val
            py_scale /= len(transform_batch)
            scale_list_pro.append(py_scale)

        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            scale_list_pro = torch.cat(scale_list_pro, dim=1)
            # (cells, features, samples)
            scale_list_pro = scale_list_pro.permute(1, 2, 0)
        else:
            scale_list_pro = torch.cat(scale_list_pro, dim=0)

        if return_mean is True and n_samples > 1:
            scale_list_pro = torch.mean(scale_list_pro, dim=-1)

        scale_list_pro = scale_list_pro.cpu().numpy()
        if return_numpy is None or return_numpy is False:
            pro_df = pd.DataFrame(
                scale_list_pro,
                columns=self.adata.uns["protein_names"][protein_mask],
                index=self.adata.obs_names[indices],
            )

            return pro_df
        else:
            return scale_list_pro

    @torch.no_grad()
    def get_protein_foreground_probability(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            protein_list: Optional[Sequence[str]] = None,
            n_samples: int = 1,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ):
        r"""
        Returns the foreground probability for proteins.

        This is denoted as :math:`(1 - \pi_{nt})` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        - **foreground_probability** - probability foreground for each protein

        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)

        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = adata.uns['protein_names']
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        py_mixings = []
        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        for data in adata_batch:
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            py_mixing = torch.zeros_like(input_pros[..., protein_mask])
            if n_samples > 1:
                py_mixing = torch.stack(n_samples * [py_mixing])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_pros, ref_adata, batch_index, compute_loss=False,
                                                       transform_batch=b)
                py_ = generative_outputs
                # probability of background
                py_mixing += torch.sigmoid(py_["mixing"])[..., protein_mask].cpu()
            py_mixing /= len(transform_batch)
            py_mixings += [py_mixing]
        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            py_mixings = torch.cat(py_mixings, dim=1)
            # (cells, features, samples)
            py_mixings = py_mixings.permute(1, 2, 0)
        else:
            py_mixings = torch.cat(py_mixings, dim=0)

        if return_mean is True and n_samples > 1:
            py_mixings = torch.mean(py_mixings, dim=-1)

        py_mixings = py_mixings.cpu().numpy()

        if return_numpy is True:
            return 1 - py_mixings
        else:
            pro_names = self.adata.uns["protein_names"]
            foreground_prob = pd.DataFrame(
                1 - py_mixings,
                columns=pro_names[protein_mask],
                index=adata.obs_names[indices],
            )
            return foreground_prob

    @torch.no_grad()
    def get_protein_likelihood_parameters(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        back_rate_list = []
        fore_rate_list = []
        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        for data in adata_batch:
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            _, _, generative_outputs = self.module(input_pros, ref_adata, batch_index, compute_loss=False)
            py_ = generative_outputs
            back_rate_list += [py_['back_rate']]
            fore_rate_list += [py_['fore_rate']]
        back_rate = torch.cat(back_rate_list, dim=0).cpu().numpy()
        fore_rate = torch.cat(fore_rate_list, dim=0).cpu().numpy()
        return back_rate, fore_rate

    def _expression_for_de(
            self,
            adata=None,
            indices=None,
            n_samples_overall=None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            scale_protein=False,
            batch_size: Optional[int] = None,
            sample_protein_mixing=False,
            include_protein_background=False,
            protein_prior_count=0.5,
    ):
        rna, protein = self.get_normalized_expression(
            adata=adata,
            indices=indices,
            n_samples_overall=n_samples_overall,
            transform_batch=transform_batch,
            return_numpy=True,
            n_samples=1,
            batch_size=batch_size,
            scale_protein=scale_protein,
            sample_protein_mixing=sample_protein_mixing,
            include_protein_background=include_protein_background,
        )
        protein += protein_prior_count

        joint = np.concatenate([rna, protein], axis=1)
        return joint

    # def differential_expression(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     groupby: Optional[str] = None,
    #     group1: Optional[Iterable[str]] = None,
    #     group2: Optional[str] = None,
    #     idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
    #     mode: Literal["vanilla", "change"] = "change",
    #     delta: float = 0.25,
    #     batch_size: Optional[int] = None,
    #     all_stats: bool = True,
    #     batch_correction: bool = False,
    #     batchid1: Optional[Iterable[str]] = None,
    #     batchid2: Optional[Iterable[str]] = None,
    #     fdr_target: float = 0.05,
    #     silent: bool = False,
    #     protein_prior_count: float = 0.1,
    #     scale_protein: bool = False,
    #     sample_protein_mixing: bool = False,
    #     include_protein_background: bool = False,
    #     **kwargs,
    # ) -> pd.DataFrame:
    #     r"""
    #     A unified method for differential expression analysis.
    #
    #     Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.
    #
    #     Parameters
    #     ----------
    #     {doc_differential_expression}
    #     protein_prior_count
    #         Prior count added to protein expression before LFC computation
    #     scale_protein
    #         Force protein values to sum to one in every single cell (post-hoc normalization)
    #     sample_protein_mixing
    #         Sample the protein mixture component, i.e., use the parameter to sample a Bernoulli
    #         that determines if expression is from foreground/background.
    #     include_protein_background
    #         Include the protein background component as part of the protein expression
    #     **kwargs
    #         Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`
    #
    #     Returns
    #     -------
    #     Differential expression DataFrame.
    #     """
    #     adata = self._validate_anndata(adata)
    #     model_fn = partial(
    #         self._expression_for_de,
    #         scale_protein=scale_protein,
    #         sample_protein_mixing=sample_protein_mixing,
    #         include_protein_background=include_protein_background,
    #         protein_prior_count=protein_prior_count,
    #         batch_size=batch_size,
    #     )
    #     col_names = np.concatenate(
    #         [
    #             np.asarray(_get_var_names_from_setup_anndata(adata)),
    #             self.scvi_setup_dict_["protein_names"],
    #         ]
    #     )
    #     result = _de_core(
    #         adata,
    #         model_fn,
    #         groupby,
    #         group1,
    #         group2,
    #         idx1,
    #         idx2,
    #         all_stats,
    #         cite_seq_raw_counts_properties,
    #         col_names,
    #         mode,
    #         batchid1,
    #         batchid2,
    #         delta,
    #         batch_correction,
    #         fdr_target,
    #         silent,
    #         **kwargs,
    #     )
    #
    #     return result

    # @torch.no_grad()
    # def posterior_predictive_sample(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: int = 1,
    #     batch_size: Optional[int] = None,
    #     gene_list: Optional[Sequence[str]] = None,
    #     protein_list: Optional[Sequence[str]] = None,
    # ) -> np.ndarray:
    #     r"""
    #     Generate observation samples from the posterior predictive distribution.
    #
    #     The posterior predictive distribution is written as :math:`p(\hat{x}, \hat{y} \mid x, y)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of required samples for each cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     gene_list
    #         Names of genes of interest
    #     protein_list
    #         Names of proteins of interest
    #
    #     Returns
    #     -------
    #     x_new : :class:`~numpy.ndarray`
    #         tensor with shape (n_cells, n_genes, n_samples)
    #     """
    #     if self.module.gene_likelihood not in ["nb"]:
    #         raise ValueError("Invalid gene_likelihood")
    #
    #     adata = self._validate_anndata(adata)
    #     if gene_list is None:
    #         gene_mask = slice(None)
    #     else:
    #         all_genes = _get_var_names_from_setup_anndata(adata)
    #         gene_mask = [True if gene in gene_list else False for gene in all_genes]
    #     if protein_list is None:
    #         protein_mask = slice(None)
    #     else:
    #         all_proteins = self.scvi_setup_dict_["protein_names"]
    #         protein_mask = [True if p in protein_list else False for p in all_proteins]
    #
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         rna_sample, protein_sample = self.module.sample(
    #             tensors, n_samples=n_samples
    #         )
    #         rna_sample = rna_sample[..., gene_mask]
    #         protein_sample = protein_sample[..., protein_mask]
    #         data = torch.cat([rna_sample, protein_sample], dim=-1).numpy()
    #
    #         scdl_list += [data]
    #         if n_samples > 1:
    #             scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #     scdl_list = np.concatenate(scdl_list, axis=0)
    #
    #     return scdl_list
    #
    # @torch.no_grad()
    # def _get_denoised_samples(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 25,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[int] = None,
    # ) -> np.ndarray:
    #     """
    #     Return samples from an adjusted posterior predictive.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         indices of `adata` to use
    #     n_samples
    #         How may samples per cell
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         int of which batch to condition on for all cells
    #     """
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #
    #     scdl_list = []
    #     for tensors in scdl:
    #         x = tensors[_CONSTANTS.X_KEY]
    #         y = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
    #
    #         generative_kwargs = dict(transform_batch=transform_batch)
    #         inference_kwargs = dict(n_samples=n_samples)
    #         with torch.no_grad():
    #             inference_outputs, generative_outputs, = self.module.forward(
    #                 tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )
    #         px_ = generative_outputs["px_"]
    #         py_ = generative_outputs["py_"]
    #         device = px_["r"].device
    #
    #         pi = 1 / (1 + torch.exp(-py_["mixing"]))
    #         mixing_sample = torch.distributions.Bernoulli(pi).sample()
    #         protein_rate = py_["rate_fore"]
    #         rate = torch.cat((rna_size_factor * px_["scale"], protein_rate), dim=-1)
    #         if len(px_["r"].size()) == 2:
    #             px_dispersion = px_["r"]
    #         else:
    #             px_dispersion = torch.ones_like(x).to(device) * px_["r"]
    #         if len(py_["r"].size()) == 2:
    #             py_dispersion = py_["r"]
    #         else:
    #             py_dispersion = torch.ones_like(y).to(device) * py_["r"]
    #
    #         dispersion = torch.cat((px_dispersion, py_dispersion), dim=-1)
    #
    #         # This gamma is really l*w using scVI manuscript notation
    #         p = rate / (rate + dispersion)
    #         r = dispersion
    #         l_train = torch.distributions.Gamma(r, (1 - p) / p).sample()
    #         data = l_train.cpu().numpy()
    #         # make background 0
    #         data[:, :, self.adata.shape[1] :] = (
    #             data[:, :, self.adata.shape[1] :] * (1 - mixing_sample).cpu().numpy()
    #         )
    #         scdl_list += [data]
    #
    #         scdl_list[-1] = np.transpose(scdl_list[-1], (1, 2, 0))
    #
    #     return np.concatenate(scdl_list, axis=0)
    #
    # @torch.no_grad()
    # def get_feature_correlation_matrix(
    #     self,
    #     adata=None,
    #     indices=None,
    #     n_samples: int = 10,
    #     batch_size: int = 64,
    #     rna_size_factor: int = 1000,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     correlation_type: Literal["spearman", "pearson"] = "spearman",
    #     log_transform: bool = False,
    # ) -> pd.DataFrame:
    #     """
    #     Generate gene-gene correlation matrix using scvi uncertainty and expression.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     rna_size_factor
    #         size factor for RNA prior to sampling gamma distribution
    #     transform_batch
    #         Batches to condition on.
    #         If transform_batch is:
    #
    #         - None, then real observed batch is used
    #         - int, then batch transform_batch is used
    #         - list of int, then values are averaged over provided batches.
    #     correlation_type
    #         One of "pearson", "spearman".
    #     log_transform
    #         Whether to log transform denoised values prior to correlation calculation.
    #
    #     Returns
    #     -------
    #     Gene-protein-gene-protein correlation matrix
    #     """
    #     from scipy.stats import spearmanr
    #
    #     adata = self._validate_anndata(adata)
    #
    #     if not isinstance(transform_batch, IterableClass):
    #         transform_batch = [transform_batch]
    #
    #     transform_batch = _get_batch_code_from_category(adata, transform_batch)
    #
    #     corr_mats = []
    #     for b in transform_batch:
    #         denoised_data = self._get_denoised_samples(
    #             n_samples=n_samples,
    #             batch_size=batch_size,
    #             rna_size_factor=rna_size_factor,
    #             transform_batch=b,
    #         )
    #         flattened = np.zeros(
    #             (denoised_data.shape[0] * n_samples, denoised_data.shape[1])
    #         )
    #         for i in range(n_samples):
    #             flattened[
    #                 denoised_data.shape[0] * (i) : denoised_data.shape[0] * (i + 1)
    #             ] = denoised_data[:, :, i]
    #         if log_transform is True:
    #             flattened[:, : self.n_genes] = np.log(
    #                 flattened[:, : self.n_genes] + 1e-8
    #             )
    #             flattened[:, self.n_genes :] = np.log1p(flattened[:, self.n_genes :])
    #         if correlation_type == "pearson":
    #             corr_matrix = np.corrcoef(flattened, rowvar=False)
    #         else:
    #             corr_matrix, _ = spearmanr(flattened, axis=0)
    #         corr_mats.append(corr_matrix)
    #
    #     corr_matrix = np.mean(np.stack(corr_mats), axis=0)
    #     var_names = _get_var_names_from_setup_anndata(adata)
    #     names = np.concatenate(
    #         [np.asarray(var_names), self.scvi_setup_dict_["protein_names"]]
    #     )
    #     return pd.DataFrame(corr_matrix, index=names, columns=names)
    #
    # @torch.no_grad()
    # def get_likelihood_parameters(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     n_samples: Optional[int] = 1,
    #     give_mean: Optional[bool] = False,
    #     batch_size: Optional[int] = None,
    # ) -> Dict[str, np.ndarray]:
    #     r"""
    #     Estimates for the parameters of the likelihood :math:`p(x, y \mid z)`.
    #
    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     give_mean
    #         Return expected value of parameters or a samples
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     """
    #     raise NotImplementedError
    #
    # def _validate_anndata(
    #     self, adata: Optional[AnnData] = None, copy_if_view: bool = True
    # ):
    #     adata = super()._validate_anndata(adata, copy_if_view)
    #     error_msg = "Number of {} in anndata different from when setup_anndata was run. Please rerun setup_anndata."
    #     if _CONSTANTS.PROTEIN_EXP_KEY in adata.uns["_scvi"]["data_registry"].keys():
    #         pro_exp = get_from_registry(adata, _CONSTANTS.PROTEIN_EXP_KEY)
    #         if self.summary_stats["n_proteins"] != pro_exp.shape[1]:
    #             raise ValueError(error_msg.format("proteins"))
    #         is_nonneg_int = _check_nonnegative_integers(pro_exp)
    #         if not is_nonneg_int:
    #             warnings.warn(
    #                 "Make sure the registered protein expression in anndata contains unnormalized count data."
    #             )
    #     else:
    #         raise ValueError("No protein data found, please setup or transfer anndata")
    #
    #     return adata

    # @torch.no_grad()
    # def get_protein_background_mean(self, adata, indices, batch_size):
    #     adata = self._validate_anndata(adata)
    #     scdl = self._make_data_loader(
    #         adata=adata, indices=indices, batch_size=batch_size
    #     )
    #     background_mean = []
    #     for tensors in scdl:
    #         _, inference_outputs, _ = self.module.forward(tensors)
    #         b_mean = inference_outputs["py_"]["rate_back"]
    #         background_mean += [b_mean.cpu().numpy()]
    #     return np.concatenate(background_mean)

    # @staticmethod
    # @setup_anndata_dsp.dedent
    # def setup_anndata(
    #     adata: AnnData,
    #     protein_expression_obsm_key: str,
    #     protein_names_uns_key: Optional[str] = None,
    #     batch_key: Optional[str] = None,
    #     layer: Optional[str] = None,
    #     categorical_covariate_keys: Optional[List[str]] = None,
    #     continuous_covariate_keys: Optional[List[str]] = None,
    #     copy: bool = False,
    # ) -> Optional[AnnData]:
    #     """
    #     %(summary)s.
    #
    #     Parameters
    #     ----------
    #     %(param_adata)s
    #     protein_expression_obsm_key
    #         key in `adata.obsm` for protein expression data.
    #     protein_names_uns_key
    #         key in `adata.uns` for protein names. If None, will use the column names of `adata.obsm[protein_expression_obsm_key]`
    #         if it is a DataFrame, else will assign sequential names to proteins.
    #     %(param_batch_key)s
    #     %(param_layer)s
    #     %(param_cat_cov_keys)s
    #     %(param_cont_cov_keys)s
    #     %(param_copy)s
    #
    #     Returns
    #     -------
    #     %(returns)s
    #     """
    #     return _setup_anndata(
    #         adata,
    #         batch_key=batch_key,
    #         layer=layer,
    #         protein_expression_obsm_key=protein_expression_obsm_key,
    #         protein_names_uns_key=protein_names_uns_key,
    #         categorical_covariate_keys=categorical_covariate_keys,
    #         continuous_covariate_keys=continuous_covariate_keys,
    #         copy=copy,
    #     )
    #
    #
    #


class SERNAProtein(nn.Module):
    """
    Single-cell self-expression model for RNA and protein in original data space.
    """

    def __init__(
            self,
            adata: AnnData,
            n_latent: int = 10,
            gene_dispersion: Literal[
                "gene", "gene-batch", "gene-label", "gene-cell"
            ] = "gene",
            protein_dispersion: Literal[
                "protein", "protein-batch", "protein-label"
            ] = "protein",
            gene_likelihood: Literal["zinb", "nb"] = "nb",
            protein_likelihood: Literal["nbm", "nb", "poisson"] = "nbm",
            latent_distribution: Literal["normal", "ln"] = "normal",
            empirical_protein_background_prior: Optional[bool] = None,
            n_sample_ref: Optional[int] = 100,
            batch_size: Optional[int] = 128,
            **model_kwargs,
    ):
        super(SERNAProtein, self).__init__()
        self.adata = adata
        if "batch" not in adata.obs.keys():
            adata.obs["batch"] = np.zeros(adata.shape[0], dtype=np.int64)
        if "n_batch" not in adata.uns.keys():
            adata.uns["n_batch"] = 1

        n_cats_per_cov = (
            adata.uns["n_cats_per_key"]
            if "n_cats_per_key" in adata.uns.keys()
            else None
        )
        n_batch = self.adata.uns["n_batch"]
        library_log_means, library_log_vars = init_library_size(adata, n_batch)
        emp_prior = (
            empirical_protein_background_prior
            if empirical_protein_background_prior is not None
            else (adata.obsm["protein_expression"].shape[1] >= 10)
        )
        if emp_prior:
            batch_mask = None
            prior_mean, prior_scale = get_protein_priors(adata, batch_mask)
        else:
            prior_mean, prior_scale = None, None
        # prior_mean[1] = 5
        # prior_mean[3] = 5
        # prior_mean[7] = 6
        # prior_mean[8] = 3.2

        train_adata, val_adata = data_splitter(
            self.adata,
            train_size=0.9,
            validation_size=0.1,
            use_gpu=False,
        )
        self.train_adata = train_adata
        self.val_adata = val_adata
        self.batch_size = batch_size
        if n_sample_ref > train_adata.shape[0]:
            ref_idx = np.arange(train_adata.shape[0])
        else:
            ref_idx = get_anchor_index(train_adata.X, n_pcs=20, n_clusters=n_sample_ref)
            print(f"Choose {n_sample_ref} anchor samples for self-expressive learning")
        # ref_idx = np.random.permutation(train_adata.shape[1])[0:500]
        self.ref_adata = train_adata[ref_idx, :]

        self.module = RNAProteinSE(
            n_input_genes=adata.X.shape[1],
            n_input_proteins=adata.obsm['protein_expression'].shape[1],
            n_batch=n_batch,
            n_latent=n_latent,
            n_continuous_cov=0,
            n_cats_per_cov=n_cats_per_cov,
            gene_dispersion=gene_dispersion,
            protein_dispersion=protein_dispersion,
            gene_likelihood=gene_likelihood,
            protein_likelihood=protein_likelihood,
            latent_distribution=latent_distribution,
            protein_background_prior_mean=prior_mean,
            protein_background_prior_scale=prior_scale,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            use_observed_lib_size=True,
            **model_kwargs,
        )

    def train(
            self,
            max_epochs: Optional[int] = 400,
            lr: float = 4e-4,
            use_gpu: Optional[Union[str, int, bool]] = None,
            train_size: float = 0.9,
            validation_size: Optional[float] = None,
            batch_size: int = 256,
            early_stopping: bool = False,
            check_val_every_n_epoch: Optional[int] = None,
            reduce_lr_on_plateau: bool = True,
            n_steps_kl_warmup: Union[int, None] = None,
            n_epochs_kl_warmup: Union[int, None] = None,
            adversarial_classifier: Optional[bool] = None,
            plan_kwargs: Optional[dict] = None,
            weight_decay=1e-6,
            record_loss=True,
            **kwargs,
    ):
        """
        Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping` is `True`
            or `reduce_lr_on_plateau` is `True`. If either of the latter conditions are met, val is checked
            every epoch.
        reduce_lr_on_plateau
            Reduce learning rate on plateau of validation metric (default is ELBO).
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_classifier
            Whether to use adversarial classifier in the latent space. This helps mixing when
            there are missing proteins in any of the batches. Defaults to `True` is missing proteins
            are detected.
        plan_kwargs
            Keyword args for :class:`~scvi.train.AdversarialTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        weight_decay
            weight decay
        record_loss
            whether save loss
        **kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.module.cuda(device=device)
        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        train_adata = self.train_adata
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        optimizer = optim.Adam(self.module.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        epoch_loss = []
        epoch_rec_loss = {}
        epoch_kl_loss = {}
        epoch_se_loss = {}
        self.module.train()
        for epoch in tqdm(range(max_epochs), desc="training", disable=False):
            train_adata_batch = batch_sampler(train_adata, batch_size, shuffle=True)
            batch_loss = []
            batch_rec_loss = {}
            batch_kl_loss = {}
            batch_se_loss = {}
            create_dict = True
            for train_batch in train_adata_batch:
                input_genes = torch.tensor(train_batch.layers['rna_expression'], dtype=torch.float32)
                input_proteins = torch.tensor(train_batch.obsm['protein_expression'], dtype=torch.float32)
                batch_index = torch.tensor(train_batch.obs['batch'], dtype=torch.float32).reshape(-1, 1)
                if "cat_covs" in train_batch.obsm.keys():
                    cat_covs = torch.tensor(train_batch.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
                else:
                    cat_covs = None
                # batch_index = (batch_index, batch_index)
                # print(len(batch_index))
                _, _, _, all_loss = self.module(input_genes, input_proteins, ref_adata, batch_index=batch_index,
                                                cat_covs=cat_covs)
                loss, reconst_loss, kl_local, se_losses = all_loss
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.module.parameters(), 0.001)
                optimizer.step()
                # batch_loss.append(loss.data.numpy())
                if create_dict:
                    for k in reconst_loss.keys():
                        batch_rec_loss[k] = []
                    for k in kl_local.keys():
                        batch_kl_loss[k] = []
                    for k in se_losses.keys():
                        batch_se_loss[k] = []
                    create_dict = False
                batch_loss.append(loss.cpu().data.numpy())
                if record_loss:
                    for k in batch_rec_loss.keys():
                        batch_rec_loss[k].append(torch.mean(reconst_loss[k]).cpu().data.numpy())
                    for k in batch_kl_loss.keys():
                        batch_kl_loss[k].append(torch.mean(kl_local[k]).cpu().data.numpy())
                    for k in batch_se_loss.keys():
                        batch_se_loss[k].append(torch.mean(se_losses[k]).cpu().data.numpy())

            if epoch % 20 == 0 or epoch == (max_epochs - 1):
                print("epoch: {}; loss: {} ".format(epoch, np.mean(batch_loss)))
            scheduler.step()
            # self.module.eval()
            if epoch == 0:
                for k in batch_rec_loss.keys():
                    epoch_rec_loss[k] = []
                for k in batch_kl_loss.keys():
                    epoch_kl_loss[k] = []
                for k in batch_se_loss.keys():
                    epoch_se_loss[k] = []

            epoch_loss.append(np.mean(batch_loss))
            if record_loss:
                for k in epoch_rec_loss.keys():
                    epoch_rec_loss[k].append(np.mean(batch_rec_loss[k]))
                for k in epoch_kl_loss.keys():
                    epoch_kl_loss[k].append(np.mean(batch_kl_loss[k]))
                for k in epoch_se_loss.keys():
                    epoch_se_loss[k].append(np.mean(batch_se_loss[k]))
        self.history = dict(
            epoch_loss=epoch_loss,
            epoch_rec_loss=epoch_rec_loss,
            epoch_kl_loss=epoch_kl_loss,
            epoch_se_loss=epoch_se_loss,
        )

    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        latent_gene = []
        latent_pro = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            enc_gene_outputs, enc_pro_outputs = self.module.inference(input_genes, input_proteins,
                                                                      batch_index=batch_index, cat_covs=cat_covs)
            qz_m_gene = enc_gene_outputs["qz_m"]
            qz_v_gene = enc_gene_outputs["qz_v"]
            z_gene = enc_gene_outputs["z"]
            qz_m_pro = enc_pro_outputs["qz_m"]
            qz_v_pro = enc_pro_outputs["qz_v"]
            z_pro = enc_pro_outputs["z"]
            if give_mean:
                samples = Normal(qz_m_gene, qz_v_gene.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z_gene = self.module.encoder_genes.z_transformation(samples)
                z_gene = z_gene.mean(dim=0)
                samples = Normal(qz_m_pro, qz_v_pro.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                z_pro = self.module.encoder_proteins.z_transformation(samples)
                z_pro = z_pro.mean(dim=0)
            else:
                z_gene = qz_m_gene
                z_pro = qz_m_pro

            latent_gene += [z_gene.cpu()]
            latent_pro += [z_pro.cpu()]
        lat_gene = torch.cat(latent_gene).numpy()
        lat_pro = torch.cat(latent_pro).numpy()

        return lat_gene, lat_pro

    @torch.no_grad()
    def get_reconstruct_latent(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        ref_adata = batch_sampler(self.ref_adata, batch_size, shuffle=False)
        ref_z_gene_list = []
        ref_z_pro_list = []
        rec_z_gene_list = []
        rec_z_pro_list = []
        for data in ref_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_proteins = torch.tensor(data.obsm["protein_expression"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                refer_cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                refer_cat_covs = None
            ref_outputs_gene, ref_outputs_pro = self.module.inference(refer_genes, refer_proteins,
                                                                      batch_index=refer_index, cat_covs=refer_cat_covs)
            ref_z_gene_list.append(ref_outputs_gene['z'])
            ref_z_pro_list.append(ref_outputs_pro['z'])
        ref_z_genes = torch.cat(ref_z_gene_list, dim=0)
        ref_z_pros = torch.cat(ref_z_pro_list, dim=0)
        if not give_mean:
            mc_samples = 1
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm["protein_expression"], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            inference_genes, inference_pros = self.module.inference(input_genes, input_proteins, batch_index,
                                                                    n_samples=mc_samples, cat_covs=cat_covs)
            z_genes = inference_genes['z'].mean(dim=0)
            z_pros = inference_pros['z'].mean(dim=0)
            se_gene_outputs, se_pro_outputs = self.module.self_expressiveness(z_genes, z_pros, ref_z_genes, ref_z_pros)
            rec_z_genes = se_gene_outputs['rec_queries']
            rec_z_pros = se_pro_outputs['rec_queries']
            rec_z_gene_list.append(rec_z_genes.cpu())
            rec_z_pro_list.append(rec_z_pros.cpu())
        rec_z_gs = torch.cat(rec_z_gene_list, dim=0).numpy()
        rec_z_ps = torch.cat(rec_z_pro_list, dim=0).numpy()
        return rec_z_gs, rec_z_ps

    @torch.no_grad()
    def get_coeff(
            self,
            adata: Optional[AnnData] = None,
            batch_size: int = 128,
            pro_w=1,
            ord=1,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        coeff_genes_list = []
        coeff_pros_list = []
        ref_adata = self.ref_adata
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            _, senet_outputs, _ = self.module(input_genes, input_proteins, ref_adata, batch_index=batch_index,
                                              cat_covs=cat_covs, compute_loss=False)
            se_gene_outputs, se_pro_outputs = senet_outputs
            coeff_genes_list.append(se_gene_outputs['coeff'])
            coeff_pros_list.append(se_pro_outputs['coeff'])
        coeff_genes = torch.cat(coeff_genes_list, dim=0)
        coeff_proteins = torch.cat(coeff_pros_list, dim=0)
        coeff = coeff_genes + pro_w * coeff_proteins
        coeff_genes = coeff_genes.cpu().numpy()
        coeff_proteins = coeff_proteins.cpu().numpy()
        coeff = coeff.cpu().numpy()
        if ord is not None:
            coeff_genes = coeff_genes / norm(coeff_genes, ord=ord, axis=1, keepdims=True)
            coeff_proteins = coeff_proteins / norm(coeff_proteins, ord=ord, axis=1, keepdims=True)
            coeff = coeff / norm(coeff, ord=ord, axis=1, keepdims=True)
        return coeff_genes, coeff_proteins, coeff

    @torch.no_grad()
    def get_latent_library_size(
            self,
            adata: Optional[AnnData] = None,
            give_mean: bool = True,
            mc_samples: int = 5000,
            batch_size: Optional[int] = None,
    ) -> np.ndarray:
        r"""
        Returns the latent library size for each cell.

        This is denoted as :math:`\ell_n` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        give_mean
            Return the mean or a sample from the posterior distribution.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        if self.module.training:
            self.module.eval()
        libraries = []
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            outputs, _ = self.module.inference(input_genes, input_proteins, batch_index, cat_covs=cat_covs)
            ql_m = outputs["ql_m"]
            ql_v = outputs["ql_v"]
            library_gene = outputs["library_gene"]
            if give_mean and (not self.module.use_observed_lib_size):
                untran_l = Normal(ql_m, ql_v.sqrt()).sample([mc_samples])
                # z = torch.nn.functional.softmax(samples, dim=-1)
                library = torch.exp(untran_l)
                library = library.mean(dim=0)
            else:
                library = library_gene

            libraries += [library.cpu()]
        return torch.cat(libraries).numpy()

    @torch.no_grad()
    def get_normalized_expression(
            self,
            adata=None,
            indices=None,
            n_samples_overall: Optional[int] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            gene_list: Optional[Sequence[str]] = None,
            protein_list: Optional[Sequence[str]] = None,
            library_size: Optional[Union[float, Literal["latent"]]] = 1,
            n_samples: int = 1,
            sample_protein_mixing: bool = False,
            scale_protein: bool = False,
            include_protein_background: bool = False,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        r"""
        Returns the normalized gene expression and protein expression.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to use in total
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude.
        n_samples
            Get sample scale from multiple samples.
        sample_protein_mixing
            Sample mixing bernoulli, setting background to zero
        scale_protein
            Make protein expression sum to 1
        include_protein_background
            Include background component for protein expression
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a `np.ndarray` instead of a `pd.DataFrame`. Includes gene
            names as columns. If either n_samples=1 or return_mean=True, defaults to False.
            Otherwise, it defaults to True.

        Returns
        -------
        - **gene_normalized_expression** - normalized expression for RNA
        - **protein_normalized_expression** - normalized expression for proteins

        If ``n_samples`` > 1 and ``return_mean`` is False, then the shape is ``(samples, cells, genes)``.
        Otherwise, shape is ``(cells, genes)``. Return type is ``pd.DataFrame`` unless ``return_numpy`` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]
        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = adata.uns['protein_names']
            protein_mask = [True if p in protein_list else False for p in all_proteins]
        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True

        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        scale_list_gene = []
        scale_list_pro = []

        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            px_scale = torch.zeros_like(input_genes)
            py_scale = torch.zeros_like(input_proteins)
            if n_samples > 1:
                px_scale = torch.stack(n_samples * [px_scale])
                py_scale = torch.stack(n_samples * [py_scale])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_genes, input_proteins, ref_adata, batch_index,
                                                       cat_covs=cat_covs, compute_loss=False, transform_batch=b)
                px_, py_ = generative_outputs
                if library_size == "latent":
                    px_scale += px_["rate"].cpu()
                else:
                    px_scale += px_["scale"].cpu()
                px_scale = px_scale[..., gene_mask]

                # probability of background
                protein_mixing = 1 / (1 + torch.exp(-py_["mixing"].cpu()))
                if sample_protein_mixing is True:
                    protein_mixing = torch.distributions.Bernoulli(
                        protein_mixing
                    ).sample()
                protein_val = py_["fore_rate"].cpu() * (1 - protein_mixing)
                if include_protein_background is True:
                    protein_val += py_["back_rate"].cpu() * protein_mixing

                if scale_protein is True:
                    protein_val = torch.nn.functional.normalize(
                        protein_val, p=1, dim=-1
                    )
                protein_val = protein_val[..., protein_mask]
                py_scale += protein_val
            px_scale /= len(transform_batch)
            py_scale /= len(transform_batch)
            scale_list_gene.append(px_scale)
            scale_list_pro.append(py_scale)

        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            scale_list_gene = torch.cat(scale_list_gene, dim=1)
            scale_list_pro = torch.cat(scale_list_pro, dim=1)
            # (cells, features, samples)
            scale_list_gene = scale_list_gene.permute(1, 2, 0)
            scale_list_pro = scale_list_pro.permute(1, 2, 0)
        else:
            scale_list_gene = torch.cat(scale_list_gene, dim=0)
            scale_list_pro = torch.cat(scale_list_pro, dim=0)

        if return_mean is True and n_samples > 1:
            scale_list_gene = torch.mean(scale_list_gene, dim=-1)
            scale_list_pro = torch.mean(scale_list_pro, dim=-1)

        scale_list_gene = scale_list_gene.cpu().numpy()
        scale_list_pro = scale_list_pro.cpu().numpy()
        if return_numpy is None or return_numpy is False:
            gene_df = pd.DataFrame(
                scale_list_gene,
                columns=self.adata.var_names[gene_mask],
                index=self.adata.obs_names[indices],
            )
            pro_df = pd.DataFrame(
                scale_list_pro,
                columns=self.adata.uns["protein_names"][protein_mask],
                index=self.adata.obs_names[indices],
            )

            return gene_df, pro_df
        else:
            return scale_list_gene, scale_list_pro

    @torch.no_grad()
    def get_protein_foreground_probability(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            transform_batch: Optional[Sequence[Union[Number, str]]] = None,
            protein_list: Optional[Sequence[str]] = None,
            n_samples: int = 1,
            batch_size: Optional[int] = None,
            return_mean: bool = True,
            return_numpy: Optional[bool] = None,
    ):
        r"""
        Returns the foreground probability for proteins.

        This is denoted as :math:`(1 - \pi_{nt})` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
            - List[int], then average over batches in list
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        - **foreground_probability** - probability foreground for each protein

        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)

        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = adata.uns['protein_names']
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        py_mixings = []
        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            py_mixing = torch.zeros_like(input_pros[..., protein_mask])
            if n_samples > 1:
                py_mixing = torch.stack(n_samples * [py_mixing])
            for b in transform_batch:
                _, _, generative_outputs = self.module(input_genes, input_pros, ref_adata, batch_index,
                                                       cat_covs=cat_covs, compute_loss=False, transform_batch=b)
                _, py_ = generative_outputs
                # probability of background
                py_mixing += torch.sigmoid(py_["mixing"])[..., protein_mask].cpu()
            py_mixing /= len(transform_batch)
            py_mixings += [py_mixing]
        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            py_mixings = torch.cat(py_mixings, dim=1)
            # (cells, features, samples)
            py_mixings = py_mixings.permute(1, 2, 0)
        else:
            py_mixings = torch.cat(py_mixings, dim=0)

        if return_mean is True and n_samples > 1:
            py_mixings = torch.mean(py_mixings, dim=-1)

        py_mixings = py_mixings.cpu().numpy()

        if return_numpy is True:
            return 1 - py_mixings
        else:
            pro_names = self.adata.uns["protein_names"]
            foreground_prob = pd.DataFrame(
                1 - py_mixings,
                columns=pro_names[protein_mask],
                index=adata.obs_names[indices],
            )
            return foreground_prob

    @torch.no_grad()
    def get_protein_likelihood_parameters(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
    ):
        if self.module.training:
            self.module.eval()
        adata = adata if adata is not None else self.adata
        batch_size = batch_size if batch_size is not None else self.batch_size
        ref_adata = self.ref_adata  # ref_adata features not mask
        if indices is None:
            indices = np.arange(adata.n_obs)
        adata = adata[indices]
        adata_batch = batch_sampler(adata, batch_size, shuffle=False)
        back_rate_list = []
        fore_rate_list = []
        # transform_batch = _get_batch_code_from_category(adata, transform_batch)
        for data in adata_batch:
            input_genes = torch.tensor(data.layers['rna_expression'], dtype=torch.float32)
            input_pros = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            batch_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                cat_covs = None
            _, _, generative_outputs = self.module(input_genes, input_pros, ref_adata, batch_index, cat_covs=cat_covs,
                                                   compute_loss=False)
            _, py_ = generative_outputs
            back_rate_list += [py_['back_rate']]
            fore_rate_list += [py_['fore_rate']]
        back_rate = torch.cat(back_rate_list, dim=0).cpu().numpy()
        fore_rate = torch.cat(fore_rate_list, dim=0).cpu().numpy()
        return back_rate, fore_rate


