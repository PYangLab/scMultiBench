# -*- coding: utf-8 -*-
"""Main module.
Created by Chengming Zhang, Sep 1st, 2022
"""
from typing import Dict, Iterable, Optional, Tuple, Union, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

from .distributions import (
    NegativeBinomial,
    NegativeBinomialMixture,
    ZeroInflatedNegativeBinomial,
)
from .base_components import EncoderGene, EncoderProtein, EncoderPeak, DecoderProtein, SENet, DecoderSCVI, DecoderPeak
from .base_components import EncoderImage, DecoderImage
from .utils import regularizer, regularizer_l12, one_hot

torch.backends.cudnn.benchmark = True


# VAE model
class RNAProteinSVAE(nn.Module):
    """
    Single cell multi-omics self-expressive integration.

    Implements the SCMSI model for single cell multi-omics data.

    Parameters
    ----------
    n_input_genes
        Number of input genes
    n_input_proteins
        Number of input proteins
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    gene_dispersion
        One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    protein_dispersion
        One of the following

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    protein_batch_mask
        Dictionary where each key is a batch code, and value is for each protein, whether it was observed or not.
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    protein_background_prior_mean
        Array of proteins by batches, the prior initialization for the protein background mean (log scale)
    protein_background_prior_scale
        Array of proteins by batches, the prior initialization for the protein background scale (log scale)
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_senet_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_layers_senet: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.1,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_senet: float = 0.0,
        gene_dispersion: str = "gene",
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        protein_likelihood: str = "nbm",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        protein_background_prior_mean: Optional[np.ndarray] = None,
        protein_background_prior_scale: Optional[np.ndarray] = None,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.protein_likelihood = protein_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.n_input_proteins = n_input_proteins
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size
        self.warm_up = 0
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        # parameters for prior on back_rate (background protein mean)
        if protein_background_prior_mean is None:
            if n_batch > 0:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins, n_batch)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins, n_batch), -10, 1)
                )
            else:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins), -10, 1)
                )
        else:
            if protein_background_prior_mean.shape[1] == 1 and n_batch != 1:
                init_mean = protein_background_prior_mean.ravel()
                init_scale = protein_background_prior_scale.ravel()
            else:
                init_mean = protein_background_prior_mean
                init_scale = protein_background_prior_scale
            self.background_pro_alpha = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)))
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:  # gene-cell
            pass

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_batch)
            )
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_labels)
            )
        else:  # protein-cell
            pass

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # n_input = n_input_genes + self.n_input_proteins
        n_input_genes_encoder = n_input_genes + n_continuous_cov * encode_covariates
        n_input_proteins_encoder = n_input_proteins + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        # cat_list = [n_batch, n_batch]
        # print(cat_list)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_genes = EncoderGene(
            n_input_genes_encoder,
            self.n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.encoder_proteins = EncoderProtein(
            n_input_proteins_encoder,
            n_latent,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.decoder_genes = DecoderSCVI(
            n_latent + n_continuous_cov,
            n_input_genes,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.decoder_proteins = DecoderProtein(
            n_latent + n_continuous_cov,
            n_input_proteins,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.senet_genes = SENet(
            n_latent,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.senet_proteins = SENet(
            n_latent,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        cont_covs=None,
        cat_covs=None,
    ):
        """
        Internal helper function to compute necessary inference quantities.

        We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
        The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
        `scale` refers to the quantity upon which differential expression is performed. For genes,
        this can be viewed as the mean of the underlying gamma distribution.

        We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
        `fore_rate` refers to foreground mean, while `back_rate` refers to background mean. ``scale`` refers to
        foreground mean adjusted for background probability and scaled to reside in simplex.
        ``back_mean`` and ``back_var`` are the posterior parameters for ``back_rate``.  ``fore_scale`` is the scaling
        factor that enforces `fore_rate` > `back_rate`.

        ``px_["r"]`` and ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        label
            tensor of cell-types labels with shape (batch_size, n_labels)
        n_samples
            Number of samples to sample from approximate posterior
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        batch_index = batch_index.to(device)
        x_ = x
        y_ = y
        if cat_covs is not None:
            cat_covs = cat_covs.to(device)
        if self.use_observed_lib_size:
            library_gene = x.sum(1).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
            y_ = torch.log(1 + y_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input_genes = torch.cat((x_, cont_covs), dim=-1)
            encoder_input_proteins = torch.cat((y_, cont_covs), dim=-1)
        else:
            encoder_input_genes = x_
            encoder_input_proteins = y_
        # cat_covs = batch_index
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        enc_gene_outputs = self.encoder_genes(
            encoder_input_genes, batch_index, *categorical_input
        )
        enc_pro_outputs = self.encoder_proteins(
            encoder_input_proteins, batch_index, *categorical_input
        )

        if self.use_observed_lib_size:
            enc_gene_outputs['library_gene'] = library_gene

        if n_samples > 1:
            # genes latent space sampling
            qz_m, qz_v, z, untran_z, ql_m, ql_v, _, untran_l = enc_gene_outputs.values()
            enc_gene_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_gene_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_gene_outputs['untran_z'] = Normal(enc_gene_outputs['qz_m'], enc_gene_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_gene_outputs['z'] = self.encoder_genes.z_transformation(enc_gene_outputs['untran_z'])
            enc_gene_outputs['ql_m'] = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            enc_gene_outputs['ql_v'] = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            enc_gene_outputs['untran_l'] = Normal(enc_gene_outputs['ql_m'], enc_gene_outputs['ql_v'].sqrt()).sample()
            if self.use_observed_lib_size:
                enc_gene_outputs['library_gene'] = library_gene.unsqueeze(0).expand(
                    (n_samples, library_gene.size(0), library_gene.size(1))
                )
            else:
                enc_gene_outputs['library_gene'] = self.encoder_genes.l_transformation(untran_l)

            # proteins latent space sampling
            qz_m, qz_v, z, untran_z = enc_pro_outputs.values()
            enc_pro_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_pro_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_pro_outputs['untran_z'] = Normal(enc_pro_outputs['qz_m'], enc_pro_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_pro_outputs['z'] = self.encoder_proteins.z_transformation(enc_pro_outputs['untran_z'])

        # Background regularization
        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)
        if self.n_batch > 0:
            py_back_mean_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_var_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_mean_prior = self.background_pro_alpha
            py_back_var_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(py_back_mean_prior, py_back_var_prior)

        return enc_gene_outputs, enc_pro_outputs

    def self_expressiveness(
        self,
        z_genes: torch.Tensor,
        z_proteins: torch.Tensor,
        ref_z_genes: torch.Tensor,
        ref_z_proteins: torch.Tensor,
    ):
        """
        Use self-expressive network to reconstruct the latent space
        Parameters
        ----------
        z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        z_proteins
            tensor of values with shape ``(batch_size, n_latent_proteins)``
        ref_z_genes
            tensor of values with shape ``(training_size, n_latent_genes)``
        ref_z_proteins
            tensor of values with shape ``(training_size, n_latent_proteins)``

        Returns
        -------
        rec_z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        rec_z_proteins
            tensor of values with shape ``(batch_size, n_latent_proteins)``
        se_gene_outputs
            genes dict with "rec_queries" and "c_list"
        se_pro_outputs
            genes dict with "rec_queries" and "c_list"
        """

        se_gene_outputs = self.senet_genes(z_genes, ref_z_genes)
        se_pro_outputs = self.senet_proteins(z_proteins, ref_z_proteins)
        return se_gene_outputs, se_pro_outputs

    def generative(
        self,
        z_genes: torch.Tensor,
        z_proteins: torch.Tensor,
        library_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        transform_batch: Optional[int] = None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_index = batch_index.to(device)
        if cat_covs is not None:
            cat_covs = cat_covs.to(device)
        if cont_covs is not None:
            decoder_input_genes = torch.cat((z_genes, cont_covs), dim=-1)
            decoder_input_proteins = torch.cat((z_proteins, cont_covs), dim=-1)
        else:
            decoder_input_genes = z_genes
            decoder_input_proteins = z_proteins
        # cat_covs = batch_index
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
            if torch.cuda.is_available():
                batch_index = batch_index.to('cuda')

        px_ = self.decoder_genes(
            decoder_input_genes, library_gene, batch_index, *categorical_input
        )
        py_ = self.decoder_proteins(
            decoder_input_proteins, batch_index, *categorical_input
        )

        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_["r"] = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_["r"] = torch.exp(py_r)

        return px_, py_

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px_dict: Dict[str, torch.Tensor],
        py_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        px_ = px_dict
        py_ = py_dict
        # Reconstruction Loss
        if self.gene_likelihood == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )

        if self.protein_likelihood == "nbm":
            py_conditional = NegativeBinomialMixture(
                mu1=py_["back_rate"],
                mu2=py_["fore_rate"],
                theta1=py_["r"],
                mixture_logits=py_["mixing"],
            )
        elif self.protein_likelihood == "nb":
            py_conditional = NegativeBinomial(mu=py_['rate'], theta=py_["r"])
        else:
            py_conditional = Poisson(py_['rate'])

        reconst_loss_protein_full = -py_conditional.log_prob(y)
        reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_gene, reconst_loss_protein

    def get_self_expressiveness_loss(
        self,
        inference_outputs,
        senet_outputs,
        lmbd=0.9,
    ):
        """Compute self-expressiveness loss."""
        enc_gene_outputs, enc_pro_outputs = inference_outputs
        z_gene = enc_gene_outputs['z']
        z_protein = enc_pro_outputs['z']
        se_gene_outputs, se_pro_outputs = senet_outputs
        rec_z_gene, coeff_gene = se_gene_outputs['rec_queries'], se_gene_outputs['coeff']
        rec_z_protein, coeff_protein = se_pro_outputs['rec_queries'], se_pro_outputs['coeff']
        se_loss_gene = torch.sum(torch.pow(z_gene - rec_z_gene, 2), dim=-1)
        se_loss_protein = torch.sum(torch.pow(z_protein - rec_z_protein, 2), dim=-1)

        c_reg_gene = regularizer(coeff_gene, lmbd).sum(dim=-1)
        c_reg_protein = regularizer(coeff_protein, lmbd).sum(dim=-1)
        # c_reg_genes = regularizer_l12(coeff_gene, lmbd) * torch.ones(coeff_gene.shape[0])
        # c_reg_proteins = regularizer_l12(coeff_protein, lmbd) * torch.ones(coeff_protein.shape[0])
        c_contrast = torch.sum(torch.pow(coeff_gene - coeff_protein, 2), dim=-1)

        return se_loss_gene, se_loss_protein, c_reg_gene, c_reg_protein, c_contrast

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        inference_outputs,
        senet_outputs,
        generative_outputs,
        batch_index: Optional[torch.Tensor] = None,
        pro_recons_weight=1.0,  # double check these defaults
        kl_weight=1.0,
        se_weight=1.0,
    ):
        """
        Returns the reconstruction loss and the Kullback-Leibler divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        inference_outputs
            inference network outputs
        senet_outputs
            self-expressive network outputs
        generative_outputs
            generative network outputs
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        pro_recons_weight
            proteins reconstruction weight
        kl_weight
            Kullback-Leibler divergences loss weight
        se_weight
            self-expressive loss weight
        Returns
        -------
        type
            the reconstruction loss, the KL divergences and self-expressive loss
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        batch_index = batch_index.to(device)
        enc_gene_outputs, enc_pro_outputs = inference_outputs
        qz_m_gene = enc_gene_outputs['qz_m']
        qz_v_gene = enc_gene_outputs['qz_v']
        ql_m_gene = enc_gene_outputs['ql_m']
        ql_v_gene = enc_gene_outputs['ql_v']
        qz_m_protein = enc_pro_outputs['qz_m']
        qz_v_protein = enc_pro_outputs['qz_v']
        px_, py_ = generative_outputs

        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x, y, px_, py_,
        )
        se_loss_gene, se_loss_protein, c_reg_genes, c_reg_proteins, c_contrast = self.get_self_expressiveness_loss(
            inference_outputs, senet_outputs, lmbd=0.95
        )

        # KL Divergence
        kl_div_z_gene = kl(Normal(qz_m_gene, torch.sqrt(qz_v_gene)), Normal(0, 1)).sum(dim=1)
        kl_div_z_protein = kl(Normal(qz_m_protein, torch.sqrt(qz_v_protein)), Normal(0, 1)).sum(dim=1)
        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                one_hot(batch_index, n_batch), self.library_log_means
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index, n_batch), self.library_log_vars
            )
            kl_div_l_gene = kl(
                Normal(ql_m_gene, torch.sqrt(ql_v_gene)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_div_l_gene = 0.0 * torch.ones(x.shape[0])
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            kl_div_l_gene = kl_div_l_gene.to(device)

        if self.protein_likelihood == "nbm":
            kl_div_back_pro_full = kl(
                Normal(py_["back_log_rate_mean"], py_["back_log_rate_var"].sqrt()), self.back_mean_prior
            )
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=-1)
        else:
            kl_div_back_pro = 0.0 * torch.ones(x.shape[0])
        losses = torch.mean(
            reconst_loss_gene
            + pro_recons_weight * reconst_loss_protein
            + kl_weight * kl_div_z_gene
            + kl_weight * kl_div_z_protein
            + kl_div_l_gene
            + kl_weight * kl_div_back_pro
            + se_weight * (se_loss_gene + c_reg_genes)
            + se_weight * (se_loss_protein + c_reg_proteins)
            + se_weight * c_contrast * 0.1
        )

        reconst_losses = dict(
            reconst_loss_gene=reconst_loss_gene,
            reconst_loss_protein=reconst_loss_protein,
        )
        kl_local = dict(
            kl_div_z_gene=kl_div_z_gene,
            kl_div_l_gene=kl_div_l_gene,
            kl_div_z_protein=kl_div_z_protein,
            kl_div_back_pro=kl_div_back_pro,
        )
        se_losses = dict(
            se_loss_gene=se_loss_gene,
            se_loss_protein=se_loss_protein,
            c_reg_genes=c_reg_genes,
            c_reg_proteins=c_reg_proteins,
            c_contrast=c_contrast
        )

        return losses, reconst_losses, kl_local, se_losses

    def forward(
        self,
        input_genes: torch.Tensor,
        input_proteins: torch.Tensor,
        refer_adata,
        batch_index: Optional[torch.Tensor] = None,
        cat_covs: Optional[torch.Tensor] = None,
        compute_loss=True,
        transform_batch: Optional[int] = None,
    ):
        """
        Forward pass through the network.

        Parameters
        ----------
        self
        input_genes
            tensor with shape ``(batch, n_input_gens)``
        input_proteins
            tensor with shape ``(batch, n_input_proteins)``
        refer_adata
            adata with shape ``(training_size, )``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        cat_covs
            categorical variable of each cell
        compute_loss
            whether compute losses
        transform_batch
            batch to condition on
        """
        ref_z_genes_list = []
        ref_z_proteins_list = []
        for data in refer_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                refer_cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                refer_cat_covs = None
            # refer_index = (refer_index, refer_index)
            enc_ref_gene_outputs, enc_ref_protein_outputs = self.inference(refer_genes, refer_proteins, batch_index=refer_index, cat_covs=refer_cat_covs)
            ref_z_genes_list.append(enc_ref_gene_outputs['z'])
            ref_z_proteins_list.append(enc_ref_protein_outputs['z'])
        ref_z_genes = torch.cat(ref_z_genes_list, dim=0)
        ref_z_proteins = torch.cat(ref_z_proteins_list, dim=0)

        inference_outputs = self.inference(input_genes, input_proteins, batch_index=batch_index, cat_covs=cat_covs)
        enc_gene_outputs, enc_pro_outputs = inference_outputs
        z_genes = enc_gene_outputs['z']
        library_gene = enc_gene_outputs['library_gene']
        z_proteins = enc_pro_outputs['z']
        senet_outputs = self.self_expressiveness(z_genes, z_proteins, ref_z_genes, ref_z_proteins)
        se_gene_outputs, se_pro_outputs = senet_outputs
        rec_z_genes = se_gene_outputs['rec_queries']
        rec_z_proteins = se_pro_outputs['rec_queries']
        generative_outputs = self.generative(rec_z_genes, rec_z_proteins, library_gene, batch_index=batch_index, cat_covs=cat_covs, transform_batch=transform_batch)
        if compute_loss:
            loss = self.loss(
                input_genes, input_proteins, inference_outputs, senet_outputs, generative_outputs, batch_index
            )
            return inference_outputs, senet_outputs, generative_outputs, loss
        else:
            return inference_outputs, senet_outputs, generative_outputs


class CiteRnaSVAE(nn.Module):
    """
    Single cell multi-omics self-expressive integration.

    Implements the SCMSI model for single cell multi-omics data.

    Parameters
    ----------
    n_input_genes
        Number of input genes
    n_input_proteins
        Number of input proteins
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    gene_dispersion
        One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    protein_dispersion
        One of the following

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    protein_batch_mask
        Dictionary where each key is a batch code, and value is for each protein, whether it was observed or not.
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    protein_background_prior_mean
        Array of proteins by batches, the prior initialization for the protein background mean (log scale)
    protein_background_prior_scale
        Array of proteins by batches, the prior initialization for the protein background scale (log scale)
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_senet_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_layers_senet: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.1,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_senet: float = 0.0,
        gene_dispersion: str = "gene",
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        protein_likelihood: str = "nbm",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        protein_background_prior_mean: Optional[np.ndarray] = None,
        protein_background_prior_scale: Optional[np.ndarray] = None,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.protein_likelihood = protein_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.n_input_proteins = n_input_proteins
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size
        self.warm_up = 0
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        # parameters for prior on back_rate (background protein mean)
        if protein_background_prior_mean is None:
            if n_batch > 0:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins, n_batch)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins, n_batch), -10, 1)
                )
            else:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins), -10, 1)
                )
        else:
            if protein_background_prior_mean.shape[1] == 1 and n_batch != 1:
                init_mean = protein_background_prior_mean.ravel()
                init_scale = protein_background_prior_scale.ravel()
            else:
                init_mean = protein_background_prior_mean
                init_scale = protein_background_prior_scale
            self.background_pro_alpha = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)))
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:  # gene-cell
            pass

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_batch)
            )
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_labels)
            )
        else:  # protein-cell
            pass

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # n_input = n_input_genes + self.n_input_proteins
        n_input_genes_encoder = n_input_genes + n_continuous_cov * encode_covariates
        n_input_proteins_encoder = n_input_proteins + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        # cat_list = [n_batch, n_batch]
        # print(cat_list)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_genes = EncoderGene(
            n_input_genes_encoder,
            self.n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.encoder_proteins = EncoderProtein(
            n_input_proteins_encoder,
            n_latent,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.decoder_genes = DecoderSCVI(
            n_latent + n_continuous_cov,
            n_input_genes,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.decoder_proteins = DecoderProtein(
            n_latent + n_continuous_cov,
            n_input_proteins,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.senet_genes = SENet(
            n_latent,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.senet_proteins = SENet(
            n_latent,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        cont_covs=None,
        cat_covs=None,
    ):
        """
        Internal helper function to compute necessary inference quantities.

        We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
        The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
        `scale` refers to the quantity upon which differential expression is performed. For genes,
        this can be viewed as the mean of the underlying gamma distribution.

        We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
        `fore_rate` refers to foreground mean, while `back_rate` refers to background mean. ``scale`` refers to
        foreground mean adjusted for background probability and scaled to reside in simplex.
        ``back_mean`` and ``back_var`` are the posterior parameters for ``back_rate``.  ``fore_scale`` is the scaling
        factor that enforces `fore_rate` > `back_rate`.

        ``px_["r"]`` and ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        label
            tensor of cell-types labels with shape (batch_size, n_labels)
        n_samples
            Number of samples to sample from approximate posterior
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        batch_index = batch_index.to(device)
        x_ = x
        y_ = y
        if cat_covs is not None:
            cat_covs = cat_covs.to(device)
        if self.use_observed_lib_size:
            library_gene = x.sum(1).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
            y_ = torch.log(1 + y_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input_genes = torch.cat((x_, cont_covs), dim=-1)
            encoder_input_proteins = torch.cat((y_, cont_covs), dim=-1)
        else:
            encoder_input_genes = x_
            encoder_input_proteins = y_
        # cat_covs = batch_index
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        enc_gene_outputs = self.encoder_genes(
            encoder_input_genes, batch_index, *categorical_input
        )
        enc_pro_outputs = self.encoder_proteins(
            encoder_input_proteins, batch_index, *categorical_input
        )

        if self.use_observed_lib_size:
            enc_gene_outputs['library_gene'] = library_gene

        if n_samples > 1:
            # genes latent space sampling
            qz_m, qz_v, z, untran_z, ql_m, ql_v, _, untran_l = enc_gene_outputs.values()
            enc_gene_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_gene_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_gene_outputs['untran_z'] = Normal(enc_gene_outputs['qz_m'], enc_gene_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_gene_outputs['z'] = self.encoder_genes.z_transformation(enc_gene_outputs['untran_z'])
            enc_gene_outputs['ql_m'] = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            enc_gene_outputs['ql_v'] = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            enc_gene_outputs['untran_l'] = Normal(enc_gene_outputs['ql_m'], enc_gene_outputs['ql_v'].sqrt()).sample()
            if self.use_observed_lib_size:
                enc_gene_outputs['library_gene'] = library_gene.unsqueeze(0).expand(
                    (n_samples, library_gene.size(0), library_gene.size(1))
                )
            else:
                enc_gene_outputs['library_gene'] = self.encoder_genes.l_transformation(untran_l)

            # proteins latent space sampling
            qz_m, qz_v, z, untran_z = enc_pro_outputs.values()
            enc_pro_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_pro_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_pro_outputs['untran_z'] = Normal(enc_pro_outputs['qz_m'], enc_pro_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_pro_outputs['z'] = self.encoder_proteins.z_transformation(enc_pro_outputs['untran_z'])

        # Background regularization
        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)
        if self.n_batch > 0:
            py_back_mean_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_var_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_mean_prior = self.background_pro_alpha
            py_back_var_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(py_back_mean_prior, py_back_var_prior)

        return enc_gene_outputs, enc_pro_outputs

    def self_expressiveness(
        self,
        z_genes: torch.Tensor,
        z_proteins: torch.Tensor,
        ref_z_genes: torch.Tensor,
        ref_z_proteins: torch.Tensor,
    ):
        """
        Use self-expressive network to reconstruct the latent space
        Parameters
        ----------
        z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        z_proteins
            tensor of values with shape ``(batch_size, n_latent_proteins)``
        ref_z_genes
            tensor of values with shape ``(training_size, n_latent_genes)``
        ref_z_proteins
            tensor of values with shape ``(training_size, n_latent_proteins)``

        Returns
        -------
        rec_z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        rec_z_proteins
            tensor of values with shape ``(batch_size, n_latent_proteins)``
        se_gene_outputs
            genes dict with "rec_queries" and "c_list"
        se_pro_outputs
            genes dict with "rec_queries" and "c_list"
        """

        se_gene_outputs = self.senet_genes(z_genes, ref_z_genes)
        se_pro_outputs = self.senet_proteins(z_proteins, ref_z_proteins)
        return se_gene_outputs, se_pro_outputs

    def generative(
        self,
        z_genes: torch.Tensor,
        z_proteins: torch.Tensor,
        library_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        transform_batch: Optional[int] = None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_index = batch_index.to(device)
        if cat_covs is not None:
            cat_covs = cat_covs.to(device)
        if cont_covs is not None:
            decoder_input_genes = torch.cat((z_genes, cont_covs), dim=-1)
            decoder_input_proteins = torch.cat((z_proteins, cont_covs), dim=-1)
        else:
            decoder_input_genes = z_genes
            decoder_input_proteins = z_proteins
        # cat_covs = batch_index
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
            if torch.cuda.is_available():
                batch_index = batch_index.to('cuda')

        px_ = self.decoder_genes(
            decoder_input_genes, library_gene, batch_index, *categorical_input
        )
        py_ = self.decoder_proteins(
            decoder_input_proteins, batch_index, *categorical_input
        )

        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_["r"] = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_["r"] = torch.exp(py_r)

        return px_, py_

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px_dict: Dict[str, torch.Tensor],
        py_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        px_ = px_dict
        py_ = py_dict
        # Reconstruction Loss
        if self.gene_likelihood == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )

        if self.protein_likelihood == "nbm":
            py_conditional = NegativeBinomialMixture(
                mu1=py_["back_rate"],
                mu2=py_["fore_rate"],
                theta1=py_["r"],
                mixture_logits=py_["mixing"],
            )
        elif self.protein_likelihood == "nb":
            py_conditional = NegativeBinomial(mu=py_['rate'], theta=py_["r"])
        else:
            py_conditional = Poisson(py_['rate'])

        reconst_loss_protein_full = -py_conditional.log_prob(y)
        reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_gene, reconst_loss_protein

    def get_self_expressiveness_loss(
        self,
        inference_outputs,
        senet_outputs,
        lmbd=0.9,
    ):
        """Compute self-expressiveness loss."""
        enc_gene_outputs, enc_pro_outputs = inference_outputs
        z_gene = enc_gene_outputs['z']
        z_protein = enc_pro_outputs['z']
        se_gene_outputs, se_pro_outputs = senet_outputs
        rec_z_gene, coeff_gene = se_gene_outputs['rec_queries'], se_gene_outputs['coeff']
        rec_z_protein, coeff_protein = se_pro_outputs['rec_queries'], se_pro_outputs['coeff']
        se_loss_gene = torch.sum(torch.pow(z_gene - rec_z_gene, 2), dim=-1)
        se_loss_protein = torch.sum(torch.pow(z_protein - rec_z_protein, 2), dim=-1)

        c_reg_gene = regularizer(coeff_gene, lmbd).sum(dim=-1)
        c_reg_protein = regularizer(coeff_protein, lmbd).sum(dim=-1)
        # c_reg_genes = regularizer_l12(coeff_gene, lmbd) * torch.ones(coeff_gene.shape[0])
        # c_reg_proteins = regularizer_l12(coeff_protein, lmbd) * torch.ones(coeff_protein.shape[0])
        c_contrast = torch.sum(torch.pow(coeff_gene - coeff_protein, 2), dim=-1)

        return se_loss_gene, se_loss_protein, c_reg_gene, c_reg_protein, c_contrast

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        inference_outputs,
        senet_outputs,
        generative_outputs,
        batch_index: Optional[torch.Tensor] = None,
        pro_recons_weight=1.0,  # double check these defaults
        kl_weight=1.0,
        se_weight=1.0,
    ):
        """
        Returns the reconstruction loss and the Kullback-Leibler divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        inference_outputs
            inference network outputs
        senet_outputs
            self-expressive network outputs
        generative_outputs
            generative network outputs
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        pro_recons_weight
            proteins reconstruction weight
        kl_weight
            Kullback-Leibler divergences loss weight
        se_weight
            self-expressive loss weight
        Returns
        -------
        type
            the reconstruction loss, the KL divergences and self-expressive loss
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        batch_index = batch_index.to(device)
        enc_gene_outputs, enc_pro_outputs = inference_outputs
        qz_m_gene = enc_gene_outputs['qz_m']
        qz_v_gene = enc_gene_outputs['qz_v']
        ql_m_gene = enc_gene_outputs['ql_m']
        ql_v_gene = enc_gene_outputs['ql_v']
        qz_m_protein = enc_pro_outputs['qz_m']
        qz_v_protein = enc_pro_outputs['qz_v']
        px_, py_ = generative_outputs

        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x, y, px_, py_,
        )
        se_loss_gene, se_loss_protein, c_reg_genes, c_reg_proteins, c_contrast = self.get_self_expressiveness_loss(
            inference_outputs, senet_outputs, lmbd=0.95
        )

        # KL Divergence
        kl_div_z_gene = kl(Normal(qz_m_gene, torch.sqrt(qz_v_gene)), Normal(0, 1)).sum(dim=1)
        kl_div_z_protein = kl(Normal(qz_m_protein, torch.sqrt(qz_v_protein)), Normal(0, 1)).sum(dim=1)
        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                one_hot(batch_index, n_batch), self.library_log_means
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index, n_batch), self.library_log_vars
            )
            kl_div_l_gene = kl(
                Normal(ql_m_gene, torch.sqrt(ql_v_gene)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_div_l_gene = 0.0 * torch.ones(x.shape[0])
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            kl_div_l_gene = kl_div_l_gene.to(device)

        if self.protein_likelihood == "nbm":
            kl_div_back_pro_full = kl(
                Normal(py_["back_log_rate_mean"], py_["back_log_rate_var"].sqrt()), self.back_mean_prior
            )
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=-1)
        else:
            kl_div_back_pro = 0.0 * torch.ones(x.shape[0])

        pro_mask = y.sum(dim=1) != 0
        scale = len(pro_mask) / pro_mask.sum()
        reconst_loss_protein = torch.where(pro_mask, reconst_loss_protein*scale, torch.zeros_like(reconst_loss_protein))
        se_loss_protein = torch.where(pro_mask, se_loss_protein*scale, torch.zeros_like(se_loss_protein))
        c_reg_proteins = torch.where(pro_mask, c_reg_proteins*scale, torch.zeros_like(c_reg_proteins))
        c_contrast = torch.where(pro_mask, c_contrast*scale, torch.zeros_like(c_contrast))

        losses = torch.mean(
            reconst_loss_gene
            + pro_recons_weight * reconst_loss_protein
            + kl_weight * kl_div_z_gene
            + kl_weight * kl_div_z_protein
            + kl_div_l_gene
            + kl_weight * kl_div_back_pro
            + se_weight * (se_loss_gene + c_reg_genes)
            + se_weight * (se_loss_protein + c_reg_proteins)
            + se_weight * c_contrast * 0.1
        )

        reconst_losses = dict(
            reconst_loss_gene=reconst_loss_gene,
            reconst_loss_protein=reconst_loss_protein,
        )
        kl_local = dict(
            kl_div_z_gene=kl_div_z_gene,
            kl_div_l_gene=kl_div_l_gene,
            kl_div_z_protein=kl_div_z_protein,
            kl_div_back_pro=kl_div_back_pro,
        )
        se_losses = dict(
            se_loss_gene=se_loss_gene,
            se_loss_protein=se_loss_protein,
            c_reg_genes=c_reg_genes,
            c_reg_proteins=c_reg_proteins,
            c_contrast=c_contrast
        )

        return losses, reconst_losses, kl_local, se_losses

    def forward(
        self,
        input_genes: torch.Tensor,
        input_proteins: torch.Tensor,
        refer_adata,
        batch_index: Optional[torch.Tensor] = None,
        cat_covs: Optional[torch.Tensor] = None,
        compute_loss=True,
        transform_batch: Optional[int] = None,
    ):
        """
        Forward pass through the network.

        Parameters
        ----------
        self
        input_genes
            tensor with shape ``(batch, n_input_gens)``
        input_proteins
            tensor with shape ``(batch, n_input_proteins)``
        refer_adata
            adata with shape ``(training_size, )``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        cat_covs
            categorical variable of each cell
        compute_loss
            whether compute losses
        transform_batch
            batch to condition on
        """
        ref_z_genes_list = []
        ref_z_proteins_list = []
        for data in refer_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                refer_cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                refer_cat_covs = None
            # refer_index = (refer_index, refer_index)
            enc_ref_gene_outputs, enc_ref_protein_outputs = self.inference(refer_genes, refer_proteins, batch_index=refer_index, cat_covs=refer_cat_covs)
            ref_z_genes_list.append(enc_ref_gene_outputs['z'])
            ref_z_proteins_list.append(enc_ref_protein_outputs['z'])
        ref_z_genes = torch.cat(ref_z_genes_list, dim=0)
        ref_z_proteins = torch.cat(ref_z_proteins_list, dim=0)

        inference_outputs = self.inference(input_genes, input_proteins, batch_index=batch_index, cat_covs=cat_covs)
        enc_gene_outputs, enc_pro_outputs = inference_outputs
        z_genes = enc_gene_outputs['z']
        library_gene = enc_gene_outputs['library_gene']
        z_proteins = enc_pro_outputs['z']
        senet_outputs = self.self_expressiveness(z_genes, z_proteins, ref_z_genes, ref_z_proteins)
        se_gene_outputs, se_pro_outputs = senet_outputs
        rec_z_genes = se_gene_outputs['rec_queries']
        rec_z_proteins = se_pro_outputs['rec_queries']
        generative_outputs = self.generative(rec_z_genes, rec_z_proteins, library_gene, batch_index=batch_index, cat_covs=cat_covs, transform_batch=transform_batch)
        if compute_loss:
            loss = self.loss(
                input_genes, input_proteins, inference_outputs, senet_outputs, generative_outputs, batch_index
            )
            return inference_outputs, senet_outputs, generative_outputs, loss
        else:
            return inference_outputs, senet_outputs, generative_outputs


class RNAPeakSVAE(nn.Module):
    """
    Single cell multi-omics self-expressive integration.

    Implements the SCMSI model for single cell multi-omics data.

    Parameters
    ----------
    n_input_genes
        Number of input genes
    n_input_peaks
        Number of input peaks
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    gene_dispersion
        One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_peaks: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden_gene: int = 256,
        n_hidden_peak: int = 256,
        n_senet_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_layers_senet: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.1,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_senet: float = 0.0,
        gene_dispersion: str = "gene",
        log_variational: bool = True,
        convert_binary: bool = False,
        gene_likelihood: str = "nb",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        use_cell_factors: bool = True,
        use_region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        cc_weight: float = 1e-4
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.convert_binary = convert_binary
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.n_input_peaks = n_input_peaks
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size
        self.use_cell_factors = use_cell_factors
        self.use_region_factors = use_region_factors
        self.cc_weight=cc_weight
        self.warm_up = 0

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:  # gene-cell
            pass
        self.region_factors = None
        if self.use_region_factors:
            self.region_factors = torch.nn.Parameter(torch.zeros(self.n_input_peaks))

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # n_input = n_input_genes + self.n_input_proteins
        n_input_genes_encoder = n_input_genes + n_continuous_cov * encode_covariates
        n_input_peaks_encoder = n_input_peaks + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        # cat_list = [n_batch, n_batch]
        # print(cat_list)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_genes = EncoderGene(
            n_input_genes_encoder,
            self.n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden_gene,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.encoder_peaks = EncoderPeak(
            n_input_peaks_encoder,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden_peak,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=False,
            use_layer_norm=True,
        )
        self.decoder_genes = DecoderSCVI(
            n_latent + n_continuous_cov,
            n_input_genes,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden_gene,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.decoder_peaks = DecoderPeak(
            n_latent + n_continuous_cov,
            n_input_peaks_encoder,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden_peak,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=False,
            use_layer_norm=True,
        )
        self.senet_genes = SENet(
            n_latent,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.senet_peaks = SENet(
            n_latent,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        cont_covs=None,
        cat_covs=None,
    ):
        """
        Internal helper function to compute necessary inference quantities.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_peaks)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        label
            tensor of cell-types labels with shape (batch_size, n_labels)
        n_samples
            Number of samples to sample from approximate posterior
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        batch_index = batch_index.to(device)
        x_ = x
        y_ = y
        if self.use_observed_lib_size:
            library_gene = x.sum(1).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
            y_ = torch.log(1 + y_)
        if self.convert_binary:
            y_ = (y_ > 0).float()

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input_genes = torch.cat((x_, cont_covs), dim=-1)
            encoder_input_peaks = torch.cat((y_, cont_covs), dim=-1)
        else:
            encoder_input_genes = x_
            encoder_input_peaks = y_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        enc_gene_outputs = self.encoder_genes(
            encoder_input_genes, batch_index, *categorical_input
        )
        enc_peak_outputs = self.encoder_peaks(
            encoder_input_peaks, batch_index, *categorical_input
        )

        if self.use_observed_lib_size:
            enc_gene_outputs['library_gene'] = library_gene
        if not self.use_cell_factors:
            enc_peak_outputs['lib'] = 1

        if n_samples > 1:
            # genes latent space sampling
            qz_m, qz_v, z, untran_z, ql_m, ql_v, _, untran_l = enc_gene_outputs.values()
            enc_gene_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_gene_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_gene_outputs['untran_z'] = Normal(enc_gene_outputs['qz_m'], enc_gene_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_gene_outputs['z'] = self.encoder_genes.z_transformation(enc_gene_outputs['untran_z'])
            enc_gene_outputs['ql_m'] = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            enc_gene_outputs['ql_v'] = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            enc_gene_outputs['untran_l'] = Normal(enc_gene_outputs['ql_m'], enc_gene_outputs['ql_v'].sqrt()).sample()
            if self.use_observed_lib_size:
                enc_gene_outputs['library_gene'] = library_gene.unsqueeze(0).expand(
                    (n_samples, library_gene.size(0), library_gene.size(1))
                )
            else:
                enc_gene_outputs['library_gene'] = self.encoder_genes.l_transformation(untran_l)

            # proteins latent space sampling
            qz_m, qz_v, _, _, lib = enc_peak_outputs.values()
            qz_m, qz_v, _, _, lib = enc_peak_outputs.values()
            if not self.use_cell_factors:
                lib = torch.ones_like(lib)
            enc_peak_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_peak_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_peak_outputs['untran_z'] = Normal(enc_peak_outputs['qz_m'], enc_peak_outputs[
                'qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_peak_outputs['z'] = self.encoder_peaks.z_transformation(enc_peak_outputs['untran_z'])
            enc_peak_outputs['lib'] = lib.unsqueeze(0).expand((n_samples, lib.size(0), lib.size(1)))

        # Background regularization
        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        return enc_gene_outputs, enc_peak_outputs

    def self_expressiveness(
        self,
        z_genes: torch.Tensor,
        z_peaks: torch.Tensor,
        ref_z_genes: torch.Tensor,
        ref_z_peaks: torch.Tensor,
    ):
        """
        Use self-expressive network to reconstruct the latent space
        Parameters
        ----------
        z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        z_peaks
            tensor of values with shape ``(batch_size, n_latent_peaks)``
        ref_z_genes
            tensor of values with shape ``(training_size, n_latent_genes)``
        ref_z_peaks
            tensor of values with shape ``(training_size, n_latent_peaks)``

        Returns
        -------
        rec_z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        rec_z_proteins
            tensor of values with shape ``(batch_size, n_latent_peaks)``
        se_gene_outputs
            genes dict with "rec_queries" and "c_list"
        se_pro_outputs
            genes dict with "rec_queries" and "c_list"
        """

        se_gene_outputs = self.senet_genes(z_genes, ref_z_genes)
        se_peak_outputs = self.senet_peaks(z_peaks, ref_z_peaks)
        return se_gene_outputs, se_peak_outputs

    def generative(
        self,
        z_genes: torch.Tensor,
        z_peaks: torch.Tensor,
        library_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        transform_batch: Optional[int] = None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_index = batch_index.to(device)
        if cont_covs is not None:
            decoder_input_genes = torch.cat((z_genes, cont_covs), dim=-1)
            decoder_input_peaks = torch.cat((z_peaks, cont_covs), dim=-1)
        else:
            decoder_input_genes = z_genes
            decoder_input_peaks = z_peaks
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
            if torch.cuda.is_available():
                batch_index = batch_index.to('cuda')

        px_ = self.decoder_genes(
            decoder_input_genes, library_gene, batch_index, *categorical_input
        )
        py_ = self.decoder_peaks(
            decoder_input_peaks, batch_index, *categorical_input
        )

        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_["r"] = torch.exp(px_r)

        py_["rf"] = torch.sigmoid(self.region_factors) if self.use_region_factors is not None else 1

        return px_, py_

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px_dict: Dict[str, torch.Tensor],
        py_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        px_ = px_dict
        py_ = py_dict
        # Reconstruction Loss
        if self.gene_likelihood == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )
        reconst_loss_protein = torch.nn.BCELoss(reduction="none")(py_['prob'] * py_['cf'] * py_['rf'], (y > 0).float()).sum(dim=-1)

        return reconst_loss_gene, reconst_loss_protein

    def get_self_expressiveness_loss(
        self,
        inference_outputs,
        senet_outputs,
        lmbd=0.9,
    ):
        """Compute self-expressiveness loss."""
        enc_gene_outputs, enc_peak_outputs = inference_outputs
        z_gene = enc_gene_outputs['z']
        z_peak = enc_peak_outputs['z']
        se_gene_outputs, se_peak_outputs = senet_outputs
        rec_z_gene, coeff_gene = se_gene_outputs['rec_queries'], se_gene_outputs['coeff']
        rec_z_peak, coeff_peak = se_peak_outputs['rec_queries'], se_peak_outputs['coeff']
        se_loss_gene = torch.sum(torch.pow(z_gene - rec_z_gene, 2), dim=-1)
        se_loss_peak = torch.sum(torch.pow(z_peak - rec_z_peak, 2), dim=-1)

        c_reg_gene = regularizer(coeff_gene, lmbd).sum(dim=-1)
        c_reg_peak = regularizer(coeff_peak, lmbd).sum(dim=-1)
        # c_reg_genes = regularizer_l12(coeff_gene, lmbd) * torch.ones(coeff_gene.shape[0])
        # c_reg_proteins = regularizer_l12(coeff_protein, lmbd) * torch.ones(coeff_protein.shape[0])
        c_contrast = torch.sum(torch.pow(coeff_gene - coeff_peak, 2), dim=-1)

        return se_loss_gene, se_loss_peak, c_reg_gene, c_reg_peak, c_contrast

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        inference_outputs,
        senet_outputs,
        generative_outputs,
        batch_index: Optional[torch.Tensor] = None,
        peak_recons_weight=1.0,  # double check these defaults
        kl_weight=1.0,
        se_weight=1.0,
    ):
        """
        Returns the reconstruction loss and the Kullback-Leibler divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_peaks)``
        inference_outputs
            inference network outputs
        senet_outputs
            self-expressive network outputs
        generative_outputs
            generative network outputs
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        peak_recons_weight
            proteins reconstruction weight
        kl_weight
            Kullback-Leibler divergences loss weight
        se_weight
            self-expressive loss weight
        Returns
        -------
        type
            the reconstruction loss, the KL divergences and self-expressive loss
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        batch_index = batch_index.to(device)
        enc_gene_outputs, enc_peak_outputs = inference_outputs
        qz_m_gene = enc_gene_outputs['qz_m']
        qz_v_gene = enc_gene_outputs['qz_v']
        ql_m_gene = enc_gene_outputs['ql_m']
        ql_v_gene = enc_gene_outputs['ql_v']
        qz_m_peak = enc_peak_outputs['qz_m']
        qz_v_peak = enc_peak_outputs['qz_v']
        px_, py_ = generative_outputs
        py_['cf'] = enc_peak_outputs['lib']
        reconst_loss_gene, reconst_loss_peak = self.get_reconstruction_loss(
            x, y, px_, py_,
        )
        se_loss_gene, se_loss_peak, c_reg_genes, c_reg_peaks, c_contrast = self.get_self_expressiveness_loss(
            inference_outputs, senet_outputs, lmbd=0.95
        )

        # KL Divergence
        kl_div_z_gene = kl(Normal(qz_m_gene, torch.sqrt(qz_v_gene)), Normal(0, 1)).sum(dim=1)
        kl_div_z_peak = kl(Normal(qz_m_peak, torch.sqrt(qz_v_peak)), Normal(0, 1)).sum(dim=1)
        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                one_hot(batch_index, n_batch), self.library_log_means
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index, n_batch), self.library_log_vars
            )
            kl_div_l_gene = kl(
                Normal(ql_m_gene, torch.sqrt(ql_v_gene)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_div_l_gene = 0.0 * torch.ones(x.shape[0])
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            kl_div_l_gene = kl_div_l_gene.to(device)

        losses = torch.mean(
            reconst_loss_gene
            + peak_recons_weight * reconst_loss_peak
            + kl_weight * kl_div_z_gene
            + kl_weight * kl_div_z_peak
            + kl_div_l_gene
            + se_weight * (se_loss_gene + c_reg_genes)
            + se_weight * (se_loss_peak + c_reg_peaks)
            + se_weight * c_contrast * self.cc_weight
        )

        reconst_losses = dict(
            reconst_loss_gene=reconst_loss_gene,
            reconst_loss_peak=reconst_loss_peak,
        )
        kl_local = dict(
            kl_div_z_gene=kl_div_z_gene,
            kl_div_l_gene=kl_div_l_gene,
            kl_div_z_peak=kl_div_z_peak,
        )
        se_losses = dict(
            se_loss_gene=se_loss_gene,
            se_loss_peak=se_loss_peak,
            c_reg_genes=c_reg_genes,
            c_reg_peaks=c_reg_peaks,
            c_contrast=c_contrast
        )

        return losses, reconst_losses, kl_local, se_losses

    def forward(
        self,
        input_genes: torch.Tensor,
        input_peaks: torch.Tensor,
        refer_adata,
        batch_index: Optional[torch.Tensor] = None,
        compute_loss=True,
        transform_batch: Optional[int] = None,
    ):
        """
        Forward pass through the network.

        Parameters
        ----------
        self
        input_genes
            tensor with shape ``(batch, n_input_gens)``
        input_peaks
            tensor with shape ``(batch, n_input_proteins)``
        refer_adata
            adata with shape ``(training_size, )``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        compute_loss
            whether compute losses
        transform_batch
            batch to condition on
        """
        ref_z_genes_list = []
        ref_z_peaks_list = []
        for data in refer_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_peaks = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_gene_outputs, enc_ref_peak_outputs = self.inference(refer_genes, refer_peaks, refer_index)
            ref_z_genes_list.append(enc_ref_gene_outputs['z'])
            ref_z_peaks_list.append(enc_ref_peak_outputs['z'])
        ref_z_genes = torch.cat(ref_z_genes_list, dim=0)
        ref_z_peaks = torch.cat(ref_z_peaks_list, dim=0)

        inference_outputs = self.inference(input_genes, input_peaks, batch_index)
        enc_gene_outputs, enc_peak_outputs = inference_outputs
        z_genes = enc_gene_outputs['z']
        library_gene = enc_gene_outputs['library_gene']
        z_peaks = enc_peak_outputs['z']
        senet_outputs = self.self_expressiveness(z_genes, z_peaks, ref_z_genes, ref_z_peaks)
        se_gene_outputs, se_pro_outputs = senet_outputs
        rec_z_genes = se_gene_outputs['rec_queries']
        rec_z_peaks = se_pro_outputs['rec_queries']
        generative_outputs = self.generative(rec_z_genes, rec_z_peaks, library_gene, batch_index, transform_batch)
        if compute_loss:
            loss = self.loss(
                input_genes, input_peaks, inference_outputs, senet_outputs, generative_outputs, batch_index
            )
            return inference_outputs, senet_outputs, generative_outputs, loss
        else:
            return inference_outputs, senet_outputs, generative_outputs


# RNA SEVAE model
class RNASVAE(nn.Module):
    """
    scRNA-seq self-expressive clustering.
    Implements the SCMSI model for single cell RNA data.

    Parameters
    ----------
    n_input_genes
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    gene_dispersion
        One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'zip'`` - Zero-inflated poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_senet_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_layers_senet: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.0,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_senet: float = 0.1,
        gene_dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size
        self.warm_up = 0.0
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:  # gene-cell
            pass

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # n_input = n_input_genes + self.n_input_proteins
        n_input_genes_encoder = n_input_genes + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_genes = EncoderGene(
            n_input_genes_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        self.senet_genes = SENet(
            n_latent,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=True,
            use_layer_norm=False,
        )

        self.decoder_genes = DecoderSCVI(
            n_latent + n_continuous_cov,
            n_input_genes,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def inference(
        self,
        x: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """
        Internal helper function to compute necessary inference quantities.

        We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
        The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
        `scale` refers to the quantity upon which differential expression is performed. For genes,
        this can be viewed as the mean of the underlying gamma distribution.

        ``px_["r"]`` is the inverse dispersion parameters for genes.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        n_samples
            Number of samples to sample from approximate posterior
        """
        x_ = x
        if self.use_observed_lib_size:
            library_gene = x.sum(1).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates is True:
            enc_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            enc_input = x_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=-1)
        else:
            categorical_input = tuple()
        enc_outputs = self.encoder_genes(
            enc_input, batch_index, *categorical_input
        )

        if self.use_observed_lib_size:
            enc_outputs['library_gene'] = library_gene

        if n_samples > 1:
            # genes latent space sampling
            qz_m, qz_v, _, _, ql_m, ql_v, _, _ = enc_outputs.values()
            enc_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_outputs['untran_z'] = Normal(enc_outputs['qz_m'], enc_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_outputs['z'] = self.encoder_genes.z_transformation(enc_outputs['untran_z'])
            enc_outputs['ql_m'] = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            enc_outputs['ql_v'] = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            enc_outputs['untran_l'] = Normal(enc_outputs['ql_m'], enc_outputs['ql_v'].sqrt()).sample()
            if self.use_observed_lib_size:
                enc_outputs['library_gene'] = library_gene.unsqueeze(0).expand(
                    (n_samples, library_gene.size(0), library_gene.size(1))
                )
            else:
                enc_outputs['library_gene'] = self.encoder_genes.l_transformation(enc_outputs['untran_l'])

        return enc_outputs

    def self_expressiveness(
            self,
            z: torch.Tensor,
            ref_z: torch.Tensor,
    ):
        """
        Use self-expressive network to reconstruct the latent space
        Parameters
        ----------
        z
            tensor of values with shape ``(batch_size, n_latent_genes)``
        ref_z
            tensor of values with shape ``(training_size, n_latent_genes)``
        Returns
        -------
        rec_z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        se_gene_outputs
            genes dict with "rec_queries" and "coeff"
        """
        se_outputs = self.senet_genes(z, ref_z)
        return se_outputs

    def generative(
        self,
        z: torch.Tensor,
        library_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        transform_batch: Optional[int] = None,
    ):
        # decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cont_covs is not None:
            decoder_input = torch.cat((z, cont_covs), dim=-1)
        else:
            decoder_input = z
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        px_ = self.decoder_genes(
            decoder_input, library_gene, batch_index, *categorical_input
        )

        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_["r"] = torch.exp(px_r)

        return px_

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        px_: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        # px_ = px_dict
        # Reconstruction Loss

        if self.gene_likelihood == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -Poisson(px_["rate"])
                .log_prob(x)
                .sum(dim=-1))

        return reconst_loss_gene

    def get_self_expressiveness_loss(
        self,
        inference_outputs,
        senet_outputs,
        lmbd=0.9,
    ):
        """Compute self-expressiveness loss."""
        enc_outputs = inference_outputs
        z = enc_outputs['z']
        rec_z, coeff = senet_outputs['rec_queries'], senet_outputs['coeff']
        se_loss = torch.sum(torch.pow(z - rec_z, 2), dim=-1)
        coeff_reg = regularizer(coeff, lmbd).sum(dim=-1)
        # c_contrast = torch.sum(torch.pow(c_genes - c_proteins, 2), dim=-1)

        return se_loss, coeff_reg

    def loss(
        self,
        x: torch.Tensor,
        inference_outputs,
        senet_outputs,
        generative_outputs,
        batch_index: Optional[torch.Tensor] = None,
        kl_weight=1.0,
        se_weight=1.0,
    ):
        """
        Returns the reconstruction loss and the Kullback-Leibler divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        inference_outputs
            inference network outputs
        senet_outputs
            self-expressive network outputs
        generative_outputs
            generative network outputs
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        kl_weight
            Kullback-Leibler divergences loss weight
        se_weight
            self-expressive loss weight
        Returns
        -------
        type
            the reconstruction loss, the KL divergences and self-expressive loss
        """
        self.warm_up += 1
        kl_weight = 1.0 if self.warm_up > 100 else 0.0
        enc_gene_outputs = inference_outputs
        qz_m = enc_gene_outputs['qz_m']
        qz_v = enc_gene_outputs['qz_v']
        ql_m = enc_gene_outputs['ql_m']
        ql_v = enc_gene_outputs['ql_v']
        px_ = generative_outputs

        reconst_loss = self.get_reconstruction_loss(x, px_)

        # KL Divergence
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                one_hot(batch_index, n_batch), self.library_log_means
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index, n_batch), self.library_log_vars
            )
            kl_div_l = kl(
                Normal(ql_m, torch.sqrt(ql_v)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_div_l = 0.0

        se_loss, coeff_reg = self.get_self_expressiveness_loss(
            inference_outputs, senet_outputs)

        loss = torch.mean(
            reconst_loss
            + kl_weight * kl_div_z
            + kl_div_l
            + se_weight * (se_loss + coeff_reg)
        )

        kl_local = dict(
            kl_div_z=kl_div_z,
            kl_div_l=kl_div_l,
        )
        se_losses = dict(
            se_loss=se_loss,
            coeff_reg=coeff_reg,
        )

        return loss, reconst_loss, kl_local, se_losses

    def forward(
        self,
        input_genes: torch.Tensor,
        refer_adata,
        batch_index: Optional[torch.Tensor] = None,
        compute_loss=True,
        transform_batch: Optional[int] = None,
    ):
        """
        Forward pass through the network.

        Parameters
        ----------
        self
        input_genes
            tensor with shape ``(batch, n_input_gens)``
        refer_adata
            adata with shape ``(training_size, )``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        compute_loss
            whether compute losses
        transform_batch
            batch to condition on
        """
        ref_z_list = []
        for data in refer_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.inference(refer_genes, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_genes = torch.cat(ref_z_list, dim=0)

        inference_outputs = self.inference(input_genes, batch_index)
        enc_gene_outputs = inference_outputs
        z_genes = enc_gene_outputs['z']
        library_gene = enc_gene_outputs['library_gene']
        senet_outputs = self.self_expressiveness(z_genes, ref_z_genes)
        rec_z = senet_outputs['rec_queries']
        generative_outputs = self.generative(rec_z, library_gene, batch_index, transform_batch)
        if compute_loss:
            loss = self.loss(
                input_genes, inference_outputs, senet_outputs, generative_outputs, batch_index
            )
            return inference_outputs, senet_outputs, generative_outputs, loss
        else:
            return inference_outputs, senet_outputs, generative_outputs


# Protein SVAE model
class ProteinSVAE(nn.Module):
    """
    single cell protein self-expressive clustering.

    Implements the SCMSI model for single cell multi-omics data.

    Parameters
    ----------
    n_input_proteins
        Number of input proteins
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    protein_dispersion
        One of the following

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    protein_background_prior_mean
        Array of proteins by batches, the prior initialization for the protein background mean (log scale)
    protein_background_prior_scale
        Array of proteins by batches, the prior initialization for the protein background scale (log scale)

    """

    def __init__(
        self,
        n_input_proteins: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_senet_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_layers_senet: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.1,
        dropout_rate_encoder: float = 0.0,
        dropout_rate_senet: float = 0.0,
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        protein_likelihood: str = "nbm",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        protein_background_prior_mean: Optional[np.ndarray] = None,
        protein_background_prior_scale: Optional[np.ndarray] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_proteins = n_input_proteins
        self.protein_dispersion = protein_dispersion
        self.protein_likelihood = protein_likelihood
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates

        # parameters for prior on back_rate (background protein mean)
        if protein_background_prior_mean is None:
            if n_batch > 0:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins, n_batch)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins, n_batch), -10, 1)
                )
            else:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins), -10, 1)
                )
        else:
            # mean_shape = protein_background_prior_mean.shape
            # protein_background_prior_mean += np.random.randn(mean_shape[0], mean_shape[1])*0.01
            if protein_background_prior_mean.shape[1] == 1 and n_batch != 1:
                init_mean = protein_background_prior_mean.ravel()
                init_scale = protein_background_prior_scale.ravel()
            else:
                init_mean = protein_background_prior_mean
                init_scale = protein_background_prior_scale
            self.background_pro_alpha = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)))
            )

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_batch)
            )
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_labels)
            )
        else:  # protein-cell
            pass

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # n_input = n_input_genes + self.n_input_proteins
        n_input_proteins_encoder = n_input_proteins + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_proteins = EncoderProtein(
            n_input_proteins_encoder,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.senet_proteins = SENet(
            n_latent,
            n_senet_hidden,  # n_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.decoder_proteins = DecoderProtein(
            n_latent + n_continuous_cov,
            n_input_proteins,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def inference(
        self,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        cont_covs=None,
        cat_covs=None,
    ):
        """
        Internal helper function to compute necessary inference quantities.

        We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
        `fore_rate` refers to foreground mean, while `back_rate` refers to background mean. ``scale`` refers to
        foreground mean adjusted for background probability and scaled to reside in simplex.
        ``back_mean`` and ``back_var`` are the posterior parameters for ``back_rate``.  ``fore_scale`` is the scaling
        factor that enforces `fore_rate` > `back_rate`.

        ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.

        Parameters
        ----------
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        label
            tensor of cell-types labels with shape (batch_size, n_labels)
        n_samples
            Number of samples to sample from approximate posterior
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        """
        y_ = y
        if self.log_variational:
            y_ = torch.log(1 + y_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input_proteins = torch.cat((y_, cont_covs), dim=-1)
        else:
            encoder_input_proteins = y_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        enc_pro_outputs = self.encoder_proteins(
            encoder_input_proteins, batch_index, *categorical_input
        )

        if n_samples > 1:
            # proteins latent space sampling
            qz_m, qz_v, z, untran_z = enc_pro_outputs.values()
            enc_pro_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_pro_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_pro_outputs['untran_z'] = Normal(enc_pro_outputs['qz_m'], enc_pro_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_pro_outputs['z'] = self.encoder_proteins.z_transformation(enc_pro_outputs['untran_z'])

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)
        if self.n_batch > 0:
            py_back_mean_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_var_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_mean_prior = self.background_pro_alpha
            py_back_var_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(py_back_mean_prior, py_back_var_prior)

        return enc_pro_outputs

    def self_expressiveness(
            self,
            z: torch.Tensor,
            ref_z: torch.Tensor,
    ):
        """
        Use self-expressive network to reconstruct the latent space
        Parameters
        ----------
        z
            tensor of values with shape ``(batch_size, n_latent_genes)``
        ref_z
            tensor of values with shape ``(training_size, n_latent_genes)``
        Returns
        -------
        rec_z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        se_gene_outputs
            genes dict with "rec_queries" and "coeff"
        """
        se_outputs = self.senet_proteins(z, ref_z)

        return se_outputs

    def generative(
        self,
        z: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        transform_batch: Optional[int] = None,
    ):
        # decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cont_covs is not None:
            decoder_input_proteins = torch.cat((z, cont_covs), dim=-1)
        else:
            decoder_input_proteins = z
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        py_ = self.decoder_proteins(
            decoder_input_proteins, batch_index, *categorical_input
        )

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_["r"] = torch.exp(py_r)

        return py_

    def get_reconstruction_loss(
        self,
        y: torch.Tensor,
        py_dict: Dict[str, torch.Tensor],
    ):
        """Compute reconstruction loss."""
        py_ = py_dict
        # Reconstruction Loss
        if self.protein_likelihood == "nbm":
            py_conditional = NegativeBinomialMixture(
                mu1=py_["back_rate"],
                mu2=py_["fore_rate"],
                theta1=py_["r"],
                mixture_logits=py_["mixing"],
            )
        elif self.protein_likelihood == "nb":
            py_conditional = NegativeBinomial(mu=py_['rate'], theta=py_["r"])
        else:
            py_conditional = Poisson(py_['rate'])

        reconst_loss_protein_full = -py_conditional.log_prob(y)
        reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_protein

    def get_self_expressiveness_loss(
        self,
        inference_outputs,
        senet_outputs,
        lmbd=0.9,
    ):
        """Compute self-expressiveness loss."""
        enc_outputs = inference_outputs
        z = enc_outputs['z']
        rec_z, coeff = senet_outputs['rec_queries'], senet_outputs['coeff']
        se_loss = torch.sum(torch.pow(z - rec_z, 2), dim=-1)
        coeff_reg = regularizer(coeff, lmbd).sum(dim=-1)
        coeff_reg = regularizer_l12(coeff, lmbd) * torch.ones(coeff.shape[0])
        # c_contrast = torch.sum(torch.pow(c_genes - c_proteins, 2), dim=-1)

        return se_loss, coeff_reg

    def loss(
        self,
        y: torch.Tensor,
        inference_outputs,
        senet_outputs,
        generative_outputs,
        batch_index: Optional[torch.Tensor] = None,
        pro_recons_weight=1.0,  # double check these defaults
        kl_weight=1.0,
        se_weight=1.0,
    ):
        """
        Returns the reconstruction loss and the Kullback-Leibler divergences.

        Parameters
        ----------
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        inference_outputs
            inference network outputs
        senet_outputs
            self-expressive network outputs
        generative_outputs
            generative network outputs
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        pro_recons_weight
            proteins reconstruction weight
        kl_weight
            Kullback-Leibler divergences loss weight
        se_weight
            self-expressive loss weight
        Returns
        -------
        type
            the reconstruction loss, the KL divergences and self-expressive loss
        """
        enc_pro_outputs = inference_outputs
        qz_m = enc_pro_outputs['qz_m']
        qz_v = enc_pro_outputs['qz_v']
        py_ = generative_outputs

        reconst_loss = self.get_reconstruction_loss(y, py_)
        se_loss, coeff_reg = self.get_self_expressiveness_loss(inference_outputs, senet_outputs, lmbd=0.95)

        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        if self.protein_likelihood == "nbm":
            kl_div_back_pro_full = kl(
                Normal(py_["back_log_rate_mean"], py_["back_log_rate_var"].sqrt()), self.back_mean_prior
            )
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=-1)
        else:
            kl_div_back_pro = 0.0
        losses = torch.mean(
            + pro_recons_weight * reconst_loss
            + kl_weight * kl_div_z
            + kl_weight * kl_div_back_pro
            + se_weight * (se_loss + coeff_reg)
        )

        kl_local = dict(
            kl_div_z=kl_div_z,
            kl_div_back=kl_div_back_pro,
        )
        se_losses = dict(
            se_loss=se_loss,
            coeff_reg=coeff_reg,
        )

        return losses, reconst_loss, kl_local, se_losses

    def forward(
        self,
        input_proteins: torch.Tensor,
        refer_adata,
        batch_index: Optional[torch.Tensor] = None,
        compute_loss=True,
        transform_batch: Optional[int] = None,
    ):
        """
        Forward pass through the network.

        Parameters
        ----------
        self
        input_proteins
            tensor with shape ``(batch, n_input_proteins)``
        refer_adata
            adata with shape ``(training_size, )``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        compute_loss
            whether compute losses
        transform_batch
            batch to condition on
        """
        ref_z_list = []
        for data in refer_adata:
            refer_proteins = torch.tensor(data.obsm["protein_expression"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.inference(refer_proteins, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_proteins = torch.cat(ref_z_list, dim=0)

        inference_outputs = self.inference(input_proteins, batch_index)
        enc_pro_outputs = inference_outputs
        z_proteins = enc_pro_outputs['z']
        senet_outputs = self.self_expressiveness(z_proteins, ref_z_proteins)
        rec_z = senet_outputs['rec_queries']
        generative_outputs = self.generative(rec_z, batch_index, transform_batch)
        if compute_loss:
            loss = self.loss(
                input_proteins, inference_outputs, senet_outputs, generative_outputs, batch_index
            )
            return inference_outputs, senet_outputs, generative_outputs, loss
        else:
            return inference_outputs, senet_outputs, generative_outputs


# ATAC SEVAE model
class PeakSVAE(nn.Module):
    """
    scATAC-seq self-expressive clustering.
    Implements the SCMSI model for single cell RNA data.

    Parameters
    ----------
    n_input_peaks
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    """

    def __init__(
        self,
        n_input_peaks: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 512,
        n_senet_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        n_layers_senet: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.0,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_senet: float = 0.0,
        # peak_likelihood: str = "zp",
        latent_distribution: str = "normal",
        log_variational: bool = False,
        convert_binary: bool = False,
        encode_covariates: bool = True,
        use_cell_factors: bool = True,
        use_region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
    ):
        super().__init__()
        # self.peak_likelihood = peak_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_peaks = n_input_peaks
        self.n_hidden = int(np.sqrt(self.n_input_regions)) if n_hidden is None else n_hidden
        self.n_latent = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
        self.latent_distribution = latent_distribution
        self.log_variational = log_variational
        self.convert_binary = convert_binary
        self.encode_covariates = encode_covariates
        self.use_cell_factors = use_cell_factors
        self.use_region_factors = use_region_factors

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        self.region_factors = None
        if self.use_region_factors:
            self.region_factors = torch.nn.Parameter(torch.zeros(self.n_input_peaks))

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_peaks_encoder = n_input_peaks + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_peaks = EncoderPeak(
            n_input_peaks_encoder,
            n_output=self.n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers_encoder,
            n_hidden=self.n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        self.senet_peaks = SENet(
            self.n_latent,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=True,
            use_layer_norm=False,
        )

        self.decoder_peaks = DecoderPeak(
            n_latent + n_continuous_cov,
            n_input_peaks,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def inference(
        self,
        x: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """
        Internal helper function to compute necessary inference quantities.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        n_samples
            Number of samples to sample from approximate posterior
        """
        if self.log_variational:
            x = torch.log(1 + x)
        if self.convert_binary:
            x = (x > 0).float()
        if cont_covs is not None and self.encode_covariates is True:
            enc_input = torch.cat((x, cont_covs), dim=-1)
        else:
            enc_input = x
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=-1)
        else:
            categorical_input = tuple()
        enc_outputs = self.encoder_peaks(
            enc_input, batch_index, *categorical_input
        )
        if not self.use_cell_factors:
            enc_outputs['lib'] = 1

        if n_samples > 1:
            qz_m, qz_v, _, _, lib = enc_outputs.values()
            if not self.use_cell_factors:
                lib = torch.ones_like(lib)
            enc_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_outputs['untran_z'] = Normal(enc_outputs['qz_m'], enc_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_outputs['z'] = self.encoder_peaks.z_transformation(enc_outputs['untran_z'])
            enc_outputs['lib'] = lib.unsqueeze(0).expand((n_samples, lib.size(0), lib.size(1)))

        return enc_outputs

    def self_expressiveness(
        self,
        z: torch.Tensor,
        ref_z: torch.Tensor,
    ):
        """
        Use self-expressive network to reconstruct the latent space
        Parameters
        ----------
        z
            tensor of values with shape ``(batch_size, n_latent_genes)``
        ref_z
            tensor of values with shape ``(training_size, n_latent_genes)``
        Returns
        -------
        rec_z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        se_gene_outputs
            genes dict with "rec_queries" and "coeff"
        """
        se_outputs = self.senet_peaks(z, ref_z)
        return se_outputs

    def generative(
        self,
        z: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        transform_batch: Optional[int] = None,
    ):
        # decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cont_covs is not None:
            decoder_input = torch.cat((z, cont_covs), dim=-1)
        else:
            decoder_input = z
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        px_ = self.decoder_peaks(
            decoder_input, batch_index, *categorical_input
        )

        px_["rf"] = torch.sigmoid(self.region_factors) if self.use_region_factors is not None else 1

        return px_

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        px_: Dict[str, torch.Tensor],
    ):
        """Compute reconstruction loss."""
        # px_ = px_dict
        # Reconstruction Loss
        rl = torch.nn.BCELoss(reduction="none")(px_['prob'] * px_['cf'] * px_['rf'], (x > 0).float()).sum(dim=-1)
        # if self.gene_likelihood == "zinb":
        #     reconst_loss_gene = (
        #         -ZeroInflatedNegativeBinomial(
        #             mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
        #         )
        #         .log_prob(x)
        #         .sum(dim=-1)
        #     )
        # elif self.gene_likelihood == "nb":
        #     reconst_loss_gene = (
        #         -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
        #         .log_prob(x)
        #         .sum(dim=-1)
        #     )
        # else:
        #     reconst_loss_gene = (
        #         -Poisson(px_["rate"])
        #         .log_prob(x)
        #         .sum(dim=-1))

        return rl

    def get_self_expressiveness_loss(
        self,
        inference_outputs,
        senet_outputs,
        lmbd=0.9,
    ):
        """Compute self-expressiveness loss."""
        enc_outputs = inference_outputs
        z = enc_outputs['z']
        rec_z, coeff = senet_outputs['rec_queries'], senet_outputs['coeff']
        se_loss = torch.sum(torch.pow(z - rec_z, 2), dim=-1)
        coeff_reg = regularizer(coeff, lmbd).sum(dim=-1)
        # c_contrast = torch.sum(torch.pow(c_genes - c_proteins, 2), dim=-1)

        return se_loss, coeff_reg

    def loss(
        self,
        x: torch.Tensor,
        inference_outputs,
        senet_outputs,
        generative_outputs,
        batch_index: Optional[torch.Tensor] = None,
        kl_weight=1.0,
        se_weight=1.0,
    ):
        """
        Returns the reconstruction loss and the Kullback-Leibler divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        inference_outputs
            inference network outputs
        senet_outputs
            self-expressive network outputs
        generative_outputs
            generative network outputs
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        kl_weight
            Kullback-Leibler divergences loss weight
        se_weight
            self-expressive loss weight
        Returns
        -------
        type
            the reconstruction loss, the KL divergences and self-expressive loss
        """
        enc_peak_outputs = inference_outputs
        qz_m = enc_peak_outputs['qz_m']
        qz_v = enc_peak_outputs['qz_v']
        px_ = generative_outputs
        px_['cf'] = enc_peak_outputs['lib']

        reconst_loss = self.get_reconstruction_loss(x, px_)

        # KL Divergence
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        se_loss, coeff_reg = self.get_self_expressiveness_loss(
            inference_outputs, senet_outputs)

        loss = torch.mean(
            reconst_loss
            + kl_weight * kl_div_z
            + se_weight * (se_loss + coeff_reg)
        )

        kl_local = dict(
            kl_div_z=kl_div_z,
        )
        se_losses = dict(
            se_loss=se_loss,
            coeff_reg=coeff_reg,
        )

        return loss, reconst_loss, kl_local, se_losses

    def forward(
        self,
        input_peaks: torch.Tensor,
        refer_adata,
        batch_index: Optional[torch.Tensor] = None,
        compute_loss=True,
        transform_batch: Optional[int] = None,
    ):
        """
        Forward pass through the network.

        Parameters
        ----------
        self
        input_peaks
            tensor with shape ``(batch, n_input_gens)``
        refer_adata
            adata with shape ``(training_size, )``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        compute_loss
            whether compute losses
        transform_batch
            batch to condition on
        """
        ref_z_list = []
        for data in refer_adata:
            refer_peaks = torch.tensor(data.obsm['peak_counts'], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.inference(refer_peaks, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_peaks = torch.cat(ref_z_list, dim=0)

        inference_outputs = self.inference(input_peaks, batch_index)
        enc_peak_outputs = inference_outputs
        z_peaks = enc_peak_outputs['z']
        senet_outputs = self.self_expressiveness(z_peaks, ref_z_peaks)
        rec_z = senet_outputs['rec_queries']
        generative_outputs = self.generative(rec_z, batch_index, transform_batch)
        if compute_loss:
            loss = self.loss(
                input_peaks, inference_outputs, senet_outputs, generative_outputs, batch_index
            )
            return inference_outputs, senet_outputs, generative_outputs, loss
        else:
            return inference_outputs, senet_outputs, generative_outputs

    # @torch.no_grad()
    # def sample(self, tensors, n_samples=1):
    #     inference_kwargs = dict(n_samples=n_samples)
    #     with torch.no_grad():
    #         inference_outputs, generative_outputs, = self.forward(
    #             tensors,
    #             inference_kwargs=inference_kwargs,
    #             compute_loss=False,
    #         )
    #
    #     px_ = generative_outputs["px_"]
    #     py_ = generative_outputs["py_"]
    #
    #     rna_dist = NegativeBinomial(mu=px_["rate"], theta=px_["r"])
    #     protein_dist = NegativeBinomialMixture(
    #         mu1=py_["rate_back"],
    #         mu2=py_["rate_fore"],
    #         theta1=py_["r"],
    #         mixture_logits=py_["mixing"],
    #     )
    #     rna_sample = rna_dist.sample().cpu()
    #     protein_sample = protein_dist.sample().cpu()
    #
    #     return rna_sample, protein_sample
    #
    # @torch.no_grad()
    # def marginal_ll(self, tensors, n_mc_samples):
    #     x = tensors["X"]
    #     batch_index = tensors["batch_indices"]
    #     to_sum = torch.zeros(x.size()[0], n_mc_samples)
    #
    #     for i in range(n_mc_samples):
    #         # Distribution parameters and sampled variables
    #         inference_outputs, generative_outputs, losses = self.forward(tensors)
    #         # outputs = self.module.inference(x, y, batch_index, labels)
    #         qz_m = inference_outputs["qz_m"]
    #         qz_v = inference_outputs["qz_v"]
    #         ql_m = inference_outputs["ql_m"]
    #         ql_v = inference_outputs["ql_v"]
    #         py_ = generative_outputs["py_"]
    #         log_library = inference_outputs["untran_l"]
    #         # really need not softmax transformed random variable
    #         z = inference_outputs["untran_z"]
    #         log_pro_back_mean = generative_outputs["log_pro_back_mean"]
    #
    #         # Reconstruction Loss
    #         reconst_loss = losses._reconstruction_loss
    #         reconst_loss_gene = reconst_loss["reconst_loss_gene"]
    #         reconst_loss_protein = reconst_loss["reconst_loss_protein"]
    #
    #         # Log-probabilities
    #         log_prob_sum = torch.zeros(qz_m.shape[0]).to(self.device)
    #
    #         if not self.use_observed_lib_size:
    #             n_batch = self.library_log_means.shape[1]
    #             local_library_log_means = F.linear(
    #                 one_hot(batch_index, n_batch), self.library_log_means
    #             )
    #             local_library_log_vars = F.linear(
    #                 one_hot(batch_index, n_batch), self.library_log_vars
    #             )
    #             p_l_gene = (
    #                 Normal(local_library_log_means, local_library_log_vars.sqrt())
    #                 .log_prob(log_library)
    #                 .sum(dim=-1)
    #             )
    #             q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(log_library).sum(dim=-1)
    #
    #             log_prob_sum += p_l_gene - q_l_x
    #
    #         p_z = Normal(0, 1).log_prob(z).sum(dim=-1)
    #         p_mu_back = self.back_mean_prior.log_prob(log_pro_back_mean).sum(dim=-1)
    #         p_xy_zl = -(reconst_loss_gene + reconst_loss_protein)
    #         q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
    #         q_mu_back = (
    #             Normal(py_["back_alpha"], py_["back_beta"])
    #             .log_prob(log_pro_back_mean)
    #             .sum(dim=-1)
    #         )
    #         log_prob_sum += p_z + p_mu_back + p_xy_zl - q_z_x - q_mu_back
    #
    #         to_sum[:, i] = log_prob_sum
    #
    #     batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
    #     log_lkl = torch.sum(batch_log_lkl).item()
    #     return log_lkl


# IMAGE SEVAE model
class ImageSVAE(nn.Module):
    """
    scRNA-seq self-expressive clustering.
    Implements the SCMSI model for single cell RNA data.

    Parameters
    ----------
    n_input_genes
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'zip'`` - Zero-inflated poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_senet_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_layers_senet: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.0,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_senet: float = 0.1,
        log_variational: bool = False,
        gene_likelihood: str = "gaussian",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        use_observed_lib_size: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size
        self.warm_up = 0.0

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # n_input = n_input_genes + self.n_input_proteins
        n_input_genes_encoder = n_input_genes + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_genes = EncoderImage(
            n_input_genes_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        self.senet_genes = SENet(
            n_latent,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=True,
            use_layer_norm=False,
        )

        self.decoder_genes = DecoderImage(
            n_latent + n_continuous_cov,
            n_input_genes,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def inference(
        self,
        x: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """
        Internal helper function to compute necessary inference quantities.

        We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
        The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
        `scale` refers to the quantity upon which differential expression is performed. For genes,
        this can be viewed as the mean of the underlying gamma distribution.

        ``px_["r"]`` is the inverse dispersion parameters for genes.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        n_samples
            Number of samples to sample from approximate posterior
        """
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates is True:
            enc_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            enc_input = x_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=-1)
        else:
            categorical_input = tuple()
        enc_outputs = self.encoder_genes(
            enc_input, batch_index, *categorical_input
        )

        if n_samples > 1:
            # genes latent space sampling
            qz_m, qz_v, _, _ = enc_outputs.values()
            enc_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_outputs['untran_z'] = Normal(enc_outputs['qz_m'], enc_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_outputs['z'] = self.encoder_genes.z_transformation(enc_outputs['untran_z'])

        return enc_outputs

    def self_expressiveness(
        self,
        z: torch.Tensor,
        ref_z: torch.Tensor,
    ):
        """
        Use self-expressive network to reconstruct the latent space
        Parameters
        ----------
        z
            tensor of values with shape ``(batch_size, n_latent_genes)``
        ref_z
            tensor of values with shape ``(training_size, n_latent_genes)``
        Returns
        -------
        rec_z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        se_gene_outputs
            genes dict with "rec_queries" and "coeff"
        """
        se_outputs = self.senet_genes(z, ref_z)
        return se_outputs

    def generative(
        self,
        z: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        transform_batch: Optional[int] = None,
    ):
        # decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cont_covs is not None:
            decoder_input = torch.cat((z, cont_covs), dim=-1)
        else:
            decoder_input = z
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        px_ = self.decoder_genes(
            decoder_input, batch_index, *categorical_input
        )

        return px_

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        px_: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        # px_ = px_dict
        # Reconstruction Loss

        if self.gene_likelihood == "gaussian":
            reconst_loss_gene = (
                torch.pow(px_["scale"] - x, 2).sum(dim=-1)
            )

        return reconst_loss_gene

    def get_self_expressiveness_loss(
        self,
        inference_outputs,
        senet_outputs,
        lmbd=0.9,
    ):
        """Compute self-expressiveness loss."""
        enc_outputs = inference_outputs
        z = enc_outputs['z']
        rec_z, coeff = senet_outputs['rec_queries'], senet_outputs['coeff']
        se_loss = torch.sum(torch.pow(z - rec_z, 2), dim=-1)
        coeff_reg = regularizer(coeff, lmbd).sum(dim=-1)
        # c_contrast = torch.sum(torch.pow(c_genes - c_proteins, 2), dim=-1)

        return se_loss, coeff_reg

    def loss(
        self,
        x: torch.Tensor,
        inference_outputs,
        senet_outputs,
        generative_outputs,
        batch_index: Optional[torch.Tensor] = None,
        kl_weight=1.0,
        se_weight=1.0,
    ):
        """
        Returns the reconstruction loss and the Kullback-Leibler divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        inference_outputs
            inference network outputs
        senet_outputs
            self-expressive network outputs
        generative_outputs
            generative network outputs
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        kl_weight
            Kullback-Leibler divergences loss weight
        se_weight
            self-expressive loss weight
        Returns
        -------
        type
            the reconstruction loss, the KL divergences and self-expressive loss
        """
        self.warm_up += 1
        kl_weight = 1.0 if self.warm_up > 100 else 0.0
        enc_gene_outputs = inference_outputs
        qz_m = enc_gene_outputs['qz_m']
        qz_v = enc_gene_outputs['qz_v']
        px_ = generative_outputs

        reconst_loss = self.get_reconstruction_loss(x, px_)

        # KL Divergence
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        se_loss, coeff_reg = self.get_self_expressiveness_loss(
            inference_outputs, senet_outputs)

        loss = torch.mean(
            reconst_loss
            + kl_weight * kl_div_z
            + se_weight * (se_loss + coeff_reg)
        )

        kl_local = dict(
            kl_div_z=kl_div_z,
        )
        se_losses = dict(
            se_loss=se_loss,
            coeff_reg=coeff_reg,
        )

        return loss, reconst_loss, kl_local, se_losses

    def forward(
        self,
        input_genes: torch.Tensor,
        refer_adata,
        batch_index: Optional[torch.Tensor] = None,
        compute_loss=True,
        transform_batch: Optional[int] = None,
    ):
        """
        Forward pass through the network.

        Parameters
        ----------
        self
        input_genes
            tensor with shape ``(batch, n_input_gens)``
        refer_adata
            adata with shape ``(training_size, )``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        compute_loss
            whether compute losses
        transform_batch
            batch to condition on
        """
        ref_z_list = []
        for data in refer_adata:
            refer_genes = torch.tensor(data.obsm["X_morphology"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.inference(refer_genes, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_genes = torch.cat(ref_z_list, dim=0)

        inference_outputs = self.inference(input_genes, batch_index)
        enc_gene_outputs = inference_outputs
        z_genes = enc_gene_outputs['z']
        senet_outputs = self.self_expressiveness(z_genes, ref_z_genes)
        rec_z = senet_outputs['rec_queries']
        generative_outputs = self.generative(rec_z, batch_index, transform_batch)
        if compute_loss:
            loss = self.loss(
                input_genes, inference_outputs, senet_outputs, generative_outputs, batch_index
            )
            return inference_outputs, senet_outputs, generative_outputs, loss
        else:
            return inference_outputs, senet_outputs, generative_outputs

    @torch.no_grad()
    def sample(self, tensors, n_samples=1):
        inference_kwargs = dict(n_samples=n_samples)
        with torch.no_grad():
            inference_outputs, generative_outputs, = self.forward(
                tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )

        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]

        rna_dist = NegativeBinomial(mu=px_["rate"], theta=px_["r"])
        protein_dist = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )
        rna_sample = rna_dist.sample().cpu()
        protein_sample = protein_dist.sample().cpu()

        return rna_sample, protein_sample

    @torch.no_grad()
    def marginal_ll(self, tensors, n_mc_samples):
        x = tensors["X"]
        batch_index = tensors["batch_indices"]
        to_sum = torch.zeros(x.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, generative_outputs, losses = self.forward(tensors)
            # outputs = self.module.inference(x, y, batch_index, labels)
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            py_ = generative_outputs["py_"]
            log_library = inference_outputs["untran_l"]
            # really need not softmax transformed random variable
            z = inference_outputs["untran_z"]
            log_pro_back_mean = generative_outputs["log_pro_back_mean"]

            # Reconstruction Loss
            reconst_loss = losses._reconstruction_loss
            reconst_loss_gene = reconst_loss["reconst_loss_gene"]
            reconst_loss_protein = reconst_loss["reconst_loss_protein"]

            # Log-probabilities
            log_prob_sum = torch.zeros(qz_m.shape[0]).to(self.device)

            if not self.use_observed_lib_size:
                n_batch = self.library_log_means.shape[1]
                local_library_log_means = F.linear(
                    one_hot(batch_index, n_batch), self.library_log_means
                )
                local_library_log_vars = F.linear(
                    one_hot(batch_index, n_batch), self.library_log_vars
                )
                p_l_gene = (
                    Normal(local_library_log_means, local_library_log_vars.sqrt())
                    .log_prob(log_library)
                    .sum(dim=-1)
                )
                q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(log_library).sum(dim=-1)

                log_prob_sum += p_l_gene - q_l_x

            p_z = Normal(0, 1).log_prob(z).sum(dim=-1)
            p_mu_back = self.back_mean_prior.log_prob(log_pro_back_mean).sum(dim=-1)
            p_xy_zl = -(reconst_loss_gene + reconst_loss_protein)
            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            q_mu_back = (
                Normal(py_["back_alpha"], py_["back_beta"])
                .log_prob(log_pro_back_mean)
                .sum(dim=-1)
            )
            log_prob_sum += p_z + p_mu_back + p_xy_zl - q_z_x - q_mu_back

            to_sum[:, i] = log_prob_sum

        batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl


# scVI VAE model
class scVIVAE(nn.Module):
    """
    scVI re-coding.

    Parameters
    ----------
    n_input_genes
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    gene_dispersion
        One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'zip'`` - Zero-inflated poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.0,
        dropout_rate_encoder: float = 0.1,
        gene_dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:  # gene-cell
            pass

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # n_input = n_input_genes + self.n_input_proteins
        n_input_genes_encoder = n_input_genes + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_genes = EncoderGene(
            n_input_genes_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        self.decoder_genes = DecoderSCVI(
            n_latent + n_continuous_cov,
            n_input_genes,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def inference(
        self,
        x: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """
        Internal helper function to compute necessary inference quantities.

        We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
        The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
        `scale` refers to the quantity upon which differential expression is performed. For genes,
        this can be viewed as the mean of the underlying gamma distribution.

        ``px_["r"]`` is the inverse dispersion parameters for genes.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        n_samples
            Number of samples to sample from approximate posterior
        """
        x_ = x
        if self.use_observed_lib_size:
            library_gene = x.sum(1).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates is True:
            enc_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            enc_input = x_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=-1)
        else:
            categorical_input = tuple()
        enc_outputs = self.encoder_genes(
            enc_input, batch_index, *categorical_input
        )

        if self.use_observed_lib_size:
            enc_outputs['library_gene'] = library_gene

        if n_samples > 1:
            # genes latent space sampling
            qz_m, qz_v, _, _, ql_m, ql_v, _, _ = enc_outputs.values()
            enc_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_outputs['untran_z'] = Normal(enc_outputs['qz_m'], enc_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_outputs['z'] = self.encoder_genes.z_transformation(enc_outputs['untran_z'])
            enc_outputs['ql_m'] = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            enc_outputs['ql_v'] = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            enc_outputs['untran_l'] = Normal(enc_outputs['ql_m'], enc_outputs['ql_v'].sqrt()).sample()
            if self.use_observed_lib_size:
                enc_outputs['library_gene'] = library_gene.unsqueeze(0).expand(
                    (n_samples, library_gene.size(0), library_gene.size(1))
                )
            else:
                enc_outputs['library_gene'] = self.encoder_genes.l_transformation(enc_outputs['untran_l'])

        return enc_outputs

    def generative(
        self,
        z: torch.Tensor,
        library_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        transform_batch: Optional[int] = None,
    ):
        # decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cont_covs is not None:
            decoder_input = torch.cat((z, cont_covs), dim=-1)
        else:
            decoder_input = z
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        px_ = self.decoder_genes(
            decoder_input, library_gene, batch_index, *categorical_input
        )

        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_["r"] = torch.exp(px_r)

        return px_

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        px_: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        # px_ = px_dict
        # Reconstruction Loss

        if self.gene_likelihood == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -Poisson(px_["rate"])
                .log_prob(x)
                .sum(dim=-1))

        return reconst_loss_gene

    def loss(
        self,
        x: torch.Tensor,
        inference_outputs,
        generative_outputs,
        batch_index: Optional[torch.Tensor] = None,
        kl_weight=1.0,
    ):
        """
        Returns the reconstruction loss and the Kullback-Leibler divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        inference_outputs
            inference network outputs
        generative_outputs
            generative network outputs
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        kl_weight
            Kullback-Leibler divergences loss weight
        Returns
        -------
        type
            the reconstruction loss, the KL divergences and self-expressive loss
        """
        enc_gene_outputs = inference_outputs
        qz_m = enc_gene_outputs['qz_m']
        qz_v = enc_gene_outputs['qz_v']
        ql_m = enc_gene_outputs['ql_m']
        ql_v = enc_gene_outputs['ql_v']
        px_ = generative_outputs

        reconst_loss = self.get_reconstruction_loss(x, px_)

        # KL Divergence
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                one_hot(batch_index, n_batch), self.library_log_means
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index, n_batch), self.library_log_vars
            )
            kl_div_l = kl(
                Normal(ql_m, torch.sqrt(ql_v)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_div_l = 0.0

        loss = torch.mean(
            reconst_loss
            + kl_weight * kl_div_z
            + kl_div_l
        )

        kl_local = dict(
            kl_div_z=kl_div_z,
            kl_div_l=kl_div_l,
        )

        return loss, reconst_loss, kl_local

    def forward(
        self,
        input_genes: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        compute_loss=True,
    ):
        """
        Forward pass through the network.

        Parameters
        ----------
        self
        input_genes
            tensor with shape ``(batch, n_input_gens)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        compute_loss
            whether compute losses
        """
        inference_outputs = self.inference(input_genes, batch_index)
        enc_gene_outputs = inference_outputs
        z_genes = enc_gene_outputs['z']
        library_gene = enc_gene_outputs['library_gene']
        generative_outputs = self.generative(z_genes, library_gene, batch_index)
        if compute_loss:
            loss = self.loss(
                input_genes, inference_outputs, generative_outputs, batch_index
            )
            return inference_outputs, generative_outputs, loss
        else:
            return inference_outputs, generative_outputs

    @torch.no_grad()
    def sample(self, tensors, n_samples=1):
        inference_kwargs = dict(n_samples=n_samples)
        with torch.no_grad():
            inference_outputs, generative_outputs, = self.forward(
                tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )

        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]

        rna_dist = NegativeBinomial(mu=px_["rate"], theta=px_["r"])
        protein_dist = NegativeBinomialMixture(
            mu1=py_["rate_back"],
            mu2=py_["rate_fore"],
            theta1=py_["r"],
            mixture_logits=py_["mixing"],
        )
        rna_sample = rna_dist.sample().cpu()
        protein_sample = protein_dist.sample().cpu()

        return rna_sample, protein_sample

    @torch.no_grad()
    def marginal_ll(self, tensors, n_mc_samples):
        x = tensors["X"]
        batch_index = tensors["batch_indices"]
        to_sum = torch.zeros(x.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, generative_outputs, losses = self.forward(tensors)
            # outputs = self.module.inference(x, y, batch_index, labels)
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            py_ = generative_outputs["py_"]
            log_library = inference_outputs["untran_l"]
            # really need not softmax transformed random variable
            z = inference_outputs["untran_z"]
            log_pro_back_mean = generative_outputs["log_pro_back_mean"]

            # Reconstruction Loss
            reconst_loss = losses._reconstruction_loss
            reconst_loss_gene = reconst_loss["reconst_loss_gene"]
            reconst_loss_protein = reconst_loss["reconst_loss_protein"]

            # Log-probabilities
            log_prob_sum = torch.zeros(qz_m.shape[0]).to(self.device)

            if not self.use_observed_lib_size:
                n_batch = self.library_log_means.shape[1]
                local_library_log_means = F.linear(
                    one_hot(batch_index, n_batch), self.library_log_means
                )
                local_library_log_vars = F.linear(
                    one_hot(batch_index, n_batch), self.library_log_vars
                )
                p_l_gene = (
                    Normal(local_library_log_means, local_library_log_vars.sqrt())
                    .log_prob(log_library)
                    .sum(dim=-1)
                )
                q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(log_library).sum(dim=-1)

                log_prob_sum += p_l_gene - q_l_x

            p_z = Normal(0, 1).log_prob(z).sum(dim=-1)
            p_mu_back = self.back_mean_prior.log_prob(log_pro_back_mean).sum(dim=-1)
            p_xy_zl = -(reconst_loss_gene + reconst_loss_protein)
            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            q_mu_back = (
                Normal(py_["back_alpha"], py_["back_beta"])
                .log_prob(log_pro_back_mean)
                .sum(dim=-1)
            )
            log_prob_sum += p_z + p_mu_back + p_xy_zl - q_z_x - q_mu_back

            to_sum[:, i] = log_prob_sum

        batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl


# RNA SEVAE model
class RNAVAE(nn.Module):
    """
    scRNA-seq self-expressive clustering.
    Implements the SCMSI model for single cell RNA data.

    Parameters
    ----------
    n_input_genes
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    gene_dispersion
        One of the following

        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'zip'`` - Zero-inflated poisson distribution
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    """

    def __init__(
        self,
        n_input_genes: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_senet_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_layers_senet: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.0,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_senet: float = 0.1,
        gene_dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size
        self.warm_up = 0.0
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:  # gene-cell
            pass

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # n_input = n_input_genes + self.n_input_proteins
        n_input_genes_encoder = n_input_genes + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_genes = EncoderGene(
            n_input_genes_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )

        self.senet_genes = SENet(
            n_latent,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=True,
            use_layer_norm=False,
        )

        self.decoder_genes = DecoderSCVI(
            n_latent + n_continuous_cov,
            n_input_genes,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def inference(
        self,
        x: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """
        Internal helper function to compute necessary inference quantities.

        We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
        The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
        `scale` refers to the quantity upon which differential expression is performed. For genes,
        this can be viewed as the mean of the underlying gamma distribution.

        ``px_["r"]`` is the inverse dispersion parameters for genes.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        n_samples
            Number of samples to sample from approximate posterior
        """
        x_ = x
        if self.use_observed_lib_size:
            library_gene = x.sum(1).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates is True:
            enc_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            enc_input = x_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=-1)
        else:
            categorical_input = tuple()
        enc_outputs = self.encoder_genes(
            enc_input, batch_index, *categorical_input
        )

        if self.use_observed_lib_size:
            enc_outputs['library_gene'] = library_gene

        if n_samples > 1:
            # genes latent space sampling
            qz_m, qz_v, _, _, ql_m, ql_v, _, _ = enc_outputs.values()
            enc_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_outputs['untran_z'] = Normal(enc_outputs['qz_m'], enc_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_outputs['z'] = self.encoder_genes.z_transformation(enc_outputs['untran_z'])
            enc_outputs['ql_m'] = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            enc_outputs['ql_v'] = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            enc_outputs['untran_l'] = Normal(enc_outputs['ql_m'], enc_outputs['ql_v'].sqrt()).sample()
            if self.use_observed_lib_size:
                enc_outputs['library_gene'] = library_gene.unsqueeze(0).expand(
                    (n_samples, library_gene.size(0), library_gene.size(1))
                )
            else:
                enc_outputs['library_gene'] = self.encoder_genes.l_transformation(enc_outputs['untran_l'])

        return enc_outputs

    def self_expressiveness(
            self,
            z: torch.Tensor,
            ref_z: torch.Tensor,
    ):
        """
        Use self-expressive network to reconstruct the latent space
        Parameters
        ----------
        z
            tensor of values with shape ``(batch_size, n_latent_genes)``
        ref_z
            tensor of values with shape ``(training_size, n_latent_genes)``
        Returns
        -------
        rec_z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        se_gene_outputs
            genes dict with "rec_queries" and "coeff"
        """
        se_outputs = self.senet_genes(z, ref_z)
        se_outputs['rec_queries'] = z  # make se module invalid

        return se_outputs

    def generative(
        self,
        z: torch.Tensor,
        library_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        transform_batch: Optional[int] = None,
    ):
        # decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cont_covs is not None:
            decoder_input = torch.cat((z, cont_covs), dim=-1)
        else:
            decoder_input = z
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        px_ = self.decoder_genes(
            decoder_input, library_gene, batch_index, *categorical_input
        )

        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_["r"] = torch.exp(px_r)

        return px_

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        px_: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        # px_ = px_dict
        # Reconstruction Loss

        if self.gene_likelihood == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -Poisson(px_["rate"])
                .log_prob(x)
                .sum(dim=-1))

        return reconst_loss_gene

    def get_self_expressiveness_loss(
        self,
        inference_outputs,
        senet_outputs,
        lmbd=0.9,
    ):
        """Compute self-expressiveness loss."""
        enc_outputs = inference_outputs
        z = enc_outputs['z']
        rec_z, coeff = senet_outputs['rec_queries'], senet_outputs['coeff']
        se_loss = torch.sum(torch.pow(z - rec_z, 2), dim=-1)
        coeff_reg = regularizer(coeff, lmbd).sum(dim=-1)
        # c_contrast = torch.sum(torch.pow(c_genes - c_proteins, 2), dim=-1)

        return se_loss, coeff_reg

    def loss(
        self,
        x: torch.Tensor,
        inference_outputs,
        senet_outputs,
        generative_outputs,
        batch_index: Optional[torch.Tensor] = None,
        kl_weight=1.0,
        se_weight=0.0,
    ):
        """
        Returns the reconstruction loss and the Kullback-Leibler divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        inference_outputs
            inference network outputs
        senet_outputs
            self-expressive network outputs
        generative_outputs
            generative network outputs
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        kl_weight
            Kullback-Leibler divergences loss weight
        se_weight
            self-expressive loss weight
        Returns
        -------
        type
            the reconstruction loss, the KL divergences and self-expressive loss
        """
        self.warm_up += 1
        kl_weight = 1.0 if self.warm_up > 100 else 0.0
        enc_gene_outputs = inference_outputs
        qz_m = enc_gene_outputs['qz_m']
        qz_v = enc_gene_outputs['qz_v']
        ql_m = enc_gene_outputs['ql_m']
        ql_v = enc_gene_outputs['ql_v']
        px_ = generative_outputs

        reconst_loss = self.get_reconstruction_loss(x, px_)

        # KL Divergence
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                one_hot(batch_index, n_batch), self.library_log_means
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index, n_batch), self.library_log_vars
            )
            kl_div_l = kl(
                Normal(ql_m, torch.sqrt(ql_v)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_div_l = 0.0

        se_loss, coeff_reg = self.get_self_expressiveness_loss(
            inference_outputs, senet_outputs)

        loss = torch.mean(
            reconst_loss
            + kl_weight * kl_div_z
            + kl_div_l
            + se_weight * (se_loss + coeff_reg)
        )

        kl_local = dict(
            kl_div_z=kl_div_z,
            kl_div_l=kl_div_l,
        )
        se_losses = dict(
            se_loss=se_loss,
            coeff_reg=coeff_reg,
        )

        return loss, reconst_loss, kl_local, se_losses

    def forward(
        self,
        input_genes: torch.Tensor,
        refer_adata,
        batch_index: Optional[torch.Tensor] = None,
        compute_loss=True,
        transform_batch: Optional[int] = None,
    ):
        """
        Forward pass through the network.

        Parameters
        ----------
        self
        input_genes
            tensor with shape ``(batch, n_input_gens)``
        refer_adata
            adata with shape ``(training_size, )``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        compute_loss
            whether compute losses
        transform_batch
            batch to condition on
        """
        ref_z_list = []
        for data in refer_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.inference(refer_genes, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_genes = torch.cat(ref_z_list, dim=0)

        inference_outputs = self.inference(input_genes, batch_index)
        enc_gene_outputs = inference_outputs
        z_genes = enc_gene_outputs['z']
        library_gene = enc_gene_outputs['library_gene']
        senet_outputs = self.self_expressiveness(z_genes, ref_z_genes)
        rec_z = senet_outputs['rec_queries']
        generative_outputs = self.generative(rec_z, library_gene, batch_index, transform_batch)
        if compute_loss:
            loss = self.loss(
                input_genes, inference_outputs, senet_outputs, generative_outputs, batch_index
            )
            return inference_outputs, senet_outputs, generative_outputs, loss
        else:
            return inference_outputs, senet_outputs, generative_outputs


# Protein SVAE model
class ProteinVAE(nn.Module):
    """
    single cell protein self-expressive clustering.

    Implements the SCMSI model for single cell multi-omics data.

    Parameters
    ----------
    n_input_proteins
        Number of input proteins
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer for encoder and decoder
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    protein_dispersion
        One of the following

        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    protein_background_prior_mean
        Array of proteins by batches, the prior initialization for the protein background mean (log scale)
    protein_background_prior_scale
        Array of proteins by batches, the prior initialization for the protein background scale (log scale)

    """

    def __init__(
        self,
        n_input_proteins: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_senet_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_layers_senet: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.1,
        dropout_rate_encoder: float = 0.0,
        dropout_rate_senet: float = 0.0,
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        protein_likelihood: str = "nbm",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        protein_background_prior_mean: Optional[np.ndarray] = None,
        protein_background_prior_scale: Optional[np.ndarray] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_proteins = n_input_proteins
        self.protein_dispersion = protein_dispersion
        self.protein_likelihood = protein_likelihood
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates

        # parameters for prior on back_rate (background protein mean)
        if protein_background_prior_mean is None:
            if n_batch > 0:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins, n_batch)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins, n_batch), -10, 1)
                )
            else:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins), -10, 1)
                )
        else:
            # mean_shape = protein_background_prior_mean.shape
            # protein_background_prior_mean += np.random.randn(mean_shape[0], mean_shape[1])*0.01
            if protein_background_prior_mean.shape[1] == 1 and n_batch != 1:
                init_mean = protein_background_prior_mean.ravel()
                init_scale = protein_background_prior_scale.ravel()
            else:
                init_mean = protein_background_prior_mean
                init_scale = protein_background_prior_scale
            self.background_pro_alpha = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)))
            )

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_batch)
            )
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_labels)
            )
        else:  # protein-cell
            pass

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # n_input = n_input_genes + self.n_input_proteins
        n_input_proteins_encoder = n_input_proteins + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_proteins = EncoderProtein(
            n_input_proteins_encoder,
            n_latent,
            n_layers=n_layers_encoder,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.senet_proteins = SENet(
            n_latent,
            n_senet_hidden,  # n_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.decoder_proteins = DecoderProtein(
            n_latent + n_continuous_cov,
            n_input_proteins,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def inference(
        self,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        cont_covs=None,
        cat_covs=None,
    ):
        """
        Internal helper function to compute necessary inference quantities.

        We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
        `fore_rate` refers to foreground mean, while `back_rate` refers to background mean. ``scale`` refers to
        foreground mean adjusted for background probability and scaled to reside in simplex.
        ``back_mean`` and ``back_var`` are the posterior parameters for ``back_rate``.  ``fore_scale`` is the scaling
        factor that enforces `fore_rate` > `back_rate`.

        ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.

        Parameters
        ----------
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        label
            tensor of cell-types labels with shape (batch_size, n_labels)
        n_samples
            Number of samples to sample from approximate posterior
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        """
        y_ = y
        if self.log_variational:
            y_ = torch.log(1 + y_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input_proteins = torch.cat((y_, cont_covs), dim=-1)
        else:
            encoder_input_proteins = y_
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        enc_pro_outputs = self.encoder_proteins(
            encoder_input_proteins, batch_index, *categorical_input
        )

        if n_samples > 1:
            # proteins latent space sampling
            qz_m, qz_v, z, untran_z = enc_pro_outputs.values()
            enc_pro_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_pro_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_pro_outputs['untran_z'] = Normal(enc_pro_outputs['qz_m'], enc_pro_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_pro_outputs['z'] = self.encoder_proteins.z_transformation(enc_pro_outputs['untran_z'])

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)
        if self.n_batch > 0:
            py_back_mean_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_var_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_mean_prior = self.background_pro_alpha
            py_back_var_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(py_back_mean_prior, py_back_var_prior)

        return enc_pro_outputs

    def self_expressiveness(
            self,
            z: torch.Tensor,
            ref_z: torch.Tensor,
    ):
        """
        Use self-expressive network to reconstruct the latent space
        Parameters
        ----------
        z
            tensor of values with shape ``(batch_size, n_latent_genes)``
        ref_z
            tensor of values with shape ``(training_size, n_latent_genes)``
        Returns
        -------
        rec_z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        se_gene_outputs
            genes dict with "rec_queries" and "coeff"
        """
        se_outputs = self.senet_proteins(z, ref_z)
        se_outputs['rec_queries'] = z  # make se module invalid

        return se_outputs

    def generative(
        self,
        z: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        transform_batch: Optional[int] = None,
    ):
        # decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        if cont_covs is not None:
            decoder_input_proteins = torch.cat((z, cont_covs), dim=-1)
        else:
            decoder_input_proteins = z
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        py_ = self.decoder_proteins(
            decoder_input_proteins, batch_index, *categorical_input
        )

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_["r"] = torch.exp(py_r)

        return py_

    def get_reconstruction_loss(
        self,
        y: torch.Tensor,
        py_dict: Dict[str, torch.Tensor],
    ):
        """Compute reconstruction loss."""
        py_ = py_dict
        # Reconstruction Loss
        if self.protein_likelihood == "nbm":
            py_conditional = NegativeBinomialMixture(
                mu1=py_["back_rate"],
                mu2=py_["fore_rate"],
                theta1=py_["r"],
                mixture_logits=py_["mixing"],
            )
        elif self.protein_likelihood == "nb":
            py_conditional = NegativeBinomial(mu=py_['rate'], theta=py_["r"])
        else:
            py_conditional = Poisson(py_['rate'])

        reconst_loss_protein_full = -py_conditional.log_prob(y)
        reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_protein

    def get_self_expressiveness_loss(
        self,
        inference_outputs,
        senet_outputs,
        lmbd=0.9,
    ):
        """Compute self-expressiveness loss."""
        enc_outputs = inference_outputs
        z = enc_outputs['z']
        rec_z, coeff = senet_outputs['rec_queries'], senet_outputs['coeff']
        se_loss = torch.sum(torch.pow(z - rec_z, 2), dim=-1)
        coeff_reg = regularizer(coeff, lmbd).sum(dim=-1)
        coeff_reg = regularizer_l12(coeff, lmbd) * torch.ones(coeff.shape[0])
        # c_contrast = torch.sum(torch.pow(c_genes - c_proteins, 2), dim=-1)

        return se_loss, coeff_reg

    def loss(
        self,
        y: torch.Tensor,
        inference_outputs,
        senet_outputs,
        generative_outputs,
        batch_index: Optional[torch.Tensor] = None,
        pro_recons_weight=1.0,  # double check these defaults
        kl_weight=1.0,
        se_weight=1.0,
    ):
        """
        Returns the reconstruction loss and the Kullback-Leibler divergences.

        Parameters
        ----------
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        inference_outputs
            inference network outputs
        senet_outputs
            self-expressive network outputs
        generative_outputs
            generative network outputs
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        pro_recons_weight
            proteins reconstruction weight
        kl_weight
            Kullback-Leibler divergences loss weight
        se_weight
            self-expressive loss weight
        Returns
        -------
        type
            the reconstruction loss, the KL divergences and self-expressive loss
        """
        enc_pro_outputs = inference_outputs
        qz_m = enc_pro_outputs['qz_m']
        qz_v = enc_pro_outputs['qz_v']
        py_ = generative_outputs

        reconst_loss = self.get_reconstruction_loss(y, py_)
        se_loss, coeff_reg = self.get_self_expressiveness_loss(inference_outputs, senet_outputs, lmbd=0.95)

        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        if self.protein_likelihood == "nbm":
            kl_div_back_pro_full = kl(
                Normal(py_["back_log_rate_mean"], py_["back_log_rate_var"].sqrt()), self.back_mean_prior
            )
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=-1)
        else:
            kl_div_back_pro = 0.0
        losses = torch.mean(
            + pro_recons_weight * reconst_loss
            + kl_weight * kl_div_z
            + kl_weight * kl_div_back_pro
            + se_weight * (se_loss + coeff_reg)
        )

        kl_local = dict(
            kl_div_z=kl_div_z,
            kl_div_back=kl_div_back_pro,
        )
        se_losses = dict(
            se_loss=se_loss,
            coeff_reg=coeff_reg,
        )

        return losses, reconst_loss, kl_local, se_losses

    def forward(
        self,
        input_proteins: torch.Tensor,
        refer_adata,
        batch_index: Optional[torch.Tensor] = None,
        compute_loss=True,
        transform_batch: Optional[int] = None,
    ):
        """
        Forward pass through the network.

        Parameters
        ----------
        self
        input_proteins
            tensor with shape ``(batch, n_input_proteins)``
        refer_adata
            adata with shape ``(training_size, )``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        compute_loss
            whether compute losses
        transform_batch
            batch to condition on
        """
        ref_z_list = []
        for data in refer_adata:
            refer_proteins = torch.tensor(data.obsm["protein_expression"], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            enc_ref_outputs = self.inference(refer_proteins, refer_index)
            ref_z_list.append(enc_ref_outputs['z'])
        ref_z_proteins = torch.cat(ref_z_list, dim=0)

        inference_outputs = self.inference(input_proteins, batch_index)
        enc_pro_outputs = inference_outputs
        z_proteins = enc_pro_outputs['z']
        senet_outputs = self.self_expressiveness(z_proteins, ref_z_proteins)
        rec_z = senet_outputs['rec_queries']
        generative_outputs = self.generative(rec_z, batch_index, transform_batch)
        if compute_loss:
            loss = self.loss(
                input_proteins, inference_outputs, senet_outputs, generative_outputs, batch_index
            )
            return inference_outputs, senet_outputs, generative_outputs, loss
        else:
            return inference_outputs, senet_outputs, generative_outputs


# MSE model
class RNAProteinSE(nn.Module):
    """
    Self-expression model for RNA and protein data (original data space).
    """

    def __init__(
        self,
        n_input_genes: int,
        n_input_proteins: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 256,
        n_senet_hidden: int = 128,
        n_latent: int = 10,
        n_layers_encoder: int = 1,
        n_layers_decoder: int = 1,
        n_layers_senet: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate_decoder: float = 0.1,
        dropout_rate_encoder: float = 0.1,
        dropout_rate_senet: float = 0.0,
        gene_dispersion: str = "gene",
        protein_dispersion: str = "protein",
        log_variational: bool = True,
        gene_likelihood: str = "nb",
        protein_likelihood: str = "nbm",
        latent_distribution: str = "normal",
        encode_covariates: bool = True,
        protein_background_prior_mean: Optional[np.ndarray] = None,
        protein_background_prior_scale: Optional[np.ndarray] = None,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
    ):
        super().__init__()
        self.gene_dispersion = gene_dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.protein_likelihood = protein_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.n_input_genes = n_input_genes
        self.n_input_proteins = n_input_proteins
        self.protein_dispersion = protein_dispersion
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size
        self.warm_up = 0
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )

        # parameters for prior on back_rate (background protein mean)
        if protein_background_prior_mean is None:
            if n_batch > 0:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins, n_batch)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins, n_batch), -10, 1)
                )
            else:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins), -10, 1)
                )
        else:
            if protein_background_prior_mean.shape[1] == 1 and n_batch != 1:
                init_mean = protein_background_prior_mean.ravel()
                init_scale = protein_background_prior_scale.ravel()
            else:
                init_mean = protein_background_prior_mean
                init_scale = protein_background_prior_scale
            self.background_pro_alpha = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)))
            )

        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        else:  # gene-cell
            pass

        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_batch)
            )
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(
                2 * torch.rand(self.n_input_proteins, n_labels)
            )
        else:  # protein-cell
            pass

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # n_input = n_input_genes + self.n_input_proteins
        n_input_genes_encoder = n_input_genes + n_continuous_cov * encode_covariates
        n_input_proteins_encoder = n_input_proteins + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        # cat_list = [n_batch, n_batch]
        # print(cat_list)
        encoder_cat_list = cat_list if encode_covariates else None
        self.encoder_genes = EncoderGene(
            n_input_genes_encoder,
            self.n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.encoder_proteins = EncoderProtein(
            n_input_proteins_encoder,
            n_latent,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_encoder,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.decoder_genes = DecoderSCVI(
            n_input_genes_encoder + n_continuous_cov,
            n_input_genes,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.decoder_proteins = DecoderProtein(
            n_input_proteins_encoder + n_continuous_cov,
            n_input_proteins,
            n_layers=n_layers_decoder,
            n_cat_list=cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate_decoder,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.senet_genes = SENet(
            n_input_genes_encoder,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )
        self.senet_proteins = SENet(
            n_input_proteins_encoder,
            n_senet_hidden,
            n_cat_list=None,
            n_layers=n_layers_senet,
            n_hidden=n_senet_hidden,
            dropout_rate=dropout_rate_senet,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

    def inference(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        n_samples=1,
        cont_covs=None,
        cat_covs=None,
    ):
        """
        Internal helper function to compute necessary inference quantities.

        We use the dictionary ``px_`` to contain the parameters of the ZINB/NB for genes.
        The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
        `scale` refers to the quantity upon which differential expression is performed. For genes,
        this can be viewed as the mean of the underlying gamma distribution.

        We use the dictionary ``py_`` to contain the parameters of the Mixture NB distribution for proteins.
        `fore_rate` refers to foreground mean, while `back_rate` refers to background mean. ``scale`` refers to
        foreground mean adjusted for background probability and scaled to reside in simplex.
        ``back_mean`` and ``back_var`` are the posterior parameters for ``back_rate``.  ``fore_scale`` is the scaling
        factor that enforces `fore_rate` > `back_rate`.

        ``px_["r"]`` and ``py_["r"]`` are the inverse dispersion parameters for genes and protein, respectively.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        label
            tensor of cell-types labels with shape (batch_size, n_labels)
        n_samples
            Number of samples to sample from approximate posterior
        cont_covs
            Continuous covariates to condition on
        cat_covs
            Categorical covariates to condition on
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        batch_index = batch_index.to(device)
        x_ = x
        y_ = y
        if cat_covs is not None:
            cat_covs = cat_covs.to(device)
        if self.use_observed_lib_size:
            library_gene = x.sum(1).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
            y_ = torch.log(1 + y_)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input_genes = torch.cat((x_, cont_covs), dim=-1)
            encoder_input_proteins = torch.cat((y_, cont_covs), dim=-1)
        else:
            encoder_input_genes = x_
            encoder_input_proteins = y_
        # cat_covs = batch_index
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        enc_gene_outputs = self.encoder_genes(
            encoder_input_genes, batch_index, *categorical_input
        )
        enc_pro_outputs = self.encoder_proteins(
            encoder_input_proteins, batch_index, *categorical_input
        )

        if self.use_observed_lib_size:
            enc_gene_outputs['library_gene'] = library_gene

        if n_samples > 1:
            # genes latent space sampling
            qz_m, qz_v, z, untran_z, ql_m, ql_v, _, untran_l = enc_gene_outputs.values()
            enc_gene_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_gene_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_gene_outputs['untran_z'] = Normal(enc_gene_outputs['qz_m'], enc_gene_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_gene_outputs['z'] = self.encoder_genes.z_transformation(enc_gene_outputs['untran_z'])
            enc_gene_outputs['ql_m'] = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            enc_gene_outputs['ql_v'] = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            enc_gene_outputs['untran_l'] = Normal(enc_gene_outputs['ql_m'], enc_gene_outputs['ql_v'].sqrt()).sample()
            if self.use_observed_lib_size:
                enc_gene_outputs['library_gene'] = library_gene.unsqueeze(0).expand(
                    (n_samples, library_gene.size(0), library_gene.size(1))
                )
            else:
                enc_gene_outputs['library_gene'] = self.encoder_genes.l_transformation(untran_l)

            # proteins latent space sampling
            qz_m, qz_v, z, untran_z = enc_pro_outputs.values()
            enc_pro_outputs['qz_m'] = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            enc_pro_outputs['qz_v'] = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            enc_pro_outputs['untran_z'] = Normal(enc_pro_outputs['qz_m'], enc_pro_outputs['qz_v'].sqrt()).sample()  # why not use reparameterization trick
            enc_pro_outputs['z'] = self.encoder_proteins.z_transformation(enc_pro_outputs['untran_z'])

        # Background regularization
        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)
        if self.n_batch > 0:
            py_back_mean_prior = F.linear(
                one_hot(batch_index, self.n_batch), self.background_pro_alpha
            )
            py_back_var_prior = F.linear(
                one_hot(batch_index, self.n_batch),
                torch.exp(self.background_pro_log_beta),
            )
        else:
            py_back_mean_prior = self.background_pro_alpha
            py_back_var_prior = torch.exp(self.background_pro_log_beta)
        self.back_mean_prior = Normal(py_back_mean_prior, py_back_var_prior)

        return enc_gene_outputs, enc_pro_outputs

    def self_expressiveness(
        self,
        z_genes: torch.Tensor,
        z_proteins: torch.Tensor,
        ref_z_genes: torch.Tensor,
        ref_z_proteins: torch.Tensor,
    ):
        """
        Use self-expressive network to reconstruct the latent space
        Parameters
        ----------
        z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        z_proteins
            tensor of values with shape ``(batch_size, n_latent_proteins)``
        ref_z_genes
            tensor of values with shape ``(training_size, n_latent_genes)``
        ref_z_proteins
            tensor of values with shape ``(training_size, n_latent_proteins)``

        Returns
        -------
        rec_z_genes
            tensor of values with shape ``(batch_size, n_latent_genes)``
        rec_z_proteins
            tensor of values with shape ``(batch_size, n_latent_proteins)``
        se_gene_outputs
            genes dict with "rec_queries" and "c_list"
        se_pro_outputs
            genes dict with "rec_queries" and "c_list"
        """

        se_gene_outputs = self.senet_genes(z_genes, ref_z_genes)
        se_pro_outputs = self.senet_proteins(z_proteins, ref_z_proteins)
        return se_gene_outputs, se_pro_outputs

    def generative(
        self,
        z_genes: torch.Tensor,
        z_proteins: torch.Tensor,
        library_gene: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        cont_covs=None,
        cat_covs=None,
        transform_batch: Optional[int] = None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_index = batch_index.to(device)
        if cat_covs is not None:
            cat_covs = cat_covs.to(device)
        if cont_covs is not None:
            decoder_input_genes = torch.cat((z_genes, cont_covs), dim=-1)
            decoder_input_proteins = torch.cat((z_proteins, cont_covs), dim=-1)
        else:
            decoder_input_genes = z_genes
            decoder_input_proteins = z_proteins
        # cat_covs = batch_index
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
            if torch.cuda.is_available():
                batch_index = batch_index.to('cuda')

        px_ = self.decoder_genes(
            decoder_input_genes, library_gene, batch_index, *categorical_input
        )
        py_ = self.decoder_proteins(
            decoder_input_proteins, batch_index, *categorical_input
        )

        if self.gene_dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(one_hot(label, self.n_labels), self.px_r)
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_["r"] = torch.exp(px_r)

        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(one_hot(label, self.n_labels), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(one_hot(batch_index, self.n_batch), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_["r"] = torch.exp(py_r)

        return px_, py_

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        px_dict: Dict[str, torch.Tensor],
        py_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction loss."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        px_ = px_dict
        py_ = py_dict
        # Reconstruction Loss
        if self.gene_likelihood == "zinb":
            reconst_loss_gene = (
                -ZeroInflatedNegativeBinomial(
                    mu=px_["rate"], theta=px_["r"], zi_logits=px_["dropout"]
                )
                .log_prob(x)
                .sum(dim=-1)
            )
        else:
            reconst_loss_gene = (
                -NegativeBinomial(mu=px_["rate"], theta=px_["r"])
                .log_prob(x)
                .sum(dim=-1)
            )

        if self.protein_likelihood == "nbm":
            py_conditional = NegativeBinomialMixture(
                mu1=py_["back_rate"],
                mu2=py_["fore_rate"],
                theta1=py_["r"],
                mixture_logits=py_["mixing"],
            )
        elif self.protein_likelihood == "nb":
            py_conditional = NegativeBinomial(mu=py_['rate'], theta=py_["r"])
        else:
            py_conditional = Poisson(py_['rate'])

        reconst_loss_protein_full = -py_conditional.log_prob(y)
        reconst_loss_protein = reconst_loss_protein_full.sum(dim=-1)

        return reconst_loss_gene, reconst_loss_protein

    def get_self_expressiveness_loss(
        self,
        inference_outputs,
        senet_outputs,
        lmbd=0.9,
    ):
        """Compute self-expressiveness loss."""
        z_gene, z_protein = inference_outputs
        # z_gene = enc_gene_outputs['z']
        # z_protein = enc_pro_outputs['z']
        se_gene_outputs, se_pro_outputs = senet_outputs
        rec_z_gene, coeff_gene = se_gene_outputs['rec_queries'], se_gene_outputs['coeff']
        rec_z_protein, coeff_protein = se_pro_outputs['rec_queries'], se_pro_outputs['coeff']
        se_loss_gene = torch.sum(torch.pow(z_gene - rec_z_gene, 2), dim=-1)
        se_loss_protein = torch.sum(torch.pow(z_protein - rec_z_protein, 2), dim=-1)

        c_reg_gene = regularizer(coeff_gene, lmbd).sum(dim=-1)
        c_reg_protein = regularizer(coeff_protein, lmbd).sum(dim=-1)
        # c_reg_genes = regularizer_l12(coeff_gene, lmbd) * torch.ones(coeff_gene.shape[0])
        # c_reg_proteins = regularizer_l12(coeff_protein, lmbd) * torch.ones(coeff_protein.shape[0])
        c_contrast = torch.sum(torch.pow(coeff_gene - coeff_protein, 2), dim=-1)

        return se_loss_gene, se_loss_protein, c_reg_gene, c_reg_protein, c_contrast

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        inference_outputs,
        senet_outputs,
        generative_outputs,
        batch_index: Optional[torch.Tensor] = None,
        pro_recons_weight=1.0,  # double check these defaults
        kl_weight=1.0,
        se_weight=1.0,
    ):
        """
        Returns the reconstruction loss and the Kullback-Leibler divergences.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_input_genes)``
        y
            tensor of values with shape ``(batch_size, n_input_proteins)``
        inference_outputs
            inference network outputs
        senet_outputs
            self-expressive network outputs
        generative_outputs
            generative network outputs
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        pro_recons_weight
            proteins reconstruction weight
        kl_weight
            Kullback-Leibler divergences loss weight
        se_weight
            self-expressive loss weight
        Returns
        -------
        type
            the reconstruction loss, the KL divergences and self-expressive loss
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        y = y.to(device)
        batch_index = batch_index.to(device)
        enc_gene_outputs, enc_pro_outputs = inference_outputs
        qz_m_gene = enc_gene_outputs['qz_m']
        qz_v_gene = enc_gene_outputs['qz_v']
        ql_m_gene = enc_gene_outputs['ql_m']
        ql_v_gene = enc_gene_outputs['ql_v']
        qz_m_protein = enc_pro_outputs['qz_m']
        qz_v_protein = enc_pro_outputs['qz_v']
        px_, py_ = generative_outputs

        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x, y, px_, py_,
        )
        inference_outputs = (x, y)
        se_loss_gene, se_loss_protein, c_reg_genes, c_reg_proteins, c_contrast = self.get_self_expressiveness_loss(
            inference_outputs, senet_outputs, lmbd=0.95
        )

        # KL Divergence
        kl_div_z_gene = kl(Normal(qz_m_gene, torch.sqrt(qz_v_gene)), Normal(0, 1)).sum(dim=1)
        kl_div_z_protein = kl(Normal(qz_m_protein, torch.sqrt(qz_v_protein)), Normal(0, 1)).sum(dim=1)
        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                one_hot(batch_index, n_batch), self.library_log_means
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index, n_batch), self.library_log_vars
            )
            kl_div_l_gene = kl(
                Normal(ql_m_gene, torch.sqrt(ql_v_gene)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_div_l_gene = 0.0 * torch.ones(x.shape[0])
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            kl_div_l_gene = kl_div_l_gene.to(device)

        if self.protein_likelihood == "nbm":
            kl_div_back_pro_full = kl(
                Normal(py_["back_log_rate_mean"], py_["back_log_rate_var"].sqrt()), self.back_mean_prior
            )
            kl_div_back_pro = kl_div_back_pro_full.sum(dim=-1)
        else:
            kl_div_back_pro = 0.0 * torch.ones(x.shape[0])
        losses = torch.mean(
            # reconst_loss_gene
            # + pro_recons_weight * reconst_loss_protein
            # + kl_weight * kl_div_z_gene
            # + kl_weight * kl_div_z_protein
            # + kl_div_l_gene
            # + kl_weight * kl_div_back_pro
            se_weight * (se_loss_gene + c_reg_genes)
            + se_weight * (se_loss_protein + c_reg_proteins)
            + se_weight * c_contrast * 0.1
        )

        reconst_losses = dict(
            reconst_loss_gene=reconst_loss_gene,
            reconst_loss_protein=reconst_loss_protein,
        )
        kl_local = dict(
            kl_div_z_gene=kl_div_z_gene,
            kl_div_l_gene=kl_div_l_gene,
            kl_div_z_protein=kl_div_z_protein,
            kl_div_back_pro=kl_div_back_pro,
        )
        se_losses = dict(
            se_loss_gene=se_loss_gene,
            se_loss_protein=se_loss_protein,
            c_reg_genes=c_reg_genes,
            c_reg_proteins=c_reg_proteins,
            c_contrast=c_contrast
        )

        return losses, reconst_losses, kl_local, se_losses

    def forward(
        self,
        input_genes: torch.Tensor,
        input_proteins: torch.Tensor,
        refer_adata,
        batch_index: Optional[torch.Tensor] = None,
        cat_covs: Optional[torch.Tensor] = None,
        compute_loss=True,
        transform_batch: Optional[int] = None,
    ):
        """
        Forward pass through the network.

        Parameters
        ----------
        self
        input_genes
            tensor with shape ``(batch, n_input_gens)``
        input_proteins
            tensor with shape ``(batch, n_input_proteins)``
        refer_adata
            adata with shape ``(training_size, )``
        batch_index
            array that indicates which batch the cells belong to with shape ``batch_size``
        cat_covs
            categorical variable of each cell
        compute_loss
            whether compute losses
        transform_batch
            batch to condition on
        """
        ref_z_genes_list = []
        ref_z_proteins_list = []
        for data in refer_adata:
            refer_genes = torch.tensor(data.layers["rna_expression"], dtype=torch.float32)
            refer_proteins = torch.tensor(data.obsm['protein_expression'], dtype=torch.float32)
            refer_index = torch.tensor(data.obs['batch'], dtype=torch.float32).reshape(-1, 1)
            if "cat_covs" in data.obsm.keys():
                refer_cat_covs = torch.tensor(data.obsm['cat_covs'], dtype=torch.float32).reshape(-1, 1)
            else:
                refer_cat_covs = None
            # refer_index = (refer_index, refer_index)
            enc_ref_gene_outputs, enc_ref_protein_outputs = self.inference(refer_genes, refer_proteins, batch_index=refer_index, cat_covs=refer_cat_covs)
            # ref_z_genes_list.append(enc_ref_gene_outputs['z'])
            # ref_z_proteins_list.append(enc_ref_protein_outputs['z'])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            refer_genes = refer_genes.to(device)
            refer_proteins = refer_proteins.to(device)
            ref_z_genes_list.append(torch.log(1 + refer_genes))
            ref_z_proteins_list.append(torch.log(1 + refer_proteins))
        ref_z_genes = torch.cat(ref_z_genes_list, dim=0)
        ref_z_proteins = torch.cat(ref_z_proteins_list, dim=0)

        inference_outputs = self.inference(input_genes, input_proteins, batch_index=batch_index, cat_covs=cat_covs)
        enc_gene_outputs, enc_pro_outputs = inference_outputs

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_genes = input_genes.to(device)
        input_proteins = input_proteins.to(device)

        # z_genes = enc_gene_outputs['z']
        z_genes = torch.log(1 + input_genes)
        library_gene = enc_gene_outputs['library_gene']
        # z_proteins = enc_pro_outputs['z']
        z_proteins = torch.log(1 + input_proteins)
        senet_outputs = self.self_expressiveness(z_genes, z_proteins, ref_z_genes, ref_z_proteins)
        se_gene_outputs, se_pro_outputs = senet_outputs
        rec_z_genes = se_gene_outputs['rec_queries']
        rec_z_proteins = se_pro_outputs['rec_queries']
        generative_outputs = self.generative(rec_z_genes, rec_z_proteins, library_gene, batch_index=batch_index, cat_covs=cat_covs, transform_batch=transform_batch)
        if compute_loss:
            loss = self.loss(
                z_genes, z_proteins, inference_outputs, senet_outputs, generative_outputs, batch_index
            )
            return inference_outputs, senet_outputs, generative_outputs, loss
        else:
            return inference_outputs, senet_outputs, generative_outputs