import collections
from typing import Callable, Iterable, List, Optional, Literal

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal
from torch.nn import ModuleList
from torch.nn import init
from .utils import one_hot


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


def identity(x):
    return x


class FCLayers(nn.Module):
    """
    A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn=nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                    zip(layers_dim[:-1], layers_dim[1:])
                )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """
        Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(batch_size, n_in)``
        cat_list
            list of category membership(s) for this sample
        x: torch.Tensor

        Returns
        -------
        py:class:`torch.Tensor`
            tensor of shape ``(n_out,)``

        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list):
            raise ValueError(
                "nb. categorical args provided doesn't match init. params."
            )
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand(
                                        (x.size(0), o.size(0), o.size(1))
                                    )
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                        x = layer(x)
        return x


# Encoder

# Encoder
class Encoder(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        When `None`, defaults to `torch.exp`.
    **kwargs
        Keyword args for :class:`~scvi.module._base.FCLayers`
    """

    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_cat_list: Iterable[int] = None,
            n_layers: int = 1,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            distribution: str = "normal",
            var_eps: float = 1e-4,
            var_activation: Optional[Callable] = None,
            **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (batch, n_input)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        latent = self.z_transformation(reparameterize_gaussian(q_m, q_v))
        return q_m, q_v, latent


class EncoderGene(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input_genes
        The dimensionality of the input (gene space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of the latent space, one of

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm
    """

    def __init__(
        self,
        n_input_genes: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout_rate: float = 0.1,
        distribution: str = "ln",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.gene_encoder = FCLayers(
            n_in=n_input_genes,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_gene_encoder = nn.Linear(n_hidden, n_output)
        self.z_var_gene_encoder = nn.Linear(n_hidden, n_output)

        self.l_gene_encoder = FCLayers(
            n_in=n_input_genes,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.l_mean_gene_encoder = nn.Linear(n_hidden, 1)
        self.l_var_gene_encoder = nn.Linear(n_hidden, 1)

        self.distribution = distribution

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

        self.l_transformation = torch.exp

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(self, data_gene: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. latent distribution

        The dictionary ``latent`` contains the samples of the latent variables, while ``untran_latent``
        contains the untransformed versions of these latent variables. For example, the library size is log normally distributed,
        so ``untran_latent["l"]`` gives the normal sample that was later exponentiated to become ``latent["l"]``.
        The logistic normal distribution is equivalent to applying softmax to a normal sample.

        Parameters
        ----------
        data_gene
            tensor with shape ``(batch_size, n_input_genes)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple. `dict` of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for gene latent distribution
        qz = self.gene_encoder(data_gene, *cat_list)
        qz_m = self.z_mean_gene_encoder(qz)
        qz_v = torch.exp(self.z_var_gene_encoder(qz)) + 1e-4
        z, untran_z = self.reparameterize_transformation(qz_m, qz_v)

        ql = self.l_gene_encoder(data_gene, *cat_list)
        ql_m = self.l_mean_gene_encoder(ql)
        ql_v = torch.exp(self.l_var_gene_encoder(ql)) + 1e-4
        log_library_gene = torch.clamp(reparameterize_gaussian(ql_m, ql_v), max=15)
        library_gene = self.l_transformation(log_library_gene)

        return dict(
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z,
            ql_m=ql_m,
            ql_v=ql_v,
            library_gene=library_gene,
            untran_l=log_library_gene,
        )


class EncoderProtein(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input_proteins
        The dimensionality of the input (proteins space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of the latent space, one of

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm
    """

    def __init__(
            self,
            n_input_proteins: int,
            n_output: int,
            n_cat_list: Iterable[int] = None,
            n_layers: int = 2,
            n_hidden: int = 128,
            dropout_rate: float = 0.1,
            distribution: str = "ln",
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
    ):
        super().__init__()

        self.protein_encoder = FCLayers(
            n_in=n_input_proteins,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_protein_encoder = nn.Linear(n_hidden, n_output)
        self.z_var_protein_encoder = nn.Linear(n_hidden, n_output)

        self.distribution = distribution

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(self, data_protein: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. latent distribution

        The dictionary ``latent`` contains the samples of the latent variables, while ``untran_latent``
        contains the untransformed versions of these latent variables. For example, the library size is log normally distributed,
        so ``untran_latent["z"]`` gives the normal sample that was later softmax to become ``latent["z"]``.
        The logistic normal distribution is equivalent to applying softmax to a normal sample.

        Parameters
        ----------
        data_protein
            tensor with shape ``(batch_size, n_input_proteins)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple. `dict` of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for protein latent distribution
        qz = self.protein_encoder(data_protein, *cat_list)
        qz_m = self.z_mean_protein_encoder(qz)
        qz_v = torch.exp(self.z_var_protein_encoder(qz)) + 1e-4
        # qz_m = torch.clamp(qz_m, min=-20, max=20)
        # print(qz_m.isnan().sum(), qz_v.isnan().sum())
        z, untran_z = self.reparameterize_transformation(qz_m, qz_v)

        return dict(
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z
        )


class EncoderPeak(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input_peaks
        The dimensionality of the input (peak space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of the latent space, one of

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm
    """

    def __init__(
        self,
        n_input_peaks: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout_rate: float = 0.1,
        distribution: str = "ln",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.z_peak_encoder = FCLayers(
            n_in=n_input_peaks,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            activation_fn=nn.LeakyReLU
        )
        self.z_mean_peak_encoder = nn.Linear(n_hidden, n_output)
        self.z_var_peak_encoder = nn.Linear(n_hidden, n_output)

        self.l_peak_encoder = FCLayers(
            n_in=n_input_peaks,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0.0,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            activation_fn=nn.LeakyReLU
        )
        self.l_peak = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, 1), torch.nn.Sigmoid()
        )

        self.distribution = distribution

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(self, data_peak: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. latent distribution

        The dictionary ``latent`` contains the samples of the latent variables, while ``untran_latent``
        contains the untransformed versions of these latent variables. For example, the library size is log normally distributed,
        so ``untran_latent["l"]`` gives the normal sample that was later exponentiated to become ``latent["l"]``.
        The logistic normal distribution is equivalent to applying softmax to a normal sample.

        Parameters
        ----------
        data_peak
            tensor with shape ``(batch_size, n_input_peaks)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple. `dict` of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for peank latent distribution
        qz = self.z_peak_encoder(data_peak, *cat_list)
        qz_m = self.z_mean_peak_encoder(qz)
        qz_v = torch.exp(self.z_var_peak_encoder(qz)) + 1e-4
        z, untran_z = self.reparameterize_transformation(qz_m, qz_v)

        ql = self.l_peak_encoder(data_peak, *cat_list)
        lib = self.l_peak(ql)

        return dict(
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z,
            lib=lib,
        )


class EncoderImage(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input_genes
        The dimensionality of the input (gene space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of the latent space, one of

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm
    """

    def __init__(
        self,
        n_input_genes: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout_rate: float = 0.1,
        distribution: str = "ln",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.gene_encoder = FCLayers(
            n_in=n_input_genes,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_gene_encoder = nn.Linear(n_hidden, n_output)
        self.z_var_gene_encoder = nn.Linear(n_hidden, n_output)

        self.distribution = distribution

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

        self.l_transformation = torch.exp

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(self, data_gene: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. latent distribution

        The dictionary ``latent`` contains the samples of the latent variables, while ``untran_latent``
        contains the untransformed versions of these latent variables. For example, the library size is log normally distributed,
        so ``untran_latent["l"]`` gives the normal sample that was later exponentiated to become ``latent["l"]``.
        The logistic normal distribution is equivalent to applying softmax to a normal sample.

        Parameters
        ----------
        data_gene
            tensor with shape ``(batch_size, n_input_genes)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple. `dict` of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for gene latent distribution
        qz = self.gene_encoder(data_gene, *cat_list)
        qz_m = self.z_mean_gene_encoder(qz)
        qz_v = torch.exp(self.z_var_gene_encoder(qz)) + 1e-4
        z, untran_z = self.reparameterize_transformation(qz_m, qz_v)

        return dict(
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            untran_z=untran_z,
        )


class DecoderGene(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output`` dimensions.

    Uses a linear decoder.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output_genes
        The dimensionality of the output (gene space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output_genes: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.n_output_genes = n_output_genes

        linear_args = dict(
            n_layers=1,
            use_activation=False,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
        )

        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean gamma
        self.px_scale_decoder = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_genes,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        # dropout (ZI probability for genes)
        self.px_dropout_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.px_dropout_decoder_gene = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_genes,
            n_cat_list=n_cat_list,
            **linear_args,
        )

    def forward(self, z: torch.Tensor, library_gene: torch.Tensor, *cat_list: int):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns local parameters for the ZINB distribution for genes

         We use the dictionary `px_` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quantity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

        Parameters
        ----------
        z
            tensor with shape ``(batch_size, n_input)``
        library_gene
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        1-tuple (px:class:`dict`)
            parameters for the ZINB distribution of expression

        """
        px_ = {}

        px = self.px_decoder(z, *cat_list)
        # px_cat_z = torch.cat([px, z], dim=-1)
        unnorm_px_scale = self.px_scale_decoder(px, *cat_list)
        px_["scale"] = nn.Softmax(dim=-1)(unnorm_px_scale)
        px_["rate"] = library_gene * px_["scale"]

        px_dropout = self.px_dropout_decoder(z, *cat_list)
        # px_dropout_cat_z = torch.cat([px_dropout, z], dim=-1)
        px_["dropout"] = self.px_dropout_decoder_gene(px_dropout, *cat_list)

        return px_


class DecoderSCVI(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output`` dimensions.

    Uses a linear decoder.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output_genes
        The dimensionality of the output (gene space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        dropout_rate: float = 0,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.n_output = n_output

        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=inject_covariates,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=-1),
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, z: torch.Tensor, library_gene: torch.Tensor, *cat_list: int):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns local parameters for the ZINB distribution for genes

         We use the dictionary `px_` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quantity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

        Parameters
        ----------
        z
            tensor with shape ``(batch_size, n_input)``
        library_gene
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        1-tuple (px:class:`dict`)
            parameters for the ZINB distribution of expression

        """
        px_ = {}

        px = self.px_decoder(z, *cat_list)
        px_["scale"] = self.px_scale_decoder(px)
        px_["dropout"] = self.px_dropout_decoder(px)
        px_["rate"] = library_gene * px_["scale"]

        return px_


class DecoderProtein(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output`` dimensions.

    Uses a linear decoder.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output_proteins
        The dimensionality of the output (protein space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
            self,
            n_input: int,
            n_output_proteins: int,
            n_cat_list: Iterable[int] = None,
            n_layers: int = 1,
            n_hidden: int = 256,
            dropout_rate: float = 0,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
    ):
        super().__init__()
        self.n_output_proteins = n_output_proteins

        linear_args = dict(
            n_layers=1,
            use_activation=False,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
        )

        # background mean first decoder
        self.py_back_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        # background mean parameters second decoder
        self.py_back_log_rate_mean = FCLayers(
            n_in=n_hidden,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )
        self.py_back_log_rate_var = FCLayers(
            n_in=n_hidden,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        # foreground increment decoder step 1
        self.py_fore_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        # foreground increment decoder step 2
        self.py_fore_scale_decoder = FCLayers(
            n_in=n_hidden,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
            # activation_fn=nn.ReLU,
        )

        # dropout (mixing component for proteins)
        self.py_mixing_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.py_mixing_decoder_protein = FCLayers(
            n_in=n_hidden,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

    def forward(self, z: torch.Tensor, *cat_list: int):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns local parameters for the Mixture NB distribution for proteins

         We use the dictionary `py_` to contain the parameters of the Mixture NB distribution for proteins.
         `fore_rate` refers to foreground mean, while `back_rate` refers to background mean. `scale` refers to
         foreground mean adjusted for background probability and scaled to reside in simplex.
         `back_rate_mu` and `back_rate_var` are the posterior parameters for `back_rate`.  `fore_scale` is the scaling
         factor that enforces `fore_rate` > `back_rate`.

        Parameters
        ----------
        z
            tensor with shape ``(batch_size, n_input)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple (first 1-tuple :py:class:`dict`, last :py:class:`torch.Tensor`)
            parameters for the mixture NB distribution of expression

        """
        py_ = {}

        py_back = self.py_back_decoder(z, *cat_list)
        # py_back_cat_z = torch.cat([py_back, z], dim=-1)

        back_log_rate_mean = self.py_back_log_rate_mean(py_back, *cat_list)
        py_["rate"] = torch.exp(torch.clamp(back_log_rate_mean, max=12))  # NB or Poisson likelihood mu
        py_["back_log_rate_mean"] = torch.clamp(back_log_rate_mean, max=12)
        py_["back_log_rate_var"] = torch.exp(self.py_back_log_rate_var(py_back, *cat_list))
        py_["back_log_rate"] = Normal(py_["back_log_rate_mean"], py_["back_log_rate_var"].sqrt()).rsample()
        py_["back_rate"] = torch.exp(py_["back_log_rate"])

        # py_fore = self.py_fore_decoder(z, *cat_list)
        # py_fore_cat_z = torch.cat([py_fore, z], dim=-1)
        py_["fore_scale"] = (self.py_fore_scale_decoder(py_back, *cat_list) + 1 + 1e-8)
        py_["fore_rate"] = py_["back_rate"] * py_["fore_scale"]

        # py_mixing = self.py_mixing_decoder(z, *cat_list)
        # py_mixing_cat_z = torch.cat([py_mixing, z], dim=-1)
        py_['mixing'] = self.py_mixing_decoder_protein(py_back, *cat_list)

        protein_mixing = 1 / (1 + torch.exp(-py_['mixing']))  # nn.Sigmoid(py_["mixing"])
        py_["scale"] = torch.nn.functional.normalize((1 - protein_mixing) * py_["fore_rate"], p=1, dim=-1)

        return py_


class DecoderPeak(torch.nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers. If False (default),
        covairates will only be included in the input layer.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            activation_fn=nn.LeakyReLU,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=deep_inject_covariates,
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_output), torch.nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, *cat_list: int):
        px_ = {'prob': self.output(self.px_decoder(z, *cat_list))}
        return px_


class DecoderImage(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output`` dimensions.

    Uses a linear decoder.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output_genes
        The dimensionality of the output (gene space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        dropout_rate: float = 0,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.n_output = n_output

        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=inject_covariates,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, z: torch.Tensor, *cat_list: int):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns local parameters for the ZINB distribution for genes

         We use the dictionary `px_` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quantity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

        Parameters
        ----------
        z
            tensor with shape ``(batch_size, n_input)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        1-tuple (px:class:`dict`)
            parameters for the ZINB distribution of expression

        """
        px_ = {}

        px = self.px_decoder(z, *cat_list)
        px_["scale"] = self.px_scale_decoder(px)

        return px_


class EncoderSCMSI(nn.Module):
    """
    Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input_genes
        The dimensionality of the input (gene space)
    n_input_proteins
        The dimensionality of the input (proteins space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of the latent space, one of

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm
    """

    def __init__(
            self,
            n_input_genes: int,
            n_input_proteins: int,
            n_output: int,
            n_cat_list: Iterable[int] = None,
            n_layers: int = 2,
            n_hidden: int = 256,
            dropout_rate: float = 0.1,
            distribution: str = "ln",
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
    ):
        super().__init__()

        self.gene_encoder = FCLayers(
            n_in=n_input_genes,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_gene_encoder = nn.Linear(n_hidden, n_output)
        self.z_var_gene_encoder = nn.Linear(n_hidden, n_output)

        self.l_gene_encoder = FCLayers(
            n_in=n_input_genes,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.l_gene_mean_encoder = nn.Linear(n_hidden, 1)
        self.l_gene_var_encoder = nn.Linear(n_hidden, 1)

        self.distribution = distribution

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = identity

        self.l_transformation = torch.exp

        self.protein_encoder = FCLayers(
            n_in=n_input_proteins,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_protein_encoder = nn.Linear(n_hidden, n_output)
        self.z_var_protein_encoder = nn.Linear(n_hidden, n_output)

    def reparameterize_transformation(self, mu, var):
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(self, data_gene: torch.Tensor, data_protein: torch.Tensor, *cat_list: int):
        r"""
        The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. latent distribution

        The dictionary ``latent`` contains the samples of the latent variables, while ``untran_latent``
        contains the untransformed versions of these latent variables. For example, the library size is log normally distributed,
        so ``untran_latent["l"]`` gives the normal sample that was later exponentiated to become ``latent["l"]``.
        The logistic normal distribution is equivalent to applying softmax to a normal sample.

        Parameters
        ----------
        data_gene
            tensor with shape ``(batch_size, n_input_genes)``
        data_protein
            tensor with shape ``(batch_size, n_input_proteins)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple. `dict` of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for gene latent distribution
        q_gene = self.gene_encoder(data_gene, *cat_list)
        qz_m_gene = self.z_mean_gene_encoder(q_gene)
        qz_v_gene = torch.exp(self.z_var_gene_encoder(q_gene)) + 1e-4
        z_gene, untran_z_gene = self.reparameterize_transformation(qz_m_gene, qz_v_gene)

        ql_gene = self.l_gene_encoder(data_gene, *cat_list)
        ql_m = self.l_gene_mean_encoder(ql_gene)
        ql_v = torch.exp(self.l_gene_var_encoder(ql_gene)) + 1e-4
        log_library_gene = torch.clamp(reparameterize_gaussian(ql_m, ql_v), max=15)
        library_gene = self.l_transformation(log_library_gene)

        # Parameters for protein latent distribution
        q_protein = self.protein_encoder(data_protein, *cat_list)
        qz_m_protein = self.z_mean_protein_encoder(q_protein)
        qz_v_protein = torch.exp(self.z_var_protein_encoder(q_protein)) + 1e-4
        z_protein, untran_z_protein = self.reparameterize_transformation(qz_m_protein, qz_v_protein)

        latent = {}
        untran_latent = {}
        latent["z_gene"] = z_gene
        latent["l_gene"] = library_gene
        latent["z_protein"] = z_protein
        untran_latent["z_gene"] = untran_z_gene
        untran_latent["l_gene"] = log_library_gene
        untran_latent["untran_z_protein"] = untran_z_protein

        return latent, untran_latent


class DecoderSCMSI(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output`` dimensions.

    Uses a linear decoder.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output_genes
        The dimensionality of the output (gene space)
    n_output_proteins
        The dimensionality of the output (protein space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    scale_activation
        Activation layer to use for px_scale_decoder
    """

    def __init__(
            self,
            n_input: int,
            n_output_genes: int,
            n_output_proteins: int,
            n_cat_list: Iterable[int] = None,
            n_layers: int = 1,
            n_hidden: int = 256,
            dropout_rate: float = 0,
            use_batch_norm: bool = True,
            use_layer_norm: bool = False,
            scale_activation: Literal["softmax", "softplus"] = "softmax",
    ):
        super().__init__()
        self.n_output_genes = n_output_genes
        self.n_output_proteins = n_output_proteins

        linear_args = dict(
            n_layers=1,
            use_activation=False,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
        )

        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean gamma
        self.px_scale_decoder = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_genes,
            n_cat_list=n_cat_list,
            **linear_args,
        )
        if scale_activation == "softmax":
            self.px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            self.px_scale_activation = nn.Softplus()

        # background mean first decoder
        self.py_back_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        # background mean parameters second decoder
        self.py_back_rate_log_mean = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )
        self.py_back_rate_log_var = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        # foreground increment decoder step 1
        self.py_fore_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        # foreground increment decoder step 2
        self.py_fore_scale_decoder = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
            # activation_fn=nn.ReLU,
        )

        # dropout (mixing component for proteins, ZI probability for genes)
        self.px_dropout_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.px_dropout_decoder_gene = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_genes,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        self.py_mixing_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.py_mixing_decoder_protein = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

    def forward(self, z_gene: torch.Tensor, z_protein: torch.Tensor, library_gene: torch.Tensor, *cat_list: int):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns local parameters for the ZINB distribution for genes
         #. Returns local parameters for the Mixture NB distribution for proteins

         We use the dictionary `px_` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quanity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

         We use the dictionary `py_` to contain the parameters of the Mixture NB distribution for proteins.
         `rate_fore` refers to foreground mean, while `rate_back` refers to background mean. `scale` refers to
         foreground mean adjusted for background probability and scaled to reside in simplex.
         `back_rate_mean` and `back_rate_var` are the posterior parameters for `rate_back`.  `fore_scale` is the scaling
         factor that enforces `rate_fore` > `rate_back`.

        Parameters
        ----------
        z_gene
            tensor with shape ``(batch_size, n_latent)``
        z_protein
            tensor with shape ``(batch_size, n_latent)``
        library_gene
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple (first 2-tuple :py:class:`dict`, last :py:class:`torch.Tensor`)
            parameters for the ZINB distribution of expression

        """
        px_ = {}
        py_ = {}

        px = self.px_decoder(z_gene, *cat_list)
        px_cat_z = torch.cat([px, z_gene], dim=-1)
        unnorm_px_scale = self.px_scale_decoder(px_cat_z, *cat_list)
        px_["scale"] = self.px_scale_activation(unnorm_px_scale)
        px_["rate"] = library_gene * px_["scale"]
        px_dropout = self.dropout_decoder(z_gene, *cat_list)
        px_dropout_cat_z = torch.cat([px_dropout, z_gene], dim=-1)
        px_["dropout"] = self.px_dropout_decoder_gene(px_dropout_cat_z, *cat_list)

        py_back = self.py_back_decoder(z_protein, *cat_list)
        py_back_cat_z = torch.cat([py_back, z_protein], dim=-1)

        py_["back_rate_log_mean"] = self.py_back_rate_log_mean(py_back_cat_z, *cat_list)
        py_["back_rate_var"] = torch.exp(self.py_back_mean_log_beta(py_back_cat_z, *cat_list))
        py_["back_log_rate"] = Normal(py_["back_rate_log_mean"], py_["back_rate_var"]).rsample()
        py_["rate_back"] = torch.exp(py_["back_log_rate"])

        py_fore = self.py_fore_decoder(z_protein, *cat_list)
        py_fore_cat_z = torch.cat([py_fore, z_protein], dim=-1)
        py_["fore_scale"] = (self.py_fore_scale_decoder(py_fore_cat_z, *cat_list) + 1 + 1e-8)
        py_["fore_rate"] = py_["back_rate"] * py_["fore_scale"]

        py_mixing = self.py_mixing_decoder(z_protein, *cat_list)
        py_mixing_cat_z = torch.cat([py_mixing, z_protein], dim=-1)
        py_["mixing"] = self.py_mixing_decoder_protein(py_mixing_cat_z, *cat_list)

        protein_mixing = 1 / (1 + torch.exp(-py_["mixing"]))
        py_["scale"] = torch.nn.functional.normalize(
            (1 - protein_mixing) * py_["fore_rate"], p=1, dim=-1
        )

        return px_, py_


class MLP(nn.Module):

    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=False):
        super(MLP, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.output_dims = out_dims
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dims, self.hid_dims[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(self.hid_dims[i], self.hid_dims[i + 1]))
            self.layers.append(nn.ReLU())

        self.out_layer = nn.Linear(self.hid_dims[-1], self.output_dims)
        if kaiming_init:
            self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.out_layer.weight)
        init.zeros_(self.out_layer.bias)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        h = self.out_layer(h)
        h = torch.tanh_(h)
        return h


class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))

    def forward(self, c):
        # a = torch.bernoulli(torch.tensor(0.1))
        # if a == 1:
        #     print(self.bias)
        # torch.relu(torch.abs(c) - self.bias)
        return torch.relu(c) * torch.relu(torch.abs(c) - self.bias)
        # torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class SENet0(nn.Module):
    """
    Learning a Self-Expressive Network from latent space for subspace clustering.
    Modified from https://github.com/zhangsz1998/Self-Expressive-Network.

    Parameters
    ----------
    input_dims
        The dimensionality of the input (latent space)
    hid_dims
        The dimensionality of the hidden layer (hidden space)
    out_dims
        The dimensionality of the output (output space)
    kaiming_init
        Whether to use kaiming initialization in layers
    """

    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=True):
        super(SENet0, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.kaiming_init = kaiming_init
        self.shrink = 1.0 / out_dims

        self.net_q = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.net_k = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.thres = AdaptiveSoftThreshold(1)

    def query_embedding(self, queries):
        q_emb = self.net_q(queries)
        return q_emb

    def key_embedding(self, keys):
        k_emb = self.net_k(keys)
        return k_emb

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c

    def forward(self, queries, keys):
        q = self.query_embedding(queries)
        # k = self.key_embedding(keys)
        k = self.query_embedding(keys)
        out = self.get_coeff(q_emb=q, k_emb=k)
        return out


class SENet(nn.Module):
    """
    Learning a Self-Expressive Network from latent space for subspace clustering.
    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (hidden space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super(SENet, self).__init__()
        self.shrink = 1.0 / n_output
        # self.shrink = 1.0
        self.net_emb = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.net_q = FCLayers(
            n_in=n_hidden,
            n_out=n_output,
            n_layers=1,
            n_hidden=n_hidden,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
            activation_fn=nn.SiLU
        )

        self.net_k = FCLayers(
            n_in=n_hidden,
            n_out=n_output,
            n_layers=1,
            n_hidden=n_hidden,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
            activation_fn=nn.SiLU
        )

        self.thres = AdaptiveSoftThreshold(1)

    def query_embedding(self, queries):
        emb = self.net_emb(queries)
        # q_emb = emb
        q_emb = self.net_q(emb)
        return q_emb

    def key_embedding(self, keys):
        emb = self.net_emb(keys)
        k_emb = self.net_k(emb)
        # k_emb = self.net_q(emb)
        return k_emb

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c

    def get_rec_queries(self, queries, keys):
        N = keys.shape[0]
        n_block = N // 5000 + 1
        block_size = min(N, 5000)
        q_queries = self.query_embedding(queries)
        # k_queries = self.key_embedding(queries)
        rec_queries = torch.zeros_like(queries)  # set cuda
        c_list = []
        for i in range(n_block):
            if i < n_block - 1:
                block = keys[i * block_size: (i + 1) * block_size]  # set cuda
            else:
                block = keys[i * block_size:]
            k_block = self.key_embedding(block)
            c = self.get_coeff(q_queries, k_block)
            rec_queries = rec_queries + c.mm(block)
            c_list.append(c)
        # diag_c = self.thres((q_queries*k_queries).sum(dim=1, keepdim=True)) * self.shrink
        # rec_queries = rec_queries - diag_c * queries
        return rec_queries, torch.cat(c_list, dim=-1)

    def forward(self, queries, anchors):
        """

        Parameters
        ----------
        queries
            tensor of values with shape ``(batch_size, n_latent)``
        anchors
            tensor of values with shape ``(anchor_size, n_latent)``

        Returns
        -------
        rec_queries
            tensor of values with shape ``(batch_size, n_latent)``
        coeff
            self-expressive coefficients with shape ``(batch_size, ref_n_samples)``
        """
        # N = keys.shape[0]
        # n_block = N // 5000 + 1
        # block_size = min(N, 5000)
        queries_emb = self.query_embedding(queries)
        # k_queries = self.key_embedding(queries)
        # rec_queries = torch.zeros_like(queries)  # set cuda
        # c_list = []
        # for i in range(n_block):
        #     if i < n_block - 1:
        #         block = keys[i * block_size: (i + 1) * block_size]  # set cuda
        #     else:
        #         block = keys[i * block_size:]
        # anchors_emb = self.key_embedding(anchors)
        anchors_emb = self.query_embedding(anchors)
        coeff = self.get_coeff(queries_emb, anchors_emb)
        rec_queries = coeff.mm(anchors)
            # c_list.append(c)
        # diag_c = self.thres((q_queries*k_queries).sum(dim=1, keepdim=True)) * self.shrink
        # rec_queries = rec_queries - diag_c * queries
        # coeff = torch.cat(c_list, dim=-1)
        # s = torch.bernoulli(torch.tensor(0.01))
        # if s == 1:
        #     print((coeff != 0).sum(axis=1))
        return dict(rec_queries=rec_queries, coeff=coeff)
        # return dict(rec_queries=queries, coeff=coeff)
