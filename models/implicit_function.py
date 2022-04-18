# -*- coding: utf-8 -*-
#
# Developed by Haiyang Bai 


import torch

from typing import Tuple
from common.linear_with_repeat import LinearWithRepeat
from pytorch3d.renderer import ray_bundle_to_ray_points, RayBundle


def _xavier_init(linear):
    """ Performs the Xavier weight initialization of the linear layer `linear`. """
    torch.nn.init.xavier_uniform_(linear.weight.data)


class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        (i.e. vector along the last dimension) in `x`
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]:
            ```
            [
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i],              # only present if append_input is True.
            ]
            ```
        where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.

        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`

        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`

        Note that `x` is also premultiplied by the base frequency `omega_0`
        before evaluating the harmonic functions.

        Args:
            n_harmonic_functions: int, number of harmonic
                features
            omega_0: float, base frequency
            logspace: bool, Whether to space the frequencies in
                logspace or linear space
            append_input: bool, whether to concat the original
                input to the harmonic embedding. If true the
                output is of the form (x, embed.sin(), embed.cos()

        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", frequencies * omega_0, persistent=False)
        self.append_input = append_input

    def forward(self, x: torch.Tensor) :   # -> torch.Tensor
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., (n_harmonic_functions * 2 + int(append_input)) * dim]
        """
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        embed = torch.cat(
            (embed.sin(), embed.cos(), x)
            if self.append_input
            else (embed.sin(), embed.cos()),
            dim=-1,
        )
        return embed

    @staticmethod
    def get_output_dim_static(
        input_dims: int,
        n_harmonic_functions: int,
        append_input: bool) :    # -> int
        """
        Utility to help predict the shape of the output of `forward`.

        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: int = 3) :   # -> int
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(
            input_dims, len(self._frequencies), self.append_input
        )


class MLPWithInputSkips(torch.nn.Module):
    """
    Implements the multi-layer perceptron architecture of the Neural Radiance Field.

    As such, `MLPWithInputSkips` is a multi layer perceptron consisting
    of a sequence of linear layers with ReLU activations.

    Additionally, for a set of predefined layers `input_skips`, the forward pass
    appends a skip tensor `z` to the output of the preceding layer.

    Note that this follows the architecture described in the Supplementary
    Material (Fig. 7) of [1].

    References:
        [1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik
            and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng:
            NeRF: Representing Scenes as Neural Radiance Fields for View
            Synthesis, ECCV2020
    """

    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips: Tuple[int] = (),
    ):
        """
        Args:
            n_layers: The number of linear layers of the MLP.
            input_dim: The number of channels of the input tensor.
            output_dim: The number of channels of the output.
            skip_dim: The number of channels of the tensor `z` appended when
                evaluating the skip layers.
            hidden_dim: The number of hidden units of the MLP.
            input_skips: The list of layer indices at which we append the skip
                tensor `z`.
        """
        super().__init__()
        layers = []
        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim
            linear = torch.nn.Linear(dimin, dimout)
            _xavier_init(linear)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))
        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor): #  -> torch.Tensor
        """
        Args:
            x: The input tensor of shape `(..., input_dim)`.
            z: The input skip tensor of shape `(..., skip_dim)` which is appended
                to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape `(..., output_dim)`.
        """
        y = x
        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)
            y = layer(y)
        return y




class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        append_xyz: Tuple[int] = (5,),
        use_multiple_streams: bool = True,
        **kwargs):

        super(NeuralRadianceField, self).__init__()
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        self.mlp_xyz = MLPWithInputSkips(n_layers_xyz, embedding_dim_xyz, 
        n_hidden_neurons_xyz, embedding_dim_xyz, n_hidden_neurons_xyz, input_skips=append_xyz)

        self.intermediate_linear = torch.nn.Linear(n_hidden_neurons_xyz, n_hidden_neurons_xyz)
        _xavier_init(self.intermediate_linear)

        self.density_layer = torch.nn.Linear(n_hidden_neurons_xyz, 1)
        _xavier_init(self.density_layer)

        # Zero the bias of the density layer to avoid a completely transparent initialization.
        self.density_layer.bias.data[:] = 0.0  # fixme: Sometimes this is not enough
        self.color_layer = torch.nn.Sequential(
            LinearWithRepeat(n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir),
            torch.nn.ReLU(True),
            torch.nn.Linear(n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),
        )
        self.use_multiple_streams = use_multiple_streams


    def _get_densities(
        self,
        features: torch.Tensor,
        depth_values: torch.Tensor,
        density_noise_std: float): # -> torch.Tensor
        """
        This function takes `features` predicted by `self.mlp_xyz`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later re-weighted using the depth step sizes
        and mapped to [0-1] range with 1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        deltas = torch.cat((depth_values[..., 1:] - depth_values[..., :-1], 1e10 * torch.ones_like(depth_values[..., :1]),), dim=-1,)[..., None]
        if density_noise_std > 0.0:
            raw_densities = (raw_densities + torch.randn_like(raw_densities) * density_noise_std)
        densities = 1 - (-deltas * torch.relu(raw_densities)).exp()
        return densities


    def _get_colors(self, features: torch.Tensor, rays_directions: torch.Tensor):    # -> torch.Tensor
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

        return self.color_layer((self.intermediate_linear(features), rays_embedding))


    def _get_densities_and_colors(self, features: torch.Tensor, ray_bundle: RayBundle, density_noise_std: float): 
        # -> Tuple[torch.Tensor, torch.Tensor]
        """
        The second part of the forward calculation.

        Args:
            features: the output of the common mlp (the prior part of the
                calculation), shape
                (minibatch x ... x self.n_hidden_neurons_xyz).
            ray_bundle: As for forward().
            density_noise_std:  As for forward().

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        if self.use_multiple_streams and features.is_cuda:
            current_stream = torch.cuda.current_stream(features.device)
            other_stream = torch.cuda.Stream(features.device)
            other_stream.wait_stream(current_stream)

            with torch.cuda.stream(other_stream):
                rays_densities = self._get_densities(
                    features, ray_bundle.lengths, density_noise_std
                )
                # rays_densities.shape = [minibatch x ... x 1] in [0-1]

            rays_colors = self._get_colors(features, ray_bundle.directions)
            # rays_colors.shape = [minibatch x ... x 3] in [0-1]

            current_stream.wait_stream(other_stream)
        else:
            # Same calculation as above, just serial.
            rays_densities = self._get_densities(
                features, ray_bundle.lengths, density_noise_std
            )
            rays_colors = self._get_colors(features, ray_bundle.directions)
        return rays_densities, rays_colors


    def forward(self, ray_bundle: RayBundle, density_noise_std: float = 0.0, **kwargs): 
        # -> Tuple[torch.Tensor, torch.Tensor]
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            density_noise_std: A floating point value representing the
                variance of the random normal noise added to the output of
                the opacity function. This can prevent floating artifacts.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = self.harmonic_embedding_xyz(rays_points_world)
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz, embeds_xyz)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        rays_densities, rays_colors = self._get_densities_and_colors(
            features, ray_bundle, density_noise_std
        )
        return rays_densities, rays_colors


