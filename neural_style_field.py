import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class ProgressiveEncoding(nn.Module):
    def __init__(self, mapping_size, T, d=3, apply=True):
        super(ProgressiveEncoding, self).__init__()
        self._t = 0
        self.n = mapping_size
        self.T = T
        self.d = d
        self._tau = 2 * self.n / self.T
        self.indices = torch.tensor([i for i in range(self.n)], device=device)
        self.apply = apply
    def forward(self, x):
        alpha = ((self._t - self._tau * self.indices) / self._tau).clamp(0, 1).repeat(
            2)  # no need to reduce d or to check cases
        if not self.apply:
            alpha = torch.ones_like(alpha, device=device)  ## this layer means pure ffn without progress.
        alpha = torch.cat([torch.ones(self.d, device=device), alpha], dim=0)
        self._t += 1
        return x * alpha


class NeuralStyleField(nn.Module):
    # Same base then split into two separate modules 
    def __init__(self, width, depth, colordepth=2, normdepth=2, labeldepth=2, input_dim=3, num_frequencies=10, min_freq=0.0, max_freq=8.0, normal_output_dim=3, color_output_dim=3, label_output_dim=1, use_tcnn_encoding=False, print_model=False):
        super(NeuralStyleField, self).__init__()
        layers = []
        self.depth = depth
        self.colordepth = colordepth
        self.normdepth = normdepth
        self.labeldepth = labeldepth # TODO: not sure whether adding separate layers for labels or appending the labels as additional channels to the rgb output is better
        self.color_output_dim = color_output_dim
        self.normal_output_dim = normal_output_dim
        self.label_output_dim = label_output_dim

        assert num_frequencies is not None and min_freq is not None and max_freq is not None, "Please provide num_frequencies, min_freq, and max_freq for positional encoding."

        if use_tcnn_encoding:
            import tinycudann as tcnn
            encoding_config =  {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 15,
                "base_resolution": 16,
                "per_level_scale": 1.5
	        }
            encoding = tcnn.Encoding(n_input_dims=input_dim, encoding_config=encoding_config, dtype=torch.float32) # output_dim = 32
            base_input_dim = encoding.n_output_dims
        else:
            encoding = PositionalEncoding(in_dim=input_dim, num_frequencies=num_frequencies, min_freq=min_freq, max_freq=max_freq, include_input=True)
            base_input_dim = encoding.get_out_dim()

        # 20240509 Modify NN architecture.
        layers.append(encoding)


        if self.colordepth > 0 and self.normdepth > 0 and self.labeldepth > 0:
            base_out_dim = width
            # NN branches out for two modalities. The base layer output intermediate features.
            if self.depth > 0:
                branch_input_dim = width
            else:
                branch_input_dim = base_input_dim
        elif self.colordepth == 0 and self.normdepth == 0 and self.labeldepth == 0:
            # no branched layers, directly output features
            base_out_dim = self.color_output_dim + self.normal_output_dim + self.label_output_dim
        else:
            raise ValueError(f"colordepth, normdepth, labeldepth should be both greater than 0 or both equal to 0. Now we have {self.colordepth} and {self.normdepth} and {self.labeldepth}")
        
        if self.depth > 0:
            for i in range(self.depth-1):
                layers.append(nn.Linear(base_input_dim, width))
                layers.append(nn.ReLU())
                base_input_dim = width
            layers.append(nn.Linear(base_input_dim, base_out_dim))
        self.base = nn.ModuleList(layers)
        
        # Branches 
        color_layers = []
        if self.colordepth > 0:
            branch_color_input_dim = branch_input_dim # 32
            if self.depth > 0:
                color_layers.append(nn.ReLU())
            for i in range(self.colordepth-1):
                color_layers.append(nn.Linear(branch_color_input_dim, width))
                color_layers.append(nn.ReLU())
                branch_color_input_dim = width
            color_layers.append(nn.Linear(branch_color_input_dim, color_output_dim))            
        self.mlp_rgb = nn.ModuleList(color_layers)
        

        normal_layers = []
        if self.normdepth > 0:
            branch_normal_input_dim = branch_input_dim # 32
            if self.depth > 0:
                normal_layers.append(nn.ReLU())
            for i in range(normdepth-1):
                normal_layers.append(nn.Linear(branch_normal_input_dim, width))
                normal_layers.append(nn.ReLU())
                branch_normal_input_dim = width
            normal_layers.append(nn.Linear(branch_normal_input_dim, normal_output_dim))
        self.mlp_delta_normal = nn.ModuleList(normal_layers)


        label_layers = []
        if self.labeldepth > 0:
            branch_label_input_dim = branch_input_dim # 32
            if self.depth > 0:
                label_layers.append(nn.ReLU())
            for i in range(labeldepth-1):
                label_layers.append(nn.Linear(branch_label_input_dim, width))
                label_layers.append(nn.ReLU())
                branch_label_input_dim = width
            label_layers.append(nn.Linear(branch_label_input_dim, label_output_dim))
        self.mlp_label = nn.ModuleList(label_layers)
        
        if print_model:
            print("Check layers for NeuralStyleField")
            print(f"base layers:")
            print(self.base)
            print(f"color layers:")
            print(self.mlp_rgb)
            print(f"normal layers:")
            print(self.mlp_delta_normal)
            print(f"label layers:")
            print(self.mlp_label)


    def reset_weights(self):
        self.mlp_rgb[-1].weight.data.zero_()
        self.mlp_rgb[-1].bias.data.zero_()
        self.mlp_delta_normal[-1].weight.data.zero_()
        self.mlp_delta_normal[-1].bias.data.zero_()
        self.mlp_label[-1].weight.data.zero_()
        self.mlp_label[-1].bias.data.zero_()

    def forward(self, x):
        for layer in self.base:
            x = layer(x)
        
        if self.colordepth > 0:
            colors = self.mlp_rgb[0](x)
            for layer in self.mlp_rgb[1:]:
                colors = layer(colors)
        else:
            # take the first few channels as color
            colors = x[:, :self.color_output_dim]

        if self.normdepth > 0:
            normal = self.mlp_delta_normal[0](x)
            for layer in self.mlp_delta_normal[1:]:
                normal = layer(normal)
        else:
            # take the last few channels as normal
            normal = x[:, self.color_output_dim:self.color_output_dim+self.normal_output_dim]
        
        if self.labeldepth > 0:
            label = self.mlp_label[0](x)
            for layer in self.mlp_label[1:]:
                label = layer(label)
        else:
            # take the last few channels as label
            label = x[:, -self.label_output_dim:]

        # apply activation functions / clamping
        colors = F.sigmoid(colors) # range [0, 1]
        normal = F.tanh(normal) # range [-1, 1]

        # 20241026: Since we change to use cross-entry loss, we do not need to apply softmax here.
        # # Apply softmax and argmax for label output
        # label_probs = F.softmax(label, dim=-1)  # softmax across the class dimension
        # label_class = torch.argmax(label_probs, dim=-1, keepdim=True)  # class index for each point
    
        # add [0, 0, 1] as the normal direction so that we learn the residual
        normal = normal + torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
        
        # normalized normal to unit vector
        normal = F.normalize(normal, p=2, dim=-1)

        output = torch.cat([colors, normal, label], dim=-1) # [texture_size*texture_size, 8] # 8 = 3 (rgb) + 3 (normal) + 2 (label)
        return output


def save_model(model, loss, iter, optim, output_dir):
    save_dict = {
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss
    }

    path = os.path.join(output_dir, 'checkpoint.pth.tar')

    torch.save(save_dict, path)


class PositionalEncoding(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, in_dim, num_frequencies, min_freq=1, max_freq=10, include_input=True):
        super().__init__()
        self.in_dim = in_dim
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_frequencies = num_frequencies
        self.include_input =  include_input

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(self, x):
        
        """
        Calculates positioanal encoding. 

        Ref: nerfstudio ->field_components->encodings.py

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """

        # print(f"In forward of PositionalEncoding, check x shape: {x.shape}") # [262146, 3] (N_vertices, xyz)
        # print(f"min_freq: {self.min_freq}, max_freq: {self.max_freq}, num_frequencies: {self.num_frequencies}") # 0.0 8.0 10

        scaled_x = 2 * torch.pi * x # scale to [0, 2* pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies, device=x.device) # shape [10]

        
        scaled_inputs = scaled_x.unsqueeze(-1) * freqs # [..., "input_dim", "num_frequencies"] [262146, 3, 10]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1) # [..., "input_dim" * "num_frequencies"] [262146, 30]

        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1)) # [262146, 60]
        
        if self.include_input:
            encoded_inputs = torch.cat([x, encoded_inputs], dim=-1)

        return encoded_inputs



# Use networks implemented by threestudio
# Ref: https://github.com/threestudio-project/threestudio/blob/cd462fb0b73a89b6be17160f7802925fe6cf34cd/threestudio/models/networks.py
class VanillaMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, config: dict):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = (
            config["n_neurons"],
            config["n_hidden_layers"],
        )
        layers = [
            self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    self.n_neurons, self.n_neurons, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        layers += [
            self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(config.get("output_activation", None))

    def forward(self, x):
        # disable autocast
        # strange that the parameters will have empty gradients if autocast is enabled in AMP
        with torch.cuda.amp.autocast(enabled=False):
            x = self.layers(x)
            x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=False)
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)
# helper function for activation functions. 
# Ref: https://github.com/threestudio-project/threestudio/blob/cd462fb0b73a89b6be17160f7802925fe6cf34cd/threestudio/utils/ops.py#L78
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))
trunc_exp = _TruncExp.apply

def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")



class Updateable:
    def do_update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step(
                    epoch, global_step, on_load_weights=on_load_weights
                )
        self.update_step(epoch, global_step, on_load_weights=on_load_weights)

    def do_update_step_end(self, epoch: int, global_step: int):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step_end(epoch, global_step)
        self.update_step_end(epoch, global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # override this method to implement custom update logic
        # if on_load_weights is True, you should be careful doing things related to model evaluations,
        # as the models and tensors are not guarenteed to be on the same device
        pass

    def update_step_end(self, epoch: int, global_step: int):
        pass


def update_if_possible(module: Any, epoch: int, global_step: int) -> None:
    if isinstance(module, Updateable):
        module.do_update_step(epoch, global_step)


def update_end_if_possible(module: Any, epoch: int, global_step: int) -> None:
    if isinstance(module, Updateable):
        module.do_update_step_end(epoch, global_step)


class NetworkWithInputEncoding(nn.Module, Updateable):
    def __init__(self, encoding, network):
        super().__init__()
        self.encoding, self.network = encoding, network

    def forward(self, x):
        encoding = self.encoding(x)
        return self.network(encoding)
        # return self.network(self.encoding(x))
