import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


class NeuralStyleField(nn.Module):
    # Network architecture: a shared encoding and base layer for all modalities, which then branches out to separate layers for color, normal, and label
    def __init__(self, width, depth, colordepth=2, normdepth=2, labeldepth=2, input_dim=3, color_output_dim=3, normal_output_dim=3, label_output_dim=1, print_model=False):
        super(NeuralStyleField, self).__init__()
        layers = []
        self.depth = depth
        self.colordepth = colordepth
        self.normdepth = normdepth
        self.labeldepth = labeldepth
        self.color_output_dim = color_output_dim
        self.normal_output_dim = normal_output_dim
        self.label_output_dim = label_output_dim

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

