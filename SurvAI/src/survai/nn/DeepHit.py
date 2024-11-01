import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepHit(nn.Module):
    def __init__(self, input_dims, network_settings):
        super(DeepHit, self).__init__()

        # INPUT DIMENSIONS
        self.x_dim = input_dims['x_dim']
        self.num_Event = input_dims['num_Event']
        self.num_Category = input_dims['num_Category']

        # NETWORK HYPER-PARAMETERS
        self.h_dim_shared = network_settings['h_dim_shared']
        self.h_dim_CS = network_settings['h_dim_CS']
        self.num_layers_shared = network_settings['num_layers_shared']
        self.num_layers_CS = network_settings['num_layers_CS']
        self.active_fn = network_settings['active_fn']
        self.initial_W = network_settings['initial_W']

        # Shared subnetwork
        self.shared_layers = self._build_layers(self.x_dim, self.h_dim_shared, self.num_layers_shared)

        # Cause-specific subnetworks
        self.cs_layers = nn.ModuleList(
            [self._build_layers(self.h_dim_shared + self.x_dim, self.h_dim_CS, self.num_layers_CS) for _ in
             range(self.num_Event)])

        # Output layer
        self.output_layer = nn.Linear(self.num_Event * self.h_dim_CS, self.num_Event * self.num_Category)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def _build_layers(self, input_dim, hidden_dim, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._activation_fn(self.active_fn))
            input_dim = hidden_dim
        return nn.Sequential(*layers)

    def _activation_fn(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, x, keep_prob=1.0):
        # Shared network
        shared_out = self.shared_layers(x)
        h = torch.cat([x, shared_out], dim=1)  # Residual connection

        # Cause-specific networks
        cs_out = [cs_layer(h) for cs_layer in self.cs_layers]
        out = torch.cat(cs_out, dim=1)
        out = F.dropout(out, p=1 - keep_prob, training=self.training)

        # Final output layer with softmax
        out = self.output_layer(out)
        out = F.softmax(out.view(-1, self.num_Event, self.num_Category), dim=-1)

        return out

    def loss_log_likelihood(self, out, fc_mask1, k):
        I_1 = torch.sign(k)
        tmp1 = torch.sum(torch.sum(fc_mask1 * out, dim=2), dim=1, keepdim=True)
        tmp1 = I_1 * torch.log(tmp1 + 1e-8)

        tmp2 = torch.sum(torch.sum(fc_mask1 * out, dim=2), dim=1, keepdim=True)
        tmp2 = (1. - I_1) * torch.log(tmp2 + 1e-8)

        return -torch.mean(tmp1 + tmp2)

    def loss_ranking(self, out, fc_mask2, t, k):
        sigma1 = 0.1
        eta = []

        for e in range(self.num_Event):
            I_2 = (k == e + 1).float().squeeze()
            tmp_e = out[:, e, :]
            R = torch.matmul(tmp_e, fc_mask2.t())
            diag_R = R.diag().unsqueeze(1)
            R = diag_R - R
            T = torch.relu((t.unsqueeze(1) - t.unsqueeze(0)).sign())
            T = T * I_2.unsqueeze(1)
            tmp_eta = torch.mean(T * torch.exp(-R / sigma1), dim=1, keepdim=True)
            eta.append(tmp_eta)

        eta = torch.stack(eta, dim=1)
        return torch.sum(eta.mean(dim=1))

    def loss_calibration(self, out, fc_mask2, k):
        eta = []

        for e in range(self.num_Event):
            I_2 = (k == e + 1).float().squeeze()
            tmp_e = out[:, e, :]
            r = torch.sum(tmp_e * fc_mask2, dim=0)
            tmp_eta = torch.mean((r - I_2) ** 2, dim=0, keepdim=True)
            eta.append(tmp_eta)

        eta = torch.stack(eta, dim=1)
        return torch.sum(eta.mean(dim=1))

    def get_cost(self, x, k, t, fc_mask1, fc_mask2, alpha, beta, gamma, keep_prob):
        out = self.forward(x, keep_prob)
        loss1 = self.loss_log_likelihood(out, fc_mask1, k)
        loss2 = self.loss_ranking(out, fc_mask2, t, k)
        loss3 = self.loss_calibration(out, fc_mask2, k)
        return alpha * loss1 + beta * loss2 + gamma * loss3


# # Example of model initialization
# input_dims = {'x_dim': 30, 'num_Event': 3, 'num_Category': 10}
# network_settings = {'h_dim_shared': 128, 'h_dim_CS': 64, 'num_layers_shared': 3, 'num_layers_CS': 2,
#                     'active_fn': 'relu', 'initial_W': 'xavier'}

# model = DeepHit(input_dims, network_settings)
#
# print(model)
