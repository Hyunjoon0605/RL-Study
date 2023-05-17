import dgl
import torch
import torch.nn as nn

from model.neuralnet import *


class Actor(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, action_dim,
                 edge_type, node_type, aggregator, con_range):
        super(Actor, self).__init__()

        self.target_node = node_type
        num_graph_layer = len(edge_type)
        node_input_dims = [node_input_dim] + (num_graph_layer - 1) * [hidden_dim]
        node_output_dims = num_graph_layer * [hidden_dim]

        self.graph_layers = nn.ModuleList()
        for idx in range(num_graph_layer):

            ei_dim = 2 * node_input_dims[idx] + edge_input_dim
            eo_dim = hidden_dim
            ni_dim = node_input_dims[idx] + len(edge_type[idx]) * eo_dim
            no_dim = node_output_dims[idx]

            edge_model = [MLP(ei_dim, eo_dim, [64]) for _ in range(len(edge_type[idx]))]
            node_model = MLP(ni_dim, no_dim, [64])

            self.graph_layers.append(GraphLayer(node_model=node_model, edge_model=edge_model,
                                                edge_read_out=edge_type[idx], node_read_out=node_type,
                                                aggregator=aggregator, con_range=con_range[idx]))

        self.mlp = MLP(in_dim=hidden_dim, out_dim=action_dim, hidden=[64, 64], out_activation=nn.Tanh())

    def forward(self, gs):

        nf = gs.ndata['nf']
        ef = gs.edata['ef']

        for graph_layer in self.graph_layers:
            nf, ef = graph_layer(gs, nf, ef)

        return self.mlp(nf[gs.ndata['node_type'] == self.target_node])


class Critic(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, action_dim,
                 aggregator, edge_type, node_type, con_range):
        super(Critic, self).__init__()

        self.target_node = node_type
        num_graph_layer = len(edge_type)
        node_input_dims = [node_input_dim] + (num_graph_layer - 1) * [hidden_dim]
        node_output_dims = num_graph_layer * [hidden_dim]

        self.graph_layers = nn.ModuleList()
        for idx in range(num_graph_layer):
            ei_dim = edge_input_dim + 2 * node_input_dims[idx] + 2 * action_dim
            eo_dim = hidden_dim
            ni_dim = node_input_dims[idx] + len(edge_type[idx]) * eo_dim + action_dim
            no_dim = node_output_dims[idx]

            edge_model = [MLP(ei_dim, eo_dim, [64]) for _ in range(len(edge_type[idx]))]
            node_model = MLP(ni_dim, no_dim, [64])
            edge_update_type = edge_type[idx]
            node_update_type = node_type
            cur_con_range = con_range[idx]

            self.graph_layers.append(GraphLayer(node_model=node_model, edge_model=edge_model,
                                                edge_read_out=edge_update_type, node_read_out=node_update_type,
                                                aggregator=aggregator, con_range=cur_con_range))

        self.mlp = MLP(in_dim=hidden_dim + action_dim, out_dim=1, hidden=[64, 64], critic=True)

    def forward(self, gs, action):

        nf = gs.ndata['nf']
        ef = gs.edata['ef']
        target_node = gs.ndata['node_type'] == self.target_node

        for graph_layer in self.graph_layers:

            action_all = torch.zeros(nf.shape[0], action.shape[-1]).to(gs.device)
            action_all[target_node] = action
            nf, ef = graph_layer(gs, torch.cat([nf, action_all], dim=-1), ef)

        return self.mlp(nf[target_node], action).reshape(-1)

