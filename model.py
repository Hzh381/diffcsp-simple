"""A compact CSP denoising network following the original DiffCSP idea."""

import math

import torch
import torch.nn as nn


def scatter_mean(values, index, size):
    """Mean aggregation without torch_scatter."""
    output = values.new_zeros((size, values.size(1)))
    count = values.new_zeros(size)
    output.index_add_(0, index, values)
    count.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return output / count.clamp_min(1).unsqueeze(1)


def full_edge_index(num_atoms, node2graph):
    """Create a fully connected graph for each crystal in the batch."""
    edges = []
    offset = 0
    for num_atom in num_atoms.tolist():
        nodes = torch.arange(num_atom, device=node2graph.device) + offset
        edges.append(torch.cartesian_prod(nodes, nodes).t())
        offset += num_atom
    return torch.cat(edges, dim=1)


class SinusoidsEmbedding(nn.Module):
    """Fourier features for fractional coordinate differences."""

    def __init__(self, num_frequencies=10):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.register_buffer(
            "frequencies",
            2 * math.pi * torch.arange(num_frequencies),
        )
        self.dim = 2 * 3 * num_frequencies

    def forward(self, values):
        embedded = values.unsqueeze(-1) * self.frequencies
        embedded = embedded.reshape(-1, 3 * self.num_frequencies)
        return torch.cat([embedded.sin(), embedded.cos()], dim=-1)


class CSPLayer(nn.Module):
    """One residual message-passing layer."""

    def __init__(self, hidden_dim, distance_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 9 + distance_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff):
        source, target = edge_index
        normalized = self.norm(node_features)
        lattice_products = lattices @ lattices.transpose(-1, -2)
        edge_input = torch.cat(
            [
                normalized[source],
                normalized[target],
                lattice_products[edge2graph].reshape(len(source), 9),
                frac_diff,
            ],
            dim=1,
        )
        edge_features = self.edge_mlp(edge_input)
        aggregated = scatter_mean(edge_features, source, node_features.size(0))
        updated = self.node_mlp(torch.cat([normalized, aggregated], dim=1))
        return node_features + updated


class CSPNet(nn.Module):
    """Predict lattice noise and fractional-coordinate score."""

    def __init__(
        self,
        hidden_dim=128,
        time_dim=256,
        num_layers=4,
        max_atoms=100,
        num_frequencies=10,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.max_atoms = max_atoms
        self.distance_embedding = SinusoidsEmbedding(num_frequencies)
        self.atom_embedding = nn.Embedding(max_atoms + 1, hidden_dim)
        self.atom_time_embedding = nn.Linear(hidden_dim + time_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [CSPLayer(hidden_dim, self.distance_embedding.dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias=False)

    def forward(self, time_emb, atom_types, frac_coords, lattices, num_atoms, node2graph):
        edge_index = full_edge_index(num_atoms, node2graph)
        source, target = edge_index
        frac_diff = (frac_coords[target] - frac_coords[source]) % 1.0
        frac_diff = self.distance_embedding(frac_diff)
        edge2graph = node2graph[source]

        time_per_atom = time_emb.repeat_interleave(num_atoms, dim=0)
        node_features = self.atom_embedding(atom_types.long())
        node_features = self.atom_time_embedding(
            torch.cat([node_features, time_per_atom], dim=1)
        )

        for layer in self.layers:
            node_features = layer(
                node_features,
                frac_coords,
                lattices,
                edge_index,
                edge2graph,
                frac_diff,
            )

        node_features = self.norm(node_features)
        coord_out = self.coord_out(node_features)
        graph_features = scatter_mean(
            node_features,
            node2graph,
            lattices.size(0),
        )
        lattice_out = self.lattice_out(graph_features).view(-1, 3, 3)
        return torch.bmm(lattice_out, lattices), coord_out
