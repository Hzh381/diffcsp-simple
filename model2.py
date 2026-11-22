"""A readable invariant-atom / equivariant-lattice Transformer.

PyTorch only. No PyG or DGL dependency.

Shapes
------
atom:        [B, N, D]       invariant atom features
lattice:     [B, 3, 3]       lattice matrix; columns are lattice vectors
edge_index:  [2, E]         directed edges: source -> destination
edge_attr:   [B, E, F]      optional invariant edge features
padding_mask:[B, N]         True for padded atoms

The lattice stream stores vector features with shape [B, 3, 3, D]:
[batch, lattice-vector, Cartesian-coordinate, feature-channel].
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class CrystalDualStreamConfig:
    """Hyper-parameters used by the dual-stream denoiser adapter."""

    hidden_dim: int = 384
    num_layers: int = 4
    heads: int = 8
    time_dim: int = 128
    max_atomic_number: int = 100
    expansion: int = 4
    coord_frequencies: int = 10
    dropout: float = 0.0
    # Build edge features only for the sum(n_i^2) real atom pairs instead of
    # the dense padded [B, max_atoms^2, ...] grid. Kept as a flag so the two
    # layouts can be compared on the same model.
    sparse_edges: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.time_dim % 2:
            raise ValueError("time_dim must be even")


def segment_softmax(scores, segment, num_segments):
    """Softmax over the last ``E`` edges, grouped by ``segment``.

    ``scores`` is ``[E, H]`` and ``segment`` is ``[E]``; each group is one
    destination atom, so this reproduces a dense row-wise softmax without
    materialising the ``[B, N, N]`` matrix.
    """
    index = segment[:, None].expand_as(scores)
    maximum = scores.new_full(
        (num_segments, scores.size(1)), float("-inf")
    ).scatter_reduce(0, index, scores, reduce="amax", include_self=False)
    weights = (scores - maximum[segment].detach()).exp()
    total = scores.new_zeros((num_segments, scores.size(1)))
    total = total.index_add(0, segment, weights)
    return weights / total[segment].clamp_min(1e-12)


def build_sparse_edges(num_atoms, nodes, device):
    """Real edges of a padded batch as ``(edge_batch, edge_src, edge_dst)``.

    Only a ``[B, nodes^2]`` boolean mask is built; padded atoms and
    cross-graph pairs are dropped, leaving ``E = sum_i n_i^2`` edges.
    ``edge_dst`` receives the message, ``edge_src`` sends it.
    """
    grid = torch.cartesian_prod(
        torch.arange(nodes, device=device),
        torch.arange(nodes, device=device),
    )
    source, target = grid[:, 1], grid[:, 0]
    counts = num_atoms.to(device=device, dtype=torch.long)[:, None]
    valid = (source[None, :] < counts) & (target[None, :] < counts)
    edge_batch = torch.arange(
        num_atoms.numel(), device=device
    )[:, None].expand_as(valid)[valid]
    return (
        edge_batch,
        source[None, :].expand_as(valid)[valid],
        target[None, :].expand_as(valid)[valid],
    )


class InvariantGraphAttention(nn.Module):
    """Dense graph attention over invariant atom features.

    For edge j -> i the edge embedding is added to the key and the value
    before attention:
        k[i,j] = W_k h[j] + W_ek e[i,j]
        v[i,j] = W_v h[j] + W_ev e[i,j]
        score[i,j] = q[i] dot k[i,j] / sqrt(d)
        output[i] = sum_j softmax_j(score[i,j]) * v[i,j]

    A dense attention matrix is used here for readability. For small crystal
    graphs this is usually practical; edge_index defines which pairs connect.
    """

    def __init__(self, dim, heads=4, edge_dim=0, dropout=0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.edge_dim = edge_dim

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        if edge_dim > 0:
            self.edge_key = nn.Linear(edge_dim, dim, bias=False)
            self.edge_value = nn.Linear(edge_dim, dim, bias=False)
        else:
            self.edge_key = None
            self.edge_value = None

    def _make_graph_mask(self, edge_index, batch, nodes, device):
        """Return [B, N, N] mask; True means that an edge is allowed."""
        graph_mask = torch.eye(nodes, dtype=torch.bool, device=device)
        graph_mask = graph_mask.unsqueeze(0).expand(batch, -1, -1).clone()

        if edge_index is None:
            return torch.ones_like(graph_mask)

        source = edge_index[0].to(device=device, dtype=torch.long)
        target = edge_index[1].to(device=device, dtype=torch.long)
        graph_mask[:, target, source] = True
        return graph_mask

    def _make_edge_features(self, edge_index, edge_attr, batch, nodes, device):
        """Place edge features in a dense [B, N, N, F] tensor."""
        if self.edge_dim == 0:
            return None
        if edge_attr is None:
            raise ValueError("edge_attr is required when edge_dim > 0")

        if edge_attr.ndim == 2:
            edge_attr = edge_attr.unsqueeze(0).expand(batch, -1, -1)
        if edge_attr.shape != (batch, edge_index.shape[-1], self.edge_dim):
            raise ValueError("edge_attr must have shape [B, E, edge_dim]")

        dense = edge_attr.new_zeros(batch, nodes, nodes, self.edge_dim)
        source = edge_index[0].to(device=device, dtype=torch.long)
        target = edge_index[1].to(device=device, dtype=torch.long)
        dense[:, target, source] = edge_attr
        return dense

    def forward(self, atom, edge_index=None, edge_attr=None, padding_mask=None,
                edges=None):
        if edges is not None:
            return self._forward_sparse(atom, edges, edge_attr)

        batch, nodes, _ = atom.shape

        # Q, K, V: [B, heads, N, head_dim]
        q, k, v = self.qkv(atom).chunk(3, dim=-1)
        q = q.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)

        if edge_index is None:
            graph_mask = torch.ones(
                batch, nodes, nodes, dtype=torch.bool, device=atom.device
            )
        else:
            graph_mask = self._make_graph_mask(
                edge_index, batch, nodes, atom.device
            )

        if self.edge_dim > 0:
            if edge_index is None:
                raise ValueError("edge_index is required when edge_dim > 0")
            dense_edge_attr = self._make_edge_features(
                edge_index, edge_attr, batch, nodes, atom.device
            )
            # Add the edge embedding to the keys and values before attention,
            # so both the score and the message see the pair geometry.
            edge_key = self.edge_key(dense_edge_attr).view(
                batch, nodes, nodes, self.heads, self.head_dim
            ).permute(0, 3, 1, 2, 4)
            edge_value = self.edge_value(dense_edge_attr).view(
                batch, nodes, nodes, self.heads, self.head_dim
            ).permute(0, 3, 1, 2, 4)
            key = k[:, :, None, :, :] + edge_key       # [B, h, i, j, head_dim]
            value = v[:, :, None, :, :] + edge_value
            score = (q[:, :, :, None, :] * key).sum(dim=-1)
            score = score / math.sqrt(self.head_dim)
        else:
            score = q @ k.transpose(-2, -1)
            score = score / math.sqrt(self.head_dim)
            value = None

        if padding_mask is not None:
            # Padded atoms cannot be used as keys/values.
            graph_mask = graph_mask & (~padding_mask[:, None, :])

        score = score.masked_fill(~graph_mask[:, None, :, :], -1e9)
        attention = self.dropout(torch.softmax(score, dim=-1))

        if value is None:
            # Weighted sum of atom values.
            message = attention @ v
            message = message.transpose(1, 2).reshape(batch, nodes, self.dim)
        else:
            # Weighted sum over j of the edge-conditioned values.
            message = (attention.unsqueeze(-1) * value).sum(dim=3)
            message = message.transpose(1, 2).reshape(batch, nodes, self.dim)

        return self.output(message)

    def _forward_sparse(self, atom, edges, edge_attr):
        """Real-edge variant of ``forward``; builds no [B, N, N] tensor."""
        edge_batch, edge_src, edge_dst = edges
        batch, nodes, _ = atom.shape
        heads, head_dim = self.heads, self.head_dim
        count = edge_batch.numel()

        # The dense branch fuses Q, K, V into one projection. Here the query
        # comes from the destination atom and the key/value from the source.
        query = self.qkv(atom[edge_batch, edge_dst])[..., :self.dim]
        key_value = self.qkv(atom[edge_batch, edge_src])
        q = query.view(count, heads, head_dim)
        k = key_value[..., self.dim:2 * self.dim].view(count, heads, head_dim)
        v = key_value[..., 2 * self.dim:].view(count, heads, head_dim)

        if self.edge_dim > 0:
            if edge_attr is None:
                raise ValueError("edge_attr is required when edge_dim > 0")
            k = k + self.edge_key(edge_attr).view(count, heads, head_dim)
            v = v + self.edge_value(edge_attr).view(count, heads, head_dim)

        score = (q * k).sum(dim=-1) / math.sqrt(head_dim)      # [E, heads]
        segment = edge_batch * nodes + edge_dst
        weight = self.dropout(segment_softmax(score, segment, batch * nodes))

        message = (weight.unsqueeze(-1) * v).view(count, self.dim)
        pooled = atom.new_zeros(batch * nodes, self.dim)
        pooled = pooled.index_add(0, segment, message)
        return self.output(pooled.view(batch, nodes, self.dim))


class LatticeVectorEmbedding(nn.Module):
    """Initialize equivariant vector channels from the lattice matrix.

    Z_E[b, m, xyz, d] = L[b, xyz, m] * scale[d]
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) / math.sqrt(dim))

    def forward(self, lattice):
        if lattice.shape[-2:] != (3, 3):
            raise ValueError("lattice must have shape [B, 3, 3]")
        return lattice.transpose(1, 2).unsqueeze(-1) * self.scale


class EquivariantRMSNorm(nn.Module):
    """Scale each lattice vector by a rotation-invariant RMS.

    The scalar is shared by the three Cartesian components, so a common
    rotation still commutes with the normalization. A plain ``LayerNorm``
    over the feature axis would rescale each component differently and
    break equivariance.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, vector):
        rms = vector.pow(2).mean(dim=(2, 3), keepdim=True)
        return vector * torch.rsqrt(rms + self.eps) * self.weight


class EquivariantVectorAttention(nn.Module):
    """Self-attention over the three lattice vectors.

    The attention score is a sum of Cartesian inner products, hence invariant:
        score[i,j] = sum_xyz,channel Q[i] * K[j]
    The weighted sum of vector values is therefore equivariant.
    """

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vector):
        # vector: [B, 3 lattice vectors, 3 xyz coordinates, D]
        query = self.query(vector)
        key = self.key(vector)
        value = self.value(vector)

        # [B, 3, 3]; each score is invariant under a common 3D rotation.
        score = torch.einsum("bixd,bjxd->bij", query, key)
        score = score / math.sqrt(3 * vector.shape[-1])
        attention = self.dropout(torch.softmax(score, dim=-1))

        # [B, 3, 3, D]; a scalar-weighted sum preserves equivariance.
        result = torch.einsum("bij,bjxd->bixd", attention, value)
        return self.output(result)


class InvariantCrossAttention(nn.Module):
    """Atom queries attend to invariant descriptors of lattice geometry."""

    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.metric_embedding = nn.Sequential(
            nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, atom, vector):
        # G[i,j] = <Z_E[i], Z_E[j]>; G is invariant under rotation.
        # Divide by the feature dimension so Gram entries stay O(1).
        gram = torch.einsum("bixd,bjxd->bij", vector, vector)
        gram = gram / vector.shape[-1]
        lattice_tokens = self.metric_embedding(gram)
        update, _ = self.attention(
            atom, lattice_tokens, lattice_tokens, need_weights=False
        )
        return update


class EquivariantCrossAttention(nn.Module):
    """Lattice-vector queries attend to atoms using scalar atom features.

    Atom features generate scalar gates for vector keys and values. The
    attention score contracts Cartesian coordinates, producing a scalar.
    """

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key_vector = nn.Linear(dim, dim, bias=False)
        self.key_atom = nn.Linear(dim, dim, bias=False)
        self.value_vector = nn.Linear(dim, dim, bias=False)
        self.value_atom = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vector, atom, padding_mask=None):
        # Query: [B, 3, 3, D]. Atom gates: [B, N, D].
        query = self.query(vector)
        atom_key = self.key_atom(atom)
        atom_value = self.value_atom(atom)

        # Build one vector key/value for every lattice-vector/atom pair.
        key = self.key_vector(vector).unsqueeze(2)
        key = key * atom_key[:, None, :, None, :]
        value = self.value_vector(vector).unsqueeze(2)
        value = value * atom_value[:, None, :, None, :]

        # [B, 3 lattice vectors, N atoms], invariant scalar scores.
        score = (query.unsqueeze(2) * key).sum(dim=(-2, -1))
        score = score / math.sqrt(3 * vector.shape[-1])
        if padding_mask is not None:
            score = score.masked_fill(padding_mask[:, None, :], -1e9)
        attention = self.dropout(torch.softmax(score, dim=-1))

        # Sum over atoms; scalar weights preserve vector equivariance.
        update = (attention[..., None, None] * value).sum(dim=2)
        return self.output(update)


class CrystalDualStreamBlock(nn.Module):
    """One block containing both the invariant and equivariant streams."""

    def __init__(self, dim, heads=4, edge_dim=0, expansion=4, dropout=0.0):
        super().__init__()
        hidden = dim * expansion

        # Invariant atom stream.
        self.atom_norm1 = nn.LayerNorm(dim)
        self.atom_norm2 = nn.LayerNorm(dim)
        self.atom_norm3 = nn.LayerNorm(dim)
        self.graph_attention = InvariantGraphAttention(
            dim, heads, edge_dim, dropout
        )
        self.atom_cross_attention = InvariantCrossAttention(dim, heads, dropout)
       
        self.atom_ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

        # Equivariant lattice stream.
        self.vector_norm1 = EquivariantRMSNorm(dim)
        self.vector_norm2 = EquivariantRMSNorm(dim)
        self.vector_norm3 = EquivariantRMSNorm(dim)
        self.vector_attention = EquivariantVectorAttention(dim, dropout)
        self.vector_cross_attention = EquivariantCrossAttention(dim, dropout)
        self.vector_ffn_in = nn.Linear(dim, hidden, bias=False)
        self.vector_ffn_out = nn.Linear(hidden, dim, bias=False)
        self.vector_gate = nn.Sequential(
            nn.Linear(dim + 3, hidden), nn.GELU()
        )

    def forward(self, atom, vector, edge_index=None, edge_attr=None,
                padding_mask=None, edges=None):
        # 1. Intra-stream attention: pre-norm, residual bypass.
        atom = atom + self.graph_attention(
            self.atom_norm1(atom), edge_index, edge_attr, padding_mask,
            edges=edges,
        )
        vector = vector + self.vector_attention(self.vector_norm1(vector))

        # 2. Cross-stream attention: both blocks consume the pre-cross
        # normalized features of both streams.
        atom_normed = self.atom_norm2(atom)
        vector_normed = self.vector_norm2(vector)
        atom = atom + self.atom_cross_attention(atom_normed, vector_normed)
        vector = vector + self.vector_cross_attention(
            vector_normed, atom_normed, padding_mask
        )

        # 3. Atom feed forward: pre-norm, residual bypass.
        atom = atom + self.atom_ffn(self.atom_norm3(atom))

        # 4. Equivariant feed forward: channel mixing gated by invariant
        # scalars. An elementwise activation on xyz would break equivariance.
        atom_ffn_normed = self.atom_norm3(atom)
        if padding_mask is None:
            atom_context = atom_ffn_normed.mean(dim=1)
        else:
            valid = (~padding_mask).to(atom.dtype).unsqueeze(-1)
            atom_context = (atom_ffn_normed * valid).sum(dim=1)
            atom_context = atom_context / valid.sum(dim=1).clamp_min(1.0)

        vector_ffn_normed = self.vector_norm3(vector)
        gram = torch.einsum("bixd,bjxd->bij", vector_ffn_normed, vector_ffn_normed)
        gram = gram / vector.shape[-1]
        lattice_context = torch.cat((
            atom_context[:, None, :].expand(-1, 3, -1),
            gram,
        ), dim=-1)
        gate = torch.tanh(self.vector_gate(lattice_context)).unsqueeze(2)

        vector_hidden = self.vector_ffn_in(vector_ffn_normed)
        vector = vector + self.vector_ffn_out(vector_hidden * gate)
        return atom, vector


class CrystalDualStreamTransformer(nn.Module):
    """Stack of dual-stream blocks operating on precomputed features."""

    def __init__(self, dim, depth=4, heads=4, edge_dim=0, expansion=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrystalDualStreamBlock(dim, heads, edge_dim, expansion, dropout)
            for _ in range(depth)
        ])
        self.atom_norm = nn.LayerNorm(dim)

    def forward(self, atom, vector, edge_index=None, edge_attr=None,
                padding_mask=None, edges=None):
        for block in self.blocks:
            atom, vector = block(
                atom, vector, edge_index, edge_attr, padding_mask, edges=edges
            )
        return self.atom_norm(atom), vector


class CrystalDualStreamDenoiser(nn.Module):
    """Adapt :class:`CrystalDualStreamTransformer` to the flat-batch interface.

    ``diffusion.DiffCSP`` stores a batch as concatenated atoms plus graph
    boundaries and supplies one sinusoidal time embedding per graph. This
    wrapper embeds the atom types, adds the time embedding and a Fourier
    embedding of the fractional coordinates, pads atoms into ``[B, N]``,
    builds the padding mask, and flattens the per-atom score back to the
    original concatenated order.
    """

    def __init__(self, config: Optional[CrystalDualStreamConfig] = None):
        super().__init__()
        self.config = config or CrystalDualStreamConfig()
        config = self.config
        self.time_dim = config.time_dim

        self.atom_embedding = nn.Embedding(
            config.max_atomic_number + 1, config.hidden_dim
        )
        self.register_buffer(
            "frequencies",
            2 * math.pi * torch.arange(config.coord_frequencies),
        )
        # Edge features: Fourier embedding of the wrapped fractional
        # displacement (6 * F) plus the 9 lattice-metric products, as in CSPNet.
        self.edge_dim = 6 * config.coord_frequencies + 9
        self.lattice_embedding = LatticeVectorEmbedding(config.hidden_dim)
        self.time_projection = nn.Linear(config.time_dim, config.hidden_dim)
        self.net = CrystalDualStreamTransformer(
            config.hidden_dim,
            depth=config.num_layers,
            heads=config.heads,
            edge_dim=self.edge_dim,
            expansion=config.expansion,
            dropout=config.dropout,
        )
        self.frac_head = nn.Linear(config.hidden_dim, 3, bias=False)
        # Contract the equivariant vector channels into one lattice per graph.
        self.lattice_head = nn.Linear(config.hidden_dim, 1, bias=False)

    def forward(
        self,
        time_emb: torch.Tensor,
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattice: torch.Tensor,
        num_atoms: torch.Tensor,
        node2graph: torch.Tensor,
    ) -> tuple:
        batch = lattice.size(0)
        max_atoms = int(num_atoms.max())
        device = atom_types.device

        offsets = torch.cumsum(num_atoms, 0) - num_atoms
        positions = torch.arange(atom_types.size(0), device=device) - offsets[node2graph]

        padded_types = atom_types.new_zeros((batch, max_atoms))
        padded_frac = frac_coords.new_zeros((batch, max_atoms, 3))
        node_mask = torch.zeros((batch, max_atoms), dtype=torch.bool, device=device)
        padded_types[node2graph, positions] = atom_types
        padded_frac[node2graph, positions] = frac_coords
        node_mask[node2graph, positions] = True

        # Two edge layouts are supported for A/B comparison. With sparse_edges
        # only the sum(n_i^2) real atom pairs are materialised; otherwise the
        # dense padded grid [B, max_atoms^2, ...] is built.
        if self.config.sparse_edges:
            edges = build_sparse_edges(num_atoms, max_atoms, device)
            edge_index = None
            edge_attr = self._sparse_edge_attr(padded_frac, lattice, *edges)
        else:
            edges = None
            # Pairwise edges on the padded grid, ordered as (source=j, target=i)
            # to match the dense attention layout. Padded atoms are masked out
            # as keys by the attention itself.
            grid = torch.cartesian_prod(
                torch.arange(max_atoms, device=device),
                torch.arange(max_atoms, device=device),
            )
            source, target = grid[:, 1], grid[:, 0]
            edge_index = grid[:, [1, 0]].t().contiguous()

            # Fourier embedding of the wrapped fractional displacement, plus the
            # lattice-metric products, following CSPNet's edge features.
            frac_diff = (padded_frac[:, target] - padded_frac[:, source]) % 1.0
            phase = frac_diff.unsqueeze(-1) * self.frequencies
            frac_embedding = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
            lattice_products = (lattice @ lattice.transpose(-1, -2)).flatten(-2)
            edge_attr = torch.cat((
                frac_embedding,
                lattice_products[:, None, :].expand(-1, grid.size(0), -1),
            ), dim=-1)

        atom = self.atom_embedding(padded_types.long())
        # The transformer expects columns to be lattice vectors; data.py
        # stores lattice vectors as rows.
        vector = self.lattice_embedding(lattice.transpose(-1, -2))
        # Time conditioning is fused here, before the transformer: an additive
        # shift for the invariant atom features and a per-channel gate for the
        # equivariant lattice vectors.
        time_context = self.time_projection(time_emb)
        atom = atom + time_context[:, None, :]
        vector = vector * (1.0 + torch.tanh(time_context))[:, None, None, :]

        atom, vector = self.net(
            atom,
            vector,
            edge_index,
            edge_attr,
            padding_mask=~node_mask,
            edges=edges,
        )

        frac_score = self.frac_head(atom)
        lattice_noise = self.lattice_head(vector).squeeze(-1)
        return lattice_noise, frac_score[node_mask]

    def _sparse_edge_attr(self, padded_frac, lattice, edge_batch, edge_src,
                          edge_dst):
        """Per-edge features for the real edges; mirrors the dense branch."""
        frac_diff = (
            padded_frac[edge_batch, edge_dst]
            - padded_frac[edge_batch, edge_src]
        ) % 1.0
        phase = frac_diff.unsqueeze(-1) * self.frequencies
        frac_embedding = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        lattice_products = (lattice @ lattice.transpose(-1, -2)).flatten(-2)
        return torch.cat((
            frac_embedding,
            lattice_products[edge_batch],
        ), dim=-1)


__all__ = [
    "CrystalDualStreamTransformer",
    "CrystalDualStreamBlock",
    "CrystalDualStreamConfig",
    "CrystalDualStreamDenoiser",
    "EquivariantRMSNorm",
]


if __name__ == "__main__":
    torch.manual_seed(7)
    batch, nodes, dim, edge_count, edge_dim = 2, 8, 32, 18, 5

    atom = torch.randn(batch, nodes, dim, requires_grad=True)
    lattice = torch.randn(batch, 3, 3)
    edge_index = torch.randint(0, nodes, (2, edge_count))
    edge_attr = torch.randn(batch, edge_count, edge_dim)
    padding_mask = torch.zeros(batch, nodes, dtype=torch.bool)

    model = CrystalDualStreamTransformer(
        dim=dim, depth=2, heads=4, edge_dim=edge_dim
    )
    embedding = LatticeVectorEmbedding(dim)
    vector = embedding(lattice)

    atom_out, vector_out = model(
        atom, vector, edge_index, edge_attr, padding_mask
    )
    loss = atom_out.square().mean() + vector_out.square().mean()
    loss.backward()

    print("atom output:", atom_out.shape)
    print("lattice output:", vector_out.shape)
    print("backward: passed")

    # A global rotation must leave atom features unchanged and rotate
    # every Cartesian vector channel in the lattice stream.
    model.eval()
    rotation, _ = torch.linalg.qr(torch.randn(batch, 3, 3))
    rotated_atom, rotated_vector = model(
        atom.detach(), embedding(rotation @ lattice), edge_index, edge_attr,
        padding_mask
    )
    expected_vector = torch.einsum("bij,bmjd->bmid", rotation, vector_out.detach())
    print("atom rotation error:",
          (rotated_atom - atom_out.detach()).abs().max().item())
    print("lattice rotation error:",
          (rotated_vector - expected_vector).abs().max().item())
