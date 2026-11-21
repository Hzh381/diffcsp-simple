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
    coord_frequencies: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.time_dim % 2:
            raise ValueError("time_dim must be even")


class InvariantGraphAttention(nn.Module):
    """Dense graph attention over invariant atom features.

    For edge j -> i:
        score[i,j] = q[i] dot k[j] / sqrt(d) + edge_bias[i,j]
        message[i,j] = v[j] + edge_value[i,j]
        output[i] = sum_j softmax_j(score[i,j]) * message[i,j]

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
            self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
            self.edge_value = nn.Linear(edge_dim, dim, bias=False)
        else:
            self.edge_bias = None
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

    def forward(self, atom, edge_index=None, edge_attr=None, padding_mask=None):
        batch, nodes, _ = atom.shape

        # Q, K, V: [B, heads, N, head_dim]
        q, k, v = self.qkv(atom).chunk(3, dim=-1)
        q = q.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, nodes, self.heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, heads, N, N].
        score = q @ k.transpose(-2, -1)
        score = score / math.sqrt(self.head_dim)

        if edge_index is None:
            graph_mask = torch.ones(
                batch, nodes, nodes, dtype=torch.bool, device=atom.device
            )
        else:
            graph_mask = self._make_graph_mask(
                edge_index, batch, nodes, atom.device
            )

        dense_edge_attr = None
        if self.edge_dim > 0:
            if edge_index is None:
                raise ValueError("edge_index is required when edge_dim > 0")
            dense_edge_attr = self._make_edge_features(
                edge_index, edge_attr, batch, nodes, atom.device
            )
            edge_bias = self.edge_bias(dense_edge_attr).permute(0, 3, 1, 2)
            score = score + edge_bias

        if padding_mask is not None:
            # Padded atoms cannot be used as keys/values.
            graph_mask = graph_mask & (~padding_mask[:, None, :])

        score = score.masked_fill(~graph_mask[:, None, :, :], -1e9)
        attention = self.dropout(torch.softmax(score, dim=-1))

        # Weighted sum of atom values.
        message = attention @ v
        message = message.transpose(1, 2).reshape(batch, nodes, self.dim)

        if dense_edge_attr is not None:
            edge_value = self.edge_value(dense_edge_attr)
            edge_value = edge_value.view(
                batch, nodes, nodes, self.heads, self.head_dim
            ).permute(0, 3, 1, 2, 4)
            message = message + (attention.unsqueeze(-1) * edge_value).sum(dim=3).transpose(1, 2).reshape(
                batch, nodes, self.dim
            )

        return self.output(message)


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
        self.eps = eps

    def forward(self, vector, scale=None):
        rms = vector.pow(2).mean(dim=(2, 3), keepdim=True)
        vector = vector * torch.rsqrt(rms + self.eps)
        if scale is not None:
            # Per-channel modulation; a per-channel scalar multiplies all
            # three Cartesian components equally, so equivariance is kept.
            # This is the only affine term, mirroring adaLN in DiT.
            vector = vector * (1.0 + scale)[:, None, None, :]
        return vector


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

    def __init__(self, dim, heads=4, edge_dim=0, expansion=4, dropout=0.0,
                 time_dim=256):
        super().__init__()
        hidden = dim * expansion

        # DiT-style conditioning: the time embedding produces one adaptive
        # parameter per normalization layer. The three atom LayerNorms get a
        # (scale, shift) pair each (a1/b1, a2/b2, a3/b3); the three equivariant
        # RMSNorms get a scale each (a4, a5, a6), since an additive shift on
        # Cartesian components would break rotation equivariance.
        # Zero-initialized, so every norm starts unmodulated.
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 9 * dim),
        )
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

        # Invariant atom stream. No built-in affine: the shift/scale of every
        # norm come from the time modulation, as in DiT's adaLN.
        self.atom_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.atom_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.atom_norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
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
                padding_mask=None, time_embed=None):
        if time_embed is None:
            mods = (None,) * 9
        else:
            mods = self.modulation(time_embed).chunk(9, dim=-1)
        a1, b1, a2, b2, a3, b3, a4, a5, a6 = mods

        def atom_norm(norm, x, scale, shift):
            x = norm(x)
            if scale is not None:
                x = x * (1.0 + scale[:, None]) + shift[:, None]
            return x

        # 1. Intra-stream attention: pre-norm, residual bypass.
        atom = atom + self.graph_attention(
            atom_norm(self.atom_norm1, atom, a3, b3),
            edge_index, edge_attr, padding_mask,
        )
        vector = vector + self.vector_attention(self.vector_norm1(vector, a4))

        # 2. Cross-stream attention: both blocks consume the pre-cross
        # normalized features of both streams (the crossing arrows).
        atom_normed = atom_norm(self.atom_norm2, atom, a2, b2)
        vector_normed = self.vector_norm2(vector, a5)
        atom = atom + self.atom_cross_attention(atom_normed, vector_normed)
        vector = vector + self.vector_cross_attention(
            vector_normed, atom_normed, padding_mask
        )

        # 3. Atom feed forward.
        atom = atom + self.atom_ffn(atom_norm(self.atom_norm3, atom, a1, b1))

        # 4. Equivariant feed forward: channel mixing gated by invariant
        # scalars. An elementwise activation on xyz would break equivariance.
        atom_for_gate = atom_norm(self.atom_norm3, atom, a1, b1)
        if padding_mask is None:
            atom_context = atom_for_gate.mean(dim=1)
        else:
            valid = (~padding_mask).to(atom.dtype).unsqueeze(-1)
            atom_context = (atom_for_gate * valid).sum(dim=1)
            atom_context = atom_context / valid.sum(dim=1).clamp_min(1.0)

        vector_normed = self.vector_norm3(vector, a6)
        gram = torch.einsum("bixd,bjxd->bij", vector_normed, vector_normed)
        gram = gram / vector.shape[-1]
        lattice_context = torch.cat((
            atom_context[:, None, :].expand(-1, 3, -1),
            gram,
        ), dim=-1)
        gate = torch.tanh(self.vector_gate(lattice_context)).unsqueeze(2)

        vector_hidden = self.vector_ffn_in(vector_normed)
        vector = vector + self.vector_ffn_out(vector_hidden * gate)
        return atom, vector


class CrystalDualStreamTransformer(nn.Module):
    """Complete multi-layer model; includes its own graph attention layers."""

    def __init__(self, dim, depth=4, heads=4, edge_dim=0, expansion=4, dropout=0.0,
                 time_dim=256):
        super().__init__()
        self.lattice_embedding = LatticeVectorEmbedding(dim)
        self.blocks = nn.ModuleList([
            CrystalDualStreamBlock(
                dim, heads, edge_dim, expansion, dropout, time_dim
            )
            for _ in range(depth)
        ])
        self.atom_norm = nn.LayerNorm(dim)

    def forward(self, atom, lattice, edge_index=None, edge_attr=None,
                padding_mask=None, time_embed=None):
        vector = self.lattice_embedding(lattice)
        for block in self.blocks:
            atom, vector = block(
                atom, vector, edge_index, edge_attr, padding_mask, time_embed
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
        self.coord_embedding = nn.Linear(
            6 * config.coord_frequencies, config.hidden_dim
        )
        self.net = CrystalDualStreamTransformer(
            config.hidden_dim,
            depth=config.num_layers,
            heads=config.heads,
            edge_dim=0,
            expansion=config.expansion,
            dropout=config.dropout,
            time_dim=config.time_dim,
        )
        self.frac_head = nn.Linear(config.hidden_dim, 3, bias=False)
        # Contract the equivariant vector channels into one lattice per graph.
        self.lattice_head = nn.Linear(config.hidden_dim, 1, bias=False)
        # Turn the invariant lattice metric into a 3x3 coefficient matrix.
        self.lattice_coeff = nn.Sequential(
            nn.Linear(9, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 9),
        )

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

        phase = padded_frac.unsqueeze(-1) * self.frequencies
        coord_features = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(-2)

        atom = self.atom_embedding(padded_types.long())
        atom = atom + self.coord_embedding(coord_features)
        # The transformer expects columns to be lattice vectors; data.py
        # stores lattice vectors as rows.
        atom, vector = self.net(
            atom,
            lattice.transpose(-1, -2),
            edge_index=None,
            edge_attr=None,
            padding_mask=~node_mask,
            time_embed=time_emb,
        )

        frac_score = self.frac_head(atom)
        # Row Gram: G[i,j] = sum_xyz H[i,xyz] * H[j,xyz] is rotation invariant
        # because H -> H R^T gives H R^T R H^T = H H^T. The coefficient matrix
        # is therefore invariant, and C @ L rotates together with the lattice.
        head = self.lattice_head(vector).squeeze(-1)
        gram = torch.bmm(head, head.transpose(-1, -2))
        coeff = self.lattice_coeff(gram.flatten(-2)).view(-1, 3, 3)
        lattice_noise = torch.bmm(coeff, lattice)
        return lattice_noise, frac_score[node_mask]


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
    time_embed = torch.randn(batch, 256)

    atom_out, vector_out = model(
        atom, lattice, edge_index, edge_attr, padding_mask, time_embed
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
        atom.detach(), rotation @ lattice, edge_index, edge_attr,
        padding_mask, time_embed
    )
    expected_vector = torch.einsum("bij,bmjd->bmid", rotation, vector_out.detach())
    print("atom rotation error:",
          (rotated_atom - atom_out.detach()).abs().max().item())
    print("lattice rotation error:",
          (rotated_vector - expected_vector).abs().max().item())
