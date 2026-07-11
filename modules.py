from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cpu_style_init_parameter(shape, device=None, generator=None):
    if len(shape) != 2:
        raise ValueError("cpu_style_init_parameter expects a 2D weight shape")
    fan_in, fan_out = shape
    weight = torch.randn(fan_in, fan_out, device=device, generator=generator) * math.sqrt(2.0 / fan_in)
    return nn.Parameter(weight)


class GraphPropagation(nn.Module):
    def __init__(self, d_h: int, generator=None, device=None):
        super().__init__()
        self.Wg = cpu_style_init_parameter((d_h, d_h), device=device, generator=generator)
        self.Ws = cpu_style_init_parameter((d_h, d_h), device=device, generator=generator)
        self.bg = nn.Parameter(torch.zeros(d_h, device=device))

    def forward(self, H: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        agg  = torch.matmul(A_hat, torch.matmul(H, self.Wg))
        skip = torch.matmul(H, self.Ws)
        return F.relu(agg + skip + self.bg)


class TemporalGraphEncoder(nn.Module):
    """Temporal graph encoder for anchor and auxiliary nodes."""

    def __init__(self, d_x: int, d_h: int, generator=None, device=None):
        super().__init__()
        self.d_x = d_x
        self.d_h = d_h
        self.gru = nn.GRU(input_size=d_x, hidden_size=d_h, batch_first=True)

        for name, param in self.gru.named_parameters():
            if "weight" in name and param.dim() == 2:
                fan_in, _ = param.shape
                with torch.no_grad():
                    rand = torch.randn(param.shape, device=param.device, dtype=param.dtype)
                    param.copy_(rand * math.sqrt(2.0 / fan_in))
            elif "bias" in name:
                with torch.no_grad():
                    param.zero_()

        self.graph_prop = GraphPropagation(d_h, generator=generator, device=device)
        self.readout_proj = nn.Linear(5 * d_h, d_h)
        nn.init.xavier_uniform_(self.readout_proj.weight)
        nn.init.zeros_(self.readout_proj.bias)
        self.pred_head = nn.Linear(d_h, d_x)
        nn.init.xavier_uniform_(self.pred_head.weight)
        nn.init.zeros_(self.pred_head.bias)

    def forward(
        self,
        X:       torch.Tensor,
        A_hat:   torch.Tensor,
        n_anchor: int,
        use_graph:        bool = True,
        return_graph_seq: bool = False,
        return_pred:      bool = False,
    ) -> Tuple:
        B, T, n_k, _ = X.shape
        self.gru.flatten_parameters()

        seq   = X.permute(0, 2, 1, 3).reshape(B * n_k, T, self.d_x)
        h_seq, _ = self.gru(seq)
        h_seq = h_seq.reshape(B, n_k, T, self.d_h).permute(0, 2, 1, 3)

        H_tilde = h_seq
        G = self.graph_prop(H_tilde, A_hat) if use_graph else H_tilde

        z_mean = G.mean(dim=(1, 2))                                        
        z_max  = G.amax(dim=(1, 2))                                        
        z_std  = G.std(dim=(1, 2))                                         
        G_anc  = G[:, :, :n_anchor, :]                                            
        z_anc  = G_anc.mean(dim=(1, 2))                                    
        n_k    = G.size(2)
        if n_k > n_anchor:
            z_aux = G[:, :, n_anchor:, :].mean(dim=(1, 2))                 
        else:
            z_aux = torch.zeros_like(z_mean)
        z = self.readout_proj(
            torch.cat([z_mean, z_max, z_std, z_anc, z_aux], dim=-1))
        if return_pred:
            if G.size(1) > 1:
                pred_next = self.pred_head(G[:, :-1])                      
            else:
                pred_next = None
            return (z, G if return_graph_seq else None, pred_next)
        return (z, G) if return_graph_seq else (z, None)

class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor) -> None:
        self.mask.data.copy_(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class MADEBlock(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.d      = d
        self.hidden = hidden
        self.fc1      = MaskedLinear(d, hidden)
        self.fc_mu    = MaskedLinear(hidden, d)
        self.fc_log_s = MaskedLinear(hidden, d)
        self._build_masks()
        for layer in [self.fc1, self.fc_mu, self.fc_log_s]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _build_masks(self) -> None:
        d, hidden = self.d, self.hidden
        m_h   = torch.arange(hidden) % max(1, d - 1)
        m_in  = torch.arange(d)
        m_out = torch.arange(d)
        mask1 = (m_h[:, None] >= m_in[None, :]).float()
        mask2 = (m_out[:, None] > m_h[None, :]).float()
        self.fc1.set_mask(mask1)
        self.fc_mu.set_mask(mask2)
        self.fc_log_s.set_mask(mask2)

    def forward(self, x: torch.Tensor):
        h     = F.relu(self.fc1(x))
        mu    = self.fc_mu(h)
        log_s = torch.clamp(self.fc_log_s(h), min=-5.0, max=3.0)
        return mu, log_s


class MAF(nn.Module):
    def __init__(self, d: int, n_blocks: int = 3, hidden: int = 48):
        super().__init__()
        self.blocks = nn.ModuleList([MADEBlock(d, hidden) for _ in range(n_blocks)])
        self.register_buffer("log_2pi",
            torch.tensor(math.log(2.0 * math.pi), dtype=torch.float32))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if z.dim() == 1:
            z       = z.unsqueeze(0)
            squeeze = True
        u = z
        log_det_total = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        for block in self.blocks:
            mu, log_s      = block(u)
            u              = (u - mu) * torch.exp(-log_s)
            log_det_total  = log_det_total - log_s.sum(dim=-1)
        log_p0 = -0.5 * (u.pow(2) + self.log_2pi).sum(dim=-1)
        out    = log_p0 + log_det_total
        return out.squeeze(0) if squeeze else out

def graph_smoothness_reg(G: torch.Tensor, A_raw: torch.Tensor) -> torch.Tensor:
    n = G.size(2)
    D  = torch.diag(A_raw.sum(dim=1))
    L  = D - A_raw
    LG = torch.einsum("ij,btjd->btid", L, G)
    trace_term = (G * LG).sum(dim=(-1, -2))
    return (2.0 / (n * n)) * trace_term.mean()


def temporal_reg(G: torch.Tensor) -> torch.Tensor:
    if G.size(1) <= 1:
        return torch.zeros((), device=G.device, dtype=G.dtype)
    n    = G.size(2)
    diff = G[:, 1:] - G[:, :-1]
    return (diff.pow(2).sum(dim=(-1, -2)) / n).mean()


def patch_continuity_reg(G: torch.Tensor) -> torch.Tensor:
    if G.size(1) <= 1:
        return torch.zeros((), device=G.device, dtype=G.dtype)
    n    = G.size(2)
    diff = G[:, 1:] - G[:, :-1]
    return (diff.pow(2).sum(dim=(-1, -2)) / n).mean()

class NPFormerGPEncoder(nn.Module):
    """
    Stage I encoder used by FedHGF.

    Pipeline:
        input  X:        [B, T, n_k, d_x]
        patches:         [B, n_k, P, L*d_x]    (P = (T-L)//S + 1)
        E (proj+posenc): [B*n_k, P, d_h]
        Y = Transformer(E):                    node-wise shared encoder
        reshape ->       [B, P, n_k, d_h]
        G = GraphProp(Y, A_hat):               per-patch graph propagation
        attentive patch pooling over P, per node -> u: [B, n_k, d_h]
        node average -> z: [B, d_h]

    Forward signature is kept compatible with TemporalGraphEncoder so that
    fedgad_full.py only needs to swap the constructor.
    """

    def __init__(
        self,
        d_x: int,
        d_h: int,
        window_size: int   = 96,
        patch_len:   int   = 16,
        patch_stride: int  = 8,
        num_layers:  int   = 2,
        num_heads:   int   = 4,
        ffn_dim:     int   = 128,
        dropout:     float = 0.1,
        generator=None,
        device=None,
    ):
        super().__init__()
        if patch_len <= 0:
            raise ValueError(f"patch_len must be positive, got {patch_len}.")
        if patch_stride <= 0:
            raise ValueError(f"patch_stride must be positive, got {patch_stride}.")
        if patch_len > window_size:
            raise ValueError(
                f"patch_len={patch_len} cannot exceed window_size={window_size}."
            )
        if d_h % num_heads != 0:
            raise ValueError(
                f"d_h={d_h} must be divisible by num_heads={num_heads}."
            )

        self.d_x          = d_x
        self.d_h          = d_h
        self.window_size  = int(window_size)
        self.patch_len    = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.num_patches  = (self.window_size - self.patch_len) // self.patch_stride + 1

        patch_dim = self.patch_len * d_x
        self.patch_proj = nn.Linear(patch_dim, d_h)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_h))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_h, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

        self.graph_prop  = GraphPropagation(d_h, generator=generator,
                                            device=device)
        self.norm_after_graph = nn.LayerNorm(d_h)
        self.gp_dropout       = nn.Dropout(dropout)

        self.attn_fc  = nn.Linear(d_h, d_h)
        self.attn_vec = nn.Linear(d_h, 1, bias=False)
        nn.init.xavier_uniform_(self.attn_fc.weight)
        nn.init.zeros_(self.attn_fc.bias)
        nn.init.xavier_uniform_(self.attn_vec.weight)

        self.readout_proj = nn.Linear(5 * d_h, d_h)
        nn.init.xavier_uniform_(self.readout_proj.weight)
        nn.init.zeros_(self.readout_proj.bias)

        self.pred_head = nn.Linear(d_h, d_x)
        nn.init.xavier_uniform_(self.pred_head.weight)
        nn.init.zeros_(self.pred_head.bias)

        self.P = self.num_patches

    def _make_patches(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, T, n_k, d_x]  ->  [B, n_k, P, L*d_x]
        """
        B, T, n_k, D = X.shape
        L, S = self.patch_len, self.patch_stride
        x_perm = X.permute(0, 2, 1, 3).contiguous()
        x_unf  = x_perm.unfold(dimension=2, size=L, step=S)                 
        x_unf  = x_unf.permute(0, 1, 2, 4, 3).contiguous()                   
        patches = x_unf.reshape(B, n_k, x_unf.size(2), L * D)
        return patches

    def forward(
        self,
        X:       torch.Tensor,
        A_hat:   torch.Tensor,
        n_anchor: int = 0,
        use_graph:        bool = True,
        return_graph_seq: bool = False,
        return_pred:      bool = False,
    ):
        B, T, n_k, D = X.shape

        if T != self.window_size:
            raise ValueError(
                f"NPFormerGPEncoder: input window length T={T} does not "
                f"match encoder.window_size={self.window_size}. Set "
                f"cfg['window_size'] to the loader's window_len, or leave "
                f"cfg['window_size']=None so it is auto-inferred from data."
            )

        patches = self._make_patches(X)                                   
        P = patches.size(2)
                                             
        E = self.patch_proj(patches.view(B * n_k, P, self.patch_len * D))
        E = E + self.pos_embed
        Y = self.transformer(E)                                             
        Y = Y.view(B, n_k, P, self.d_h).permute(0, 2, 1, 3).contiguous()
                             

        if use_graph:
            G_prop = self.graph_prop(Y, A_hat)
            G      = self.norm_after_graph(Y + self.gp_dropout(G_prop))
        else:
            G = Y

        score = self.attn_vec(torch.tanh(self.attn_fc(G))).squeeze(-1)
        alpha = torch.softmax(score, dim=1)
        u     = (G * alpha.unsqueeze(-1)).sum(dim=1)                       

        z_mean = u.mean(dim=1)                                        
        z_max  = u.amax(dim=1)                                        
        z_std  = u.std(dim=1) if u.size(1) > 1 else torch.zeros_like(z_mean)
        if n_anchor > 0 and n_anchor <= n_k:
            z_anc = u[:, :n_anchor, :].mean(dim=1)                    
        else:
            z_anc = torch.zeros_like(z_mean)
        if n_k > n_anchor:
            z_aux = u[:, n_anchor:, :].mean(dim=1)                    
        else:
            z_aux = torch.zeros_like(z_mean)
        z = self.readout_proj(
            torch.cat([z_mean, z_max, z_std, z_anc, z_aux], dim=-1)
        )                                                             

        if return_pred:
            if G.size(1) > 1:
                pred_next = self.pred_head(G[:, :-1])                        
            else:
                pred_next = None
            return (z, G if return_graph_seq else None, pred_next)
        return (z, G) if return_graph_seq else (z, None)


def variance_floor_reg(Z: torch.Tensor, gamma: float = 1.0,
                        eps_v: float = 1e-4) -> torch.Tensor:
    if Z.size(0) < 2:
        return torch.zeros((), device=Z.device, dtype=Z.dtype)
    var = Z.var(dim=0, unbiased=False)
    std = torch.sqrt(var + eps_v)
    return F.relu(gamma - std).mean()
