from __future__ import annotations

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from modules import (MAF, TemporalGraphEncoder, NPFormerGPEncoder,
                     graph_smoothness_reg, temporal_reg,
                     patch_continuity_reg, variance_floor_reg)


                                                                             
      
                                                                             

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_norm_adj(A: np.ndarray) -> np.ndarray:
    A_hat = A + np.eye(A.shape[0], dtype=np.float32)
    D     = A_hat.sum(axis=1)
    D_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-12)))
    return (D_inv @ A_hat @ D_inv).astype(np.float32)


def build_signed_norm_adj(A_signed: np.ndarray) -> np.ndarray:
    A_hat = A_signed + np.eye(A_signed.shape[0], dtype=np.float32)
    D     = np.abs(A_signed).sum(axis=1) + 1.0
    D_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-12)))
    return (D_inv @ A_hat @ D_inv).astype(np.float32)


def clip_perturb_np(v: np.ndarray, C: float, sigma: float,
                    rng: np.random.RandomState) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm > C:
        v = v * (C / max(norm, 1e-12))
    if sigma > 0:
        v = v + rng.normal(0.0, sigma * C, size=v.shape).astype(np.float32)
    return v.astype(np.float32)


def clip_perturb_torch(v: torch.Tensor, C: float, sigma: float,
                       generator: torch.Generator | None = None) -> torch.Tensor:
    flat    = v.reshape(-1)
    norm    = torch.linalg.vector_norm(flat)
    clipped = flat * torch.clamp(C / (norm + 1e-12), max=1.0)
    if sigma > 0:
        noise   = torch.randn(clipped.shape, device=clipped.device,
                              dtype=clipped.dtype,
                              generator=generator) * (sigma * C)
        clipped = clipped + noise
    return clipped.reshape_as(v)


                                                                             
                  
                                                                             

STAGE_PRIMARY_ANCHOR: Dict[str, str] = {
    "stage1": "FIT101",
    "stage2": "FIT201",
    "stage3": "FIT301",
    "stage4": "FIT401",
    "stage5": "FIT501",
    "stage6": "FIT601",
}


def build_full_client_graph(
    A_anc:              np.ndarray,
    anchor_cols:        List[str],
    aux_cols:           List[str],
    stage_name:         str,
    rho:                float = 0.5,
    aux_corr:           Optional[np.ndarray] = None,
    aux_corr_threshold: float = 0.25,
    anchor_aux_corr:    Optional[np.ndarray] = None,
    n_top_anchors:      int   = 2,
    return_signed:      bool  = False,
) -> np.ndarray:
    n_anchor = len(anchor_cols)
    n_aux    = len(aux_cols)
    n_total  = n_anchor + n_aux

    A = np.zeros((n_total, n_total), dtype=np.float32)
    S = np.zeros((n_total, n_total), dtype=np.float32)
    A[:n_anchor, :n_anchor] = np.abs(A_anc)
    S[:n_anchor, :n_anchor] = A_anc

    if n_aux == 0:
        return (A, S) if return_signed else A

    if (anchor_aux_corr is not None
            and anchor_aux_corr.shape == (n_aux, n_anchor)):
        n_top = min(n_top_anchors, n_anchor)
        for ai in range(n_aux):
            abs_row  = np.abs(anchor_aux_corr[ai])
            top_idxs = np.argsort(abs_row)[::-1][:n_top]
            floor_w = rho / max(n_anchor, 1)
            for anc_idx in top_idxs:
                raw = float(anchor_aux_corr[ai, anc_idx])
                w = max(abs(raw) * rho, floor_w)
                A[anc_idx,       n_anchor + ai] = w
                A[n_anchor + ai, anc_idx      ] = w
                S[anc_idx,       n_anchor + ai] = np.sign(raw) * w
                S[n_anchor + ai, anc_idx      ] = np.sign(raw) * w
    else:
        primary = STAGE_PRIMARY_ANCHOR.get(stage_name)
        if primary and primary in anchor_cols:
            anc_idx = anchor_cols.index(primary)
            for ai in range(n_aux):
                A[anc_idx,       n_anchor + ai] = rho
                A[n_anchor + ai, anc_idx      ] = rho
                S[anc_idx,       n_anchor + ai] = rho
                S[n_anchor + ai, anc_idx      ] = rho
        else:
            w = rho / max(n_anchor, 1)
            A[:n_anchor, n_anchor:] = w
            A[n_anchor:, :n_anchor] = w
            S[:n_anchor, n_anchor:] = w
            S[n_anchor:, :n_anchor] = w

    if aux_corr is not None and aux_corr.shape == (n_aux, n_aux):
        for i in range(n_aux):
            for j in range(i + 1, n_aux):
                v = float(abs(aux_corr[i, j]))
                if v >= aux_corr_threshold:
                    A[n_anchor + i, n_anchor + j] = v
                    A[n_anchor + j, n_anchor + i] = v
                    sv = np.sign(float(aux_corr[i, j])) * v
                    S[n_anchor + i, n_anchor + j] = sv
                    S[n_anchor + j, n_anchor + i] = sv
    else:
        half_rho = rho * 0.5
        for i in range(n_aux):
            for j in range(i + 1, n_aux):
                A[n_anchor + i, n_anchor + j] = half_rho
                A[n_anchor + j, n_anchor + i] = half_rho
                S[n_anchor + i, n_anchor + j] = half_rho
                S[n_anchor + j, n_anchor + i] = half_rho

    return (A, S) if return_signed else A


def estimate_aux_correlation(
    X_train: np.ndarray,
    n_anchor: int,
    relation_value_weight: float = 0.5,
) -> Optional[np.ndarray]:
    N, T, n_k, _ = X_train.shape
    n_aux = n_k - n_anchor
    if n_aux <= 1:
        return None

    corr = _relation_corr_from_windows(
        X_train[:, :, n_anchor:, :],
        relation_value_weight=relation_value_weight)
    np.fill_diagonal(corr, 0.0)

    return corr.astype(np.float32)



def _safe_corrcoef(flat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = flat.shape[0]
    stds = flat.std(axis=1)                
    valid = np.where(stds > eps)[0]             
    R = np.zeros((n, n), dtype=np.float32)
    if len(valid) >= 2:
        with np.errstate(invalid='ignore', divide='ignore'):
            R_sub = np.corrcoef(flat[valid]).astype(np.float32)
        R_sub = np.nan_to_num(R_sub, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(R_sub, 0.0)
        R[np.ix_(valid, valid)] = R_sub
    return R


def _relation_corr_from_windows(
    X: np.ndarray,
    relation_value_weight: float = 0.5,
) -> np.ndarray:
    vals = X[..., 0] if X.ndim == 4 else X
    N, T, n = vals.shape
    flat_v = vals.transpose(2, 0, 1).reshape(n, N * T)
    R_v = _safe_corrcoef(flat_v)
    if T > 1:
        dvals = vals[:, 1:, :] - vals[:, :-1, :]
        flat_d = dvals.transpose(2, 0, 1).reshape(n, N * (T - 1))
        R_d = _safe_corrcoef(flat_d)
    else:
        R_d = np.zeros_like(R_v)
    lam = float(np.clip(relation_value_weight, 0.0, 1.0))
    R = lam * R_v + (1.0 - lam) * R_d
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(R, 0.0)
    return R


def estimate_anchor_aux_correlation(
    X_train: np.ndarray,
    n_anchor: int,
    relation_value_weight: float = 0.5,
) -> Optional[np.ndarray]:
    N, T, n_k, _ = X_train.shape
    n_aux = n_k - n_anchor
    if n_aux == 0 or n_anchor == 0:
        return None

    def _cross_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        a_std = np.std(a, axis=1, keepdims=True)
        b_std = np.std(b, axis=1, keepdims=True)
        a_valid = a_std[:, 0] > 1e-8
        b_valid = b_std[:, 0] > 1e-8
        a = a / (a_std + 1e-8)
        b = b / (b_std + 1e-8)
        out = (a @ b.T / max(a.shape[1], 1)).astype(np.float32)
        out[~a_valid, :] = 0.0
        out[:, ~b_valid] = 0.0
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    anc_v = X_train[:, :, :n_anchor, 0].transpose(2, 0, 1).reshape(n_anchor, N * T)
    aux_v = X_train[:, :, n_anchor:, 0].transpose(2, 0, 1).reshape(n_aux, N * T)
    corr_v = _cross_corr(aux_v, anc_v)
    if T > 1:
        dX = X_train[:, 1:, :, 0] - X_train[:, :-1, :, 0]
        anc_d = dX[:, :, :n_anchor].transpose(2, 0, 1).reshape(n_anchor, N * (T - 1))
        aux_d = dX[:, :, n_anchor:].transpose(2, 0, 1).reshape(n_aux, N * (T - 1))
        corr_d = _cross_corr(aux_d, anc_d)
    else:
        corr_d = np.zeros_like(corr_v)
    lam = float(np.clip(relation_value_weight, 0.0, 1.0))
    corr = (lam * corr_v + (1.0 - lam) * corr_d).astype(np.float32)

    return corr


                                                                             
      
                                                                             

def estimate_anchor_graph(
    clients:            List[dict],
    n_anchor:           int,
    M:                  int   = 3,
    C_g:                float = 1.0,
    sigma_g:            float = 0.1,
    use_dp:             bool  = True,
    rng:                Optional[np.random.RandomState] = None,
    use_physical_prior: bool  = False,
    prior_weight:       float = 0.3,
    relation_value_weight: float = 0.5,
) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState(42)

    total = sum(c["X_train"].shape[0] for c in clients)
    agg   = None
    for c in clients:
        X              = c["X_train"][:, :, :n_anchor, :]
        N, T, n0, d_x  = X.shape
        R              = _relation_corr_from_windows(
            X, relation_value_weight=relation_value_weight)
        r  = R[np.triu_indices(n0, k=1)].astype(np.float32)
        if use_dp:
            r = clip_perturb_np(r, C_g, sigma_g, rng)
        w   = N / total
        agg = w * r if agg is None else agg + w * r

    R_bar    = np.zeros((n_anchor, n_anchor), dtype=np.float32)
    iu       = np.triu_indices(n_anchor, k=1)
    R_bar[iu] = agg
    R_bar    = R_bar + R_bar.T

    absR     = np.abs(R_bar)
    A_signed = np.zeros_like(R_bar)
    for i in range(n_anchor):
        row    = absR[i].copy()
        row[i] = -1.0
        top    = np.argsort(row)[-M:]
        for j in top:
            if row[j] > 0:
                A_signed[i, j] = R_bar[i, j]
                A_signed[j, i] = R_bar[j, i]

    return A_signed.astype(np.float32)


                                                                             
      
                                                                             

def _check_collapse(Z: torch.Tensor, round_idx: int, n_rounds: int,
                    std_thr: float = 0.01) -> bool:
    with torch.no_grad():
        z_std   = Z.std(dim=0).mean().item()
        z_norm  = Z.norm(dim=1).mean().item()
        z_range = (Z.max() - Z.min()).item()
    status = "collapsed" if z_std < std_thr else "stable"
    print(f"      Round {round_idx}/{n_rounds} | "
          f"embed_std={z_std:.5f}  embed_norm={z_norm:.4f}  "
          f"embed_range={z_range:.4f}  {status}")
    return z_std < std_thr


                                                                             
        
                                                                             

class FedGAD:
    def __init__(
        self,
        d_x:    int   = 1,
        d_h:    int   = 32,
        n_anchor: int = 8,
        M:      int   = 3,
        rho:    float = 0.5,
        n_rounds:     int   = 10,
        local_epochs: int   = 2,
        lr:           float = 1e-3,
        batch_size:   int   = 32,
        lambda_g:     float = 0.01,
        lambda_t:     float = 0.01,
        lambda_v:     float = 0.05,
        lambda_r:     float = 1e-5,
        mu_prox:       float = 0.0,
        gamma_var:    float = 2.0,
        C_g:    float = 1.0,
        sigma_g: float = 1.0,
        C_theta: float = 1.0,
        sigma_theta: float = 1.0,
        C_c:    float = 1.0,
        sigma_c: float = 1.0,
        use_dp: bool  = True,
        flow_blocks:  int   = 3,
        flow_hidden:  int   = 128,
        flow_epochs:  int   = 15,
        flow_lr:      float = 1e-3,
        lambda_phi:   float = 1e-5,
        alpha:        float = 0.05,
        w_fusion: Tuple = (0.35, 0.35, 0.30),
        use_graph:       bool  = True,
        use_flow:        bool  = True,
        use_calibration: bool  = True,
        score_mode:      str   = "both",
        eta_s:           float = 1.0,
        collapse_patience: int   = 2,
        collapse_std_thr:  float = 0.01,
        use_graph_residual:    bool  = True,
        graph_residual_weight: float = 0.30,
        aux_corr_threshold:    float = 0.25,
        graph_residual_mode:   str   = "diff_max",
        use_anchor_graph:           bool  = True,
        n_top_anchors:              int   = 2,
        lambda_c:   float = 1.0,
        use_data_driven_cross_block: bool  = False,
        hybrid_center_alpha:   float = 0.0,
        adaptive_hybrid_alpha: bool  = False,
        use_prediction_loss:   bool  = False,
        lambda_pred:           float = 1.0,
        center_score_mode:     str   = "local",
        center_hybrid_beta:    float = 0.5,                                             
        encoder_type: str = "npformer_gp",                          
        window_size:  int = None,                                                          
        patch_len:    int = 4,
        patch_stride: int = 2,
        tf_layers:    int = 2,
        tf_heads:     int = 4,
        tf_ffn:       int = 128,
        tf_dropout:   float = 0.1,
        graph_in_encoder: bool = None,
        relation_value_weight: float = 0.5,
        device: str = "cuda",
        seed:   int = 42,
    ):
        self.cfg = {k: v for k, v in locals().items() if k != "self"}
        set_global_seed(seed)
        self.device = torch.device(
            device if device == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.rng             = np.random.RandomState(seed)
        self.torch_generator = torch.Generator(device=self.device.type)
        self.torch_generator.manual_seed(seed)

        self.encoder = None
        self.center  = None
        self.A_anc   = None

        self.client_flows:          Dict[int, MAF]                         = {}
        self.client_cal:            Dict[int, Dict[str, np.ndarray]]       = {}
        self.client_fusion_weights: Dict[int, Tuple]                       = {}
        self.client_graphs:         List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.client_signed_graphs:  List[torch.Tensor] = []
        self.client_aux_corrs:         Dict[int, Optional[np.ndarray]]             = {}
        self.client_anchor_aux_corrs:  Dict[int, Optional[np.ndarray]]             = {}
        self.client_local_centers:     Dict[int, np.ndarray]                        = {}
        self.client_alpha_ks:          Dict[int, float]                              = {}
        self.round_history:            List[dict]                                   = []
        self._stage1_checkpoints:       List[dict]                                   = []

                                                                        
    def _to_device(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _encode_dataset(self, X_np, encoder, A_hat, n_anchor,
                        use_graph, batch_size):
        zs           = []
        was_training = encoder.training
        encoder.eval()
        with torch.no_grad():
            for s in range(0, len(X_np), batch_size):
                xb   = self._to_device(X_np[s:s + batch_size])
                z, _ = encoder(xb, A_hat, n_anchor=n_anchor,
                               use_graph=use_graph, return_graph_seq=False)
                zs.append(z)
        if was_training:
            encoder.train()
        Z = (torch.cat(zs, dim=0) if zs
             else torch.empty((0, self.cfg["d_h"]), device=self.device))
        return Z, None

    def _prepare_graphs(self, clients: List[dict]) -> None:
        cfg = self.cfg
        self.client_graphs = []
        self.client_signed_graphs = []
        for k, c in enumerate(clients):
            if cfg["use_graph"]:
                A_anc_used = (
                    self.A_anc if cfg.get("use_anchor_graph", True)
                    else np.zeros_like(self.A_anc)
                )
                A_raw_np, A_signed_np = build_full_client_graph(
                    A_anc              = A_anc_used,
                    anchor_cols        = c["anchor_names"],
                    aux_cols           = c["aux_names"],
                    stage_name         = c["client_name"],
                    rho                = cfg["rho"],
                    aux_corr           = self.client_aux_corrs.get(k),
                    aux_corr_threshold = cfg["aux_corr_threshold"],
                    anchor_aux_corr    = self.client_anchor_aux_corrs.get(k),
                    n_top_anchors      = cfg.get("n_top_anchors", 2),
                    return_signed      = True,
                )
            else:
                n_k = c["n_k"]
                A_raw_np = np.zeros((n_k, n_k), dtype=np.float32)
                A_signed_np = np.zeros((n_k, n_k), dtype=np.float32)
            A_hat_np = build_norm_adj(A_raw_np)
            A_signed_hat_np = build_signed_norm_adj(A_signed_np)
            self.client_graphs.append(
                (self._to_device(A_raw_np), self._to_device(A_hat_np))
            )
            self.client_signed_graphs.append(self._to_device(A_signed_hat_np))

                                                                        
                 
                                                                        
    def _compute_graph_residual_score(
        self,
        X_np:  np.ndarray,
        A_hat: torch.Tensor,
    ) -> np.ndarray:
        mode  = self.cfg.get("graph_residual_mode", "diff_max")
        X     = torch.as_tensor(X_np, dtype=torch.float32, device=self.device)
        B, T, n_k, d_x = X.shape
        A     = A_hat[:n_k, :n_k]

        if mode == "mean_only":
            X_mean = X.mean(dim=1)                                                    
            X_pred = torch.einsum('ij,bjd->bid', A, X_mean)                           
            residual = ((X_mean - X_pred) ** 2).mean(dim=(-2, -1))         
        else:
            dX      = X[:, 1:] - X[:, :-1]                                                
            dX_pred = torch.einsum('ij,btjd->btid', A, dX)                                
            r_t     = ((dX - dX_pred) ** 2).mean(dim=(-2, -1))                  

            if mode == "diff_max":
                residual = r_t.max(dim=-1).values
            elif mode == "diff_std":
                residual = r_t.std(dim=-1)
            elif mode == "diff_mean":
                residual = r_t.mean(dim=-1)
            elif mode == "diff_q90":
                residual = torch.quantile(r_t, 0.90, dim=-1)
            elif mode == "diff_topk":
                k_top = max(1, int(0.2 * r_t.shape[-1]))
                residual = r_t.topk(k_top, dim=-1).values.mean(dim=-1)
            elif mode == "node_topk_q90":
                                                                                  
                dX2      = X[:, 1:] - X[:, :-1]                                        
                dX2_pred = torch.einsum('ij,btjd->btid', A, dX2)                        
                node_r   = ((dX2 - dX2_pred) ** 2).mean(dim=-1)                     
                k_node   = max(1, max(1, int(0.1 * node_r.shape[-1])))
                top_node = node_r.topk(k_node, dim=-1).values.mean(dim=-1)           
                residual = torch.quantile(top_node, 0.90, dim=1)                 
            else:
                raise ValueError(f"Unknown graph_residual_mode: {mode!r}")

        return residual.detach().cpu().numpy().astype(np.float32)

                                                                        
    def fit(self, clients: List[dict]) -> None:
        cfg = self.cfg
                                                              
                                                                        
                                                                    
        if cfg.get("graph_in_encoder") is None:
            cfg["graph_in_encoder"] = cfg["use_graph"]
        t0  = time.time()

        if cfg["use_graph"]:
            for k, c in enumerate(clients):
                self.client_aux_corrs[k] = estimate_aux_correlation(
                    c["X_train"], cfg["n_anchor"],
                    relation_value_weight=cfg.get("relation_value_weight", 0.5),
                )
                if cfg.get("use_data_driven_cross_block", False):
                    self.client_anchor_aux_corrs[k] = estimate_anchor_aux_correlation(
                        c["X_train"], cfg["n_anchor"],
                        relation_value_weight=cfg.get("relation_value_weight", 0.5),
                    )

        self.A_anc = (
            estimate_anchor_graph(
                clients, cfg["n_anchor"], cfg["M"],
                cfg["C_g"], cfg["sigma_g"], cfg["use_dp"], self.rng,
                relation_value_weight=cfg.get("relation_value_weight", 0.5),
            )
            if cfg["use_graph"]
            else np.zeros((cfg["n_anchor"], cfg["n_anchor"]), dtype=np.float32)
        )

        self._prepare_graphs(clients)

        encoder_type = str(cfg.get("encoder_type", "npformer_gp")).lower()
        if encoder_type in ("npformer_gp", "npformer", "patch"):
                                                                           
                                                                            
                                                                          
                                                
            win_size = cfg.get("window_size", None)
            if win_size is None:
                win_size = clients[0]["X_train"].shape[1]
                                                                            
                                                                      
            for c in clients:
                if c["X_train"].shape[1] != win_size:
                    raise ValueError(
                        f"Client {c.get('client_name', '?')} has "
                        f"window_len={c['X_train'].shape[1]}, but resolved "
                        f"encoder.window_size={win_size}. All clients must "
                        f"share the same window length."
                    )
            self.encoder = NPFormerGPEncoder(
                d_x=cfg["d_x"], d_h=cfg["d_h"],
                window_size=int(win_size),
                patch_len=int(cfg.get("patch_len", 4)),
                patch_stride=int(cfg.get("patch_stride", 2)),
                num_layers=int(cfg.get("tf_layers", 2)),
                num_heads=int(cfg.get("tf_heads", 4)),
                ffn_dim=int(cfg.get("tf_ffn", 128)),
                dropout=float(cfg.get("tf_dropout", 0.1)),
                generator=self.torch_generator, device=self.device,
            ).to(self.device)
            print(f"    Encoder: NPFormer-GP  "
                  f"T={self.encoder.window_size}  L={self.encoder.patch_len}  "
                  f"S={self.encoder.patch_stride}  P={self.encoder.num_patches}  "
                  f"layers={cfg.get('tf_layers', 2)}  heads={cfg.get('tf_heads', 4)}")
        elif encoder_type in ("gru", "temporal_gru"):
            self.encoder = TemporalGraphEncoder(
                cfg["d_x"], cfg["d_h"],
                generator=self.torch_generator, device=self.device
            ).to(self.device)
            print("    Encoder: TemporalGraphEncoder (GRU fallback)")
        else:
            raise ValueError(f"Unknown encoder_type={encoder_type!r}; "
                             f"expected one of {{'npformer_gp','gru'}}.")

        self._stage1(clients)
        print(f"    Stage I done ({time.time() - t0:.1f}s) on {self.device}")

                                                                   
        if cfg.get("center_score_mode", "global") != "global":
            self.encoder.eval()
            with torch.no_grad():
                for k, c in enumerate(clients):
                    _, A_hat_k = self.client_graphs[k]
                    Z_tr, _ = self._encode_dataset(
                        c["X_train"][:4096], self.encoder, A_hat_k,
                        cfg["n_anchor"], cfg["graph_in_encoder"], cfg["batch_size"]
                    )
                    self.client_local_centers[k] = Z_tr.detach().cpu().numpy().mean(axis=0)

        if cfg["use_flow"] and cfg["score_mode"] != "center_only":
            t1 = time.time()
            self._stage2(clients)
            print(f"    Stage II (local flow) done ({time.time() - t1:.1f}s)")

        self._calibrate(clients)

                                                                        
    def _stage1(self, clients: List[dict]):
        cfg     = self.cfg
        total   = sum(c["X_train"].shape[0] for c in clients)
        weights = [c["X_train"].shape[0] / total for c in clients]

        center = torch.zeros(cfg["d_h"], device=self.device)
        for k, c in enumerate(clients):
            _, A_hat = self.client_graphs[k]
            Z_k, _   = self._encode_dataset(
                c["X_train"][:512], self.encoder, A_hat,
                cfg["n_anchor"], cfg["graph_in_encoder"], cfg["batch_size"]
            )
            m_k = Z_k.mean(dim=0)
            if cfg["use_dp"]:
                m_k = clip_perturb_torch(m_k, cfg["C_c"], cfg["sigma_c"],
                                         generator=self.torch_generator)
            center += weights[k] * m_k
        self.center = center.detach()

        collapse_count    = 0
        collapse_patience = cfg.get("collapse_patience", 2)
        collapse_std_thr  = cfg.get("collapse_std_thr",  0.01)
        use_pred = cfg.get("use_prediction_loss", False)
                                                                        
                                                                        
                                                                         
                                                                            
        if use_pred and isinstance(self.encoder, NPFormerGPEncoder):
            raise NotImplementedError(
                "use_prediction_loss=True is not supported with "
                "encoder_type='npformer_gp' (pred_next is patch-level "
                "[B,P-1,...] but xb[:,1:] is timestamp-level [B,T-1,...]). "
                "Either set use_prediction_loss=False (recommended) or "
                "switch to encoder_type='gru'."
            )
        print(f"    Stage I: use_prediction_loss={use_pred}  "
              f"lambda_pred={cfg.get('lambda_pred',1.0)}  "
              f"lambda_c={cfg.get('lambda_c',0.02)}  "
              f"n_rounds={cfg['n_rounds']}")

        for r in range(cfg["n_rounds"]):
            global_vec = parameters_to_vector(
                self.encoder.parameters()
            ).detach().clone()
            agg_delta  = torch.zeros_like(global_vec)
            center_sum = torch.zeros(cfg["d_h"], device=self.device)

            for k, c in enumerate(clients):
                A_raw, A_hat  = self.client_graphs[k]
                local_encoder = copy.deepcopy(self.encoder).to(self.device)
                global_params = [
                    parameter.detach().clone()
                    for parameter in self.encoder.parameters()
                ]
                local_encoder.train()
                optimizer = torch.optim.Adam(
                    local_encoder.parameters(),
                    lr=cfg["lr"], weight_decay=cfg["lambda_r"]
                )
                X_tr = c["X_train"]
                N_k  = len(X_tr)
                _round_center_loss = 0.0
                _round_pred_loss   = 0.0
                _round_batches     = 0

                                                       
                                                               
                                                    
                                                       
                X_tr_gpu = torch.as_tensor(
                    X_tr, dtype=torch.float32, device=self.device
                )

                for _ in range(cfg["local_epochs"]):
                    perm = self.rng.permutation(N_k)
                    perm_t = torch.as_tensor(perm, dtype=torch.long,
                                             device=self.device)
                    for bi in range(0, N_k, cfg["batch_size"]):
                        idx      = perm_t[bi:bi + cfg["batch_size"]]
                        xb           = X_tr_gpu.index_select(0, idx)
                        use_pred_loss = cfg.get("use_prediction_loss", False) and xb.size(1) > 1
                        out = local_encoder(
                            xb, A_hat,
                            n_anchor=cfg["n_anchor"],
                            use_graph=cfg["graph_in_encoder"],
                            return_graph_seq=True,
                            return_pred=use_pred_loss,
                        )
                        if use_pred_loss:
                            z_batch, G_seq, pred_next = out
                        else:
                            z_batch, G_seq = out
                            pred_next = None
                        diff = z_batch - self.center.unsqueeze(0)
                        lambda_c = cfg.get("lambda_c", 1.0)
                        _c_loss = diff.pow(2).sum(dim=1).mean()
                        loss = lambda_c * _c_loss
                        if use_pred_loss and pred_next is not None:
                            _p_loss = F.smooth_l1_loss(pred_next, xb[:, 1:])
                            loss = loss + cfg.get("lambda_pred", 1.0) * _p_loss
                            _round_pred_loss   += _p_loss.item()
                        _round_center_loss += _c_loss.item()
                        _round_batches     += 1
                        if cfg["graph_in_encoder"] and cfg["lambda_g"] > 0:
                            loss = loss + cfg["lambda_g"] * graph_smoothness_reg(
                                G_seq, A_raw)
                        if cfg["graph_in_encoder"] and cfg["lambda_t"] > 0:
                                                                 
                                                                              
                                                                           
                                                                             
                                                      
                            if isinstance(self.encoder, NPFormerGPEncoder):
                                loss = loss + cfg["lambda_t"] * patch_continuity_reg(G_seq)
                            else:
                                loss = loss + cfg["lambda_t"] * temporal_reg(G_seq)
                        if cfg["lambda_v"] > 0 and z_batch.size(0) > 1:
                            loss = loss + cfg["lambda_v"] * variance_floor_reg(
                                z_batch, cfg["gamma_var"])
                        if cfg.get("mu_prox", 0.0) > 0:
                            prox = sum(
                                (parameter - reference).pow(2).sum()
                                for parameter, reference in zip(
                                    local_encoder.parameters(), global_params
                                )
                            )
                            loss = loss + 0.5 * cfg["mu_prox"] * prox
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            local_encoder.parameters(), max_norm=5.0)
                        optimizer.step()

                local_vec = parameters_to_vector(
                    local_encoder.parameters()).detach()
                delta = local_vec - global_vec
                if cfg["use_dp"]:
                    delta = clip_perturb_torch(
                        delta, cfg["C_theta"], cfg["sigma_theta"],
                        generator=self.torch_generator)
                agg_delta += weights[k] * delta

                Z_k, _ = self._encode_dataset(
                    X_tr[:2048], local_encoder, A_hat,
                    cfg["n_anchor"], cfg["graph_in_encoder"], cfg["batch_size"]
                )
                m_k = Z_k.mean(dim=0)
                if cfg["use_dp"]:
                    m_k = clip_perturb_torch(
                        m_k, cfg["C_c"], cfg["sigma_c"],
                        generator=self.torch_generator)
                center_sum += weights[k] * m_k

            with torch.no_grad():
                vector_to_parameters(
                    global_vec + cfg["eta_s"] * agg_delta,
                    self.encoder.parameters())
                self.center = center_sum.detach()

                                   
            if _round_batches > 0:
                _avg_c = _round_center_loss / _round_batches
                _avg_p = _round_pred_loss   / _round_batches
                if use_pred:
                    print(f"    [R{r+1:02d}] center_loss={_avg_c:.4f}  "
                          f"pred_loss={_avg_p:.4f}")
                else:
                    print(f"    [R{r+1:02d}] center_loss={_avg_c:.4f}")

            with torch.no_grad():
                _, A_hat_0 = self.client_graphs[0]
                Z_dbg, _   = self._encode_dataset(
                    clients[0]["X_train"][:512], self.encoder,
                    A_hat_0, cfg["n_anchor"], cfg["graph_in_encoder"], cfg["batch_size"]
                )
                collapsed = _check_collapse(
                    Z_dbg, r + 1, cfg["n_rounds"],
                    std_thr=collapse_std_thr)

            if collapsed:
                collapse_count += 1
                if collapse_count >= collapse_patience:
                    print(f"      Stop after {collapse_patience} low-variance rounds")
                    break
            else:
                collapse_count = 0

                                                                        
    def _stage2(self, clients: List[dict]):
        cfg = self.cfg
        self.encoder.eval()
        self.client_flows = {}
        for k, c in enumerate(clients):
            _, A_hat = self.client_graphs[k]
            Z_tr, _  = self._encode_dataset(
                c["X_train"], self.encoder, A_hat,
                cfg["n_anchor"], cfg["graph_in_encoder"], cfg["batch_size"]
            )
            flow      = MAF(cfg["d_h"], cfg["flow_blocks"],
                            cfg["flow_hidden"]).to(self.device)
            optimizer = torch.optim.Adam(
                flow.parameters(),
                lr=cfg["flow_lr"], weight_decay=cfg["lambda_phi"]
            )
            N = Z_tr.size(0)
            for _ in range(cfg["flow_epochs"]):
                perm = torch.randperm(N, device=self.device,
                                      generator=self.torch_generator)
                for bi in range(0, N, cfg["batch_size"]):
                    idx  = perm[bi:bi + cfg["batch_size"]]
                    zb   = Z_tr.index_select(0, idx)
                    loss = -flow.log_prob(zb).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            flow.eval()
            self.client_flows[k] = flow

                                                                        
    def _tail_evi(self, v: float, cal_scores: np.ndarray) -> float:
        pi = (1 + np.sum(cal_scores >= v)) / (len(cal_scores) + 1)
        return float(-np.log(max(pi, 1e-10)))

    def _flow_scores(self, k: int, Z: torch.Tensor) -> np.ndarray:
        if k not in self.client_flows:
            return np.zeros(Z.size(0), dtype=np.float32)
        with torch.no_grad():
            return (-self.client_flows[k].log_prob(Z)
                    .detach().cpu().numpy()).astype(np.float32)

    def _compute_evidence(self, s1, s2, s3, k):
        cfg    = self.cfg
        cal_s1 = self.client_cal[k]["s1"]
        cal_s2 = self.client_cal[k]["s2"]
        cal_s3 = self.client_cal[k].get("s3")

        use_residual = (
            cfg["use_graph_residual"]
            and cfg["use_graph"]
            and cal_s3 is not None
            and s3 is not None
            and len(s3) == len(s1)
        )

        if cfg["use_calibration"]:
            e1 = np.asarray([self._tail_evi(v, cal_s1) for v in s1],
                            dtype=np.float32)
            e2 = np.asarray([self._tail_evi(v, cal_s2) for v in s2],
                            dtype=np.float32)
            if use_residual:
                e3 = np.asarray([self._tail_evi(v, cal_s3) for v in s3],
                                dtype=np.float32)
        else:
            e1 = s1.astype(np.float32)
            e2 = s2.astype(np.float32)
            if use_residual:
                e3 = s3.astype(np.float32)

        mode = cfg["score_mode"]
        if mode == "center_only":
            return e1
        if mode == "flow_only":
            return e2

        stored = self.client_fusion_weights.get(k)
        if stored is not None:
            if len(stored) == 3:
                w1, w2, w3 = stored
            else:
                w1, w2 = stored
                w3 = cfg.get("graph_residual_weight", 0.30) if use_residual else 0.0
                s  = w1 + w2
                if s > 0:
                    w1 = w1 / s * (1.0 - w3)
                    w2 = w2 / s * (1.0 - w3)
        else:
            raw = cfg["w_fusion"]
            if len(raw) == 3:
                w1, w2, w3 = raw
            else:
                w1, w2 = raw
                w3 = cfg.get("graph_residual_weight", 0.30) if use_residual else 0.0
                s  = w1 + w2
                if s > 0 and use_residual:
                    w1 = w1 / s * (1.0 - w3)
                    w2 = w2 / s * (1.0 - w3)

        if use_residual:
            return w1 * e1 + w2 * e2 + w3 * e3
        else:
            s = w1 + w2
            if s > 0:
                w1, w2 = w1 / s, w2 / s
            return w1 * e1 + w2 * e2

    def _calibrate(self, clients: List[dict]):
        cfg = self.cfg
        center_np = self.center.detach().cpu().numpy()
        self.client_cal = {}
        self.client_fusion_weights = {}
        self.client_local_centers = {}
        self.client_alpha_ks = {}

        cmode = cfg.get("center_score_mode", "global")
        if cmode != "global":
            self.encoder.eval()
            with torch.no_grad():
                for k, c in enumerate(clients):
                    _, A_hat_k = self.client_graphs[k]
                    Z_tr, _ = self._encode_dataset(
                        c["X_train"][:4096], self.encoder, A_hat_k,
                        cfg["n_anchor"], cfg["graph_in_encoder"], cfg["batch_size"]
                    )
                    self.client_local_centers[k] = Z_tr.detach().cpu().numpy().mean(axis=0)

        alpha_h = cfg.get("hybrid_center_alpha", 0.0)
        if alpha_h > 0 and cfg.get("adaptive_hybrid_alpha", False):
            n_anc = cfg["n_anchor"]
            densities = [n_anc / max(ci["X_cal"].shape[2], 1) for ci in clients]
            mean_d = sum(densities) / max(len(densities), 1)
            self.client_alpha_ks = {
                k: float(np.clip(alpha_h * d / mean_d, 0.1, 0.9))
                for k, d in enumerate(densities)
            }
        else:
            self.client_alpha_ks = {k: alpha_h for k in range(len(clients))}

        for k, c in enumerate(clients):
            s1, s2, s3 = self._branch_scores(k, c, split="calibration", center_np=center_np)
            self.client_cal[k] = {"s1": s1, "s2": s2, "s3": s3}
            self.client_fusion_weights[k] = tuple(cfg.get("w_fusion", (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)))

    def _branch_scores(
        self,
        k: int,
        client: dict,
        split: str,
        center_np: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        cfg = self.cfg
        if split == "train":
            key = "X_train"
        elif split in ("cal", "calibration"):
            key = "X_cal"
        elif split == "test":
            key = "X_test"
        else:
            raise ValueError(f"unknown split: {split!r}")

        if center_np is None:
            center_np = self.center.detach().cpu().numpy()
        _, A_hat = self.client_graphs[k]
        Z, _ = self._encode_dataset(
            client[key], self.encoder, A_hat,
            cfg["n_anchor"], cfg["graph_in_encoder"], cfg["batch_size"]
        )
        Z_np = Z.detach().cpu().numpy()
        cmode = cfg.get("center_score_mode", "global")
        c_k = self.client_local_centers.get(k)
        if cmode == "local" and c_k is not None:
            s1 = ((Z_np - c_k) ** 2).sum(axis=1).astype(np.float32)
        elif cmode == "hybrid" and c_k is not None:
            beta = float(cfg.get("center_hybrid_beta", 0.5))
            s1 = (beta * ((Z_np - center_np) ** 2).sum(axis=1)
                  + (1.0 - beta) * ((Z_np - c_k) ** 2).sum(axis=1)).astype(np.float32)
        else:
            s1 = ((Z_np - center_np) ** 2).sum(axis=1).astype(np.float32)

        s2 = self._flow_scores(k, Z)
        s3 = None
        if cfg["use_graph_residual"] and cfg["use_graph"]:
            s3 = self._compute_graph_residual_score(client[key], self.client_signed_graphs[k])
        return s1, s2, s3

    def score(self, clients: List[dict], split: str = "test") -> List[dict]:
        results = []
        for k, client in enumerate(clients):
            s1, s2, s3 = self._branch_scores(k, client, split=split)
            evidence = self._compute_evidence(s1, s2, s3, k)
            score = 1.0 - np.exp(-np.clip(evidence, 0.0, 50.0))
            results.append({
                "client_id": k,
                "client_name": client.get("client_name", f"client{k}"),
                "score": score.astype(np.float32),
                "evidence": evidence.astype(np.float32),
                "s1_raw": s1.astype(np.float32),
                "s2_raw": s2.astype(np.float32),
                "s3_raw": s3.astype(np.float32) if s3 is not None else None,
            })
        return results

    def predict(self, clients: List[dict]) -> List[dict]:
        return self.score(clients, split="test")
