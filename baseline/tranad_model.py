"""
TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series
======================================================================================
Tuli, Casale & Jennings, VLDB 2022

核心组件
--------
1. Transformer Encoder-Decoder     — 并行注意力, 代替 LSTM 逐步推理
2. Two-Phase Self-Conditioning     — Phase-1 粗重建, Phase-2 用 focus score 条件化
3. Adversarial Training            — 两个 decoder, min-max 博弈放大异常误差
4. Evolutionary Loss               — ε^{-n} 系数随 epoch 平衡重建/对抗 (Eq.10)
5. MAML-style Meta Learning        — 额外的元学习步, 帮小数据训练
6. Anomaly Score = ½||O₁-W||² + ½||Ô₂-W||²   (Eq.13)

数据流 (本项目窗口化数据):
    W [B, T, n_k]  (窗口本身既当 "window" 也当 "context")
      Phase 1: F = 0          → O1, O2 = Decoder1/2(Encoder(W, F))
      Phase 2: F = (O1 - W)²  → Ô2      = Decoder2(Encoder(W, F))

联邦扩展: 支持 federated=True (FedAvg) 与集中式; 异构特征 pad-to-max.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


                                                                              
                                   
                                                                              

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


                                                                              
                              
                                                                              

class TranADNet(nn.Module):
    """
    记 W ∈ [B, T, n], F ∈ [B, T, n]. Encoder 的输入是 [W || F] ∈ [B, T, 2n].
    Window Encoder 用 masked self-attn + cross-attn (以 Encoder 输出作 K/V).
    两个 decoder 并行, 都从 Window Encoder 输出解码.
    """

    def __init__(self, n_features: int, window_len: int, d_model: int = 64,
                 n_heads: int = 8, d_ff: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.window_len = window_len
        self.d_model = d_model

                                
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.n_heads = max(1, n_heads)

                           
        self.encoder_in_proj = nn.Linear(n_features * 2, d_model)
        self.window_in_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(window_len, 64))

                                                                              
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=self.n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

                                                             
        self.win_self_attn = nn.MultiheadAttention(
            d_model, self.n_heads, dropout=dropout, batch_first=True)
        self.win_norm1 = nn.LayerNorm(d_model)
        self.win_cross_attn = nn.MultiheadAttention(
            d_model, self.n_heads, dropout=dropout, batch_first=True)
        self.win_norm2 = nn.LayerNorm(d_model)
        self.win_ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.win_norm3 = nn.LayerNorm(d_model)

                      
        self.decoder1 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, n_features))
        self.decoder2 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, n_features))

                                
        mask = torch.triu(torch.ones(window_len, window_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def _encode(self, W: torch.Tensor, F_focus: torch.Tensor):
        """返回 (Encoder 输出, Window Encoder 输出), 均为 [B, T, d]"""
        I1 = self.encoder_in_proj(torch.cat([W, F_focus], dim=-1))
        I1 = self.pos_enc(I1)
        I1_enc = self.encoder(I1)

        I2 = self.window_in_proj(W)
        I2 = self.pos_enc(I2)
        sa_out, _ = self.win_self_attn(
            I2, I2, I2, attn_mask=self.causal_mask, need_weights=False)
        I2_mid = self.win_norm1(I2 + sa_out)
        ca_out, _ = self.win_cross_attn(
            I2_mid, I1_enc, I1_enc, need_weights=False)
        I2_mid = self.win_norm2(I2_mid + ca_out)
        I2_out = self.win_norm3(I2_mid + self.win_ff(I2_mid))
        return I1_enc, I2_out

    def forward(self, W: torch.Tensor, F_focus: torch.Tensor,
                which_decoder: str = "both"):
        _, I2 = self._encode(W, F_focus)
        O1 = torch.sigmoid(self.decoder1(I2)) if which_decoder in ("both", "d1") else None
        O2 = torch.sigmoid(self.decoder2(I2)) if which_decoder in ("both", "d2") else None
        return O1, O2


                                                                              
                
                                                                              

class TranAD:
    """
    TranAD 完整方案 (集中式 + 联邦; 统一 predict 输出与 FedGAD 对齐).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 8, d_ff: int = 64,
                 dropout: float = 0.1, n_epochs: int = 15, lr: float = 1e-3,
                 batch_size: int = 128, adv_eps: float = 0.95,
                 use_maml: bool = True, maml_lr: float = 2e-3, grad_clip: float = 1.0,
                 federated: bool = False, fed_rounds: int = 10,
                 fed_local_epochs: int = 3, threshold_mode: str = "ratio",
                 ratio_clip: Tuple[float, float] = (0.01, 0.45),
                 device: str = "cuda", seed: int = 42):
        self.cfg = {k: v for k, v in locals().items() if k != "self"}
        set_seed(seed)
        self.device = torch.device(
            device if device != "cpu" and torch.cuda.is_available() else "cpu")
        self.model: Optional[TranADNet] = None
        self.client_thresholds: Dict[int, float] = {}

                                                           

    def _squeeze_and_pad(self, X: np.ndarray) -> np.ndarray:
        arr = X.squeeze(-1) if X.ndim == 4 and X.shape[-1] == 1 else X
        if hasattr(self, "n_features_common") and arr.ndim == 3:
            actual = arr.shape[2]
            target = self.n_features_common
            if actual < target:
                pad = np.zeros(
                    (arr.shape[0], arr.shape[1], target - actual), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=2)
        return arr

    def _to_device(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

                                                        

    def _two_phase_forward(self, W: torch.Tensor):
        F_zero = torch.zeros_like(W)
        O1, O2 = self.model(W, F_zero, which_decoder="both")
        F_focus = (O1 - W).pow(2).detach()
        _, O2_hat = self.model(W, F_focus, which_decoder="d2")
        return O1, O2, O2_hat

    def _compute_scores(self, X: np.ndarray, k: int = 0) -> np.ndarray:
        self.model.eval()
        X3d = self._squeeze_and_pad(X)
        n_k_valid = (self.n_features_per_client[k]
                     if hasattr(self, "n_features_per_client") else X3d.shape[2])
        scores = []
        bs = self.cfg["batch_size"]
        with torch.no_grad():
            for s in range(0, len(X3d), bs):
                xb = self._to_device(X3d[s:s + bs])
                O1, _, O2_hat = self._two_phase_forward(xb)
                err1 = (O1[:, :, :n_k_valid] - xb[:, :, :n_k_valid]).pow(2).mean(dim=(1, 2))
                err2 = (O2_hat[:, :, :n_k_valid] - xb[:, :, :n_k_valid]).pow(2).mean(dim=(1, 2))
                scores.append((0.5 * err1 + 0.5 * err2).cpu().numpy())
        return np.concatenate(scores).astype(np.float32)

                                                            

    def _pick_threshold(self, scores: np.ndarray, y: np.ndarray) -> float:
        if self.cfg["threshold_mode"] == "best_f1":
            return self._threshold_best_f1(scores, y)
        return self._threshold_ratio(scores, y)

    def _threshold_ratio(self, scores: np.ndarray, y: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return float(np.quantile(scores, 0.95))
        ratio = float(np.clip(np.mean(y), *self.cfg["ratio_clip"]))
        return float(np.quantile(scores, 1.0 - ratio))

    def _threshold_best_f1(self, scores: np.ndarray, y: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return float(np.quantile(scores, 0.95))
        best_tau, best_f1 = float(np.quantile(scores, 0.95)), -1.0
        for q in np.linspace(0.01, 0.99, 99):
            tau = float(np.quantile(scores, q))
            f = f1_score(y, (scores > tau).astype(np.int64), zero_division=0)
            if f > best_f1:
                best_f1, best_tau = f, tau
        return best_tau

                                                         

    def _step_batch(self, model: nn.Module, optimizer, xb: torch.Tensor,
                    n_k_valid: int, epoch: int) -> float:
        eps = self.cfg["adv_eps"]
        w_rec = eps ** epoch                     
        w_adv = 1.0 - w_rec                      

        F_zero = torch.zeros_like(xb)
        O1, O2 = model(xb, F_zero, which_decoder="both")
        F_focus = (O1 - xb).pow(2).detach()                       
        _, O2_hat = model(xb, F_focus, which_decoder="d2")

        diff1 = (O1[:, :, :n_k_valid] - xb[:, :, :n_k_valid]).pow(2).mean()
        diff2 = (O2[:, :, :n_k_valid] - xb[:, :, :n_k_valid]).pow(2).mean()
        diff_hat = (O2_hat[:, :, :n_k_valid] - xb[:, :, :n_k_valid]).pow(2).mean()

        L1 = w_rec * diff1 + w_adv * diff_hat
        L2 = w_rec * diff2 - w_adv * diff_hat
        loss = L1 + L2

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg["grad_clip"])
        optimizer.step()
        return float(loss.item())

    def _train_one_client(self, X_train: np.ndarray, n_k_valid: int,
                         model: nn.Module, optimizer, n_epochs: int,
                         start_epoch: int = 0) -> float:
        X3d = self._squeeze_and_pad(X_train)
        N = len(X3d)
        bs = self.cfg["batch_size"]
        total_loss, n_batches = 0.0, 0
        model.train()
        for e in range(n_epochs):
            epoch_id = start_epoch + e
            perm = np.random.permutation(N)
            for bi in range(0, N, bs):
                idx = perm[bi:bi + bs]
                xb = self._to_device(X3d[idx])
                total_loss += self._step_batch(model, optimizer, xb, n_k_valid, epoch_id)
                n_batches += 1

                                                         
            if self.cfg["use_maml"] and N > bs:
                rand_idx = np.random.choice(N, size=bs, replace=False)
                xb = self._to_device(X3d[rand_idx])
                for pg in optimizer.param_groups:
                    pg["_saved_lr"] = pg["lr"]
                    pg["lr"] = self.cfg["maml_lr"]
                self._step_batch(model, optimizer, xb, n_k_valid, epoch_id)
                for pg in optimizer.param_groups:
                    pg["lr"] = pg.pop("_saved_lr")

        return total_loss / max(n_batches, 1)

                                                        

    def _train_centralized(self, clients: List[dict]) -> None:
        X_all = np.concatenate(
            [self._squeeze_and_pad(c["X_train"]) for c in clients], axis=0)
        n_k_valid = self.n_features_common
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["lr"])
        print(f"  [TranAD] centralized training on {len(X_all)} windows")

        best_loss, patience, bad = float("inf"), 5, 0
        for ep in range(self.cfg["n_epochs"]):
            loss = self._train_one_client(
                X_all, n_k_valid, self.model, optimizer,
                n_epochs=1, start_epoch=ep)
            if (ep + 1) % max(1, self.cfg["n_epochs"] // 10) == 0:
                print(f"    epoch {ep + 1}/{self.cfg['n_epochs']}: loss={loss:.4f}")
            if loss < best_loss - 1e-5:
                best_loss, bad = loss, 0
            else:
                bad += 1
            if bad >= patience:
                print(f"    early stopping at epoch {ep + 1}")
                break

    def _train_federated(self, clients: List[dict]) -> None:
        total = sum(c["X_train"].shape[0] for c in clients)
        weights = [c["X_train"].shape[0] / total for c in clients]
        for r in range(self.cfg["fed_rounds"]):
            global_vec = parameters_to_vector(self.model.parameters()).detach().clone()
            agg_delta = torch.zeros_like(global_vec)
            round_loss = 0.0
            for k, c in enumerate(clients):
                local_model = copy.deepcopy(self.model).to(self.device)
                opt = torch.optim.AdamW(local_model.parameters(), lr=self.cfg["lr"])
                n_k_valid = self.n_features_per_client[k]
                loss_k = self._train_one_client(
                    c["X_train"], n_k_valid, local_model, opt,
                    n_epochs=self.cfg["fed_local_epochs"],
                    start_epoch=r * self.cfg["fed_local_epochs"])
                round_loss += weights[k] * loss_k
                local_vec = parameters_to_vector(local_model.parameters()).detach()
                agg_delta += weights[k] * (local_vec - global_vec)
            with torch.no_grad():
                vector_to_parameters(global_vec + agg_delta, self.model.parameters())
            print(f"    Fed round {r + 1}/{self.cfg['fed_rounds']}: "
                  f"loss={round_loss:.4f}")

                                                           

    def fit(self, clients: List[dict]) -> None:
        cfg = self.cfg
        sample = clients[0]["X_train"]
        window_len = sample.shape[1]
        self.n_features_per_client = [c["X_train"].shape[2] for c in clients]
        self.n_features_common = max(self.n_features_per_client)
        n_features = self.n_features_common

        self.model = TranADNet(
            n_features=n_features, window_len=window_len,
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"], dropout=cfg["dropout"],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        mode_str = "federated" if cfg["federated"] else "centralized"
        print(f"  [TranAD] n_features={n_features} (pad-to-max, per-client={self.n_features_per_client}), "
              f"win={window_len}, d_model={cfg['d_model']}, heads={self.model.n_heads}, "
              f"params={n_params:,}, mode={mode_str}")

        if cfg["federated"]:
            self._train_federated(clients)
        else:
            self._train_centralized(clients)
        self._calibrate(clients)

    def _calibrate(self, clients: List[dict]) -> None:
        for k, c in enumerate(clients):
            scores_cal = self._compute_scores(c["X_cal"], k)
            self.client_thresholds[k] = self._pick_threshold(scores_cal, c["y_cal"])

    def predict(self, clients: List[dict]) -> List[dict]:
        results = []
        for k, c in enumerate(clients):
            scores = self._compute_scores(c["X_test"], k).astype(np.float64)
            tau_raw = float(
                self.client_thresholds.get(k, float(np.quantile(scores, 0.95))))
            y_pred = (scores > tau_raw).astype(np.int64)

            s_min, s_max = float(scores.min()), float(scores.max())
            denom = s_max - s_min + 1e-8
            norm_score = (scores - s_min) / denom
            tau_norm = (tau_raw - s_min) / denom

            results.append({
                "client_id": k,
                "client_name": c.get("client_name", f"client{k}"),
                "y_true": c["y_test"],
                "y_pred": y_pred,
                "raw_score": scores.astype(np.float32),
                "score": norm_score.astype(np.float32),
                "tau_raw": np.float32(tau_raw),
                "tau_norm": np.float32(tau_norm),
                "tau": np.float32(tau_norm),
            })
        return results
