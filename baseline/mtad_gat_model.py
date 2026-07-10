"""
MTAD-GAT: Multivariate Time-series Anomaly Detection via Graph Attention Network
==================================================================================
Zhao et al., ICDM 2020

核心组件
--------
1. 1-D Conv (kernel=7) 预处理
2. Feature-oriented GAT: n 个特征节点的完全图
3. Time-oriented GAT:   T 个时间节点的完全图
4. GRU 时序建模
5. Forecasting head (3 层 FC, 预测下一时刻)
6. Reconstruction head (VAE, 重建整个窗口)
7. Joint optimization: L = L_for + L_rec

异常分数 (Eq. 9):
  score = Σ_i [ (ŝ_i - s_i)² + γ · (1 - p_i) ] / (1 + γ)

本实现适配联邦场景
------------------
- 原论文集中式，这里 FedAvg 包装
- 异构特征 pad-to-max，损失/分数只在有效列计算
- 每个 client 独立选阈值
- 保持与 FedGAD / FL-STAM / GANF / GDN / TranAD 统一的 predict() 格式
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


                                                                              
                        
                                                                              

class MtadGATLayer(nn.Module):
    """
    MTAD-GAT 的图注意力层 (论文 Eq. 2-4)。

    输入 {v_1, ..., v_L}, v_i ∈ R^F
      e_ij = LeakyReLU(w^T (v_i ⊕ v_j))
      α_ij = softmax_j(e_ij)
      h_i = σ( Σ_j α_ij v_j )

    输出 shape 与输入相同。
    """

    def __init__(self, feat_dim: int, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.a = nn.Linear(2 * feat_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, L, feat_dim]
        返回: [B, L, feat_dim]
        """
        B, L, F_ = X.shape
        xi = X.unsqueeze(2).expand(-1, L, L, -1)                                   
        xj = X.unsqueeze(1).expand(-1, L, L, -1)                                   
        cat = torch.cat([xi, xj], dim=-1)                                           
        e = self.leaky(self.a(cat).squeeze(-1))                                 
        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        out = torch.einsum('bij,bjf->bif', alpha, X)                            
        return torch.sigmoid(out)


                                                                              
                               
                                                                              

class VAEReconstruct(nn.Module):
    """
    VAE 用于从 GRU 末端 hidden state 重建整个窗口。

    输入: h_last [B, d_gru]
    编码: q_phi(z|h) = N(μ, σ²)
    解码: x̂ [B, T, n]
    """

    def __init__(self, d_gru: int, latent_dim: int, window_len: int, n_features: int):
        super().__init__()
        self.window_len = window_len
        self.n_features = n_features

        self.enc_mu = nn.Linear(d_gru, latent_dim)
        self.enc_logvar = nn.Linear(d_gru, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_gru),
            nn.ReLU(),
            nn.Linear(d_gru, window_len * n_features),
        )

    def forward(self, h: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.enc_mu(h)
        logvar = torch.clamp(self.enc_logvar(h), min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std) if self.training else mu
        recon_flat = self.decoder(z)
        recon = recon_flat.view(-1, self.window_len, self.n_features)
        return recon, mu, logvar

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
        ).mean()


                                                                              
                  
                                                                              

class MtadGATNet(nn.Module):
    """
    完整 MTAD-GAT 网络。

    数据流:
      X [B, T, n]
        ├── 1D Conv (k=7): [B, T, n] → transpose → Conv1d(n→n) → transpose → [B, T, n]
        ├── Feature GAT: 以每个特征为节点 [B, n, T] → GAT → [B, n, T] → transpose → [B, T, n]
        └── Time GAT:    以每个时间步为节点 [B, T, n] → GAT → [B, T, n]
      → concat 3 份 → [B, T, 3n] → GRU(d_gru) → H [B, T, d_gru]
      → Forecasting: FC3(H[:, -1, :]) → ŝ [B, n] (预测最后一步/下一步)
      → Reconstruction: VAE(H[:, -1, :]) → x_hat [B, T, n] + (μ, logvar)
    """

    def __init__(self, n_features: int, window_len: int,
                 d_gru: int = 128, latent_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.window_len = window_len
        self.d_gru = d_gru

                                       
        self.conv = nn.Conv1d(n_features, n_features, kernel_size=7, padding=3)

                                                      
        self.feat_gat = MtadGATLayer(feat_dim=window_len, dropout=dropout)
                                                    
        self.time_gat = MtadGATLayer(feat_dim=n_features, dropout=dropout)

                                                                 
        self.gru = nn.GRU(
            input_size=3 * n_features, hidden_size=d_gru,
            batch_first=True, num_layers=1,
        )

                                  
        self.forecast_head = nn.Sequential(
            nn.Linear(d_gru, d_gru), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_gru, d_gru // 2), nn.ReLU(),
            nn.Linear(d_gru // 2, n_features),
        )

                            
        self.vae = VAEReconstruct(d_gru, latent_dim, window_len, n_features)

    def forward(self, X: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        X: [B, T, n]
        返回: forecast [B, n], recon [B, T, n], mu [B, lat], logvar [B, lat]
        """
        B, T, n = X.shape

                 
        conv_in = X.transpose(1, 2)                                             
        conv_out = F.gelu(self.conv(conv_in)).transpose(1, 2)                    

                                                                      
        feat_in = X.transpose(1, 2)                                             
        feat_out = self.feat_gat(feat_in).transpose(1, 2)                       

                                                                    
        time_out = self.time_gat(X)                                             

                      
        concat = torch.cat([conv_out, feat_out, time_out], dim=-1)                
        gru_out, _ = self.gru(concat)                                                
        h_last = gru_out[:, -1, :]                                                

        forecast = self.forecast_head(h_last)                                 
        recon, mu, logvar = self.vae(h_last)                                     

        return forecast, recon, mu, logvar


                                                                              
                    
                                                                              

class MtadGAT:
    """
    MTAD-GAT 完整方案 (联邦化)。

    接口:
      fit(clients)     — FedAvg + 联合优化 (forecast + recon)
      predict(clients) — 推理，返回统一格式结果

    异常分数 (Eq. 9):
      s = Σ_i [ (ŝ_i - x_i)² + γ · (1 - p_i) ] / (1 + γ)
    其中 p_i 用重建高斯的对数似然 → sigmoid 映射到 (0,1) 作为"正常概率"。
    """

    def __init__(
        self,
        d_gru: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.1,
        n_rounds: int = 10,
        local_epochs: int = 2,
        lr: float = 1e-3,
        batch_size: int = 32,
        gamma: float = 0.8,                                 
        kl_weight: float = 0.01,                         
        threshold_mode: str = "ratio",
        ratio_clip: Tuple[float, float] = (0.01, 0.45),
        device: str = "cuda",
        seed: int = 42,
    ):
        self.cfg = {k: v for k, v in locals().items() if k != "self"}
        set_seed(seed)
        self.device = torch.device(
            device if device != "cpu" and torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[MtadGATNet] = None
        self.client_thresholds: Dict[int, float] = {}
                                                             
        self.client_recon_scale: Dict[int, float] = {}

                                                        

    def _squeeze_and_pad(self, X: np.ndarray) -> np.ndarray:
        arr = X.squeeze(-1) if X.ndim == 4 and X.shape[-1] == 1 else X
        if hasattr(self, 'n_features_common') and arr.ndim == 3:
            actual = arr.shape[2]
            target = self.n_features_common
            if actual < target:
                pad = np.zeros(
                    (arr.shape[0], arr.shape[1], target - actual),
                    dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=2)
        return arr

    def _to_device(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

                                                       

    def _compute_scores(self, X: np.ndarray, k: int = 0) -> np.ndarray:
        """
        MTAD-GAT 异常分数 (Eq. 9):
          s_i = [(ŝ_i - x_i)² + γ · (1 - p_i)] / (1 + γ)
          score = Σ_i s_i    (仅有效列求和)
        重建概率 p_i 用高斯似然估计：exp(-err² / σ_scale) 映射到 (0,1)。
        """
        self.model.eval()
        X_3d = self._squeeze_and_pad(X)
        n_k_valid = (self.n_features_per_client[k]
                     if hasattr(self, 'n_features_per_client')
                     else X_3d.shape[2])
        scale = self.client_recon_scale.get(k, 1.0)
        gamma = self.cfg["gamma"]

        scores = []
        bs = self.cfg["batch_size"]
        with torch.no_grad():
            for s in range(0, len(X_3d), bs):
                xb = self._to_device(X_3d[s:s + bs])                            
                forecast, recon, _, _ = self.model(xb)

                                              
                tgt = xb[:, -1, :]                                           
                pred_err = (forecast[:, :n_k_valid] - tgt[:, :n_k_valid]).pow(2)
                pred_err_sum = pred_err.sum(dim=1)                        

                                        
                recon_err = (recon[:, :, :n_k_valid] - xb[:, :, :n_k_valid]).pow(2)
                per_feature_err = recon_err.mean(dim=1)                        
                p_i = torch.exp(-per_feature_err / max(scale, 1e-6))            
                anom_prob_sum = (1.0 - p_i).sum(dim=1)                    

                score = (pred_err_sum + gamma * anom_prob_sum) / (1.0 + gamma)
                scores.append(score.cpu().numpy())
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

                                                          

    def fit(self, clients: List[dict]) -> None:
        cfg = self.cfg

        sample = clients[0]["X_train"]
        if sample.ndim == 4:
            _, window_len, _, _ = sample.shape
        else:
            _, window_len, _ = sample.shape

        def _nk(c):
            return c["X_train"].shape[2]
        self.n_features_per_client = [_nk(c) for c in clients]
        self.n_features_common = max(self.n_features_per_client)

        self.model = MtadGATNet(
            n_features=self.n_features_common,
            window_len=window_len,
            d_gru=cfg["d_gru"],
            latent_dim=cfg["latent_dim"],
            dropout=cfg["dropout"],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        total_samples = sum(c["X_train"].shape[0] for c in clients)
        weights = [c["X_train"].shape[0] / total_samples for c in clients]

        print(f"  [MTAD-GAT] {len(clients)} clients, features={self.n_features_per_client} "
              f"(max={self.n_features_common}, pad-to-max), "
              f"win={window_len}, d_gru={cfg['d_gru']}, γ={cfg['gamma']}, "
              f"params={n_params:,}")

                         
        for r in range(cfg["n_rounds"]):
            global_vec = parameters_to_vector(
                self.model.parameters()).detach().clone()
            agg_delta = torch.zeros_like(global_vec)
            round_loss = 0.0

            for k, c in enumerate(clients):
                local_model = copy.deepcopy(self.model).to(self.device)
                local_model.train()
                optimizer = torch.optim.Adam(local_model.parameters(), lr=cfg["lr"])

                X_tr = self._squeeze_and_pad(c["X_train"])
                n_k_valid = self.n_features_per_client[k]
                N_k = len(X_tr)
                client_loss = 0.0
                n_batches = 0

                for _ in range(cfg["local_epochs"]):
                    perm = np.random.permutation(N_k)
                    for bi in range(0, N_k, cfg["batch_size"]):
                        idx = perm[bi:bi + cfg["batch_size"]]
                        xb = self._to_device(X_tr[idx])
                        forecast, recon, mu, logvar = local_model(xb)

                                                  
                        tgt = xb[:, -1, :]
                        pred_loss = torch.sqrt(
                            F.mse_loss(
                                forecast[:, :n_k_valid],
                                tgt[:, :n_k_valid]) + 1e-8)

                                                              
                        rec_loss = F.mse_loss(
                            recon[:, :, :n_k_valid],
                            xb[:, :, :n_k_valid])
                        kl = VAEReconstruct.kl_divergence(mu, logvar)

                        loss = pred_loss + rec_loss + cfg["kl_weight"] * kl

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            local_model.parameters(), max_norm=5.0)
                        optimizer.step()

                        client_loss += loss.item()
                        n_batches += 1

                round_loss += weights[k] * client_loss / max(n_batches, 1)
                local_vec = parameters_to_vector(
                    local_model.parameters()).detach()
                agg_delta += weights[k] * (local_vec - global_vec)

            with torch.no_grad():
                vector_to_parameters(
                    global_vec + agg_delta, self.model.parameters())

            print(f"    Round {r + 1}/{cfg['n_rounds']}: loss={round_loss:.4f}")

        self._calibrate(clients)

    def _calibrate(self, clients: List[dict]) -> None:
        """每个 client 估计重建误差尺度，再用 X_cal 选阈值。"""
        self.model.eval()
        for k, c in enumerate(clients):
                                           
            X_tr = self._squeeze_and_pad(c["X_train"])
            n_k_valid = self.n_features_per_client[k]
            all_recon_err = []
            bs = self.cfg["batch_size"]
            with torch.no_grad():
                for s in range(0, len(X_tr), bs):
                    xb = self._to_device(X_tr[s:s + bs])
                    _, recon, _, _ = self.model(xb)
                    err = (recon[:, :, :n_k_valid] - xb[:, :, :n_k_valid]).pow(2)
                    all_recon_err.append(err.mean(dim=1).cpu().numpy())
            arr = np.concatenate(all_recon_err, axis=0) if all_recon_err else np.ones(1)
                                      
            self.client_recon_scale[k] = float(np.clip(np.quantile(arr, 0.75),
                                                        1e-6, None))

                 
            scores_cal = self._compute_scores(c["X_cal"], k)
            self.client_thresholds[k] = self._pick_threshold(
                scores_cal, c["y_cal"])

                                                          

    def predict(self, clients: List[dict]) -> List[dict]:
        results = []
        for k, c in enumerate(clients):
            scores = self._compute_scores(c["X_test"], k).astype(np.float64)
            tau_raw = float(self.client_thresholds.get(
                k, float(np.quantile(scores, 0.95))))
            y_pred = (scores > tau_raw).astype(np.int64)

            s_min, s_max = float(scores.min()), float(scores.max())
            denom = (s_max - s_min + 1e-8)
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
