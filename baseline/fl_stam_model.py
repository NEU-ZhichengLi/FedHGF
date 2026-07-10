"""
FL-STAM: Federated Spatio-Temporal Attention for Time Series Anomaly Detection
===============================================================================
Wang et al., ADMA 2025 (LNAI 16197, pp. 129-141, 2026)

核心组件
--------
1. Serial-Oriented GATv2  — 完全图上捕获传感器间全局依赖
2. Time-Oriented GATv2    — 完全图上捕获时间步间全局依赖
3. 1D Conv Embedding      — 初始特征嵌入
4. Transformer + ST-Attention — 双分支时空关联差异建模 (Wasserstein Distance)
5. 重建损失 + 关联差异损失 (minimax)
6. FedAvg 联邦参数聚合
"""

from __future__ import annotations

import copy
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


                                                                              
             
                                                                              

class GATv2Layer(nn.Module):
    """
    GATv2 注意力层 (Brody et al., 2021)。
    完全图上的注意力，节点间两两计算注意力分数。

    serial 模式: 节点=传感器(n), 特征=时间序列值(s)
    time   模式: 节点=时间步(s), 特征=传感器值(n)
    """

    def __init__(self, in_features: int, out_features: int,
                 n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        assert out_features % n_heads == 0

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: [B, N, F_in]  (B 个样本, N 个节点, F_in 维特征)
        返回: [B, N, F_out]
        """
        B, N, _ = X.shape
        H = self.W(X).view(B, N, self.n_heads, self.head_dim)

        Hi = H.unsqueeze(2).expand(-1, -1, N, -1, -1)               
        Hj = H.unsqueeze(1).expand(-1, N, -1, -1, -1)               

        cat_ij = torch.cat([Hi, Hj], dim=-1)                
        e = self.attn(self.leaky_relu(cat_ij)).squeeze(-1)             

        alpha = F.softmax(e, dim=2)               
        alpha = self.dropout(alpha)

        out = torch.einsum('bijn,bjnd->bind', alpha, H)
        out = out.reshape(B, N, -1)
        return torch.sigmoid(out)


                                                                              
                                                        
                                                                              

class STAttention(nn.Module):
    """
    Spatial-Temporal Multi-head Attention。

    双分支结构:
    - Spatial Association (SA): 标准 self-attention, 自适应学习全时空域最相关的关联
    - Temporal Association (TA): 基于可学习 sigma 的高斯核先验, 偏向局部时间邻近

    关联差异 = Wasserstein-1(SA, TA) 的对称形式
    → 正常点: SA ≈ TA (低差异), 异常点: SA ≠ TA (高差异)

    论文发现异常点的 Dis 值更低 (因为异常使得 SA 退化为类似 TA 的局部模式),
    因此异常分数中取负号。
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_sigma = nn.Linear(d_model, 1)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H: [B, S, d_model]
        返回: (output [B, S, d_model], discrepancy [B, S])
        """
        B, S, _ = H.shape

        Q = self.W_Q(H).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(H).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(H).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** 0.5
        SA = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / scale, dim=-1)

                                              
        sigma = torch.clamp(torch.abs(self.W_sigma(H).squeeze(-1)), min=0.1, max=S)
        positions = torch.arange(S, device=H.device, dtype=H.dtype).unsqueeze(0)
        dist = (positions.unsqueeze(-1) - positions.unsqueeze(-2)) ** 2
        sigma_sq = (sigma.unsqueeze(-1) ** 2).unsqueeze(1)
        TA = F.softmax(-dist.unsqueeze(1) / (2 * sigma_sq + 1e-8), dim=-1)
        TA = TA.expand_as(SA)

                                  
        SA_cdf = torch.cumsum(SA, dim=-1)
        TA_cdf = torch.cumsum(TA, dim=-1)
        wasserstein = torch.abs(SA_cdf - TA_cdf).sum(dim=-1)           
        discrepancy = wasserstein.mean(dim=1)          

        attn_out = torch.matmul(SA, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        out = self.out_proj(attn_out)
        return out, discrepancy


class TransformerLayer(nn.Module):
    """一层 Transformer with ST-Attention + FFN。"""

    def __init__(self, d_model: int, n_heads: int = 8,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.st_attn = STAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, disc = self.st_attn(H)
        H = self.norm1(H + attn_out)
        H = self.norm2(H + self.ff(H))
        return H, disc


                                                                              
                 
                                                                              

class FLSTAMNet(nn.Module):
    """
    完整的 FL-STAM 网络。

    数据流:
      X [B,T,n_k]
        ├── Serial GAT: X^T → [B,n_k,T] → GAT → [B,n_k,T] → 转置 → [B,T,n_k]
        ├── Time GAT:   X   → [B,T,n_k] → GAT → [B,T,n_k]
        └── 1D Conv:    X^T → Conv1d → [B,T,d_model]
      → Concat + Linear → [B,T,d_model]
      → L 层 Transformer (ST-Attention)
      → Recon head → X_hat [B,T,n_k]

    输出: (X_hat, disc)
    """

    def __init__(self, n_features: int, window_len: int, d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 3, d_ff: int = 256,
                 gat_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.window_len = window_len
        self.d_model = d_model

                           
        self.conv_embed = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

                                                     
        self.serial_gat = GATv2Layer(window_len, window_len, n_heads=gat_heads)

                                                    
        self.time_gat = GATv2Layer(n_features, n_features, n_heads=gat_heads)

              
        self.fusion_proj = nn.Linear(n_features + n_features + d_model, d_model)

                            
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

             
        self.recon_head = nn.Linear(d_model, n_features)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X: [B, T, n_k]
        返回: X_hat [B,T,n_k], disc [B,T]
        """
        B, T, n_k = X.shape

        serial_in = X.transpose(1, 2)                            
        serial_out = self.serial_gat(serial_in)                   
        serial_out = serial_out.transpose(1, 2)                   

        time_out = self.time_gat(X)                               

        conv_in = X.transpose(1, 2)                               
        conv_out = self.conv_embed(conv_in).transpose(1, 2)                   

        H = self.fusion_proj(
            torch.cat([serial_out, time_out, conv_out], dim=-1)
        )

        disc_layers = []
        for layer in self.layers:
            H, disc_l = layer(H)
            disc_layers.append(disc_l)

        disc = torch.stack(disc_layers, dim=0).mean(dim=0)          
        X_hat = self.recon_head(H)

        return X_hat, disc


                                                                              
                   
                                                                              

class FLSTAM:
    """
    FL-STAM 完整方案: 联邦训练框架 + 推理。

    接口:
      fit(clients)     — 联邦训练 (FedAvg)
      predict(clients) — 推理, 返回统一格式结果

    参数
    ----
    d_model       : Transformer 隐藏维度
    n_heads       : 注意力头数
    n_layers      : Transformer 层数
    d_ff          : FFN 中间维度
    gat_heads     : GAT 注意力头数
    n_rounds      : 联邦通信轮数
    local_epochs  : 每轮本地训练 epoch 数
    lr            : 学习率
    batch_size    : 批大小
    lambda_disc   : 关联差异损失权重 (论文 Eq.13 中的 λ)
    threshold_mode: 阈值选择策略 ('ratio' / 'best_f1')
    device        : 计算设备
    seed          : 随机种子
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        gat_heads: int = 1,
        n_rounds: int = 10,
        local_epochs: int = 2,
        lr: float = 1e-4,
        batch_size: int = 32,
        lambda_disc: float = 3.0,
        dropout: float = 0.1,
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
        self.model: Optional[FLSTAMNet] = None
        self.client_thresholds: Dict[int, float] = {}

                                                        

    def _squeeze_and_pad(self, X: np.ndarray) -> np.ndarray:
        """[N, T, n_k, 1] → [N, T, n_k]，不足 n_features_common 时零补齐（不截断）"""
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

    def _squeeze(self, X: np.ndarray) -> np.ndarray:
        """向后兼容别名，调用 _squeeze_and_pad"""
        return self._squeeze_and_pad(X)

    def _to_device(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

                                                       

    def _compute_scores(self, X: np.ndarray, k: int = 0) -> np.ndarray:
        """
        异常分数 = 重建误差（仅有效特征）+ λ × (-关联差异)
        论文: 异常点关联差异更低, 取负号后越大越异常
        """
        self.model.eval()
        X_3d = self._squeeze_and_pad(X)
        n_k_valid = (self.n_features_per_client[k]
                     if hasattr(self, 'n_features_per_client') else X_3d.shape[2])
        scores = []
        bs = self.cfg["batch_size"]
        with torch.no_grad():
            for s in range(0, len(X_3d), bs):
                xb = self._to_device(X_3d[s:s + bs])
                x_hat, disc = self.model(xb)
                recon_err = (
                    xb[:, :, :n_k_valid] - x_hat[:, :, :n_k_valid]
                ).pow(2).mean(dim=(1, 2))
                disc_score = -disc.mean(dim=1)
                score = recon_err + self.cfg["lambda_disc"] * disc_score
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

                                                          

    def fit(self, clients: List[dict], round_callback=None) -> None:
        cfg = self.cfg

        sample = clients[0]["X_train"]
                                          
        if sample.ndim == 4:
            _, window_len, n_features, _ = sample.shape
        else:
            _, window_len, n_features = sample.shape
                                                             
        def _nk(c):
            s = c["X_train"]
            return s.shape[2] if s.ndim == 3 else s.shape[2]
        self.n_features_per_client = [_nk(c) for c in clients]
        self.n_features_common = max(self.n_features_per_client)
        n_features = self.n_features_common
        window_len = window_len

        _gat_heads = cfg["gat_heads"]
        while _gat_heads > 1 and n_features % _gat_heads != 0:
            _gat_heads -= 1
        if _gat_heads != cfg["gat_heads"]:
            print(f"  [FL-STAM] gat_heads auto-adjusted: {cfg['gat_heads']} -> {_gat_heads} "
                  f"(n_features={n_features} not divisible)")
        self.model = FLSTAMNet(
            n_features=n_features, window_len=window_len,
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
            gat_heads=_gat_heads, dropout=cfg["dropout"],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        total_samples = sum(c["X_train"].shape[0] for c in clients)
        weights = [c["X_train"].shape[0] / total_samples for c in clients]

        nk_list = self.n_features_per_client
        print(f"  [FL-STAM] {len(clients)} clients, features={nk_list} "
              f"(max={n_features}, pad-to-max), "
              f"win={window_len}, d_model={cfg['d_model']}, "
              f"params={n_params:,}")

                             
        for r in range(cfg["n_rounds"]):
            global_vec = parameters_to_vector(
                self.model.parameters()
            ).detach().clone()
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
                        x_hat, disc = local_model(xb)

                        recon_loss = F.mse_loss(
                            x_hat[:, :, :n_k_valid], xb[:, :, :n_k_valid])
                                                 
                        disc_loss = -disc.mean()
                        loss = recon_loss + cfg["lambda_disc"] * disc_loss

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
            if round_callback is not None:
                round_callback(r + 1, self, clients)

                  
        self._calibrate(clients)

    def _calibrate(self, clients: List[dict]) -> None:
        for k, c in enumerate(clients):
            scores_cal = self._compute_scores(c["X_cal"], k)
            self.client_thresholds[k] = self._pick_threshold(
                scores_cal, c["y_cal"])

                                                          

    def predict(self, clients: List[dict]) -> List[dict]:
        """
        返回同时包含 raw_score / norm_score / tau_raw / tau_norm。

        说明
        ----
        - raw_score: 原始连续异常分数（推荐用于分析与可视化）
        - score    : 兼容旧接口，仍保留为 min-max 归一化后的分数
        - tau_raw  : 在 raw_score 空间中的阈值
        - tau_norm : 将 tau_raw 映射到归一化 score 空间后的阈值
        """
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
