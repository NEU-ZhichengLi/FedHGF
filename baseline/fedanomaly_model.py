"""
FedAnomaly: Federated Variational Learning for Anomaly Detection in Multivariate Time Series
==============================================================================================
Zhang et al., IPCCC 2021

核心组件
--------
1. ConvGRU Cell        — 将 GRU 门控中的点积替换为 1D 卷积
                         → 同时捕获特征级（空间）与时间依赖
2. ConvGRU-VAE         — 编码器-解码器变分自编码
                         - Encoder: ConvGRU layers → h_t → (μ_φ, σ_φ) → z
                         - Decoder: 以 z 为初始态，输出逆序重构序列
3. Loss                — ELBO = MSE(x, x_hat) + β · KL(q_φ(z|x) || N(0, I))
4. 联邦框架            — FedAvg + SGD + 1 local epoch (论文默认)
5. 阈值选择            — 校准集重构误差的最大值 (论文 "max recon error on validation")
                         本实现额外支持 ratio / best_f1 方便公平对比

数据流
------
输入 X [B, T, n_k, 1]  (来自 FedGAD 数据加载器)
  → squeeze 到 [B, T, n_k]
  → ConvGRU Encoder → h_T
  → FC → μ, log σ² → 重参数化 → z
  → ConvGRU Decoder（以 z 为 h_0）→ X_hat [B, T, n_k] (时间维反转)
  → recon loss + KL

异常分数 = 窗口内 MSE 重构误差
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


                                                                              
              
                                                                              

class ConvGRUCell(nn.Module):
    """
    Convolutional GRU 单元。与普通 GRU 的差别：
    输入-状态和状态-状态变换用 1D 卷积（沿特征/通道维）替代全连接，
    从而在时间递推中同步捕获特征级（相邻传感器）依赖。

    输入约定：
        x_t : [B, F, 1]  （沿 F 做 1D conv）
        h_{t-1} : [B, F, H_channels] （把 hidden 维作为 channel）

    实现方式：我们用 [B, 2F, L=1] 的 conv 输入来对齐原始 ConvGRU 论文公式：
        z_t = σ(W_z ∗ [h_{t-1}, x_t])
        r_t = σ(W_r ∗ [h_{t-1}, x_t])
        ĥ_t = tanh(W_h ∗ [r_t ⊙ h_{t-1}, x_t])
        h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ ĥ_t
    其中 ∗ 是沿 feature 维 (kernel_size=K) 的 1D conv。
    """

    def __init__(self, n_features: int, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        pad = kernel_size // 2

                                       
                                             
                                        
        self.conv_zr = nn.Conv1d(
            in_channels=hidden_size + 1,
            out_channels=2 * hidden_size,
            kernel_size=kernel_size, padding=pad, bias=True,
        )
                          
        self.conv_h = nn.Conv1d(
            in_channels=hidden_size + 1,
            out_channels=hidden_size,
            kernel_size=kernel_size, padding=pad, bias=True,
        )

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        x       : [B, F]       当前时刻输入 (1 channel, F features)
        h_prev  : [B, H, F]    上一时刻 hidden (H channels, F features)
        返回 h_t: [B, H, F]
        """
        x1 = x.unsqueeze(1)                                   
        inp = torch.cat([h_prev, x1], dim=1)                    
        zr = self.conv_zr(inp)                                 
        z, r = zr.chunk(2, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        inp_h = torch.cat([r * h_prev, x1], dim=1)              
        h_tilde = torch.tanh(self.conv_h(inp_h))              
        h_t = (1 - z) * h_prev + z * h_tilde
        return h_t


class ConvGRUEncoder(nn.Module):
    """Encoder: 沿时间步迭代 ConvGRU，返回最后 hidden。"""

    def __init__(self, n_features: int, hidden_size: int, kernel_size: int = 3,
                 n_layers: int = 1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cells = nn.ModuleList([
            ConvGRUCell(
                n_features=n_features,
                hidden_size=hidden_size,
                kernel_size=kernel_size,
            )
            for _ in range(n_layers)
        ])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : [B, T, F]
        返回 h_T: [B, H, F]  (多层时取最顶层最后一步)
        """
        B, T, F_dim = X.shape
        H = self.hidden_size
                          
        h_list = [torch.zeros(B, H, F_dim, device=X.device, dtype=X.dtype)
                  for _ in range(self.n_layers)]
        for t in range(T):
            x_t = X[:, t, :]                               
            inp = x_t
            for li, cell in enumerate(self.cells):
                h_list[li] = cell(inp, h_list[li])
                                                  
                                                      
                if li < self.n_layers - 1:
                    inp = h_list[li].mean(dim=1)           
        return h_list[-1]                                      


class ConvGRUDecoder(nn.Module):
    """Decoder: 从 hidden 开始，按逆序生成重构序列。"""

    def __init__(self, n_features: int, hidden_size: int, kernel_size: int = 3,
                 n_layers: int = 1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cells = nn.ModuleList([
            ConvGRUCell(
                n_features=n_features,
                hidden_size=hidden_size,
                kernel_size=kernel_size,
            )
            for _ in range(n_layers)
        ])
                                                  
        self.out_proj = nn.Conv1d(hidden_size, 1, kernel_size=1)

    def forward(self, h_init: torch.Tensor, T: int) -> torch.Tensor:
        """
        h_init : [B, H, F]  初始 hidden（来自编码器 + 重参数化后的 z）
        T      : 要生成的时间步数
        返回 X_hat : [B, T, F]  （内部按逆序生成，最终转回正序）
        """
        B, H, F_dim = h_init.shape
        h_list = [h_init.clone() for _ in range(self.n_layers)]
                
        x_prev = torch.zeros(B, F_dim, device=h_init.device, dtype=h_init.dtype)
        outputs = []
        for _ in range(T):
            inp = x_prev
            for li, cell in enumerate(self.cells):
                h_list[li] = cell(inp, h_list[li])
                if li < self.n_layers - 1:
                    inp = h_list[li].mean(dim=1)
            x_hat = self.out_proj(h_list[-1]).squeeze(1)          
            outputs.append(x_hat)
            x_prev = x_hat
        X_rev = torch.stack(outputs, dim=1)                                 
        return torch.flip(X_rev, dims=[1])                        


                                                                              
             
                                                                              

class ConvGRUVAE(nn.Module):
    """
    完整 ConvGRU-VAE。

    流程：
      X [B, T, F]
        → Encoder → h_T [B, H, F]
        → flatten → μ, logσ² (每维独立)
        → 重参数化 z
        → reshape 回 [B, H, F] 作为 Decoder 初始 hidden
        → Decoder → X_hat [B, T, F]
    """

    def __init__(self, n_features: int, hidden_size: int = 128,
                 kernel_size: int = 3, n_layers: int = 1,
                 latent_factor: float = 1.0):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        latent_dim = max(4, int(hidden_size * latent_factor))
        self.latent_dim = latent_dim

        self.encoder = ConvGRUEncoder(n_features, hidden_size, kernel_size, n_layers)
        self.decoder = ConvGRUDecoder(n_features, hidden_size, kernel_size, n_layers)

                                                           
        flat = hidden_size * n_features
        self.fc_mu = nn.Linear(flat, flat)
        self.fc_logvar = nn.Linear(flat, flat)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回: X_hat [B, T, F], μ [B, H*F], logvar [B, H*F]
        """
        B, T, F_dim = X.shape
        h_T = self.encoder(X)                                    
        flat = h_T.reshape(B, -1)                               
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
                            
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        h_init = z.reshape(B, self.hidden_size, F_dim)
        X_hat = self.decoder(h_init, T)
        return X_hat, mu, logvar


def elbo_loss(X: torch.Tensor, X_hat: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回 (total_loss, recon, kl)。
    recon = MSE(X, X_hat)  (逐元素均值)
    KL    = 0.5 * sum(μ² + σ² - log σ² - 1) / B  (每样本均值)
    total = recon + β · KL
    """
    recon = F.mse_loss(X_hat, X, reduction="mean")
    kl = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1.0)
    return recon + beta * kl, recon, kl


                                                                              
                               
                                                                              

class FedAnomaly:
    """
    FedAnomaly 完整方案。

    接口:
      fit(clients)     — 联邦训练 (FedAvg)
      predict(clients) — 推理, 返回统一格式结果（与 FedGAD/FL-STAM 一致）

    数据兼容:
      复用 FedGAD 数据加载器; 输入 X 形状 [N, T, n_k, 1]，内部 squeeze 到 [N, T, n_k]
      异构客户端 n_k 对齐策略：**零填充到全局 max n_k**（与 FL-STAM 保持一致）。

    论文关键默认:
      ConvGRU kernel=3, hidden=128, 1 local epoch, SGD lr=1e-4
    """

    def __init__(
        self,
        hidden_size: int = 128,
        kernel_size: int = 3,
        n_layers: int = 1,
        n_rounds: int = 10,
        local_epochs: int = 1,
        lr: float = 1e-4,
        batch_size: int = 32,
        beta_kl: float = 1.0,
        optimizer: str = "sgd",                                   
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
        self.model: Optional[ConvGRUVAE] = None
        self.client_thresholds: Dict[int, float] = {}

                                                        

    def _squeeze_and_pad(self, X: np.ndarray) -> np.ndarray:
        """[N, T, n_k, 1] → [N, T, n_k]，不足 n_features_common 时零补齐。"""
        arr = X.squeeze(-1) if X.ndim == 4 and X.shape[-1] == 1 else X
        if hasattr(self, "n_features_common") and arr.ndim == 3:
            actual = arr.shape[2]
            target = self.n_features_common
            if actual < target:
                pad = np.zeros(
                    (arr.shape[0], arr.shape[1], target - actual),
                    dtype=arr.dtype,
                )
                arr = np.concatenate([arr, pad], axis=2)
        return arr

    def _to_device(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

                                                       

    def _compute_scores(self, X: np.ndarray, k: int = 0) -> np.ndarray:
        """异常分数 = 窗口 MSE 重构误差（仅有效特征）。"""
        self.model.eval()
        X_3d = self._squeeze_and_pad(X)
        n_k_valid = (self.n_features_per_client[k]
                     if hasattr(self, "n_features_per_client") else X_3d.shape[2])
        scores = []
        bs = self.cfg["batch_size"]
        with torch.no_grad():
            for s in range(0, len(X_3d), bs):
                xb = self._to_device(X_3d[s:s + bs])
                x_hat, _, _ = self.model(xb)
                recon_err = (
                    xb[:, :, :n_k_valid] - x_hat[:, :, :n_k_valid]
                ).pow(2).mean(dim=(1, 2))
                scores.append(recon_err.cpu().numpy())
        return np.concatenate(scores).astype(np.float32)

                                                       

    def _pick_threshold(self, scores: np.ndarray, y: np.ndarray) -> float:
        mode = self.cfg["threshold_mode"]
        if mode == "best_f1":
            return self._threshold_best_f1(scores, y)
        if mode == "max":
                                 
            normal = scores[y == 0] if len(np.unique(y)) > 1 else scores
            if len(normal) == 0:
                return float(np.quantile(scores, 0.95))
            return float(normal.max())
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
                                             
        def _nk(c):
            s = c["X_train"]
            return s.shape[2] if s.ndim in (3, 4) else 1
        self.n_features_per_client = [_nk(c) for c in clients]
        self.n_features_common = max(self.n_features_per_client)
        n_features = self.n_features_common
        window_len = (clients[0]["X_train"].shape[1]
                      if clients[0]["X_train"].ndim >= 3 else 1)

        self.model = ConvGRUVAE(
            n_features=n_features,
            hidden_size=cfg["hidden_size"],
            kernel_size=cfg["kernel_size"],
            n_layers=cfg["n_layers"],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        total = sum(c["X_train"].shape[0] for c in clients)
        weights = [c["X_train"].shape[0] / total for c in clients]

        print(f"  [FedAnomaly] {len(clients)} clients, "
              f"features={self.n_features_per_client} "
              f"(max={n_features}, pad-to-max), "
              f"win={window_len}, hidden={cfg['hidden_size']}, "
              f"params={n_params:,}")

                             
        for r in range(cfg["n_rounds"]):
            global_vec = parameters_to_vector(
                self.model.parameters()).detach().clone()
            agg_delta = torch.zeros_like(global_vec)
            round_loss, round_recon, round_kl = 0.0, 0.0, 0.0

            for k, c in enumerate(clients):
                local_model = copy.deepcopy(self.model).to(self.device)
                local_model.train()
                if cfg["optimizer"] == "adam":
                    optimizer = torch.optim.Adam(
                        local_model.parameters(), lr=cfg["lr"])
                else:
                    optimizer = torch.optim.SGD(
                        local_model.parameters(), lr=cfg["lr"])

                X_tr = self._squeeze_and_pad(c["X_train"])
                n_k_valid = self.n_features_per_client[k]
                N_k = len(X_tr)
                c_loss, c_recon, c_kl, nb = 0.0, 0.0, 0.0, 0

                for _ in range(cfg["local_epochs"]):
                    perm = np.random.permutation(N_k)
                    for bi in range(0, N_k, cfg["batch_size"]):
                        idx = perm[bi:bi + cfg["batch_size"]]
                        xb = self._to_device(X_tr[idx])
                        x_hat, mu, logvar = local_model(xb)
                                   
                        loss, recon, kl = elbo_loss(
                            xb[:, :, :n_k_valid],
                            x_hat[:, :, :n_k_valid],
                            mu, logvar, beta=cfg["beta_kl"],
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            local_model.parameters(), max_norm=5.0)
                        optimizer.step()
                        c_loss += loss.item()
                        c_recon += recon.item()
                        c_kl += kl.item()
                        nb += 1

                round_loss += weights[k] * c_loss / max(nb, 1)
                round_recon += weights[k] * c_recon / max(nb, 1)
                round_kl += weights[k] * c_kl / max(nb, 1)

                local_vec = parameters_to_vector(
                    local_model.parameters()).detach()
                agg_delta += weights[k] * (local_vec - global_vec)

            with torch.no_grad():
                vector_to_parameters(
                    global_vec + agg_delta, self.model.parameters())

            print(f"    Round {r + 1}/{cfg['n_rounds']}: "
                  f"loss={round_loss:.4f}  recon={round_recon:.4f}  "
                  f"KL={round_kl:.4f}")
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
        返回同时包含 raw_score / norm_score / tau_raw / tau_norm（与 FL-STAM 一致）。
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
