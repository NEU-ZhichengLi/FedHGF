"""
GDN: Graph Neural Network-Based Anomaly Detection in Multivariate Time Series
==============================================================================
Deng & Hooi, AAAI 2021

核心组件
--------
1. Sensor Embedding:   每个传感器一个可学习 embedding v_i ∈ R^d
2. Graph Structure Learning: 基于 v_i 余弦相似度 TopK 构建有向稀疏图
3. Graph Attention-Based Forecasting: 基于邻居做注意力预测下一时刻
4. Graph Deviation Scoring: Err = |s - ŝ|, 用 median/IQR 归一化后取 max

数据流 (对应本项目窗口化数据):
    X [B, T, n_k]  (window_len=T)
      取 X[:, :T-1, :] 作为输入, X[:, T-1, :] 作为预测目标
      ↓ 将输入 reshape 为 [B, n_k, T-1] (节点 = 传感器)
      ↓ Graph Attention Layer (用 learn_graph() 得到的 A)
      ↓ 每节点 MLP → 预测 [B, n_k]
    异常分数: robust-normalize(|预测 - 真值|).max(dim=-1)

联邦扩展:
    - 支持 federated=True (FedAvg 参数聚合) 与集中式训练
    - 异构特征采用 pad-to-max 策略 (与 FL-STAM 一致)
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


                                                                              
                                   
                                                                              

class GDNGraphAttention(nn.Module):
    """
    GDN 的图注意力层 (论文 §3.5)。
    注意力系数: g_i = v_i ⊕ (W x_i);  π(i,j) = LeakyReLU(a^T (g_i ⊕ g_j))
    关键点: v_i 是节点专属 embedding, 用于区分不同类型的传感器。
    """

    def __init__(self, in_dim: int, out_dim: int, embed_dim: int, alpha: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.alpha = alpha

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_proj = nn.Linear(2 * (out_dim + embed_dim), 1, bias=False)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_proj.weight)

    def forward(self, X: torch.Tensor, V: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X: [B, N, in_dim]  节点特征
        V: [N, embed_dim]  传感器 embedding
        A: [N, N]          二值邻接矩阵 (TopK 后)
        返回: Z [B, N, out_dim]
        """
        B, N, _ = X.shape
        H = self.W(X)                                                   

        V_exp = V.unsqueeze(0).expand(B, -1, -1)                          
        G = torch.cat([V_exp, H], dim=-1)                                 

        Gi = G.unsqueeze(2).expand(-1, -1, N, -1)                            
        Gj = G.unsqueeze(1).expand(-1, N, -1, -1)
        G_pair = torch.cat([Gi, Gj], dim=-1)                                     

        e = F.leaky_relu(self.attn_proj(G_pair).squeeze(-1), self.alpha)             

                              
        mask = A + torch.eye(N, device=A.device, dtype=A.dtype)
        mask = (mask > 0).float()
        e = e.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))

        alpha_ij = F.softmax(e, dim=2)                            
        Z = torch.einsum('bij,bjd->bid', alpha_ij, H)                   
        return F.relu(Z)


                                                                              
             
                                                                              

class GDNNet(nn.Module):
    """
    完整 GDN 网络。
    Forward 输入 X [B, T-1, N]:
      → reshape 为 [B, N, T-1]
      → GDNGraphAttention(X, V, A)           [B, N, hidden_dim]
      → element-wise × V  (论文 Eq.9)         [B, N, hidden_dim]
      → 输出 MLP → [B, N]  (预测每个传感器下一时刻)
    """

    def __init__(self, n_features: int, window_in: int, embed_dim: int = 64,
                 hidden_dim: int = 64, topk: int = 15, out_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        assert hidden_dim == embed_dim, "GDN Eq.9 要求 hidden_dim == embed_dim"
        self.n_features = n_features
        self.window_in = window_in
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.topk = min(topk, max(1, n_features - 1))

        self.V = nn.Parameter(torch.empty(n_features, embed_dim))
        nn.init.kaiming_uniform_(self.V, a=5 ** 0.5)

        self.gat = GDNGraphAttention(window_in, hidden_dim, embed_dim)

        layers = []
        in_dim = hidden_dim
        for _ in range(out_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, 1)]
        self.out_mlp = nn.Sequential(*layers)

    def learn_graph(self) -> torch.Tensor:
        """基于 sensor embedding 余弦相似度选 TopK 邻居 (Eq.2-3)。"""
        V_norm = F.normalize(self.V, p=2, dim=-1)
        sim = V_norm @ V_norm.t()
        sim = sim - torch.eye(self.n_features, device=sim.device) * 1e9
        _, idx = torch.topk(sim, self.topk, dim=-1)
        A = torch.zeros_like(sim)
        A.scatter_(1, idx, 1.0)
        return A

    def forward(self, X_in: torch.Tensor) -> torch.Tensor:
        """X_in: [B, T-1, N] → pred: [B, N]"""
        X_node = X_in.transpose(1, 2)                               
        A = self.learn_graph()
        Z = self.gat(X_node, self.V, A)                                
        V_exp = self.V.unsqueeze(0).expand(Z.size(0), -1, -1)
        Z = Z * V_exp
        pred = self.out_mlp(Z).squeeze(-1)                     
        return pred


                                                                              
             
                                                                              

class GDN:
    """
    GDN 完整方案 (集中式 + 联邦训练, 统一接口与 FedGAD 对齐)。

    参数
    ----
    embed_dim / hidden_dim : sensor embedding 和图注意力隐藏维度 (须相等)
    topk                   : 图结构学习保留的 TopK 邻居数
    n_epochs               : 集中式训练 epochs
    federated              : True → FedAvg, False → 集中式
    threshold_mode         : 'ratio' 或 'best_f1'
    """

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 64, topk: int = 15,
                 out_layers: int = 2, dropout: float = 0.0, n_epochs: int = 30,
                 lr: float = 1e-3, batch_size: int = 64, federated: bool = False,
                 fed_rounds: int = 10, fed_local_epochs: int = 3,
                 threshold_mode: str = "ratio",
                 ratio_clip: Tuple[float, float] = (0.01, 0.45),
                 device: str = "cuda", seed: int = 42):
        self.cfg = {k: v for k, v in locals().items() if k != "self"}
        set_seed(seed)
        self.device = torch.device(
            device if device != "cpu" and torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[GDNNet] = None
        self.client_thresholds: Dict[int, float] = {}
        self.err_median: Optional[torch.Tensor] = None
        self.err_iqr: Optional[torch.Tensor] = None

                                                           

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

    def _split_in_target(self, X3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return X3d[:, :-1, :], X3d[:, -1, :]

                                                         

    def _raw_errors(self, X: np.ndarray, k: int = 0) -> np.ndarray:
        self.model.eval()
        X3d = self._squeeze_and_pad(X)
        n_k_valid = (self.n_features_per_client[k]
                     if hasattr(self, "n_features_per_client") else X3d.shape[2])
        errs = []
        bs = self.cfg["batch_size"]
        with torch.no_grad():
            for s in range(0, len(X3d), bs):
                xb = self._to_device(X3d[s:s + bs])
                X_in, X_tgt = self._split_in_target(xb)
                pred = self.model(X_in)
                err = (X_tgt - pred).abs()
                errs.append(err[:, :n_k_valid].cpu().numpy())
        return np.concatenate(errs, axis=0).astype(np.float32)

    def _compute_scores(self, X: np.ndarray, k: int = 0) -> np.ndarray:
        """Graph Deviation Scoring (Eq.12-13): a_i = (Err_i - med_i)/IQR_i; A = max_i a_i"""
        errs = self._raw_errors(X, k)
        med = self.err_median.cpu().numpy()
        iqr = self.err_iqr.cpu().numpy()
        n_k = errs.shape[1]
        a = (errs - med[:n_k]) / (iqr[:n_k] + 1e-8)
        return a.max(axis=1).astype(np.float32)

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

                                                            

    def _train_one_client(self, X_train: np.ndarray, n_k_valid: int,
                         model: nn.Module, optimizer, n_epochs: int) -> float:
        X3d = self._squeeze_and_pad(X_train)
        N = len(X3d)
        bs = self.cfg["batch_size"]
        total_loss, n_batches = 0.0, 0
        model.train()
        for _ in range(n_epochs):
            perm = np.random.permutation(N)
            for bi in range(0, N, bs):
                idx = perm[bi:bi + bs]
                xb = self._to_device(X3d[idx])
                X_in, X_tgt = self._split_in_target(xb)
                pred = model(X_in)
                loss = F.mse_loss(pred[:, :n_k_valid], X_tgt[:, :n_k_valid])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
        return total_loss / max(n_batches, 1)

                                                        

    def _train_centralized(self, clients: List[dict]) -> None:
        X_all = np.concatenate(
            [self._squeeze_and_pad(c["X_train"]) for c in clients], axis=0)
        n_k_valid = self.n_features_common
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        print(f"  [GDN] centralized training on {len(X_all)} windows")
        for ep in range(self.cfg["n_epochs"]):
            loss = self._train_one_client(
                X_all, n_k_valid, self.model, optimizer, n_epochs=1)
            if (ep + 1) % max(1, self.cfg["n_epochs"] // 10) == 0:
                print(f"    epoch {ep + 1}/{self.cfg['n_epochs']}: loss={loss:.4f}")

    def _train_federated(self, clients: List[dict]) -> None:
        total = sum(c["X_train"].shape[0] for c in clients)
        weights = [c["X_train"].shape[0] / total for c in clients]
        for r in range(self.cfg["fed_rounds"]):
            global_vec = parameters_to_vector(self.model.parameters()).detach().clone()
            agg_delta = torch.zeros_like(global_vec)
            round_loss = 0.0
            for k, c in enumerate(clients):
                local_model = copy.deepcopy(self.model).to(self.device)
                opt = torch.optim.Adam(local_model.parameters(), lr=self.cfg["lr"])
                n_k_valid = self.n_features_per_client[k]
                loss_k = self._train_one_client(
                    c["X_train"], n_k_valid, local_model, opt,
                    n_epochs=self.cfg["fed_local_epochs"])
                round_loss += weights[k] * loss_k
                local_vec = parameters_to_vector(local_model.parameters()).detach()
                agg_delta += weights[k] * (local_vec - global_vec)
            with torch.no_grad():
                vector_to_parameters(global_vec + agg_delta, self.model.parameters())
            print(f"    Fed round {r + 1}/{self.cfg['fed_rounds']}: "
                  f"loss={round_loss:.4f}")

                                                            

    def _fit_robust_stats(self, clients: List[dict]) -> None:
        errs_all = []
        for k, c in enumerate(clients):
            errs = self._raw_errors(c["X_train"], k)
            if errs.shape[1] < self.n_features_common:
                pad = np.zeros(
                    (errs.shape[0], self.n_features_common - errs.shape[1]),
                    dtype=errs.dtype)
                errs = np.concatenate([errs, pad], axis=1)
            errs_all.append(errs)
        E = np.concatenate(errs_all, axis=0)
        med = np.median(E, axis=0)
        q1 = np.quantile(E, 0.25, axis=0)
        q3 = np.quantile(E, 0.75, axis=0)
        iqr = q3 - q1
        self.err_median = torch.as_tensor(med, dtype=torch.float32, device=self.device)
        self.err_iqr = torch.as_tensor(iqr, dtype=torch.float32, device=self.device)

                                                           

    def fit(self, clients: List[dict]) -> None:
        cfg = self.cfg
        sample = clients[0]["X_train"]
        window_len = sample.shape[1]
        assert window_len >= 2, "GDN 需要 window_len >= 2"

        self.n_features_per_client = [c["X_train"].shape[2] for c in clients]
        self.n_features_common = max(self.n_features_per_client)
        n_features = self.n_features_common
        window_in = window_len - 1

        self.model = GDNNet(
            n_features=n_features, window_in=window_in,
            embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
            topk=cfg["topk"], out_layers=cfg["out_layers"],
            dropout=cfg["dropout"],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        mode_str = "federated" if cfg["federated"] else "centralized"
        print(f"  [GDN] n_features={n_features} (pad-to-max, per-client={self.n_features_per_client}), "
              f"topk={self.model.topk}, d={cfg['embed_dim']}, "
              f"params={n_params:,}, mode={mode_str}")

        if cfg["federated"]:
            self._train_federated(clients)
        else:
            self._train_centralized(clients)

        self._fit_robust_stats(clients)
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
