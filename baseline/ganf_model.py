"""
GANF: Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series
======================================================================================
Dai & Chen, ICLR 2022

核心组件
--------
1. 贝叶斯网络 (DAG) — 建模传感器间的条件独立性 (因果关系)
2. LSTM 时序编码    — 将变长历史压缩为固定长度隐状态
3. 图卷积依赖编码器 — 聚合 DAG 父节点信息 → 条件向量 d_t^i
4. 条件 MAF         — 估计 p(x_t^i | d_t^i) 得到精确密度
5. DAG 约束         — tr(e^{A∘A}) = n (NOTEARS, 可微分)
6. 增广拉格朗日法   — 联合优化 A 和 θ

密度分解:
  log p(X) = Σ_i Σ_t log p(x_t^i | pa(x^i)_{1:t}, x^i_{1:t-1})

异常分数 = -log p(X), 密度低 → 分数高 → 异常
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


                                                                              
                                              
                                                                              

class CondMaskedLinear(nn.Module):
    """带掩码的线性层。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: torch.Tensor):
        self.mask.data.copy_(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)


class CondMADEBlock(nn.Module):
    """
    条件 MADE block。
    输入 [x || cond], 输出自回归的 (mu, log_s)。
    条件信息 cond 可以连接到所有隐藏单元 (无掩码约束)。
    """

    def __init__(self, d_x: int, d_cond: int, hidden: int):
        super().__init__()
        self.d_x = d_x
        self.d_in = d_x + d_cond
        self.hidden = hidden

        self.fc1 = CondMaskedLinear(self.d_in, hidden)
        self.fc_mu = CondMaskedLinear(hidden, d_x)
        self.fc_log_s = CondMaskedLinear(hidden, d_x)
        self._build_masks(d_cond)

        for layer in [self.fc1, self.fc_mu, self.fc_log_s]:
            nn.init.xavier_uniform_(layer.linear.weight)
            if layer.linear.bias is not None:
                nn.init.zeros_(layer.linear.bias)

    def _build_masks(self, d_cond: int):
        d_x, hidden = self.d_x, self.hidden
        m_h = torch.arange(hidden) % max(1, d_x - 1)
        m_in = torch.cat([torch.arange(d_x), torch.zeros(d_cond)])
        m_out = torch.arange(d_x)

        mask1 = (m_h[:, None] >= m_in[None, :]).float()
        mask2 = (m_out[:, None] > m_h[None, :]).float()

        self.fc1.set_mask(mask1)
        self.fc_mu.set_mask(mask2)
        self.fc_log_s.set_mask(mask2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        inp = torch.cat([x, cond], dim=-1)
        h = F.relu(self.fc1(inp))
        mu = self.fc_mu(h)
        log_s = torch.clamp(self.fc_log_s(h), min=-5.0, max=3.0)
        return mu, log_s


class ConditionalMAF(nn.Module):
    """
    条件 Masked Autoregressive Flow。

    给定条件 d, 计算 log p(x | d):
      z_i = (x_i - μ_i(x_{<i}, d)) / exp(s_i(x_{<i}, d))
      log p(x|d) = log N(z; 0, I) + Σ (-s_i)
    """

    def __init__(self, d_x: int, d_cond: int,
                 n_blocks: int = 4, hidden: int = 32):
        super().__init__()
        self.blocks = nn.ModuleList([
            CondMADEBlock(d_x, d_cond, hidden) for _ in range(n_blocks)
        ])
        self.register_buffer(
            "log_2pi",
            torch.tensor(math.log(2.0 * math.pi), dtype=torch.float32)
        )

    def log_prob(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, D]
        cond: [B, d_cond]
        返回: [B]  log p(x | cond)
        """
        u = x
        log_det = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for block in self.blocks:
            mu, log_s = block(u, cond)
            u = (u - mu) * torch.exp(-log_s)
            log_det -= log_s.sum(dim=-1)
        log_p0 = -0.5 * (u.pow(2) + self.log_2pi).sum(dim=-1)
        return log_p0 + log_det


                                                                              
                                               
                                                                              

class DependencyEncoder(nn.Module):
    """
    GANF 依赖编码器 (论文 Eq.7-8)。

    步骤:
      1. 共享 LSTM 对每个节点序列编码 → h_t^i (Eq.7)
      2. 图卷积: D_t = ReLU(A·H_t·W1 + H_{t-1}·W2) · W3 (Eq.8)
         A 是 DAG 邻接矩阵, 只聚合父节点信息
    """

    def __init__(self, d_x: int, d_h: int):
        super().__init__()
        self.d_h = d_h
        self.rnn = nn.LSTM(
            input_size=d_x, hidden_size=d_h,
            batch_first=True, num_layers=1,
        )
        self.W1 = nn.Linear(d_h, d_h, bias=False)
        self.W2 = nn.Linear(d_h, d_h, bias=False)
        self.W3 = nn.Linear(d_h, d_h, bias=False)

    def forward(self, X: torch.Tensor, A: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X: [B, n, T, D]
        A: [n, n]  DAG 邻接矩阵

        返回: D_all [B,n,T,d_h], H_all [B,n,T,d_h]
        """
        B, n, T, D = X.shape

        X_flat = X.reshape(B * n, T, D)
        H_seq, _ = self.rnn(X_flat)
        H_seq = H_seq.reshape(B, n, T, self.d_h)

        D_all = []
        H_prev = torch.zeros(B, n, self.d_h, device=X.device)

        for t in range(T):
            H_t = H_seq[:, :, t, :]                    
            agg = torch.matmul(A, H_t)                   
            d_t = self.W3(F.relu(self.W1(agg) + self.W2(H_prev)))
            D_all.append(d_t)
            H_prev = H_t

        D_all = torch.stack(D_all, dim=2)
        return D_all, H_seq


                                                                              
              
                                                                              

class GANFNet(nn.Module):
    """
    完整 GANF 网络。

    数据流:
      X [B,n,T,D]
        → DependencyEncoder(X, A) → d_t^i [B,n,T,d_h]
        → ConditionalMAF(x_t^i, d_t^i) → log p(x_t^i | d_t^i)
        → 求和 → log p(X)

    DAG 约束: h(A) = tr(e^{A∘A}) - n = 0
    """

    def __init__(self, n_nodes: int, d_x: int, d_h: int,
                 flow_blocks: int = 4, flow_hidden: int = 32):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_x = d_x
        self.d_h = d_h

        self.A_raw = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        nn.init.uniform_(self.A_raw, -0.01, 0.01)

        self.dep_encoder = DependencyEncoder(d_x, d_h)
        self.flow = ConditionalMAF(d_x, d_h, flow_blocks, flow_hidden)

    @property
    def A(self) -> torch.Tensor:
        """非负邻接矩阵, 对角线为 0。"""
        A = torch.relu(self.A_raw)
        return A * (1.0 - torch.eye(self.n_nodes, device=A.device))

    def dag_constraint(self) -> torch.Tensor:
        """h(A) = tr(e^{A∘A}) - n。用多项式展开近似矩阵指数。"""
        A = self.A
        M = A * A
        n = self.n_nodes
        I = torch.eye(n, device=A.device)
        E = I.clone()
        M_power = I.clone()
        for k in range(1, n + 1):
            M_power = M_power @ M / k
            E = E + M_power
        return torch.trace(E) - n

    def forward_batched(self, X: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        X: [B, n, T, D]
        返回: log_prob [B], node_log_probs [B, n]
        """
        B, n, T, D = X.shape
        A = self.A

        D_repr, _ = self.dep_encoder(X, A)

                                    
        x_flat = X.reshape(B * n * T, D)
        d_flat = D_repr.reshape(B * n * T, self.d_h)

        lp_flat = self.flow.log_prob(x_flat, d_flat)           
        lp = lp_flat.reshape(B, n, T)

        node_log_probs = lp.sum(dim=2)           
        log_prob = node_log_probs.sum(dim=1)       
        return log_prob, node_log_probs


                                                                              
              
                                                                              

class GANF:
    """
    GANF 完整方案。

    支持两种训练模式:
      - 集中式 (federated=False): 合并所有 client 数据, 增广拉格朗日优化
      - 联邦式 (federated=True):  简单 FedAvg 包装

    接口:
      fit(clients)     — 训练
      predict(clients) — 推理, 返回统一格式结果

    参数
    ----
    d_h              : 隐状态维度
    flow_blocks      : MAF 块数
    flow_hidden      : MAF 隐藏层维度
    n_epochs         : 每次内层优化的 epoch 数
    lr               : 模型参数学习率
    lr_A             : 邻接矩阵学习率
    batch_size       : 批大小
    eta              : 增广拉格朗日惩罚增长率
    gamma            : 收缩判定系数
    max_lagrangian_iter : 最大外层迭代次数
    federated        : 是否使用 FedAvg
    fed_rounds       : 联邦轮数 (仅 federated=True)
    threshold_mode   : 阈值策略 ('ratio' / 'best_f1')
    device           : 计算设备
    seed             : 随机种子
    """

    def __init__(
        self,
        d_h: int = 32,
        flow_blocks: int = 4,
        flow_hidden: int = 32,
        n_epochs: int = 15,
        lr: float = 1e-3,
        lr_A: float = 1e-3,
        batch_size: int = 64,
        eta: float = 10.0,
        gamma: float = 0.5,
        max_lagrangian_iter: int = 8,
        federated: bool = False,
        fed_rounds: int = 10,
        fed_local_epochs: int = 5,
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
        self.model: Optional[GANFNet] = None
        self.client_thresholds: Dict[int, float] = {}

                                                        

    def _to_ganf_format(self, X: np.ndarray) -> np.ndarray:
        """[N, T, n_k, D] → [N, n_k, T, D] (GANF 格式: 节点在前)，并截断到 n_features_common"""
        arr = X if X.ndim == 4 else X[:, :, :, np.newaxis]         
        if hasattr(self, 'n_features_common'):
            arr = arr[:, :, :self.n_features_common, :]
        return arr.transpose(0, 2, 1, 3).copy()

                                                        

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """异常分数 = -log p(X)。"""
        self.model.eval()
        X_ganf = self._to_ganf_format(X)
        scores = []
        bs = self.cfg["batch_size"]
        with torch.no_grad():
            for s in range(0, len(X_ganf), bs):
                xb = torch.as_tensor(
                    X_ganf[s:s + bs], dtype=torch.float32, device=self.device)
                log_prob, _ = self.model.forward_batched(xb)
                scores.append(-log_prob.cpu().numpy())
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

                                                       

    def _train_centralized(self, X_train_all: np.ndarray) -> None:
        """增广拉格朗日法联合优化 DAG 和 Flow。"""
        import time as _time
        cfg = self.cfg
        model = self.model
        X_data = self._to_ganf_format(X_train_all)
        N = len(X_data)

                                           
        _MAX_TRAIN = cfg.get("max_train_windows", 50_000)
        if N > _MAX_TRAIN:
            print(f"    [GANF] subsampling {N} → {_MAX_TRAIN} windows")
            idx_sub = np.random.choice(N, _MAX_TRAIN, replace=False)
            X_data = X_data[idx_sub]
            N = _MAX_TRAIN

        n_batches_per_epoch = max(1, N // cfg["batch_size"])
        total_epochs = cfg["max_lagrangian_iter"] * cfg["n_epochs"]
        print(f"    [GANF] N={N}, batch_size={cfg['batch_size']}, "
              f"batches/epoch={n_batches_per_epoch}, "
              f"total_epochs={total_epochs}")

        theta_params = [p for n, p in model.named_parameters()
                        if n != "A_raw"]
        opt_theta = torch.optim.Adam(theta_params, lr=cfg["lr"])
        opt_A = torch.optim.Adam([model.A_raw], lr=cfg["lr_A"])

        lam = 0.0
        c = 1.0
        h_prev = float("inf")
        global_epoch = 0

        for lag_iter in range(cfg["max_lagrangian_iter"]):
            model.train()
            for epoch in range(cfg["n_epochs"]):
                global_epoch += 1
                t_ep = _time.time()
                perm = np.random.permutation(N)
                total_nll = 0.0
                n_batches = 0

                for bi in range(0, N, cfg["batch_size"]):
                    idx = perm[bi:bi + cfg["batch_size"]]
                    xb = torch.as_tensor(
                        X_data[idx], dtype=torch.float32, device=self.device)

                    log_prob, _ = model.forward_batched(xb)
                    nll = -log_prob.mean()
                    h = model.dag_constraint()
                    loss = nll + lam * h + (c / 2.0) * h.pow(2)

                    opt_theta.zero_grad(set_to_none=True)
                    opt_A.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_theta.step()
                    opt_A.step()

                    total_nll += nll.item()
                    n_batches += 1

                ep_nll = total_nll / max(n_batches, 1)
                ep_sec = _time.time() - t_ep
                print(f"      epoch {global_epoch}/{total_epochs}  "
                      f"NLL={ep_nll:.4f}  ({ep_sec:.1f}s)", flush=True)

            with torch.no_grad():
                h_val = model.dag_constraint().item()
            lam += c * h_val
            if abs(h_val) > cfg["gamma"] * h_prev:
                c *= cfg["eta"]
            h_prev = abs(h_val)

            avg_nll = total_nll / max(n_batches, 1)
            n_edges = int((model.A > 0.01).sum().item())
            print(f"    Lag {lag_iter + 1}: NLL={avg_nll:.4f}  "
                  f"h(A)={h_val:.6f}  edges={n_edges}  "
                  f"λ={lam:.3f}  c={c:.1f}")

            if abs(h_val) < 1e-6:
                print(f"    DAG constraint satisfied.")
                break

                                                       

    def _train_federated(self, clients: List[dict]) -> None:
        """FedAvg 包装。"""
        cfg = self.cfg
        total = sum(c["X_train"].shape[0] for c in clients)
        weights = [c["X_train"].shape[0] / total for c in clients]

        for r in range(cfg["fed_rounds"]):
            global_vec = parameters_to_vector(
                self.model.parameters()).detach().clone()
            agg_delta = torch.zeros_like(global_vec)
            round_nll = 0.0

            for k, c in enumerate(clients):
                local_model = copy.deepcopy(self.model).to(self.device)
                X_data = self._to_ganf_format(c["X_train"])
                N = len(X_data)

                optimizer = torch.optim.Adam(
                    local_model.parameters(), lr=cfg["lr"])

                local_model.train()
                batch_nll, n_b = 0.0, 0
                for _ in range(cfg["fed_local_epochs"]):
                    perm = np.random.permutation(N)
                    for bi in range(0, N, cfg["batch_size"]):
                        idx = perm[bi:bi + cfg["batch_size"]]
                        xb = torch.as_tensor(
                            X_data[idx], dtype=torch.float32,
                            device=self.device)
                        log_prob, _ = local_model.forward_batched(xb)
                        nll = -log_prob.mean()
                        h = local_model.dag_constraint()
                        loss = nll + 0.5 * h.pow(2)

                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            local_model.parameters(), 1.0)
                        optimizer.step()
                        batch_nll += nll.item()
                        n_b += 1

                round_nll += weights[k] * batch_nll / max(n_b, 1)

                local_vec = parameters_to_vector(
                    local_model.parameters()).detach()
                agg_delta += weights[k] * (local_vec - global_vec)

            with torch.no_grad():
                vector_to_parameters(
                    global_vec + agg_delta, self.model.parameters())

            print(f"    Fed round {r + 1}/{cfg['fed_rounds']}: "
                  f"NLL={round_nll:.4f}")

                                                       

    def fit(self, clients: List[dict]) -> None:
        cfg = self.cfg
        sample = clients[0]["X_train"]
                                          
        self.n_features_common = min(c["X_train"].shape[2] for c in clients)
        n_features = self.n_features_common
        d_x = sample.shape[3] if sample.ndim == 4 else 1

        self.model = GANFNet(
            n_nodes=n_features, d_x=d_x, d_h=cfg["d_h"],
            flow_blocks=cfg["flow_blocks"], flow_hidden=cfg["flow_hidden"],
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  [GANF] n_nodes={n_features}, d_x={d_x}, d_h={cfg['d_h']}, "
              f"params={n_params:,}, federated={cfg['federated']}")

        if cfg["federated"]:
            self._train_federated(clients)
        else:
            X_all = np.concatenate(
                [c["X_train"][:, :, :self.n_features_common] for c in clients],
                axis=0)
            self._train_centralized(X_all)

        self._calibrate(clients)

    def _calibrate(self, clients: List[dict]) -> None:
        for k, c in enumerate(clients):
            scores_cal = self._compute_scores(c["X_cal"])
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
            scores = self._compute_scores(c["X_test"]).astype(np.float64)
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
