"""
uFedHy-DisMTSADD: Unsupervised Federated Hypernetwork Method for Distributed
Multivariate Time Series Anomaly Detection and Diagnosis
=========================================================================
Hao, Chen, Chen & Li, Information Processing and Management 62 (2025) 104107
https://doi.org/10.1016/j.ipm.2025.104107

核心组件
--------
1. SC Nor-Transformer (客户端模型)
   - Series Normalization     : 滑动窗口实例归一化，消除聚合时偏移 (Eq.7-9)
   - Series Conversion Embed  : 每变量序列→d_model token (Fig.4, Eq.11)
   - Transformer Encoder      : 变量维度多头自注意力 (Eq.12-15)
   - Feature Projection       : 重建 T 长度序列
   - Series De-Normalization  : 恢复原始尺度 (Eq.10)

2. FederatedHypernetwork (服务端)
   - 每个客户端可学习嵌入向量 e_i
   - MLP: e_i → θ_i (完整 SCNorTransformer 参数)
   - 梯度通过链式法则更新 μ (Algorithm 1)
   - 超网络参数不传输，仅传 θ_i，通信高效

3. 异常检测
   - 重建误差作为异常分数 (Eq.16)
   - 动态阈值选择

4. 异常诊断
   - PC 算法：条件独立性检验 → 因果骨架 → DAG
   - PageRank 算法：在 DAG 上排名根因变量 (Eq.17)

异常分数: A = sqrt(Σ(x_i - x̂_i)²)
"""

from __future__ import annotations

import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


                                                                              
                                                     
                                                                              

class SeriesNormalization(nn.Module):
    """
    逐实例归一化：对每个窗口的时间维度计算均值和标准差。
    能有效消除分布式环境中模型聚合导致的时序偏移。
    Eq. 7-9: x'_i = (x_i - μ_x) / σ_x
    """

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, T, C]
        Returns: x_norm [B, T, C], mu [B, 1, C], sigma [B, 1, C]
        """
        mu = x.mean(dim=1, keepdim=True)                      
        sigma = x.std(dim=1, keepdim=True) + 1e-8             
        return (x - mu) / sigma, mu, sigma


class SeriesDeNormalization(nn.Module):
    """
    逆归一化：恢复原始尺度。
    Eq. 10: ŷ_i = σ_x ⊙ (y'_i + μ_x)
    """

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """x: [B, T, C], mu/sigma: [B, 1, C] -> [B, T, C]"""
        return x * sigma + mu


                                                                              
                                       
                                                                              

class SeriesConversionEmbedding(nn.Module):
    """
    序列转换嵌入（Series Conversion Embedding）。

    核心区别（Fig. 4）:
      Vanilla Transformer : 每个时间步嵌入 → T 个 token，每个维度 C
      SC Nor-Transformer  : 每个变量序列嵌入 → C 个 token，每个维度 d_model

    将 [B, T, C] 转换为 [B, C, d_model]：
      · 把每个变量 i 的 T 维时序 x^i_{1:T} 作为一个向量 token
      · 通过共享线性层 R^T → R^{d_model} 嵌入
      · 能有效捕捉细粒度时序特征，对残缺/碎片数据鲁棒

    Eq. 11: t_n = Embedding(X_n), R^T ↦ R^D
    """

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, C] -> [B, C, d_model]"""
        return self.proj(x.permute(0, 2, 1))                                


                                                                              
                             
                                                                              

class SCNorTransformer(nn.Module):
    """
    Series Conversion Normalization Transformer（SC Nor-Transformer）。

    完整流程:
      输入 x [B, T, C]
        ↓  SeriesNormalization           → x_norm, μ, σ
        ↓  SeriesConversionEmbedding     → tokens [B, C, d_model]
        ↓  TransformerEncoder × n_layers → encoded [B, C, d_model]
        ↓  FeatureProjection             → [B, C, T] → permute → [B, T, C]
        ↓  SeriesDeNormalization(μ, σ)   → x̂ [B, T, C]

    纯编码器架构，训练轻量、部署友好。
    模型参数不依赖变量数 C（共享 projection layer），适合异构客户端。
    """

    def __init__(
        self,
        seq_len: int = 5,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.series_norm = SeriesNormalization()
        self.embedding = SeriesConversionEmbedding(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

                                     
        self.feature_proj = nn.Linear(d_model, seq_len)
        self.series_denorm = SeriesDeNormalization()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]
        Returns: x_hat [B, T, C]
        """
        x_norm, mu, sigma = self.series_norm(x)               
        tokens = self.embedding(x_norm)                                    
        encoded = self.encoder(tokens)                                     
        out = self.feature_proj(encoded)                             
        out = out.permute(0, 2, 1)                                   
        out = self.series_denorm(out, mu, sigma)                
        return out

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """MSE 重建损失（Eq. 2）。"""
        return F.mse_loss(self(x), x)

    @torch.no_grad()
    def anomaly_scores(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算异常分数。
        Returns:
          scores:     [B]    总体 RMSE（Eq. 16），用于检测阈值
          var_scores: [B, C] 各变量 MSE，用于异常诊断
        """
        x_hat = self(x)
        err_sq = (x - x_hat) ** 2                                     
        var_scores = err_sq.mean(dim=1)                             
        scores = torch.sqrt(err_sq.sum(dim=(1, 2)))              
        return scores, var_scores


                                                                              
                               
                                                                              

class FederatedHypernetwork(nn.Module):
    """
    联邦超网络 H(μ)，部署在远程服务端。

    为每个客户端 i 生成个性化参数 θ_i = H(e_i)：
      · e_i: 可学习的客户端嵌入向量（维度 emb_dim）
      · MLP: e_i → hidden → flat θ_i（完整客户端模型参数的展平向量）

    关键属性：
      · 超网络参数 μ 从不传输给客户端
      · 客户端只接收 θ_i，通信开销与 FedAvg 相同
      · μ 可以任意大，不影响通信效率

    Algorithm 1 梯度更新：
      Δθ_i = θ̃_i - θ_i  （本地训练前后的参数差）
      μ ← μ - α·(∇_μ θ_i)^T · (-Δθ_i)
      等价于最小化代理损失：0.5·||θ_i - θ̃_i||²
    """

    def __init__(
        self,
        n_clients: int,
        emb_dim: int,
        hidden_dim: int,
        total_params: int,
    ):
        super().__init__()
        self.n_clients = n_clients
        self.total_params = total_params

                       
        self.embeddings = nn.Embedding(n_clients, emb_dim)
        nn.init.normal_(self.embeddings.weight, std=0.01)

                                                    
        self.body = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, total_params)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.zeros_(self.head.bias)

    def forward(self, client_idx: int) -> torch.Tensor:
        """
        生成客户端 client_idx 的展平参数向量（保留梯度）。
        Returns: [total_params]，requires_grad=True（用于反向传播到 μ）
        """
        e = self.embeddings.weight[client_idx]             
        return self.head(self.body(e))                           

    def generate_params(
        self,
        client_idx: int,
        param_shapes: List[torch.Size],
    ) -> List[torch.Tensor]:
        """
        生成客户端参数列表（形状与目标模型匹配）。
        保留梯度，用于后续超网络更新。
        """
        flat = self.forward(client_idx)                      
        params, offset = [], 0
        for shape in param_shapes:
            n = shape.numel()
            params.append(flat[offset: offset + n].reshape(shape))
            offset += n
        return params


                                                                              
                                 
                                                                              

def _partial_corr(
    X: np.ndarray, i: int, j: int, cond: List[int]
) -> float:
    """
    通过线性回归残差计算偏相关系数 ρ(i,j|cond)。
    """
    if len(cond) == 0:
        r = float(np.corrcoef(X[:, i], X[:, j])[0, 1])
        return float(np.clip(r, -0.9999, 0.9999))

    Z = np.column_stack([np.ones(len(X)), X[:, cond]])                 
    yi, yj = X[:, i], X[:, j]
    try:
        coef_i = np.linalg.lstsq(Z, yi, rcond=None)[0]
        coef_j = np.linalg.lstsq(Z, yj, rcond=None)[0]
        ri = yi - Z @ coef_i
        rj = yj - Z @ coef_j
        r = float(np.corrcoef(ri, rj)[0, 1])
        return float(np.clip(r, -0.9999, 0.9999))
    except Exception:
        return 0.0


def pc_algorithm(
    scores: np.ndarray,
    alpha: float = 0.05,
    max_cond_size: int = 2,
) -> np.ndarray:
    """
    简化 PC 算法：条件独立性检验建立因果骨架，再通过异常幅度定向。

    Args:
        scores: [N, C] 各变量异常分数序列
        alpha : Fisher Z 检验显著性水平
        max_cond_size: 最大条件集大小（控制复杂度）

    Returns:
        dag: [C, C]  dag[i,j]=1 表示 i→j（i 是 j 的因）
    """
    n, C = scores.shape
    if C < 2 or n < 5:
        return np.zeros((C, C))

              
    adj = np.ones((C, C)) - np.eye(C)

                                                      
    for cond_size in range(0, min(max_cond_size + 1, C - 1)):
        for i in range(C):
            nbrs_i = [k for k in range(C) if k != i and adj[i, k] > 0]
            for j in nbrs_i:
                if adj[i, j] == 0:
                    continue

                possible_cond = [k for k in nbrs_i if k != j]
                if len(possible_cond) < cond_size:
                    continue

                                                        
                candidates = possible_cond[:4]
                if cond_size == 0:
                    csets = [[]]
                elif cond_size == 1:
                    csets = [[k] for k in candidates]
                else:
                    csets = [
                        [candidates[a], candidates[b]]
                        for a in range(len(candidates))
                        for b in range(a + 1, len(candidates))
                    ]

                for cset in csets:
                    r = _partial_corr(scores, i, j, cset)
                                 
                    z = 0.5 * math.log((1.0 + abs(r)) / (1.0 - abs(r) + 1e-9))
                    dof = max(0, n - len(cset) - 3)
                    stat = math.sqrt(dof) * abs(z)
                    p_val = 2.0 * (1.0 - scipy.stats.norm.cdf(stat))
                    if p_val > alpha:
                        adj[i, j] = adj[j, i] = 0
                        break          

                                                       
                                        
    mean_scores = scores.mean(axis=0)       
    order = np.argsort(mean_scores)            
    rank = np.empty(C, dtype=int)
    rank[order] = np.arange(C)

    dag = np.zeros((C, C), dtype=float)
    for i in range(C):
        for j in range(C):
            if adj[i, j] > 0 and i != j:
                if rank[i] < rank[j]:
                    dag[i, j] = 1.0             
                elif rank[j] < rank[i]:
                    dag[j, i] = 1.0             

    return dag


def pagerank(
    dag: np.ndarray,
    weights: Optional[np.ndarray] = None,
    damping: float = 0.85,
    max_iter: int = 100,
) -> np.ndarray:
    """
    在因果 DAG 上运行 PageRank 以识别根因。

    排名越高 → 越可能是根本原因（upstream 变量影响更多下游节点）。
    使用反转 DAG 的随机游走（上游节点通过反向边获得更高传播分数）。

    Eq. 17: P_{ij} = w_{ij} / Σ_j w_{ij}  (w_{ij} ≠ 0)

    Args:
        dag    : [C, C] 因果图，dag[i,j]=1 → i 是 j 的因
        weights: [C, C] 可选边权重（默认均为 1）
        damping: 阻尼系数
    Returns:
        pr: [C] PageRank 分数（值越高越可能为根因）
    """
    C = dag.shape[0]
    if C == 0 or dag.sum() == 0:
        return np.ones(C) / max(C, 1)

    W = dag if weights is None else dag * weights

                                         
    rev = W.T.copy()

                  
    col_sum = rev.sum(axis=0)
    col_sum[col_sum == 0] = 1.0
    T = rev / col_sum[np.newaxis, :]

    pr = np.ones(C) / C
    teleport = np.ones(C) / C

    for _ in range(max_iter):
        pr_new = (1.0 - damping) * teleport + damping * T @ pr
        if np.allclose(pr, pr_new, atol=1e-8):
            break
        pr = pr_new

    return pr


                                                                              
                     
                                                                              

class uFedHyDisMTSADD:
    """
    uFedHy-DisMTSADD：面向分布式多变量时序的无监督联邦超网络异常检测与诊断。

    训练流程 (Algorithm 1):
      for R = 1..r:
        采样部分客户端
        for 每个客户端 k:
          θ_k = H(e_k)                       # 超网络生成参数
          θ̃_k ← 本地梯度下降 k 轮            # 本地微调
          Δθ_k = θ̃_k - θ_k
          μ ← μ - α·(∇_μ θ_k)^T·(-Δθ_k)    # 超网络更新（最小化 ||θ_k - θ̃_k||²）

    接口: fit(clients) / predict(clients)
    输出格式与 GANF / FL-STAM 完全兼容（同时附加诊断信息）。

    参数
    ----
    seq_len        : 滑动窗口大小（默认 5，对应论文 Table 3）
    d_model        : Transformer 模型维度
    n_heads        : 多头注意力头数
    n_layers       : Transformer 编码器层数（Depth）
    dropout        : Dropout 率
    emb_dim        : 超网络客户端嵌入维度
    hidden_dim     : 超网络隐藏层维度
    comm_rounds    : 联邦通信轮数
    local_epochs   : 本地训练轮数
    local_lr       : 本地模型学习率
    hn_lr          : 超网络学习率
    batch_size     : 批大小
    client_sample_rate: 每轮通信采样比例
    threshold_mode : 阈值策略 ('ratio' / 'best_f1')
    pc_alpha       : PC 算法显著性水平
    device         : 计算设备
    seed           : 随机种子
    """

    def __init__(
        self,
        seq_len: int = 5,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        emb_dim: int = 100,
        hidden_dim: int = 100,
        comm_rounds: int = 20,
        local_epochs: int = 1,
        local_lr: float = 1e-4,
        hn_lr: float = 1e-3,
        batch_size: int = 128,
        client_sample_rate: float = 1.0,
        threshold_mode: str = "ratio",
        ratio_clip: Tuple[float, float] = (0.01, 0.45),
        pc_alpha: float = 0.05,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.cfg = {k: v for k, v in locals().items() if k != "self"}
        set_seed(seed)
        self.device = torch.device(
            device if device != "cpu" and torch.cuda.is_available() else "cpu"
        )

        self.hypernetwork: Optional[FederatedHypernetwork] = None
        self.client_params: Dict[int, List[torch.Tensor]] = {}
        self.client_thresholds: Dict[int, float] = {}
        self.param_shapes: List[torch.Size] = []
        self.n_features_common: int = 0

                         
        self._model: Optional[SCNorTransformer] = None

                                                          

    def _prep(self, X: np.ndarray) -> np.ndarray:
        """
        [N, T, C, D] 或 [N, T, C] → [N, T, C_common]（float32）
        """
        arr = X.squeeze(-1) if X.ndim == 4 else X
        return arr[:, :, : self.n_features_common].astype(np.float32)

                                                          

    def _compute_scores(
        self,
        X: np.ndarray,
        client_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载客户端参数，批量推理。
        Returns: scores [N], var_scores [N, C]
        """
        self._load_client_params(client_idx)
        self._model.eval()

        X_data = self._prep(X)
        all_sc, all_vsc = [], []
        bs = self.cfg["batch_size"]

        with torch.no_grad():
            for s in range(0, len(X_data), bs):
                xb = torch.tensor(
                    X_data[s: s + bs], device=self.device)
                sc, vsc = self._model.anomaly_scores(xb)
                all_sc.append(sc.cpu().numpy())
                all_vsc.append(vsc.cpu().numpy())

        return (
            np.concatenate(all_sc).astype(np.float32),
            np.concatenate(all_vsc, axis=0).astype(np.float32),
        )

                                                            

    def _pick_threshold(self, scores: np.ndarray, y: np.ndarray) -> float:
        if self.cfg["threshold_mode"] == "best_f1":
            return self._best_f1_thresh(scores, y)
        return self._ratio_thresh(scores, y)

    def _ratio_thresh(self, scores: np.ndarray, y: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return float(np.quantile(scores, 0.95))
        ratio = float(np.clip(np.mean(y), *self.cfg["ratio_clip"]))
        return float(np.quantile(scores, 1.0 - ratio))

    def _best_f1_thresh(self, scores: np.ndarray, y: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return float(np.quantile(scores, 0.95))
        best_tau, best_f1 = float(np.quantile(scores, 0.95)), -1.0
        for q in np.linspace(0.01, 0.99, 99):
            tau = float(np.quantile(scores, q))
            f = f1_score(y, (scores > tau).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_tau = f, tau
        return best_tau

                                                          

    def _get_params(self) -> List[torch.Tensor]:
        return [p.detach().clone() for p in self._model.parameters()]

    def _set_params(self, params: List[torch.Tensor]) -> None:
        with torch.no_grad():
            for p, v in zip(self._model.parameters(), params):
                p.data.copy_(v)

    def _load_client_params(self, k: int) -> None:
        if k in self.client_params:
            self._set_params(self.client_params[k])

                                                             

    def _build_models(self, n_clients: int, n_features: int) -> None:
        cfg = self.cfg

                     
        self._model = SCNorTransformer(
            seq_len=cfg["seq_len"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            dropout=cfg["dropout"],
        ).to(self.device)

        total_params = sum(p.numel() for p in self._model.parameters())
        self.param_shapes = [p.shape for p in self._model.parameters()]

                
        self.hypernetwork = FederatedHypernetwork(
            n_clients=n_clients,
            emb_dim=cfg["emb_dim"],
            hidden_dim=cfg["hidden_dim"],
            total_params=total_params,
        ).to(self.device)

        hn_total = sum(p.numel() for p in self.hypernetwork.parameters())
        print(f"  [uFedHy] SCNorTransformer params: {total_params:,} | "
              f"Hypernetwork params: {hn_total:,}")

                                                            

    def _local_train(
        self,
        X_train: np.ndarray,
        theta_init: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], float]:
        """
        从 theta_init 出发，执行 local_epochs 轮本地梯度下降。
        Returns: (updated_params, avg_loss)
        """
        cfg = self.cfg
        self._set_params(theta_init)
        self._model.train()

        X_data = self._prep(X_train)
        N = len(X_data)
        opt = torch.optim.Adam(self._model.parameters(), lr=cfg["local_lr"])
        total_loss, n_batches = 0.0, 0

        for _ in range(cfg["local_epochs"]):
            perm = np.random.permutation(N)
            for bi in range(0, N, cfg["batch_size"]):
                idx = perm[bi: bi + cfg["batch_size"]]
                xb = torch.tensor(X_data[idx], device=self.device)
                loss = self._model.reconstruction_loss(xb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                n_batches += 1

        return self._get_params(), total_loss / max(n_batches, 1)

                                                           

    def fit(self, clients: List[dict], round_callback=None) -> None:
        """
        联邦超网络训练（Algorithm 1）。

        Args:
            clients: 每个元素包含 'X_train', 'X_cal', 'y_cal', ('X_test', 'y_test')
        """
        cfg = self.cfg
        n_clients = len(clients)
        self.n_features_common = min(c["X_train"].shape[2] for c in clients)

                                      
        actual_seq_len = clients[0]["X_train"].shape[1]
        if cfg["seq_len"] != actual_seq_len:
            print(f"  [uFedHy] seq_len auto-adjusted: {cfg['seq_len']} -> {actual_seq_len}")
            cfg["seq_len"] = actual_seq_len

        self._build_models(n_clients, self.n_features_common)
        print(f"  [uFedHy] n_clients={n_clients}  "
              f"features={self.n_features_common}  "
              f"rounds={cfg['comm_rounds']}  "
              f"local_epochs={cfg['local_epochs']}  "
              f"device={self.device}")

        hn_opt = torch.optim.Adam(
            self.hypernetwork.parameters(), lr=cfg["hn_lr"])

                                                                
        for rnd in range(cfg["comm_rounds"]):
            n_sample = max(1, int(n_clients * cfg["client_sample_rate"]))
            sampled = np.random.choice(n_clients, n_sample, replace=False)

            hn_opt.zero_grad()
            round_loss = 0.0

            for k in sampled:
                                                
                theta_tensors = self.hypernetwork.generate_params(
                    int(k), self.param_shapes)
                theta_flat = torch.cat([t.flatten() for t in theta_tensors])

                                    
                theta_init = [t.detach().clone() for t in theta_tensors]
                updated_params, loss = self._local_train(
                    clients[k]["X_train"], theta_init)
                round_loss += loss

                                           
                updated_flat = torch.cat([
                    p.to(self.device).flatten() for p in updated_params])
                delta_flat = updated_flat - theta_flat.detach()

                               
                                          
                                              
                                                   
                                                     
                surrogate = (theta_flat * (-delta_flat.detach())).sum()
                (surrogate / n_sample).backward()

            nn.utils.clip_grad_norm_(self.hypernetwork.parameters(), 1.0)
            hn_opt.step()

            avg_loss = round_loss / max(len(sampled), 1)
            print(f"    Round {rnd + 1:3d}/{cfg['comm_rounds']}: "
                  f"loss={avg_loss:.4f}")
            if round_callback is not None:
                self._finalize_params(n_clients)
                round_callback(rnd + 1, self, clients)

                        
        self._finalize_params(n_clients)
        self._calibrate(clients)

    def _finalize_params(self, n_clients: int) -> None:
        """将超网络为每个客户端生成的最终参数存入 client_params。"""
        self.hypernetwork.eval()
        with torch.no_grad():
            for k in range(n_clients):
                theta = self.hypernetwork.generate_params(k, self.param_shapes)
                self.client_params[k] = [t.detach().clone() for t in theta]

    def _calibrate(self, clients: List[dict]) -> None:
        """在校准集上确定每个客户端的检测阈值。"""
        for k, c in enumerate(clients):
            scores, _ = self._compute_scores(c["X_cal"], k)
            self.client_thresholds[k] = self._pick_threshold(scores, c["y_cal"])

                                                            

    def _diagnose(
        self, var_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        PC 算法 → DAG → PageRank 根因排序。

        Args:
            var_scores: [N, C] 各变量异常分数
        Returns:
            dag: [C, C], pr: [C], rank_list: 降序根因变量索引
        """
        C = var_scores.shape[1] if var_scores.ndim == 2 else 1
        if var_scores.ndim < 2 or C < 2 or len(var_scores) < 5:
            pr = np.ones(C) / max(C, 1)
            return np.zeros((C, C)), pr, list(np.argsort(-pr))

        try:
            dag = pc_algorithm(
                var_scores, alpha=self.cfg["pc_alpha"], max_cond_size=2)
            pr = pagerank(dag)
        except Exception:
            pr = var_scores.mean(axis=0)
            pr = pr / (pr.sum() + 1e-8)
            dag = np.zeros((C, C))

        return dag, pr, list(np.argsort(-pr))

                                                              

    def predict(self, clients: List[dict]) -> List[dict]:
        """
        对所有客户端执行推理，返回与 GANF/FL-STAM 兼容的结果格式。
        同时附加异常诊断信息（dag, pagerank_scores, rank_list）。
        """
        results = []

        for k, c in enumerate(clients):
            scores, var_scores = self._compute_scores(c["X_test"], k)

            tau_raw = float(
                self.client_thresholds.get(k, float(np.quantile(scores, 0.95))))
            y_pred = (scores > tau_raw).astype(np.int64)

                         
            s_min, s_max = float(scores.min()), float(scores.max())
            denom = s_max - s_min + 1e-8
            norm_score = (scores - s_min) / denom
            tau_norm = (tau_raw - s_min) / denom

                                      
            anom_idx = np.where(y_pred.astype(bool))[0]
            diag_scores = (var_scores[anom_idx]
                           if len(anom_idx) > 4 else var_scores)
            dag, pr, rank_list = self._diagnose(diag_scores)

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
                        
                "dag": dag,
                "pagerank_scores": pr.astype(np.float32),
                "rank_list": rank_list,
            })

        return results
