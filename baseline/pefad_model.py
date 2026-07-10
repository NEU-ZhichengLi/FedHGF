"""
PeFAD: A Parameter-Efficient Federated Framework for Time Series Anomaly Detection
====================================================================================
Xu et al., KDD 2024

本实现完整复现论文的算法设计，**以轻量 Transformer Encoder 替代 GPT-2** 作为 PLM body，
避免 ~500MB 权重下载、保留参数高效联邦微调的核心思想（冻结 body 前 n 层，只微调输入/
输出层 + body 后 m 层；FedAvg 只聚合可训练参数）。核心算法（ADMS、PPDS、KD、参数高
效联邦）全部按论文公式实现。

核心组件
--------
1. Patching — 非重叠分块 [B, T, F] → [B, n_patches, patch_len*F]
2. ADMS (Anomaly-Driven Mask Selection，4.1.1 节)：
     ── intra-patch: SSA 分解 → 残差分量 → R_i (异常=残差能量大)
     ── inter-patch: cos(A_i, A_{i-1}) → C_i' (异常=与前一段不相似)
     ── 融合:         Score_i = β·R_i + (1-β)·C_i'
     ── 掩码采样:     用 softmax(Score/τ) 作为权重，按 mask_ratio 抽取
3. PLM-based Local Training (4.1 节)：
     ── Input Embed (线性) → TransformerBlock ×L (PLM body) → Output Proj (线性)
     ── freeze body 前 n_frozen 层，微调输入/输出层 + body 后 n_trainable 层
     ── 掩码位置填入可学习 mask_token；输出为重构序列
4. PPDS (4.1.2 节, Privacy-Preserving Dataset Synthesis)：
     ── 每 client 本地训练一个 SmallVAE 合成 T_{s,i}
     ── 损失: L_vae + α₁·W(T_i, T_{s,i}) + α₂·I(T_i, T_{s,i})
        · W: 一维 Wasserstein-1（排序后的绝对差）近似
        · I: 用 1 - (1-ρ²) 的变换作为 MI 上界代理（Pearson 的高斯 MI 关系）
5. Knowledge Distillation (式 13)：
     ── 客户端在 D_sh 上用本地 F(θ_i) 的表征与 F(θ_g) 的表征做 MSE 一致性
     ── 总损失 L = MSE(X_hat, X) + λ·||F(θ_i, D_sh) - F(θ_g, D_sh)||
6. Parameter-Efficient FedAvg (4.2 节)：
     ── 只聚合 requires_grad=True 的参数，冻结参数保持初始化不变且不上传

数据流
------
输入 X [B, T, n_k, 1]
  → squeeze 到 [B, T, n_k]
  → patching [B, n_patch, patch_len, n_k]
  → flatten → [B, n_patch, patch_len*n_k]
  → ADMS 计算 Score_i，按权重抽取需要 mask 的 patch
  → input_embed → [B, n_patch, d_model]
  → 加 pos_embed；mask 位置替换为 mask_token
  → TransformerBlocks ×L
  → output_proj → [B, n_patch, patch_len*n_k]
  → unpatch → X_hat [B, T, n_k]

异常分数 = 窗口 MSE 重构误差（与 FedAnomaly 一致，方便对比）
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


                                                                              
                
                                                                              

def patch_sequence(X: torch.Tensor, patch_len: int
                   ) -> Tuple[torch.Tensor, int]:
    """
    X: [B, T, F]
    返回: patches [B, n_patches, patch_len * F], n_patches

    如果 T 不能被 patch_len 整除，从末尾截掉多余步（不 padding，避免干扰 recon）。
    """
    B, T, Fd = X.shape
    n_patches = T // patch_len
    if n_patches == 0:
        raise ValueError(f"patch_len={patch_len} > T={T}")
    T_use = n_patches * patch_len
    Xc = X[:, :T_use, :]                                      
    patches = Xc.reshape(B, n_patches, patch_len, Fd)                   
    patches_flat = patches.reshape(B, n_patches, patch_len * Fd)
    return patches_flat, n_patches


def unpatch_sequence(patches_flat: torch.Tensor, patch_len: int,
                     n_features: int) -> torch.Tensor:
    """
    patches_flat: [B, n_patches, patch_len * F]
    返回: X [B, n_patches * patch_len, F]
    """
    B, n_p, _ = patches_flat.shape
    X = patches_flat.reshape(B, n_p, patch_len, n_features)
    return X.reshape(B, n_p * patch_len, n_features)


                                                                              
                                         
                                                                              

def _intra_patch_residual(patch_mat: torch.Tensor, keep_top: int = 1
                          ) -> torch.Tensor:
    """
    对每个 patch 做 SVD，保留前 keep_top 个奇异分量作为主结构，
    剩余作为残差；返回残差能量 (一个标量 per patch per sample)。

    论文 Eq. (4)-(5):
      P_i = Σ σ_k u_k v_k^T
      R_i = mean(Σ_{k ∈ residual} σ_k u_k v_k^T)

    patch_mat: [B, n_patches, patch_len, F]   (未 flatten 的 patch 矩阵)
    返回:       [B, n_patches]                 (残差能量标量)

    说明：原论文通过 Hankel + SSA 实现；这里直接对 patch 矩阵做 SVD，概念一致且
         对小 patch_len（如 4）数值更稳定（Hankel 太扁会失去 SVD 自由度）。
    """
    B, n_p, pl, Fd = patch_mat.shape
    flat = patch_mat.reshape(B * n_p, pl, Fd)                         
                                                                          
    try:
        U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
    except Exception:
                                                   
        var = flat.var(dim=(1, 2))                               
        return var.reshape(B, n_p)

    k = min(keep_top, S.shape[1])
                                                        
    S_keep = S[:, :k]
                 
    main = U[:, :, :k] @ torch.diag_embed(S_keep) @ Vh[:, :k, :]
    residual = flat - main                                             
    r_score = residual.pow(2).mean(dim=(1, 2))                 
    return r_score.reshape(B, n_p)


def _inter_patch_dissim(patches_flat: torch.Tensor) -> torch.Tensor:
    """
    相邻 patch 的余弦不相似度 (1 - cos_sim)：
      C_i' = 1 - cos(A_i, A_{i-1})

    patches_flat: [B, n_patches, D]
    返回:          [B, n_patches]  (第 0 个 patch 的不相似度设为 batch 均值)
    """
    B, n_p, D = patches_flat.shape
    if n_p < 2:
        return torch.zeros(B, n_p, device=patches_flat.device)
                                             
    a = patches_flat[:, 1:, :]
    b = patches_flat[:, :-1, :]
    cos = F.cosine_similarity(a, b, dim=-1)                         
    dissim = 1.0 - cos                                          
                       
    head = dissim.mean(dim=1, keepdim=True)                     
    out = torch.cat([head, dissim], dim=1)                         
    return out


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """沿 n_patches 维度做 min-max 归一化到 [0, 1]（每 batch 独立）。"""
    s_min = scores.amin(dim=-1, keepdim=True)
    s_max = scores.amax(dim=-1, keepdim=True)
    denom = (s_max - s_min).clamp(min=1e-8)
    return (scores - s_min) / denom


def compute_adms_scores(patch_mat: torch.Tensor, patches_flat: torch.Tensor,
                        beta: float = 0.5, keep_top: int = 1) -> torch.Tensor:
    """
    论文 Eq. (7): Score_i = β·R_i + (1-β)·C_i'
    返回 [B, n_patches] 非负分数（归一化到 [0, 1]）
    """
    R = _intra_patch_residual(patch_mat, keep_top=keep_top)             
    C = _inter_patch_dissim(patches_flat)                               
    R = _normalize_scores(R)
    C = _normalize_scores(C)
    return beta * R + (1.0 - beta) * C


def sample_mask_indices(scores: torch.Tensor, mask_ratio: float,
                        temperature: float = 1.0) -> torch.Tensor:
    """
    按 softmax(score/τ) 的权重抽取 `round(n_patches * mask_ratio)` 个 patch 索引。

    scores: [B, n_patches]
    返回:    mask_bool [B, n_patches]  — True 表示该 patch 被 mask
    """
    B, n_p = scores.shape
    n_mask = max(1, int(round(n_p * mask_ratio)))
                           
    logits = scores / max(temperature, 1e-6)
                                    
    gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-10))
                        .clamp_min(1e-10))
    scored = logits + gumbel
    _, idx = scored.topk(n_mask, dim=-1)                                    
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


                                                                              
                                     
                                                                              

class SmallVAE(nn.Module):
    """
    小型 1D 卷积 VAE，输入 [B, T, F] → 合成样本 [B, T, F]。
    设计目标：轻量、能做端到端的 I 与 W 约束训练。
    """

    def __init__(self, n_features: int, window_len: int,
                 latent_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.n_features = n_features
        self.window_len = window_len
        self.latent_dim = latent_dim

                                
        self.enc = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(hidden * window_len, latent_dim)
        self.fc_logvar = nn.Linear(hidden * window_len, latent_dim)

                 
        self.fc_dec = nn.Linear(latent_dim, hidden * window_len)
        self.dec = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, n_features, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
                                  
        h = self.enc(x.transpose(1, 2))                               
        h_flat = h.reshape(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).reshape(-1, self.enc[2].out_channels,
                                    self.window_len)
        x = self.dec(h).transpose(1, 2)                               
        return x

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """直接从 N(0, I) 采 z 解码出合成样本 T_{s,i}。"""
        z = torch.randn(n, self.latent_dim, device=device)
        with torch.no_grad():
            x = self.decode(z)
        return x


def wasserstein1_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    一维 Wasserstein-1 距离（论文 Eq. 9）的近似实现：
    把两个样本展平并排序后，取元素级绝对差的均值。
    """
    xf = x.reshape(-1)
    yf = y.reshape(-1)
    n = min(xf.numel(), yf.numel())
    if n == 0:
        return torch.zeros((), device=x.device)
    xs, _ = xf[:n].sort()
    ys, _ = yf[:n].sort()
    return (xs - ys).abs().mean()


def mi_pearson_proxy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    MI 代理项（越小越好）：对每个维度计算 Pearson²，然后取均值。
    动机：对高斯情形 MI = -½ log(1-ρ²)，与 ρ² 单调递增；直接用 ρ² 作为
    最小化目标，计算稳定且无需额外 critic 网络。

    x, y: [B, T, F]  — 要求 shape 一致
    返回:  标量
    """
    if x.shape != y.shape:
        return torch.zeros((), device=x.device)
    B, T, Fd = x.shape
    x_flat = x.reshape(B, T * Fd)
    y_flat = y.reshape(B, T * Fd)
                
    x_m = x_flat - x_flat.mean(dim=0, keepdim=True)
    y_m = y_flat - y_flat.mean(dim=0, keepdim=True)
    num = (x_m * y_m).mean(dim=0)
    den = (x_m.pow(2).mean(dim=0).clamp(min=1e-8).sqrt()
           * y_m.pow(2).mean(dim=0).clamp(min=1e-8).sqrt())
    rho = (num / den).clamp(-1.0, 1.0)
    return rho.pow(2).mean()


def ppds_loss(x: torch.Tensor, x_hat: torch.Tensor,
              mu: torch.Tensor, logvar: torch.Tensor,
              x_synth: torch.Tensor,
              alpha_w: float = 1.0, alpha_mi: float = 0.1
              ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    PPDS 总损失（论文 Eq. 10）：
      L = L_vae + α₁·W(T_i, T_{s,i}) + α₂·I(T_i, T_{s,i})
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1.0)
    vae = recon + kl
    w = wasserstein1_1d(x, x_synth)
    mi = mi_pearson_proxy(x, x_synth)
    total = vae + alpha_w * w + alpha_mi * mi
    return total, {
        "vae": float(vae.item()),
        "recon": float(recon.item()),
        "kl": float(kl.item()),
        "w": float(w.item()),
        "mi": float(mi.item()),
    }


                                                                              
                                              
                                                                              

class TransformerBlock(nn.Module):
    """
    GPT-2 风格的 pre-norm Transformer 块（因果注意力）。
    论文里 GPT-2 也是 decoder-only + 因果，这里保持一致。
    """

    def __init__(self, d_model: int, n_heads: int = 8,
                 d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                                     
        h = self.ln1(x)
        L = h.size(1)
                 
        mask = torch.triu(torch.ones(L, L, device=h.device, dtype=torch.bool),
                          diagonal=1)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class PLMBody(nn.Module):
    """
    堆叠 Transformer 块作为 PLM body。

    参数冻结策略（论文 4.2 节）：
      · 前 n_frozen 层参数 requires_grad = False（代表 GPT-2 的通用知识）
      · 后 n_trainable 层 requires_grad = True （下游任务微调）
    """

    def __init__(self, d_model: int, n_layers: int = 6, n_heads: int = 8,
                 d_ff: int = 512, dropout: float = 0.1,
                 n_frozen: int = 4, n_trainable: int = 2):
        super().__init__()
        assert n_frozen + n_trainable == n_layers, (
            f"n_frozen ({n_frozen}) + n_trainable ({n_trainable}) "
            f"must equal n_layers ({n_layers})"
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
                                                               
                                                       
        for i, blk in enumerate(self.blocks):
            if i < n_frozen:
                for p in blk.parameters():
                    p.requires_grad = False

        self.n_frozen = n_frozen
        self.n_trainable = n_trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


                                                                              
               
                                                                              

class PeFADNet(nn.Module):
    """
    PeFAD 整网：input_embed + PLM_body + output_proj + mask_token。

    数据流：
      X [B, T, F]
        → patching [B, n_patches, patch_len*F]
        → input_embed (linear) → [B, n_patches, d_model]
        → mask 位置替换为 mask_token + pos_embed
        → PLM body (Transformer stack, 部分冻结)
        → output_proj (linear) → [B, n_patches, patch_len*F]
        → unpatch → X_hat [B, T_use, F]
    """

    def __init__(self, n_features: int, window_len: int, patch_len: int,
                 d_model: int = 128, n_layers: int = 6, n_heads: int = 8,
                 d_ff: int = 512, dropout: float = 0.1,
                 n_frozen: int = 4, n_trainable: int = 2,
                 max_pos: int = 64):
        super().__init__()
        self.n_features = n_features
        self.window_len = window_len
        self.patch_len = patch_len
        self.n_patches = window_len // patch_len
        self.d_model = d_model

        patch_dim = patch_len * n_features
                                                           
        self.input_embed = nn.Linear(patch_dim, d_model)
        self.output_proj = nn.Linear(d_model, patch_dim)

                                                        
        self.pos_embed = nn.Parameter(torch.zeros(1, max_pos, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

                                       
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.plm_body = PLMBody(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_ff, dropout=dropout,
            n_frozen=n_frozen, n_trainable=n_trainable,
        )

                                                    

    def embed_patches(self, patches_flat: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        patches_flat: [B, n_p, patch_len*F]
        mask:         [B, n_p] bool, True=被 mask
        返回:          [B, n_p, d_model]  (已加 pos_embed 与 mask_token)
        """
        h = self.input_embed(patches_flat)                              
        B, n_p, _ = h.shape
        if mask is not None:
            m = mask.unsqueeze(-1).to(h.dtype)
            tok = self.mask_token.expand(B, n_p, -1)
            h = h * (1.0 - m) + tok * m
        h = h + self.pos_embed[:, :n_p, :]
        return h

    def features(self, h: torch.Tensor) -> torch.Tensor:
        """PLM body 的输出表征，用于 KD 一致性损失。"""
        return self.plm_body(h)

    def reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        """从 PLM body 的表征映射回 patch 空间。"""
        return self.output_proj(h)

    def forward(self, X: torch.Tensor,
                mask_ratio: float = 0.0,
                adms_beta: float = 0.5,
                adms_temp: float = 1.0,
                return_features: bool = False
                ) -> Dict[str, torch.Tensor]:
        """
        X: [B, T, F]

        若 mask_ratio > 0 → 训练模式：用 ADMS 选 mask patches
        若 mask_ratio == 0 → 推理模式：无掩码，直接重构

        返回 dict:
          X_hat: [B, T_use, F]    重构序列
          features: [B, n_p, d]   PLM body 输出（return_features=True 时）
          mask:   [B, n_p] bool   本次使用的 mask（用于加权损失）
        """
        patches_flat, n_p = patch_sequence(X, self.patch_len)
                              
        patch_mat = patches_flat.reshape(
            patches_flat.size(0), n_p, self.patch_len, self.n_features)

        mask = None
        if mask_ratio > 0.0:
            with torch.no_grad():
                scores = compute_adms_scores(
                    patch_mat, patches_flat, beta=adms_beta)
                mask = sample_mask_indices(
                    scores, mask_ratio, temperature=adms_temp)

        h = self.embed_patches(patches_flat, mask=mask)
        feats = self.features(h)
        recon_flat = self.reconstruct(feats)                               
        X_hat = unpatch_sequence(recon_flat, self.patch_len, self.n_features)

        out: Dict[str, torch.Tensor] = {"X_hat": X_hat}
        if mask is not None:
            out["mask"] = mask
        if return_features:
            out["features"] = feats
        return out


def trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """返回所有 requires_grad=True 的参数（用于参数高效 FedAvg）。"""
    return [p for p in model.parameters() if p.requires_grad]


def trainable_named(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


                                                                              
                       
                                                                              

class PeFAD:
    """
    PeFAD 完整方案。

    接口:
      fit(clients)     — 联邦训练 (PPDS + FedAvg on trainable only + KD)
      predict(clients) — 推理，返回与 FedGAD / FL-STAM 一致的结果字典

    数据兼容:
      输入 [N, T, n_k, 1] → squeeze 到 [N, T, n_k]
      异构客户端 n_k 对齐策略：**零填充到全局 max n_k**（与 FL-STAM 一致）

    重要说明:
      本实现以 Transformer Encoder 替代 GPT-2；模型宏观结构、冻结策略、联邦聚合逻辑
      （仅聚合 trainable 参数）、ADMS、PPDS、KD 一致性损失均按论文公式复现。
    """

    def __init__(
        self,
                  
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        n_frozen: int = 4,
        n_trainable: int = 2,
        patch_len: int = 4,
                    
        adms_beta: float = 0.5,
        mask_ratio: float = 0.25,
        adms_temperature: float = 0.5,
                    
        enable_ppds: bool = True,
        ppds_latent_dim: int = 16,
        ppds_hidden: int = 32,
        ppds_epochs: int = 20,
        ppds_lr: float = 1e-3,
        ppds_batch_size: int = 64,
        ppds_synth_size: int = 128,
        alpha_w: float = 1.0,
        alpha_mi: float = 0.1,
                  
        lambda_kd: float = 1.0,
                  
        n_rounds: int = 10,
        local_epochs: int = 2,
        lr: float = 1e-3,
        batch_size: int = 32,
                  
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
        self.model: Optional[PeFADNet] = None
        self.client_thresholds: Dict[int, float] = {}
                 
        self.D_sh: Optional[torch.Tensor] = None                   

                                                        

    def _squeeze_and_pad(self, X: np.ndarray) -> np.ndarray:
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
        """异常分数 = 窗口 MSE 重构误差（仅在有效特征上）。"""
        self.model.eval()
        X_3d = self._squeeze_and_pad(X)
        n_k_valid = (self.n_features_per_client[k]
                     if hasattr(self, "n_features_per_client") else X_3d.shape[2])
                                                            
        pl = self.cfg["patch_len"]
        T_use = (X_3d.shape[1] // pl) * pl
        scores = []
        bs = self.cfg["batch_size"]
        with torch.no_grad():
            for s in range(0, len(X_3d), bs):
                xb = self._to_device(X_3d[s:s + bs])
                out = self.model(xb, mask_ratio=0.0)                 
                X_hat = out["X_hat"]
                recon_err = (
                    xb[:, :T_use, :n_k_valid] - X_hat[:, :T_use, :n_k_valid]
                ).pow(2).mean(dim=(1, 2))
                scores.append(recon_err.cpu().numpy())
        return np.concatenate(scores).astype(np.float32)

                                                       

    def _pick_threshold(self, scores: np.ndarray, y: np.ndarray) -> float:
        mode = self.cfg["threshold_mode"]
        if mode == "best_f1":
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

                                                         

    def _run_ppds(self, clients: List[dict]) -> torch.Tensor:
        """
        每 client 在本地训练 SmallVAE，生成合成样本，最后拼接成共享数据集 D_sh。
        返回 D_sh ∈ torch.Tensor [sum_i synth_size, T, F]
        """
        cfg = self.cfg
        all_synth: List[torch.Tensor] = []
        pl_train = self.window_len
        Fd = self.n_features_common

        for k, c in enumerate(clients):
            X = self._squeeze_and_pad(c["X_train"])
            n_k_valid = self.n_features_per_client[k]
                                                  
            vae = SmallVAE(
                n_features=Fd,
                window_len=pl_train,
                latent_dim=cfg["ppds_latent_dim"],
                hidden=cfg["ppds_hidden"],
            ).to(self.device)
            opt = torch.optim.Adam(vae.parameters(), lr=cfg["ppds_lr"])
            N = len(X)
            losses = []
            for ep in range(cfg["ppds_epochs"]):
                perm = np.random.permutation(N)
                for bi in range(0, N, cfg["ppds_batch_size"]):
                    idx = perm[bi:bi + cfg["ppds_batch_size"]]
                    if len(idx) < 4:
                        continue
                    xb = self._to_device(X[idx])
                    x_hat, mu, logvar = vae(xb)
                                                
                    with torch.no_grad():
                        x_synth_const = vae.sample(
                            xb.size(0), self.device)
                    loss, info = ppds_loss(
                        xb, x_hat, mu, logvar, x_synth_const,
                        alpha_w=cfg["alpha_w"], alpha_mi=cfg["alpha_mi"],
                    )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
                    opt.step()
                    losses.append(info)
            if losses:
                last = losses[-1]
                print(f"    [PPDS client {k}] VAE={last['vae']:.4f}  "
                      f"W={last['w']:.4f}  MI={last['mi']:.4f}")
                    
            synth = vae.sample(cfg["ppds_synth_size"], self.device)
            all_synth.append(synth.detach().cpu())
            del vae

        D_sh = torch.cat(all_synth, dim=0)                                  
        print(f"  [PPDS] Built shared dataset D_sh: {tuple(D_sh.shape)}")
        return D_sh.to(self.device)

                                                          

    def fit(self, clients: List[dict], round_callback=None) -> None:
        cfg = self.cfg
            
        def _nk(c):
            s = c["X_train"]
            return s.shape[2] if s.ndim in (3, 4) else 1
        self.n_features_per_client = [_nk(c) for c in clients]
        self.n_features_common = max(self.n_features_per_client)
        n_features = self.n_features_common
        self.window_len = (clients[0]["X_train"].shape[1]
                           if clients[0]["X_train"].ndim >= 3 else 1)

                                       
        pl = cfg["patch_len"]
        while pl > 1 and self.window_len % pl != 0:
            pl -= 1
        if pl != cfg["patch_len"]:
            print(f"  [PeFAD] patch_len adjusted {cfg['patch_len']} → {pl} "
                  f"(must divide window_len={self.window_len})")
            cfg["patch_len"] = pl

        n_patches = self.window_len // pl
        if n_patches < 2:
            raise ValueError(
                f"n_patches={n_patches} < 2, PeFAD cannot apply ADMS. "
                f"Reduce patch_len or use longer window.")

              
        self.model = PeFADNet(
            n_features=n_features,
            window_len=self.window_len,
            patch_len=pl,
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            dropout=cfg["dropout"],
            n_frozen=cfg["n_frozen"],
            n_trainable=cfg["n_trainable"],
            max_pos=max(64, n_patches),
        ).to(self.device)

        n_all = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(c["X_train"].shape[0] for c in clients)
        weights = [c["X_train"].shape[0] / total for c in clients]

        print(f"  [PeFAD] {len(clients)} clients, features="
              f"{self.n_features_per_client} (max={n_features}, pad-to-max), "
              f"win={self.window_len}, patch_len={pl}, n_patches={n_patches}")
        print(f"  [PeFAD] d_model={cfg['d_model']}, layers="
              f"{cfg['n_frozen']}f+{cfg['n_trainable']}t, "
              f"params total={n_all:,}, trainable={n_train:,} "
              f"({100.*n_train/max(n_all,1):.1f}%)")

                               
        if cfg["enable_ppds"]:
            self.D_sh = self._run_ppds(clients)
        else:
            self.D_sh = None
            print("  [PPDS] Disabled, KD consistency loss will be skipped.")

                        
        for r in range(cfg["n_rounds"]):
                                                             
            global_state = {n: p.detach().clone()
                            for n, p in self.model.named_parameters()}
                                                 
            trainable_names = [n for n, p in self.model.named_parameters()
                               if p.requires_grad]
            agg_delta = {n: torch.zeros_like(global_state[n])
                         for n in trainable_names}
            round_loss, round_recon, round_kd = 0.0, 0.0, 0.0

            for k, c in enumerate(clients):
                local_model = copy.deepcopy(self.model).to(self.device)
                local_model.train()
                                  
                opt = torch.optim.Adam(
                    [p for p in local_model.parameters() if p.requires_grad],
                    lr=cfg["lr"],
                )

                X_tr = self._squeeze_and_pad(c["X_train"])
                n_k_valid = self.n_features_per_client[k]
                N_k = len(X_tr)
                T_use = (self.window_len // pl) * pl

                c_loss, c_recon, c_kd, nb = 0.0, 0.0, 0.0, 0
                for _ in range(cfg["local_epochs"]):
                    perm = np.random.permutation(N_k)
                    for bi in range(0, N_k, cfg["batch_size"]):
                        idx = perm[bi:bi + cfg["batch_size"]]
                        if len(idx) < 2:
                            continue
                        xb = self._to_device(X_tr[idx])
                                              
                        out = local_model(
                            xb, mask_ratio=cfg["mask_ratio"],
                            adms_beta=cfg["adms_beta"],
                            adms_temp=cfg["adms_temperature"],
                            return_features=True,
                        )
                        X_hat = out["X_hat"]
                        recon = F.mse_loss(
                            X_hat[:, :T_use, :n_k_valid],
                            xb[:, :T_use, :n_k_valid],
                        )
                                                             
                        kd = torch.zeros((), device=self.device)
                        if self.D_sh is not None and cfg["lambda_kd"] > 0:
                            n_shared = min(xb.size(0) * 2, self.D_sh.size(0))
                            sh_idx = torch.randperm(
                                self.D_sh.size(0))[:n_shared]
                            xb_sh = self.D_sh[sh_idx]
                            feats_local = local_model.features(
                                local_model.embed_patches(
                                    patch_sequence(xb_sh, pl)[0]))
                            with torch.no_grad():
                                feats_global = self.model.features(
                                    self.model.embed_patches(
                                        patch_sequence(xb_sh, pl)[0]))
                            kd = F.mse_loss(feats_local, feats_global)

                        loss = recon + cfg["lambda_kd"] * kd
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in local_model.parameters()
                             if p.requires_grad], max_norm=5.0)
                        opt.step()
                        c_loss += loss.item()
                        c_recon += recon.item()
                        c_kd += kd.item() if kd.dim() == 0 else 0.0
                        nb += 1

                round_loss += weights[k] * c_loss / max(nb, 1)
                round_recon += weights[k] * c_recon / max(nb, 1)
                round_kd += weights[k] * c_kd / max(nb, 1)

                                    
                local_state = dict(local_model.named_parameters())
                for n in trainable_names:
                    agg_delta[n].add_(
                        weights[k] * (local_state[n].detach() - global_state[n])
                    )

                                           
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        p.add_(agg_delta[n])
                                                      

            print(f"    Round {r + 1}/{cfg['n_rounds']}: "
                  f"loss={round_loss:.4f}  recon={round_recon:.4f}  "
                  f"KD={round_kd:.4f}")
            if round_callback is not None:
                round_callback(r + 1, self, clients)

                  
        self._calibrate(clients)

    def _calibrate(self, clients: List[dict]) -> None:
        for k, c in enumerate(clients):
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
