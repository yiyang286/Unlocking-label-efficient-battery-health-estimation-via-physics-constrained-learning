import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------- 核心组件：多头回归器（精简版） ----------------------
class MultiHeadRegressor(nn.Module):
    """极简多头回归器，固定输出3个连续特征，适配EIS数据任务"""

    def __init__(self, input_dim, hidden_dim=32, n_heads=3):
        super().__init__()
        # 共享隐藏层 + 多头输出（去除冗余维度，降低参数量）
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # 多头独立输出（每个头输出1个值）
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_heads)])

    def forward(self, x):
        feat = self.shared(x)
        # 拼接多头输出 [B, 3]
        return torch.cat([head(feat) for head in self.heads], dim=-1)


class MTLEIS(nn.Module):
    def __init__(
            self,
            in_channels=3,  # 输入通道：EIS图像3通道
            latent_dim=128,  # 特征嵌入维度
            lambda_unlabel=0.5,
            dropout_rate=0.1  # 防过拟合，小数据集必备
    ):
        super().__init__()
        self.lambda_unlabel = lambda_unlabel

        # ===================== 1. 轻量化特征提取器（适配8×8小图像） =====================
        # 输入：[B, 3, 8, 8] → 输出：[B, 128]
        self.feature_extractor = nn.Sequential(
            # 8×8 → 4×4
            nn.Conv2d(in_channels, 32, 3, padding=1),  # 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 2x2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 输出 [B, 128, 1, 1]
            nn.Flatten()  # 展平 [B, 128]
        )

        # ===================== 2. 编码器（删除冗余层，极简设计） =====================
        # 128 →  latent_dim（原模型冗余128→128层直接删除，无性能损失）
        self.encoder = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.Dropout(dropout_rate)  # 新增：防过拟合
        )

        # ===================== 3. 多任务头 =====================
        self.feature_head = MultiHeadRegressor(input_dim=latent_dim, hidden_dim=32)

        # ===================== 4. SOH最终回归器 =====================
        self.soh_regressor = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1)
        )

    def forward(self, x, soh_labels=None, labeled_mask=None, feat_labels=None):
        """
        前向传播：精简逻辑，维度严格对齐
        Args:
            x: 输入图像 [B, 3, 8, 8]
            soh_labels: SOH标签 [B]
            labeled_mask: 有标签样本掩码 [B]
            feat_labels: 3维特征标签 [B, 3]
        Returns:
            soh_pred: SOH预测值 [B]
            feat_pred: 3维特征预测值 [B, 3]
            latent: 嵌入特征 [B, latent_dim]
            total_loss: 总损失
        """
        # 1. 提取图像特征
        feat_map = self.feature_extractor(x)

        # 2. 编码为低维嵌入
        latent = self.encoder(feat_map)

        # 3. 多头预测3个物理特征
        feat_pred = self.feature_head(latent)

        # 5. SOH回归预测
        soh_pred = self.soh_regressor(latent).squeeze(-1)

        # ===================== 多任务损失计算 =====================
        total_loss = torch.tensor(0.0, device=x.device)
        soh_loss = torch.tensor(0.0, device=x.device)
        feat_loss = torch.tensor(0.0, device=x.device)

        # 有标签数据：SOH损失
        if labeled_mask is not None and soh_labels is not None and labeled_mask.sum() > 0:
            soh_loss = F.mse_loss(soh_pred[labeled_mask], soh_labels[labeled_mask])
            total_loss = total_loss + soh_loss

        # 全量数据：特征回归损失（支持无标签半监督）
        if feat_labels is not None:
            feat_loss = F.mse_loss(feat_pred, feat_labels)
            total_loss = total_loss + self.lambda_unlabel * feat_loss

        return soh_pred, feat_pred, latent, total_loss
