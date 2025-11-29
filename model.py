import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Adaptive1DCNN(nn.Module):
    def __init__(self, input_channels=7, d_h=256, dropout=0.1, T_in=3000, L=21):
        super(Adaptive1DCNN, self).__init__()
        self.d_h = d_h
        self.L = L
        self.input_channels = input_channels
        self.C_out_branch = input_channels
        self.C_combined = input_channels * 5
        in_c = input_channels

        # --- W branch ---
        kernel_size_W = 64
        padding_W = kernel_size_W // 2
        self.W_branch = self._make_branch(in_c, kernel_size_W, padding_W, d_h, dropout)

        # --- N1 branch ---
        kernel_size_N1 = 15
        padding_N1 = kernel_size_N1 // 2
        self.N1_branch = self._make_branch(in_c, kernel_size_N1, padding_N1, d_h, dropout)
        
        # --- N2 branch ---
        kernel_size_N2 = 10
        padding_N2 = kernel_size_N2 // 2
        self.N2_branch = self._make_branch(in_c, kernel_size_N2, padding_N2, d_h, dropout)
        
        # --- N3 branch ---
        kernel_size_N34 = 5
        padding_N34 = kernel_size_N34 // 2
        self.N34_branch = self._make_branch(in_c, kernel_size_N34, padding_N34, d_h, dropout)
        
        # --- REM branch ---
        kernel_size_R = 3
        padding_R = kernel_size_R // 2
        self.R_branch = self._make_branch(in_c, kernel_size_R, padding_R, d_h, dropout)
        
        # --- Channel Reduction Layer ---
        # Reduce C*5 (35) back to C (7)
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(self.C_combined, self.input_channels, kernel_size=1),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU()
        )


    def _make_branch(self, in_c, kernel_size, padding, d_h, dropout):
        # 步骤 1: 1D 深度可分离卷积 (处理时间维度 T/d_h)
        conv_layers = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=(1, kernel_size), padding=(0, padding), groups=in_c),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv2d(in_c, in_c, kernel_size=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        layers = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=(1, kernel_size), padding=(0, padding), groups=in_c),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 使用 1x1 Conv2d 也可以实现跨通道的线性投影，但不是跨 d_h 维度
            nn.Conv2d(in_c, in_c, kernel_size=1), 
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        return layers

    def forward(self, x):
        B, C, L, T = x.shape
        # FFT
        fft_data = torch.fft.rfft(x, dim=-1, norm="forward")
        fft_features = torch.abs(fft_data[..., :self.d_h]) # [B, C, L, d_h]
        # Branch Feature Extraction (Output: [B, C, L, d_h // 5])
        W_feat = self.W_branch(fft_features)
        N1_feat = self.N1_branch(fft_features)
        N2_feat = self.N2_branch(fft_features)
        N34_feat = self.N34_branch(fft_features)
        R_feat = self.R_branch(fft_features)
        # Concatenate (Output: [B, C*5, L, d_h // 5])
        combined = torch.cat([W_feat, N1_feat, N2_feat, N34_feat, R_feat], dim=1) 
        # Channel Reduction (Output: [B, C, L, d_h]) 
        features = self.channel_reducer(combined)
        return features
    
    def get_branch_output(self, x, branch_name='W'):
        B, C, L, T = x.shape
        x_in = x.permute(0, 2, 1, 3).reshape(B * L, C, T) # [B*L, C, T]
        x_in = x_in.unsqueeze(2)
        if branch_name == 'W':
            output = self.W_branch(x_in)  # [B*L, C, T_out]
        elif branch_name == 'N1':
            output = self.N1_branch(x_in) 
        elif branch_name == 'N2':
            output = self.N2_branch(x_in) 
        elif branch_name == 'N3':
            output = self.N34_branch(x_in) 
        elif branch_name == 'REM':
            output = self.R_branch(x_in) 
        else:
            raise ValueError("branch_name 不支持")
            
        return output


class STPE(nn.Module):
    """时空位置编码"""
    def __init__(self, d_h):
        """
        向量化实现的时空位置编码（更高效）
        """
        super(STPE, self).__init__()
        self.d_h = d_h

    def get_positional_encoding(self, positions, d_model):
        """向量化计算位置编码"""
        batch_size = positions.shape[0]
        positions = positions.unsqueeze(-1).float()  # [batch, seq_len, 1]，显式转换为 float

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model)).to(positions.device)
        # 计算正弦和余弦部分
        angles = positions * div_term  # [batch, seq_len, d_model//2]

        pe = torch.zeros(batch_size, angles.shape[1], d_model, device=positions.device)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        return pe

    def forward(self, x_feat):
        B, C, L, d_h = x_feat.shape
        x_in = x_feat.reshape(B, C * L, d_h)  # [B, L×C, d_h]
        # 建立时空位置映射（确保与输入同一 device）
        device = x_feat.device
        time_indices = torch.arange(L, device=device).repeat_interleave(C).unsqueeze(0).expand(B, -1)
        channel_indices = torch.arange(C, device=device).repeat(L).unsqueeze(0).expand(B, -1)
        # 计算时空位置编码
        p_time = self.get_positional_encoding(time_indices, d_h)
        p_space = self.get_positional_encoding(channel_indices, d_h)
        # 组合编码
        p_combined = p_time + p_space
        # 注入位置编码
        x_st = x_in + p_combined  # 维度为[B, L*C, d_h]，可直接输入transformer
        return x_st


class STCA(nn.Module):
    """时空协同注意力机制"""
    def __init__(self, d_h=64, n_heads=8, dropout=0.1):
        super(STCA, self).__init__()
        self.d_h = d_h
        self.n_heads = n_heads
        self.d_k = d_h // n_heads
        # 线性投影层
        self.w_q = nn.Linear(d_h, d_h)
        self.w_k = nn.Linear(d_h, d_h)
        self.w_v = nn.Linear(d_h, d_h)
        self.w_o = nn.Linear(d_h, d_h)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_h)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        # 保存残差连接
        residual = x
        # 生成Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 信息聚合
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        # 输出投影和残差连接
        output = self.w_o(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_h=64, n_heads=8, d_ff=256, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = STCA(d_h, n_heads, dropout)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_h, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_h),
            nn.Dropout(dropout)
        )
        self.layer_norm_ffn = nn.LayerNorm(d_h)

    def forward(self, x, mask=None):
        # 自注意力层
        x = self.self_attn(x, mask)
        # 前馈网络 + 残差连接
        residual = x
        x = self.ffn(x)
        x = self.layer_norm_ffn(x + residual)
        return x

class CNNNet(nn.Module):
    """1D-CNN特征提取器"""
    def __init__(self, input_channels=2, d_h=64, hiden=1024,
                 num_classes=5, dropout=0.1):
        super(CNNNet, self).__init__()
        self.d_h = d_h
        # 特征提取模块
        self.feature_extractor = Adaptive1DCNN(input_channels, d_h, dropout)
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_h, hiden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hiden, hiden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hiden//2, num_classes)
        )

    def forward(self, x):
        """
        输入数据为x: [B, C, L, T]
        return_features: 是否返回特征图
        return_attentions: 是否返回注意力权重
        return: [B, L, num_classes]
        """
        # 特征提取
        features = self.feature_extractor(x)  # 形状: [B, C, L, d_h]
        # 全局平均池化
        features = features.mean(dim=1)  # 形状: [B, C, L]
        # 分类
        logits = self.classifier(features)  # 形状: [B, L, num_classes]
        return logits


class DG2TSleepNet(nn.Module):
    """完整的DG2T-SleepNet模型"""
    def __init__(self, input_channels=2, d_h=64, hiden=1024,
                 n_heads=8, n_layers=4, num_classes=5, d_ff=256, dropout=0.1):
        super(DG2TSleepNet, self).__init__()
        """
        input_channels: 输入通道数
        d_h: 1D-Conv 输出特征维度
        n_heads: 多头注意力机制中的头数
        d_ff: 前馈神经网络中间层的维度
        dropout: dropout 层的丢弃概率
        """
        self.d_h = d_h
        self.num_classes = num_classes
        # 特征提取模块
        self.feature_extractor = Adaptive1DCNN(input_channels, d_h, dropout)
        # 时空位置编码
        self.pos_encoding = STPE(d_h)
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_h, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_h, hiden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hiden, num_classes)
        )
        # 对比学习投影头
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, 128)  # 对比学习特征维度
        )
        self.dropout = nn.Dropout(dropout)
        # 层归一化，用于特征处理
        self.layer_norm = nn.LayerNorm(d_h)

    def forward(self, x, return_features=False, return_attentions=False):
        """
        输入数据为x: [B, C, L, T]
        return: [B, L, num_classes]
        """
        B, C, L, T = x.shape
        # 特征提取
        features = self.feature_extractor(x)  # 形状: [B, C, L, d_h]
        
        # 时空位置编码
        features = self.pos_encoding(features)  # 形状变为: [B, C*L, d_h]
        features = self.dropout(features)
        
        # 保存中间注意力权重（如果需要）
        attention_weights = []
        
        # Transformer编码
        for layer in self.transformer_layers:
            features = layer(features)  # 经过多层Transformer处理
        
        # 重新编码为原始维度结构
        features = features.reshape(B, C, L, self.d_h)
        
        # 沿通道维进行加权池化
        # channel_attn = torch.softmax(torch.mean(features, dim=(0, 2, 3)).unsqueeze(0).unsqueeze(2).unsqueeze(3), dim=1)
        # weighted_features = features * channel_attn
        features = torch.sum(features, dim=1)  # 沿通道维进行加权求和
        
        # 分类输出
        logits = self.classifier(features)  # [B, L, num_classes]
        
        # 根据需要返回额外信息
        if return_features and return_attentions:
            # 返回对比学习特征和注意力权重
            contrastive_features = self.contrastive_proj(features)
            return logits, contrastive_features, attention_weights
        elif return_features:
            # 返回对比学习特征
            contrastive_features = self.contrastive_proj(features)
            return logits, contrastive_features
        elif return_attentions:
            # 仅返回注意力权重
            return logits, attention_weights

        return logits


class LabelSmoothCrossEntropyLoss(nn.Module):
    """加权交叉熵损失（适用于形状[B, L, num_classes]的logits）"""
    def __init__(self, class_weights, reduction='mean'):
        super(LabelSmoothCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights  # 形状为[num_classes]的权重张量
        self.reduction = reduction

    def forward(self, logits, targets):
        # 步骤1：计算对数概率（形状保持[B, L, 5]）
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 步骤2：提取目标类别对应的对数概率（形状变为[B, L]）
        # 先将targets从[B, L]扩展为[B, L, 1]，再通过gather提取对应位置的值
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # 步骤3：计算每个样本的加权损失（形状保持[B, L]）
        # 为每个目标标签匹配对应的权重（class_weights[targets]形状为[B, L]）
        weights = self.class_weights[targets]
        loss = -weights * target_log_probs  # 加权负对数概率
        
        # 步骤4：按指定方式聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# 测试代码
if __name__ == "__main__":
    # 测试输入
    B = 8
    C = 7
    L = 21
    T = 3000
    test_input = torch.randn(B, C, L, T)

    # 创建模型实例
    # model = CNNNet(input_channels=7, d_h=512, hiden=4096,num_classes=5, dropout=0.1)

    model = DG2TSleepNet(
        input_channels=7,
        d_h=3000,  # 注意此项为每个窗口的输出维数
        n_heads=2,
        n_layers=1,
        num_classes=5
    )
    # 前向传播
    with torch.no_grad():
        output = model(test_input)
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
# 主模型部分构建完毕，需要添加滑窗
