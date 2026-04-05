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
            raise ValueError("branch_name unspport")
            
        return output


class STPE(nn.Module):
    def __init__(self, d_h):
        super(STPE, self).__init__()
        self.d_h = d_h

    def get_positional_encoding(self, positions, d_model):
        batch_size = positions.shape[0]
        positions = positions.unsqueeze(-1).float()  

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model)).to(positions.device)
        angles = positions * div_term  # [batch, seq_len, d_model//2]

        pe = torch.zeros(batch_size, angles.shape[1], d_model, device=positions.device)
        pe[:, :, 0::2] = torch.sin(angles)
        pe[:, :, 1::2] = torch.cos(angles)
        return pe

    def forward(self, x_feat):
        B, C, L, d_h = x_feat.shape
        x_in = x_feat.reshape(B, C * L, d_h)  # [B, L×C, d_h]
        device = x_feat.device
        time_indices = torch.arange(L, device=device).repeat_interleave(C).unsqueeze(0).expand(B, -1)
        channel_indices = torch.arange(C, device=device).repeat(L).unsqueeze(0).expand(B, -1)
        p_time = self.get_positional_encoding(time_indices, d_h)
        p_space = self.get_positional_encoding(channel_indices, d_h)
        p_combined = p_time + p_space
        x_st = x_in + p_combined 
        return x_st


class STCA(nn.Module):
    def __init__(self, d_h=64, n_heads=8, dropout=0.1):
        super(STCA, self).__init__()
        self.d_h = d_h
        self.n_heads = n_heads
        self.d_k = d_h // n_heads
        self.w_q = nn.Linear(d_h, d_h)
        self.w_k = nn.Linear(d_h, d_h)
        self.w_v = nn.Linear(d_h, d_h)
        self.w_o = nn.Linear(d_h, d_h)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_h)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        residual = x
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        output = self.w_o(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_h=64, n_heads=8, d_ff=256, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = STCA(d_h, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_h, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_h),
            nn.Dropout(dropout)
        )
        self.layer_norm_ffn = nn.LayerNorm(d_h)

    def forward(self, x, mask=None):
        x = self.self_attn(x, mask)
        residual = x
        x = self.ffn(x)
        x = self.layer_norm_ffn(x + residual)
        return x

class CNNNet(nn.Module):
    def __init__(self, input_channels=2, d_h=64, hiden=1024,
                 num_classes=5, dropout=0.1):
        super(CNNNet, self).__init__()
        self.d_h = d_h
        self.feature_extractor = Adaptive1DCNN(input_channels, d_h, dropout)
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
        features = self.feature_extractor(x)  # shape: [B, C, L, d_h]
        features = features.mean(dim=1)  # shape: [B, C, L]
        logits = self.classifier(features)  # shape: [B, L, num_classes]
        return logits


class DG2TSleepNet(nn.Module):
    def __init__(self, input_channels=2, d_h=64, hiden=1024,
                 n_heads=8, n_layers=4, num_classes=5, d_ff=256, dropout=0.1):
        super(DG2TSleepNet, self).__init__()
        self.d_h = d_h
        self.num_classes = num_classes
        self.feature_extractor = Adaptive1DCNN(input_channels, d_h, dropout)
        self.pos_encoding = STPE(d_h)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_h, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_h, hiden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hiden, num_classes)
        )
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, 128)  
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_h)

    def forward(self, x, return_features=False, return_attentions=False):
        B, C, L, T = x.shape
        features = self.feature_extractor(x) 
        
        features = self.pos_encoding(features)  
        features = self.dropout(features)
        
        attention_weights = []
        
        for layer in self.transformer_layers:
            features = layer(features) 
        
        features = features.reshape(B, C, L, self.d_h)
        
        features = torch.sum(features, dim=1)  
        
        logits = self.classifier(features)  # [B, L, num_classes]
        
        if return_features and return_attentions:
            contrastive_features = self.contrastive_proj(features)
            return logits, contrastive_features, attention_weights
        elif return_features:
            contrastive_features = self.contrastive_proj(features)
            return logits, contrastive_features
        elif return_attentions:
            return logits, attention_weights

        return logits


class LabelSmoothCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, reduction='mean'):
        super(LabelSmoothCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights  
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        weights = self.class_weights[targets]
        loss = -weights * target_log_probs  
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

