import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from sklearn.model_selection import KFold
import numpy as np

# ===================================================================
# 1. 常量和模型/数据结构导入 (参考 train.py)
# ===================================================================

# 常量 (来自 train.py)
B = 8
C = 7
L = 21
T = 3000
d_h = 512  # Transformer特征维度 (用于计算 T_out_branch)

# 预训练参数
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50 
K_FOLDS = 2 # 保持与 train.py 一致，但不再用于内部划分
random_state = 42

# 缓存和目录
CACHE_DIR = "dataset_output/cache"
BATCH_DIR = os.path.join(CACHE_DIR, "test_batches")
CHECKPOINT_DIR = "pretrain_output/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 标签到分支的映射: 必须与 Adaptive1DCNN 的分支名称一致
CLASS_TO_BRANCH = {
    0: 'W',    # Wake (0) -> W_branch
    1: 'N1',   # N1 (1)   -> N1_branch
    2: 'N2',   # N2 (2)   -> N2_branch
    3: 'N3',   # N3 (3)   -> N3_branch
    4: 'REM',  # REM (4)  -> REM_branch
}

# 导入模型和数据集类
try:
    # 假设 DG2TSleepNet 和 Adaptive1DCNN 包含在 model.py 中
    # 假设 BatchLoadedSleepEDFDataset 包含在 dataset.py 中
    from dataset import BatchLoadedSleepEDFDataset 
    from model import DG2TSleepNet
except ImportError:
    print("导入错误: 请确保 dataset.py 和 model.py (包含 DG2TSleepNet) 在同一目录下。")
    # 为了避免在 train.py 导入时崩溃，不在这里 exit()，而是让 train.py 处理

# 预训练时 CNN 时域波形输出长度 (必须与 model.py 中的 Adaptive1DCNN 保持一致)
T_OUT_BRANCH = d_h // 5
if T_OUT_BRANCH == 0:
    # 仅在实际运行时检查
    pass 


# ===================================================================
# 2. 损失函数：类内分散度 / 类间分散度 (保持不变)
# ===================================================================

class IntraInterRatioLoss(nn.Module):
    """
    损失函数: (目标类内分散度) / (目标类与其它类间的距离)。
    注: Loss 计算是基于频谱图 (FFT 模)
    """
    def __init__(self):
        super(IntraInterRatioLoss, self).__init__()
        
    def forward(self, time_features, labels, target_class):
        # time_features 形状: [N_total, C, T_out_branch]
        
        # 1. 转换为频谱图 (FFT 模)
        # N_total, C, T_out_branch
        fft_result = torch.fft.rfft(time_features, dim=-1)
        magnitude = torch.abs(fft_result) 
        # magnitude 形状: [N_total, C, T_freq=T_out_branch // 2 + 1]
        
        N_total = magnitude.size(0)
        
        # 目标类 (Class X) 的频谱特征
        mask_X = (labels == target_class)
        features_X = magnitude[mask_X] 
        
        # 其他类 (Class Others) 的频谱特征
        mask_Others = (labels != target_class)
        features_Others = magnitude[mask_Others]
        labels_Others = labels[mask_Others]

        if features_X.size(0) <= 1 or features_Others.size(0) == 0:
            # 如果目标类样本数太少，或没有其他类样本，则返回 0 损失
            return torch.tensor(0.0, device=time_features.device, requires_grad=True)

        # 1. 计算目标类内分散度 (Intra-Class Scatter/Variance)
        # 展平特征 [N_X, C * T_freq]
        features_X_flat = features_X.flatten(start_dim=1)
        
        # 类中心
        centroid_X = features_X_flat.mean(dim=0) 
        # 计算样本到其类中心的平方欧氏距离的平均值
        distances_sq_X = torch.sum((features_X_flat - centroid_X).pow(2), dim=1)
        mean_intra_scatter = distances_sq_X.mean()
        
        # 2. 计算类间距离 (Inter-Class Distance)
        inter_distance_sq_sum = 0.0
        n_inter_pairs = 0
        
        unique_other_labels = torch.unique(labels_Others)
        for other_label in unique_other_labels:
            mask_c = (labels_Others == other_label)
            # 获取其他类别样本的频谱特征，并展平
            features_c_flat = features_Others[mask_c].flatten(start_dim=1)
            
            centroid_c = features_c_flat.mean(dim=0) # 其他类别中心
            
            # X 的中心 vs 其他类别的中心之间的平方欧氏距离
            distance_sq = torch.sum((centroid_X - centroid_c).pow(2))
            inter_distance_sq_sum += distance_sq
            n_inter_pairs += 1
            
        # 平均 Inter-class 距离
        mean_inter_distance = inter_distance_sq_sum / n_inter_pairs if n_inter_pairs > 0 else 1.0

        # 3. 计算最终损失 (最小化)
        # 最小化 Intra-scatter，最大化 Inter-distance
        loss = mean_intra_scatter / (mean_inter_distance + 1e-6)
        
        return loss

# ===================================================================
# 3. 数据加载函数 (修改：使用外部传入的 batches 列表)
# ===================================================================

def load_train_data(train_batches):
    """
    根据传入的训练批次列表创建 DataLoader (仅用于预训练)
    """
    print(f"找到了 {len(train_batches)} 个训练批次。")
    dataset = BatchLoadedSleepEDFDataset(
        batch_metadata=train_batches,
        batch_size=B,
        use_augmentation=True,
        augmentation_config=None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    return dataloader

# ===================================================================
# 4. 单分支训练函数 (保持不变)
# ===================================================================

def pretrain_single_branch(model, dataloader, criterion, optimizer, device, 
                           num_epochs, target_class, branch_name):
    
    # 仅训练 feature_extractor (Adaptive1DCNN) 部分
    model.feature_extractor.train()
    
    # 冻结其他层 (STPE, Transformer, Classifier)
    for name, param in model.named_parameters():
        if "feature_extractor" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True # 确保 CNN 参数是可训练的

    print(f"\n--- CNN 预训练开始: 目标分支 '{branch_name}' (类别 {target_class}) ---")
    
    min_loss = float('inf')
    best_epoch = -1
    
    # 记录当前分支的检查点路径
    branch_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"branch_{branch_name}")
    os.makedirs(branch_checkpoint_dir, exist_ok=True)
    best_save_path = os.path.join(branch_checkpoint_dir, f"best_loss_{branch_name}.pth")

    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_dict in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # data 形状: [B, C, L, T], labels 形状: [B, L]
            data = batch_dict['data'].to(device)
            labels = batch_dict['label'].to(device)
            
            # 展平标签以匹配 N_total = B * L
            flat_labels = labels.reshape(-1) # [N_total]
            
            optimizer.zero_grad()
            
            # 1. 从 CNN 中提取该分支的时域波形
            # [N_total, C, T_out_branch]
            try:
                # 假设 Adaptive1DCNN 的 feature_extractor 有 get_branch_output 方法
                branch_features = model.feature_extractor.get_branch_output(data, branch_name)
            except Exception as e:
                # 预训练时，如果模型没有正确实现，跳过这个 batch
                # print(f"\nWarning: Could not get output for branch {branch_name}. Check model.py. Error: {e}")
                continue 

            # 2. 计算损失 (基于频谱图)
            branch_loss = criterion(branch_features, flat_labels, target_class)
            
            # 3. 反向传播
            if branch_loss.item() > 0:
                branch_loss.backward()
                optimizer.step()
                total_loss += branch_loss.item() 
            
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"Branch '{branch_name}' Epoch {epoch+1} 预训练平均损失: {avg_loss:.6f}")
        
        # 保存最佳检查点
        if avg_loss < min_loss and avg_loss > 0:
            min_loss = avg_loss
            best_epoch = epoch + 1
            # 仅保存 Adaptive1DCNN 的权重
            torch.save(model.feature_extractor.state_dict(), best_save_path)
            print(f"  --> 已更新最佳检查点 ({best_save_path})，损失: {min_loss:.6f}")

    print(f"--- 分支 '{branch_name}' 预训练完成 ---")
    print(f"最佳损失: {min_loss:.6f}，出现在 Epoch {best_epoch}")
    
    # 返回最佳检查点路径，供主程序加载，以确保下一个分支从最佳状态开始
    return best_save_path


# ===================================================================
# 5. 预训练主函数 (封装)
# ===================================================================

def pretrain_cnn_sequential(train_batches, fold_idx, device):
    """
    对给定的训练批次列表，按顺序对 Adaptive1DCNN 的各个分支进行预训练。
    :param train_batches: 当前 K-Fold 的训练集批次元数据列表。
    :param fold_idx: 当前折的索引，用于生成唯一的保存路径。
    :param device: 训练设备 (cpu/cuda)。
    :return: 最终预训练好的 CNN 权重文件的路径。
    """
    
    if T_OUT_BRANCH == 0:
        raise ValueError(f"d_h={d_h} 导致 d_h//5=0。请将 d_h 增大到至少 5。")
        
    print(f"--- Fold {fold_idx} CNN 预训练开始 ---")
    
    # 1. 加载数据
    train_loader = load_train_data(train_batches)

    # 2. 实例化模型 (使用 train.py 的参数)
    model = DG2TSleepNet(
        input_channels=C,  
        d_h=d_h,           
        hiden=512,
        n_heads=8,   
        n_layers=2, 
        num_classes=5,     
        dropout=0.1
    ).to(device)

    # 3. 损失函数
    criterion = IntraInterRatioLoss() 
    
    
    # 定义最终保存路径 (在 CHECKPOINT_DIR 下的子目录中)
    fold_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    final_weights_path = os.path.join(fold_checkpoint_dir, "cnn_pretrain_final_sequential.pth")
    
    
    # 4. 按照 CLASS_TO_BRANCH 的顺序，逐个分支进行训练
    for target_class, branch_name in CLASS_TO_BRANCH.items():
        
        # 重新创建优化器，确保它关联到当前模型的 feature_extractor 参数
        optimizer = optim.Adam(model.feature_extractor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # 训练当前分支
        # 注意: 这里需要调整 pretrain_single_branch 内部的检查点路径，
        # 避免不同折的训练互相覆盖。
        # 简单起见，我们继续使用原来的逻辑，但最终保存路径是唯一的。
        best_ckpt_path = pretrain_single_branch(
            model=model, 
            dataloader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=device, 
            num_epochs=NUM_EPOCHS,
            target_class=target_class,
            branch_name=branch_name
        )
        
        # 重要：加载当前分支的最佳检查点。下一个分支的训练将从这个最佳状态开始。
        if os.path.exists(best_ckpt_path):
            print(f"加载分支 '{branch_name}' 的最佳权重，以便下一个分支继续优化...")
            model.feature_extractor.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        else:
             print(f"Warning: 无法找到或加载分支 '{branch_name}' 的最佳权重。下一个分支将从上一个 Epoch 的最终状态开始。")


    print("\n==============================================")
    print(f"--- Fold {fold_idx} 所有分支的串行预训练已完成 ---")
    print("==============================================")
    
    # 5. 保存最终权重 (最后一个分支训练结束后的模型状态)
    torch.save(model.feature_extractor.state_dict(), final_weights_path)
    print(f"\nFold {fold_idx} 最终预训练的 CNN 权重已保存至: {final_weights_path}")
    
    return final_weights_path
