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

B = 8
C = 7
L = 21
T = 3000
d_h = 512  
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50 
K_FOLDS = 2 
random_state = 42

CACHE_DIR = "dataset_output-20/cache"
BATCH_DIR = os.path.join(CACHE_DIR, "test_batches")
CHECKPOINT_DIR = "pretrain_output/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CLASS_TO_BRANCH = {
    0: 'W',    # Wake (0) -> W_branch
    1: 'N1',   # N1 (1)   -> N1_branch
    2: 'N2',   # N2 (2)   -> N2_branch
    3: 'N3',   # N3 (3)   -> N3_branch
    4: 'REM',  # REM (4)  -> REM_branch
}

try:
    from dataset import BatchLoadedSleepEDFDataset 
    from model import DG2TSleepNet
except ImportError:
    print("ImportError: Please ensure that dataset.py and model.py (containing DG2TSleepNet) are in the same directory.")

T_OUT_BRANCH = d_h // 5
if T_OUT_BRANCH == 0:
    pass 

class IntraInterRatioLoss(nn.Module):
    def __init__(self):
        super(IntraInterRatioLoss, self).__init__()
        
    def forward(self, time_features, labels, target_class):
        # N_total, C, T_out_branch
        fft_result = torch.fft.rfft(time_features, dim=-1)
        magnitude = torch.abs(fft_result) 
        
        N_total = magnitude.size(0)
        
        mask_X = (labels == target_class)
        features_X = magnitude[mask_X] 
        
        mask_Others = (labels != target_class)
        features_Others = magnitude[mask_Others]
        labels_Others = labels[mask_Others]

        if features_X.size(0) <= 1 or features_Others.size(0) == 0:
            return torch.tensor(0.0, device=time_features.device, requires_grad=True)

        features_X_flat = features_X.flatten(start_dim=1)
        
        centroid_X = features_X_flat.mean(dim=0) 
        distances_sq_X = torch.sum((features_X_flat - centroid_X).pow(2), dim=1)
        mean_intra_scatter = distances_sq_X.mean()
        
        inter_distance_sq_sum = 0.0
        n_inter_pairs = 0
        
        unique_other_labels = torch.unique(labels_Others)
        for other_label in unique_other_labels:
            mask_c = (labels_Others == other_label)
            features_c_flat = features_Others[mask_c].flatten(start_dim=1)
            
            centroid_c = features_c_flat.mean(dim=0) 
            
            distance_sq = torch.sum((centroid_X - centroid_c).pow(2))
            inter_distance_sq_sum += distance_sq
            n_inter_pairs += 1
            
        mean_inter_distance = inter_distance_sq_sum / n_inter_pairs if n_inter_pairs > 0 else 1.0

        loss = mean_intra_scatter / (mean_inter_distance + 1e-6)
        
        return loss

def load_train_data(train_batches):
    print(f"find {len(train_batches)} batches")
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

def pretrain_single_branch(model, dataloader, criterion, optimizer, device, 
                           num_epochs, target_class, branch_name):
    
    model.feature_extractor.train()
    
    for name, param in model.named_parameters():
        if "feature_extractor" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True 

    print(f"\n--- start first step training: target batch '{branch_name}' (class {target_class}) ---")
    
    min_loss = float('inf')
    best_epoch = -1
    
    branch_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"branch_{branch_name}")
    os.makedirs(branch_checkpoint_dir, exist_ok=True)
    best_save_path = os.path.join(branch_checkpoint_dir, f"best_loss_{branch_name}.pth")

    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_dict in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data = batch_dict['data'].to(device)
            labels = batch_dict['label'].to(device)
            
            flat_labels = labels.reshape(-1) # [N_total]
            
            optimizer.zero_grad()
            # [N_total, C, T_out_branch]
            try:
                branch_features = model.feature_extractor.get_branch_output(data, branch_name)
            except Exception as e:
                continue 

            branch_loss = criterion(branch_features, flat_labels, target_class)
            
            if branch_loss.item() > 0:
                branch_loss.backward()
                optimizer.step()
                total_loss += branch_loss.item() 
            
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"Branch '{branch_name}' Epoch {epoch+1} avg loss: {avg_loss:.6f}")
        
        if avg_loss < min_loss and avg_loss > 0:
            min_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model.feature_extractor.state_dict(), best_save_path)

    print(f"--- batch '{branch_name}' finished ---")
    print(f"best loss: {min_loss:.6f}，appear in Epoch {best_epoch}")
    
    return best_save_path

def pretrain_cnn_sequential(train_batches, fold_idx, device):
    if T_OUT_BRANCH == 0:
        raise ValueError(f"d_h={d_h} causes d_h//5=0. Please increase d_h to at least 5.")
        
    print(f"--- Fold {fold_idx} start ---")
    
    train_loader = load_train_data(train_batches)

    model = DG2TSleepNet(
        input_channels=C,  
        d_h=d_h,           
        hiden=512,
        n_heads=8,   
        n_layers=2, 
        num_classes=5,     
        dropout=0.1
    ).to(device)

    criterion = IntraInterRatioLoss() 
    
    fold_checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_checkpoint_dir, exist_ok=True)
    final_weights_path = os.path.join(fold_checkpoint_dir, "cnn_pretrain_final_sequential.pth")
    
    for target_class, branch_name in CLASS_TO_BRANCH.items():
        optimizer = optim.Adam(model.feature_extractor.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
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
        
        if os.path.exists(best_ckpt_path):
            print(f"Loading the best weights for branch '{branch_name}' to continue optimization for the next branch...")
            model.feature_extractor.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        else:
             print(f"Warning: Failed to find or load the best weights for branch '{branch_name}'. The next branch will start from the final state of the previous Epoch.")


    print("\n==============================================")
    print(f"--- All branches for Fold {fold_idx} have finished training ---")
    print("==============================================")
    
    torch.save(model.feature_extractor.state_dict(), final_weights_path)
    
    return final_weights_path
