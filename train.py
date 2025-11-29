import os
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold # 导入 KFold

from dataset import BatchLoadedSleepEDFDataset
from model import DG2TSleepNet, LabelSmoothCrossEntropyLoss

# 导入预训练模块
import pretrain
# 导入封装好的预训练函数，用于在 K-Fold 训练集上运行
from pretrain import pretrain_cnn_sequential 


# -------------------------
# Config (保持和 pretrain.py 一致的宏定义，并定义训练特有参数)
# -------------------------
BATCH_METADATA_PATH = "dataset_output/cache/batch_metadata.pkl"

# 从 pretrain 导入常量，保持一致性
B = getattr(pretrain, "B", 8)
C = getattr(pretrain, "C", 7)
d_h = getattr(pretrain, "d_h", 512)
K_FOLDS = getattr(pretrain, "K_FOLDS", 2)

# 训练特有参数
NUM_EPOCHS = 100 
PATIENCE = 50
LEARNING_RATE_MODEL = 1e-4
WEIGHT_DECAY_MODEL = 1e-5


# -------------------------
# Utility: KFold patient splits (保持不变)
# -------------------------
def get_kfold_patient_splits(batch_metadata, K=K_FOLDS, random_state=42):
    patient_ids = [batch["patient_id"] for batch in batch_metadata]
    unique_patients = sorted(list(set(patient_ids)))
    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_patients), start=1):
        train_patients = [unique_patients[i] for i in train_idx]
        test_patients = [unique_patients[i] for i in test_idx]
        train_batches = [b for b in batch_metadata if b["patient_id"] in train_patients]
        test_batches = [b for b in batch_metadata if b["patient_id"] in test_patients]
        folds.append({
            "fold": fold_idx,
            "train_patients": train_patients,
            "test_patients": test_patients,
            "train_batches": train_batches,
            "test_batches": test_batches
        })
    return folds

# -------------------------
# DataLoader builder (保持不变)
# -------------------------
def build_dataloader_from_batches(batch_list, batch_size=B, use_augmentation=False, shuffle=False):
    dataset = BatchLoadedSleepEDFDataset(
        batch_metadata=batch_list,
        batch_size=batch_size,
        use_augmentation=use_augmentation,
        augmentation_config=None
    )
    # 注意: num_workers=0 和 pin_memory=True 应该在 DataLoader 构造函数中
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

def compute_class_weights(dataloader, num_classes=5):
    import torch
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    total = 0
    for batch in dataloader:
        labels = batch["label"].flatten()
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum().item()
        total += labels.numel()
    if total > 0:
        weights = total / (num_classes * class_counts)
        # 避免除以零，将没有样本的类别权重设为 0 或 1 (这里设为 0)
        weights = torch.where(class_counts == 0, torch.tensor(0.0), weights)
    else:
        weights = torch.ones(num_classes, dtype=torch.float32)
    return weights

# -------------------------
# Trainer class: encapsulate full flow
# -------------------------
class Trainer:
    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        os.makedirs("train_output/checkpoints", exist_ok=True)
        os.makedirs("train_output/fig_output", exist_ok=True)

    def run(self, num_folds=K_FOLDS):
        # load metadata
        with open(BATCH_METADATA_PATH, "rb") as f:
            all_batch_metadata = pickle.load(f)

        folds = get_kfold_patient_splits(all_batch_metadata, K=num_folds, random_state=42)

        for fold in folds:
            fold_idx = fold["fold"]
            print(f"=== Fold {fold_idx} / {len(folds)} ===")
            
            # 1. 构建当前折的 DataLoader
            train_batches = fold["train_batches"]
            test_batches = fold["test_batches"]

            # train_loader 用于计算权重和模型训练
            train_loader = build_dataloader_from_batches(train_batches, batch_size=B, use_augmentation=False, shuffle=True)
            test_loader = build_dataloader_from_batches(test_batches, batch_size=B, use_augmentation=False, shuffle=False)

            # Compute class weights from train_loader
            class_weights = compute_class_weights(train_loader).to(self.device)

            # 2. 在当前 K-Fold 的训练集上运行 CNN 串行预训练
            # 这一步调用了 pretrain.py 中封装的逻辑，确保使用正确的训练数据。
            print("-> Running CNN sequential pretraining on current fold's training split ...")
            
            # pretrain_cnn_sequential 返回预训练好的 CNN 权重文件的路径
            final_cnn_path = pretrain_cnn_sequential(
                train_batches=train_batches, 
                fold_idx=fold_idx, 
                device=self.device
            )

            print(f"Pretraining finished for Fold {fold_idx}. Final CNN weights saved to {final_cnn_path}")

            # 3. 冻结 CNN 并训练剩余模型 (DG2TSleepNet)
            
            # 实例化一个全新的模型
            model_full = DG2TSleepNet(
                input_channels=C,
                d_h=d_h,
                hiden=512,
                n_heads=8,
                n_layers=2,
                num_classes=5,
                dropout=0.1
            ).to(self.device)

            if os.path.exists(final_cnn_path):
                try:
                    # 加载预训练的 CNN 权重
                    model_full.feature_extractor.load_state_dict(torch.load(final_cnn_path, map_location=self.device))
                    # 冻结 CNN (feature_extractor) 的参数
                    for p in model_full.feature_extractor.parameters():
                        p.requires_grad = False
                    print("CNN weights loaded and frozen for full model training.")
                except Exception as e:
                    print(f"Warning: failed to load/freeze CNN weights: {e}. Training full model from scratch.")
            else:
                print("Warning: final CNN pretrain weights not found. Training full model from scratch.")

            # 准备损失函数和优化器
            criterion_cls = LabelSmoothCrossEntropyLoss(class_weights=class_weights, reduction='mean')
            
            # 优化器只优化 requires_grad=True (即未冻结的 STPE, Transformer, Classifier) 的参数
            trainable_params = filter(lambda p: p.requires_grad, model_full.parameters())
            optimizer_model = optim.Adam(trainable_params, lr=LEARNING_RATE_MODEL, weight_decay=WEIGHT_DECAY_MODEL)

            # 4. 训练循环与早停
            best_test_loss = float('inf')
            patience_counter = 0
            best_epoch = 0
            train_losses, test_losses = [], []
            train_accs, test_accs = [], []

            for epoch in range(NUM_EPOCHS):
                model_full.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for batch in tqdm(train_loader, desc=f"Fold{fold_idx} Train Epoch {epoch+1}"):
                    data = batch["data"].to(self.device)
                    labels = batch["label"].to(self.device)
                    optimizer_model.zero_grad()
                    outputs = model_full(data)
                    # 展平输出和标签以匹配 CrossEntropyLoss 的输入要求 [N_total, C] 和 [N_total]
                    loss = criterion_cls(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                    loss.backward()
                    optimizer_model.step()
                    # 乘以 batch 中的总样本数 B * L
                    running_loss += loss.item() * labels.numel() 
                    
                    _, pred = outputs.max(dim=-1)
                    total += labels.numel()
                    correct += pred.eq(labels).sum().item()
                
                # len(train_loader.dataset) 是总样本数 N_total
                avg_train_loss = running_loss / len(train_loader.dataset)
                train_acc = correct / total
                train_losses.append(avg_train_loss)
                train_accs.append(train_acc)

                # 评估
                model_full.eval()
                running_loss = 0.0
                correct = 0
                total = 0
                all_preds, all_labels = [], [] # 用于计算最终混淆矩阵

                with torch.no_grad():
                    for batch in tqdm(test_loader, desc=f"Fold{fold_idx} Eval Epoch {epoch+1}"):
                        data = batch["data"].to(self.device)
                        labels = batch["label"].to(self.device)
                        outputs = model_full(data)
                        loss = criterion_cls(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))
                        
                        running_loss += loss.item() * labels.numel()
                        
                        _, pred = outputs.max(dim=-1)
                        
                        # 收集所有预测和标签，用于最终评估 (只在最佳模型上做最终评估更准确)
                        if epoch + 1 == NUM_EPOCHS: # 最后一个epoch才收集
                            all_preds.extend(pred.cpu().numpy().flatten().tolist())
                            all_labels.extend(labels.cpu().numpy().flatten().tolist())
                            
                        total += labels.numel()
                        correct += pred.eq(labels).sum().item()
                        
                avg_test_loss = running_loss / len(test_loader.dataset)
                test_acc = correct / total
                test_losses.append(avg_test_loss)
                test_accs.append(test_acc)

                print(f"Epoch {epoch+1}: TrainLoss {avg_train_loss:.4f} TrainAcc {train_acc:.4f} | TestLoss {avg_test_loss:.4f} TestAcc {test_acc:.4f}")

                # 检查点 & 早停
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    # 保存最佳模型权重
                    torch.save({
                        "epoch": epoch+1,
                        "model_state_dict": model_full.state_dict(),
                        "optimizer_state_dict": optimizer_model.state_dict(),
                        "train_loss": avg_train_loss,
                        "test_loss": avg_test_loss
                    }, f"train_output/checkpoints/fold_{fold_idx}_best.pth")
                else:
                    patience_counter += 1
                    if patience_counter > PATIENCE:
                        print(f"Early stopping triggered at Epoch {epoch+1}.")
                        break

            print(f"\nFold {fold_idx} finished. Best epoch: {best_epoch}, Best test loss: {best_test_loss:.4f}")

            # 5. 加载最佳模型并进行最终评估和绘图
            ckpt_path = f"train_output/checkpoints/fold_{fold_idx}_best.pth"
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                model_full.load_state_dict(ckpt["model_state_dict"])
                model_full.eval()
                
                # 重新计算最佳模型在测试集上的预测结果
                final_preds, final_labels = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        data = batch["data"].to(self.device)
                        labels = batch["label"].to(self.device)
                        outputs = model_full(data)
                        _, pred = outputs.max(dim=-1)
                        final_preds.extend(pred.cpu().numpy().flatten().tolist())
                        final_labels.extend(labels.cpu().numpy().flatten().tolist())

                classes = ['Wake','N1','N2','N3','REM']
                
                # 绘制混淆矩阵
                if final_labels:
                    cm = confusion_matrix(final_labels, final_preds, normalize='true')
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                    fig, ax = plt.subplots(figsize=(10, 10))
                    disp.plot(ax=ax, cmap=plt.cm.Blues) # 使用 plt.cm.Blues
                    plt.title(f"Fold {fold_idx} Confusion Matrix (Best Model)")
                    plt.savefig(f"train_output/fig_output/fold_{fold_idx}_confusion.png")
                    plt.close(fig)
                    print(f"  -> Confusion Matrix saved to train_output/fig_output/fold_{fold_idx}_confusion.png")

                # 绘制损失/准确率曲线
                if train_losses:
                    # 损失曲线
                    plt.figure()
                    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
                    plt.plot(range(1, len(test_losses)+1), test_losses, label='Test Loss')
                    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
                    plt.legend()
                    plt.title(f"Fold {fold_idx} Loss Curve")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.savefig(f"train_output/fig_output/fold_{fold_idx}_loss_curve.png")
                    plt.close()
                    print(f"  -> Loss Curve saved to train_output/fig_output/fold_{fold_idx}_loss_curve.png")

                    # 准确率曲线
                    plt.figure()
                    plt.plot(range(1, len(train_accs)+1), train_accs, label='Train Acc')
                    plt.plot(range(1, len(test_accs)+1), test_accs, label='Test Acc')
                    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
                    plt.legend()
                    plt.title(f"Fold {fold_idx} Accuracy Curve")
                    plt.xlabel("Epoch")
                    plt.ylabel("Accuracy")
                    plt.savefig(f"train_output/fig_output/fold_{fold_idx}_acc_curve.png")
                    plt.close()
                    print(f"  -> Accuracy Curve saved to train_output/fig_output/fold_{fold_idx}_acc_curve.png")
            else:
                print(f"Warning: Best checkpoint for Fold {fold_idx} not found. Skipping final evaluation and plotting.")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
