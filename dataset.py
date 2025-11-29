import os
import re
import glob
import pickle
import random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
import mne
from scipy import signal

T_h = 6  # 每个 epoch 长度（秒）
T_l = 3
window_size = 6  # 脉冲密度计算窗口（秒）

# 缓存目录
CACHE_DIR = "dataset_output/cache"
BATCH_DIR = os.path.join(CACHE_DIR, "batches")
TEST_BATCH_DIR = os.path.join(CACHE_DIR, "test_batches")
os.makedirs(TEST_BATCH_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)

# 标签映射（Sleep-EDF 标准）
annotation_desc_2_event_id = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # 合并 stage 3 & 4
    "Sleep stage R": 4,
}

class EEGDataAugmenter:
    """数据增强器（采用自适应滑动窗口+随机重组分段策略）"""
    def __init__(self, sfreq=100, T_h=6, T_l=3, window_size=6, insert_counts=None):
        """
        参数:
            sfreq: 采样频率（Hz）
            T_h: 高脉冲密度阈值（对应2秒分段）
            T_l: 低脉冲密度阈值（对应4秒分段）
            window_size: 脉冲密度计算窗口（秒）
        """
        self.sfreq = sfreq
        self.T = self.sfreq * 30
        self.T_h = T_h
        self.T_l = T_l
        self.window_size = window_size
        self.insert_counts = insert_counts if insert_counts is not None else {}

    def insert_sequences(self, data, labels):
        new_data = []
        new_labels = []
        prev_label = None
        n_insertions = {}  # 记录每个标签已插入的次数
        
        for i in range(len(labels)):
            current_label = labels[i]
            new_data.append(data[:, i, :])
            new_labels.append(current_label)
            
            # 检查是否需要插入增强序列
            if current_label in self.insert_counts:
                # 获取当前标签允许的最大插入次数
                max_inserts = self.insert_counts[current_label]
                # 已插入次数（同一连续序列内）
                if current_label == prev_label:
                    count = n_insertions.get(current_label, 0) + 1
                else:
                    count = 1
                    n_insertions[current_label] = count
                
                # 插入增强序列和复制原分段（最多插入max_inserts次）
                while count <= max_inserts:
                    # 生成增强序列（对单个分段增强）
                    augmented_segment = self.__call__(data[:, i:i+1, :])[0]
                    new_data.append(augmented_segment)
                    new_labels.append(current_label)
                    
                    # 复制原分段
                    new_data.append(data[:, i, :])
                    new_labels.append(current_label)
                    
                    count += 1
            
            prev_label = current_label
        
        new_data = np.concatenate(new_data, axis=1)
        new_labels = np.array(new_labels)
        return new_data, new_labels

    def _pulse_density_transform(self, signal_data):
        """计算脉冲密度（复制_segment_epochs中的逻辑）"""
        pulses = (signal_data > 0.00005).astype(float)  # 脉冲阈值判断
        window_samples = int(self.window_size * self.sfreq)
        density = np.convolve(pulses, np.ones(window_samples), mode='valid')
        return density

    def _adaptive_sliding_window_single_channel(self, channel_data):
        """单通道自适应分段（复制_segment_epochs中的逻辑）"""
        segments = []
        current_pos = 0
        signal_length = len(channel_data)
        density = self._pulse_density_transform(channel_data)
        
        while current_pos < signal_length:
            # 确保密度索引不越界
            density_idx = min(current_pos, len(density) - 1)
            current_density = density[density_idx]
            
            # 根据脉冲密度确定分段长度（单位：采样点）
            if current_density > self.T_h:  # 高密度：2秒分段
                segment_duration = int(2 * self.sfreq)
            elif current_density < self.T_l:  # 低密度：4秒分段
                segment_duration = int(4 * self.sfreq)
            else:  # 中等密度：3秒分段
                segment_duration = int(3 * self.sfreq)
            
            # 计算分段结束位置（不超过信号总长度）
            end_pos = min(current_pos + segment_duration, signal_length)
            is_last_segment = (end_pos == signal_length)
            
            # 保存分段
            if (end_pos - current_pos > segment_duration) or is_last_segment:
                segment = channel_data[current_pos:end_pos]
                segments.append(segment)
            
            current_pos = end_pos  # 移动到下一个分段起点
        
        return segments

    def __call__(self, data):
        """
        对EEG数据应用自适应滑动窗口重组增强
        data形状: [C, L, T] 
            C: 通道数
            L: 时间序列长度（时相数）
            T: 每个时相的采样点数
        返回增强后的数据（形状不变）
        """
        C, L, T = data.shape
        augmented_data = []
        
        # 对每个时相单独进行增强（保持时相数量L不变）
        for l in range(L):
            epoch_data = data[:, l, :]  # [C, T]：单个时相的所有通道数据
            new_epoch_channels = []
            
            # 对每个通道单独处理
            for ch in range(C):
                channel_data = epoch_data[ch]  # [T]：单通道时相数据
                segments = self._adaptive_sliding_window_single_channel(channel_data)
                
                if not segments:  # 若无法分段，直接使用原始数据
                    new_channel = channel_data
                else:
                    # 随机排列分段并重组
                    permuted_segments = np.random.permutation(segments)
                    new_channel = np.concatenate(permuted_segments, axis=0)
                    
                    # 调整长度至原始时相长度（填充或截断）
                    if len(new_channel) > T:
                        new_channel = new_channel[:T]
                    else:
                        new_channel = np.pad(
                            new_channel, 
                            (0, T - len(new_channel)), 
                            mode='constant'
                        )
                
                new_epoch_channels.append(new_channel)
            
            # 组合通道数据，形状：[C, T]
            augmented_epoch = np.stack(new_epoch_channels)
            augmented_data.append(augmented_epoch)
        
        # 重组为原始形状 [C, L, T]
        return np.stack(augmented_data, axis=1)
    
# ===================================================================
# EDF 数据集处理
# ===================================================================
class SleepEDFDataset(Dataset):
    def __init__(self, data_path, L=5):
        """
        :param data_path: EDF 文件目录
        :param L: 每个样本包含多少个连续 epoch
        :param augmentation_config: 增强配置字典，若为 None 则不启用增强
        """
        self.data_path = data_path
        self.L = L
        self.file_list = self._get_file_list()
        self.samples = []  # 有效样本索引 (file_idx, start_epoch)
        self.file_metadata = []
        self.T_h = T_h  # 高阈值
        self.T_l = T_l  # 低阈值
        self.sfreq = 100  # 采样频率（Hz）
        self.augmenter = EEGDataAugmenter(**augmentation_config) if augmentation_config else None

    def augmenter_single_epoch(self, epoch_data):
        """
        对单个 epoch 数据 [C, T] 做增强，返回增强后的 [C, T]（强制统一长度）
        """
        C, T = epoch_data.shape
        # 增加一个 dummy 的 L=1 维度 -> [1, C, T]
        epoch_data_expanded = epoch_data.reshape(1, C, T)
        # 调整为 [C, 1, T] 以匹配 EEGDataAugmenter 的输入要求
        epoch_data_expanded = np.transpose(epoch_data_expanded, (1, 0, 2))  # [C, 1, T]
        # 调用 augmenter
        augmented = self.augmenter(epoch_data_expanded)  # 输入 [C, 1, T]，输出 [C, 1, T]
        augmented_epoch = augmented[:, 0, :]  # 取出单个增强时相 [C, T']

        # === 新增：强制保证长度一致 ===
        if augmented_epoch.shape[1] > T:
            augmented_epoch = augmented_epoch[:, :T]
        elif augmented_epoch.shape[1] < T:
            pad_width = T - augmented_epoch.shape[1]
            augmented_epoch = np.pad(
                augmented_epoch, ((0, 0), (0, pad_width)), mode='constant'
            )

        return augmented_epoch  # [C, T]

    def _get_file_list(self):
        """
        根据 SC 编号匹配 PSG 和 Hypnogram 文件
        支持 SC4031E0-PSG.edf <-> SC4031EC-Hypnogram.edf / SC4031EP-Hypnogram.edf 等情况
        """
        files = []
        print("开始匹配 PSG 和 Hypnogram 文件...")

        # 遍历所有 PSG 文件
        for psg_file in glob.glob(os.path.join(self.data_path, "*-PSG.edf")):
            fname = os.path.basename(psg_file)
            # 提取 SC 编号，例如 SC4031
            m = re.match(r"(SC\d{4})", fname)
            sc_id = m.group(1)

            # 在目录下找同编号的 Hypnogram
            hyp_candidates = glob.glob(os.path.join(self.data_path, sc_id + "*-Hypnogram.edf"))
            if len(hyp_candidates) > 0:
                hyp_file = hyp_candidates[0]  # 取第一个匹配的
                files.append((psg_file, hyp_file))
            else:
                print(f"警告: {fname} 没有找到对应的 Hypnogram 文件")
        print(f"共找到 {len(files)} 个有效的文件对")
        return files

    def _zscore_normalize_epoch(self, epoch_data, epoch_idx=None, file_name=None):
        """
        对单个时相进行z-score标准化
        :param epoch_data: [n_channels, epoch_length] 的时相数据
        :param epoch_idx: 时相索引（用于日志输出）
        :param file_name: 文件名（用于日志输出）
        :return: 标准化后的数据，每个通道的均值和标准差
        """
        normalized_data = np.zeros_like(epoch_data)
        channel_means = []
        channel_stds = []

        for i in range(epoch_data.shape[0]):
            channel_data = epoch_data[i]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            # 避免除零错误
            if std == 0:
                std = 1.0
            normalized_data[i] = (channel_data - mean) / std
            channel_means.append(mean)
            channel_stds.append(std)
        return normalized_data, channel_means, channel_stds
    
    def _pulse_density_transform(self, signal_data, window_size=6):
        """脉冲密度计算"""
        pulses = (signal_data > 0.00005).astype(float)  # 阈值判断
        window_samples = int(window_size * self.sfreq)
        density = np.convolve(pulses, np.ones(window_samples), mode='valid')
        return density
        
    def _adaptive_sliding_window_single_channel(self, channel_data):
        """
        单通道自适应分段（每个通道独立分段）
        :param channel_data: 单通道信号数据，形状为 [epoch_length]
        :return: 该通道的分段列表，每个元素为一个分段（[segment_length]）
        """
        segments = []
        current_pos = 0
        signal_length = len(channel_data)
        
        density = self._pulse_density_transform(channel_data)
        
        while current_pos < signal_length:
            # 确保密度索引不越界
            density_idx = min(current_pos, len(density) - 1)
            current_density = density[density_idx]
            
            # 根据脉冲密度确定分段长度（单位：采样点）
            if current_density > self.T_h:  # 高密度：2秒分段
                segment_duration = int(2 * self.sfreq)
            elif current_density < self.T_l:  # 低密度：4秒分段
                segment_duration = int(4 * self.sfreq)
            else:  # 中等密度：3秒分段
                segment_duration = int(3 * self.sfreq)
            
            # 计算分段结束位置（不超过信号总长度）
            end_pos = min(current_pos + segment_duration, signal_length)
            # 判断是否为最后一个分段（已到达信号末尾）
            is_last_segment = (end_pos == signal_length)
            
            # 最后一个分段，无论长度如何，直接保留剩余部分
            if (end_pos - current_pos > segment_duration) or is_last_segment:
                segment = channel_data[current_pos:end_pos]
                segments.append(segment)
            
            # 移动到下一个分段起点（无重叠）
            current_pos = end_pos
        
        return segments

    def _random_permute_windows(self, windows):
        """
        对窗口进行随机排列并重组为新的epoch
        参数:
        windows: 窗口列表，形状为 [n_windows, n_channels, window_size]
        返回:
        重组后的epoch数据，形状为 [n_channels, epoch_length]
        """
        # 随机打乱窗口顺序
        permuted_indices = np.random.permutation(len(windows))
        permuted_windows = windows[permuted_indices]
        # 重组为单个epoch
        return np.concatenate(permuted_windows, axis=1)

    def _segment_epochs(self, psg_file, hyp_file):
        try:
            psg_basename = os.path.basename(psg_file)
            patient_id = re.match(r"(SC\d{4})", psg_basename).group(1)
            print(f"\n🔍 处理患者 {patient_id}: {psg_basename}")

            # 1. 读取 EDF 和 Hypnogram
            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
            annots = mne.read_annotations(hyp_file)
            raw.set_annotations(annots)

            # 2. 提取事件（睡眠阶段）和对应数据
            events, _ = mne.events_from_annotations(
                raw,
                event_id=annotation_desc_2_event_id,
                chunk_duration=30.0,
                verbose='ERROR'
            )

            self.sfreq = int(raw.info["sfreq"])  # 采样率，例如 100
            self.T = self.sfreq * 30  # 每个 epoch 是30秒，单位：采样点数
            epoch_length = self.T
            n_channels = raw.info["nchan"]
            data = raw.get_data()  # shape: [n_channels, n_samples]

            epochs_data = []      # 存储每个 epoch 的数据，shape: [n_channels, T]
            epochs_labels = []    # 存储每个 epoch 的标签，如 [0, 2, 1, ...]

            valid_events = 0

            for event_idx, (onset, _, stage) in enumerate(events):
                start = onset
                end = start + epoch_length
                if end <= data.shape[1] and stage in annotation_desc_2_event_id.values():
                    # 提取单个 epoch 的原始数据 [n_channels, T]
                    segment = data[:, start:end]
                    # z-score 标准化
                    normalized_segment, means, stds = self._zscore_normalize_epoch(segment)
                    epochs_data.append(normalized_segment)
                    epochs_labels.append(stage)
                    valid_events += 1

            if len(epochs_labels) == 0:
                print(f"  ⚠️ 警告: {psg_basename} 没有找到有效时相")
                return None, None, None  # 只返回三个，下面不再使用 balanced_*

            # 最终返回：
            original_data = np.stack(epochs_data)  # shape: [n_epochs, n_channels, T]
            original_labels = np.array(epochs_labels, dtype=np.int64)  # shape: [n_epochs]

            print(f"  ✅ 原始数据有效，共 {len(original_labels)} 个 epoch，标签分布: {dict(Counter(original_labels))}")

            # 只返回原始数据，不再返回 balanced_*
            return original_data, original_labels, None  # 最后一个 None 是为了兼容（或可去掉）

        except Exception as e:
            print(f"❌ 文件 {os.path.basename(psg_file)} 加载或处理失败: {str(e)}")
            return None, None, None

    def preprocess_to_batch_files(self, batch_size=32, augmentation_config=None):
        """
        预处理 EDF 数据并保存为 batch 文件。
        流程：
          1. 读取每个 PSG/Hypnogram 文件；
          2. 提取 epoch；
          3. 测试集：直接保存；
          4. 训练集：按 insert_counts 插入 (原始 + 增强) 组；
          5. 保存批次文件。
        """
        # === 目录初始化 ===
        BATCH_DIR = os.path.join(CACHE_DIR, "batches")          # 训练集增强数据
        TEST_BATCH_DIR = os.path.join(CACHE_DIR, "test_batches")  # 测试集原始数据
        os.makedirs(BATCH_DIR, exist_ok=True)
        os.makedirs(TEST_BATCH_DIR, exist_ok=True)

        batch_metadata, file_metadata = [], []
        test_batch_idx, train_batch_idx = 0, 0
        total_valid_files = total_batches = 0

        augmenter = EEGDataAugmenter(**(augmentation_config or {})) if augmentation_config else None

        print(f"\n🔧 开始预处理 {len(self.file_list)} 个文件...")
        print("=" * 60)

        # === insert_counts 建议使用数字键（与标签编码一致） ===
        insert_counts = augmentation_config["insert_counts"]

        for file_idx, (psg_file, hyp_file) in enumerate(self.file_list):
            psg_basename = os.path.basename(psg_file)
            patient_id = re.match(r"(SC\d{4})", psg_basename).group(1)

            # === Step 1: 提取 epoch ===
            original_data, original_labels, _ = self._segment_epochs(psg_file, hyp_file)
            if original_data is None or original_labels is None:
                file_metadata.append({
                    "file_idx": file_idx,
                    "psg_file": psg_basename,
                    "patient_id": patient_id,
                    "valid": False,
                    "reason": "解析失败或无有效时相"
                })
                continue

            n_epochs = len(original_labels)
            file_metadata.append({
                "file_idx": file_idx,
                "psg_file": psg_basename,
                "patient_id": patient_id,
                "valid": True,
                "n_epochs_original": n_epochs,
                "label_distribution": dict(Counter(original_labels.tolist())),
            })

            # === Step 2: 生成测试集 ===
            data_test = np.transpose(original_data, (1, 0, 2))  # [C, n_epochs, T]
            num_test_batches = n_epochs // self.L

            for i in range(num_test_batches):
                start, end = i * self.L, (i + 1) * self.L
                batch_data = data_test[:, start:end, :]
                batch_labels = original_labels[start:end]
                test_data_path = os.path.join(TEST_BATCH_DIR, f"batch_{test_batch_idx + 1}_data.npy")
                test_label_path = os.path.join(TEST_BATCH_DIR, f"batch_{test_batch_idx + 1}_labels.npy")
                np.save(test_data_path, batch_data)
                np.save(test_label_path, batch_labels)
                batch_metadata.append({
                    "batch_idx": test_batch_idx + 1,
                    "data_path": test_data_path,
                    "label_path": test_label_path,
                    "patient_id": patient_id,
                    "type": "test"
                })
                test_batch_idx += 1
                total_batches += 1

            # === Step 3: 生成训练集 (原始 + 增强) ===
            augmented_data, augmented_labels = [], []

            for idx, (epoch_data, label) in enumerate(zip(original_data, original_labels)):
                # 保留原始 epoch
                augmented_data.append(epoch_data)
                augmented_labels.append(label)

                # 如果该标签需要增强
                if label in insert_counts:
                    n_insert = insert_counts[label]
                    for _ in range(n_insert):
                        enhanced_epoch = self.augmenter_single_epoch(epoch_data)
                        # 插入原始 + 增强各一份
                        augmented_data.extend([epoch_data, enhanced_epoch])
                        augmented_labels.extend([label, label])

            # === Step 4: 转换并分 batch ===
            if augmented_data:
                augmented_data = np.stack(augmented_data)          # [N_total, C, T]
                augmented_labels = np.array(augmented_labels)      # [N_total]
                data_train = np.transpose(augmented_data, (1, 0, 2))  # [C, N_total, T]

                num_train_batches = len(augmented_labels) // self.L
                for i in range(num_train_batches):
                    start, end = i * self.L, (i + 1) * self.L
                    batch_data = data_train[:, start:end, :]
                    batch_labels = augmented_labels[start:end]

                    train_data_path = os.path.join(BATCH_DIR, f"batch_{train_batch_idx + 1}_data.npy")
                    train_label_path = os.path.join(BATCH_DIR, f"batch_{train_batch_idx + 1}_labels.npy")
                    np.save(train_data_path, batch_data)
                    np.save(train_label_path, batch_labels)

                    batch_metadata.append({
                        "batch_idx": train_batch_idx + 1,
                        "data_path": train_data_path,
                        "label_path": train_label_path,
                        "patient_id": patient_id,
                        "type": "train",
                        "augmentation": "epoch_insertion_with_augmentation"
                    })
                    train_batch_idx += 1
                    total_batches += 1

            total_valid_files += 1
            print(f"  ✅ {psg_basename}：{len(augmented_labels)} 个训练epoch")

        # === Step 5: 保存元信息 ===
        with open(os.path.join(CACHE_DIR, "batch_metadata.pkl"), "wb") as f:
            pickle.dump(batch_metadata, f)
        with open(os.path.join(CACHE_DIR, "file_metadata.pkl"), "wb") as f:
            pickle.dump(file_metadata, f)

        print("\n✅ 预处理完成:")
        print(f"总有效文件: {total_valid_files}")
        print(f"测试批次: {test_batch_idx}, 训练批次: {train_batch_idx}, 总计: {total_batches}")
        return os.path.join(CACHE_DIR, "batch_metadata.pkl")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("请使用 BatchLoadedSleepEDFDataset 加载数据")


# ===================================================================
# 训练时加载 batch 数据集
# ===================================================================
class BatchLoadedSleepEDFDataset(Dataset):
    def __init__(self, batch_metadata=None, batch_metadata_path=None, batch_size=1, 
                 use_augmentation=False, augmentation_config=None):
        if batch_metadata is not None:
            # 直接使用传入的批次元数据字典的列表
            self.batch_metadata = batch_metadata
        elif batch_metadata_path is not None:
            # 从文件加载元数据
            with open(batch_metadata_path, "rb") as f:
                self.batch_metadata = pickle.load(f)
        else:
            raise ValueError("必须提供batch_metadata或batch_metadata_path参数")
        
        self.batch_size = batch_size  # 用于拼接的批次数量
        self.use_augmentation = use_augmentation  # 是否使用数据增强
        
        # 初始化数据增强器
        if self.use_augmentation:
            if augmentation_config is None:
                # 使用默认配置
                self.augmenter = EEGDataAugmenter()
            else:
                # 使用自定义配置
                self.augmenter = EEGDataAugmenter(**augmentation_config)
            print("数据增强已启用")
        else:
            self.augmenter = None
            print("数据增强已禁用")

        # 打印加载信息
        patient_ids = set(batch.get("patient_id", "未知") for batch in self.batch_metadata)
        print(f"批次数据集加载完成:")
        print(f"  共 {len(self.batch_metadata)} 个批次")
        print(f"  包含患者: {', '.join(sorted(patient_ids))}")
        print(f"  设置批次拼接大小: {self.batch_size}")

    def __len__(self):
        # 返回可以组成的批次组数量
        return len(self.batch_metadata)

    def __getitem__(self, idx):
        # 直接获取单个批次的数据
        meta = self.batch_metadata[idx]
        
        # 直接使用元数据中记录的路径（已区分train/test文件夹）
        data_path = meta["data_path"]
        label_path = meta["label_path"]
        
        data = np.load(data_path)    # shape: [C, L, T]
        labels = np.load(label_path) # shape: [L]

        tensor_data = torch.tensor(data, dtype=torch.float32)
        tensor_labels = torch.tensor(labels, dtype=torch.long)

        return {
            "data": tensor_data,   # [C, L, T]
            "label": tensor_labels # [L]
        }


# ===================================================================
# 测试入口
# ===================================================================
if __name__ == "__main__":
    augmentation_config = {
        "sfreq": 100,
        "T_h": 6,
        "T_l": 6,
        "window_size": 6,
        "insert_counts": {
            1: 0,   # 对连续的N1分段插入2次增强序列+复制
            2: 0,   # 对连续的N2分段插入2次增强序列+复制
            3: 0,   # 对连续的N3分段插入2次增强序列+复制
            4: 0    # 对连续的REM分段插入1次增强序列+复制
        }
    }
    
    dataset = SleepEDFDataset(
        data_path="sleepedf", L=5
    )
    batch_metadata_path = dataset.preprocess_to_batch_files(
        batch_size=1,
        augmentation_config=augmentation_config  # 传入增强配置
    )
