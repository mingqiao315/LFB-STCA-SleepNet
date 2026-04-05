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

T_h = 6  
T_l = 3
window_size = 6 

CACHE_DIR = "dataset_output/cache"
BATCH_DIR = os.path.join(CACHE_DIR, "batches")
TEST_BATCH_DIR = os.path.join(CACHE_DIR, "test_batches")
os.makedirs(TEST_BATCH_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)

annotation_desc_2_event_id = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  
    "Sleep stage R": 4,
}

class EEGDataAugmenter:
    def __init__(self, sfreq=100, T_h=6, T_l=3, window_size=6, insert_counts=None):
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
        n_insertions = {}  
        
        for i in range(len(labels)):
            current_label = labels[i]
            new_data.append(data[:, i, :])
            new_labels.append(current_label)
            
            if current_label in self.insert_counts:
                max_inserts = self.insert_counts[current_label]
                if current_label == prev_label:
                    count = n_insertions.get(current_label, 0) + 1
                else:
                    count = 1
                    n_insertions[current_label] = count
                while count <= max_inserts:
                    augmented_segment = self.__call__(data[:, i:i+1, :])[0]
                    new_data.append(augmented_segment)
                    new_labels.append(current_label)
                    
                    new_data.append(data[:, i, :])
                    new_labels.append(current_label)
                    
                    count += 1
            
            prev_label = current_label
        
        new_data = np.concatenate(new_data, axis=1)
        new_labels = np.array(new_labels)
        return new_data, new_labels

    def _pulse_density_transform(self, signal_data):
        pulses = (signal_data > 0.00005).astype(float) 
        window_samples = int(self.window_size * self.sfreq)
        density = np.convolve(pulses, np.ones(window_samples), mode='valid')
        return density

    def _adaptive_sliding_window_single_channel(self, channel_data):
        segments = []
        current_pos = 0
        signal_length = len(channel_data)
        density = self._pulse_density_transform(channel_data)
        
        while current_pos < signal_length:
            density_idx = min(current_pos, len(density) - 1)
            current_density = density[density_idx]
            
            if current_density > self.T_h: 
                segment_duration = int(2 * self.sfreq)
            elif current_density < self.T_l: 
                segment_duration = int(4 * self.sfreq)
            else:  
                segment_duration = int(3 * self.sfreq)
            
            end_pos = min(current_pos + segment_duration, signal_length)
            is_last_segment = (end_pos == signal_length)
            
            if (end_pos - current_pos > segment_duration) or is_last_segment:
                segment = channel_data[current_pos:end_pos]
                segments.append(segment)
            
            current_pos = end_pos 
        
        return segments

    def __call__(self, data):
        C, L, T = data.shape
        augmented_data = []
        
        for l in range(L):
            epoch_data = data[:, l, :]  
            new_epoch_channels = []
            
            for ch in range(C):
                channel_data = epoch_data[ch] 
                segments = self._adaptive_sliding_window_single_channel(channel_data)
                
                if not segments:  
                    new_channel = channel_data
                else:
                    permuted_segments = np.random.permutation(segments)
                    new_channel = np.concatenate(permuted_segments, axis=0)
                    if len(new_channel) > T:
                        new_channel = new_channel[:T]
                    else:
                        new_channel = np.pad(
                            new_channel, 
                            (0, T - len(new_channel)), 
                            mode='constant'
                        )
                
                new_epoch_channels.append(new_channel)
            
            augmented_epoch = np.stack(new_epoch_channels)
            augmented_data.append(augmented_epoch)
        
        return np.stack(augmented_data, axis=1)
    
class SleepEDFDataset(Dataset):
    def __init__(self, data_path, L=5):
        self.data_path = data_path
        self.L = L
        self.file_list = self._get_file_list()
        self.samples = []  
        self.file_metadata = []
        self.T_h = T_h 
        self.T_l = T_l  
        self.sfreq = 100  
        self.augmenter = EEGDataAugmenter(**augmentation_config) if augmentation_config else None

    def augmenter_single_epoch(self, epoch_data):
        C, T = epoch_data.shape
        epoch_data_expanded = epoch_data.reshape(1, C, T)
        epoch_data_expanded = np.transpose(epoch_data_expanded, (1, 0, 2))  # [C, 1, T]
        augmented = self.augmenter(epoch_data_expanded) 
        augmented_epoch = augmented[:, 0, :]  

        if augmented_epoch.shape[1] > T:
            augmented_epoch = augmented_epoch[:, :T]
        elif augmented_epoch.shape[1] < T:
            pad_width = T - augmented_epoch.shape[1]
            augmented_epoch = np.pad(
                augmented_epoch, ((0, 0), (0, pad_width)), mode='constant'
            )

        return augmented_epoch  # [C, T]

    def _get_file_list(self):
        files = []

        for psg_file in glob.glob(os.path.join(self.data_path, "*-PSG.edf")):
            fname = os.path.basename(psg_file)
            m = re.match(r"(SC\d{4})", fname)
            sc_id = m.group(1)

            hyp_candidates = glob.glob(os.path.join(self.data_path, sc_id + "*-Hypnogram.edf"))
            if len(hyp_candidates) > 0:
                hyp_file = hyp_candidates[0]  
                files.append((psg_file, hyp_file))
            else:
                print(f"warning: {fname} no match Hypnogram file")
        return files

    def _zscore_normalize_epoch(self, epoch_data, epoch_idx=None, file_name=None):
        normalized_data = np.zeros_like(epoch_data)
        channel_means = []
        channel_stds = []

        for i in range(epoch_data.shape[0]):
            channel_data = epoch_data[i]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std == 0:
                std = 1.0
            normalized_data[i] = (channel_data - mean) / std
            channel_means.append(mean)
            channel_stds.append(std)
        return normalized_data, channel_means, channel_stds
    
    def _pulse_density_transform(self, signal_data, window_size=6):
        pulses = (signal_data > 0.00005).astype(float)  
        window_samples = int(window_size * self.sfreq)
        density = np.convolve(pulses, np.ones(window_samples), mode='valid')
        return density
        
    def _adaptive_sliding_window_single_channel(self, channel_data):
        segments = []
        current_pos = 0
        signal_length = len(channel_data)
        
        density = self._pulse_density_transform(channel_data)
        
        while current_pos < signal_length:
            density_idx = min(current_pos, len(density) - 1)
            current_density = density[density_idx]
            
            if current_density > self.T_h:  
                segment_duration = int(2 * self.sfreq)
            elif current_density < self.T_l:  
                segment_duration = int(4 * self.sfreq)
            else:  
                segment_duration = int(3 * self.sfreq)
            
            end_pos = min(current_pos + segment_duration, signal_length)
            is_last_segment = (end_pos == signal_length)
            
            if (end_pos - current_pos > segment_duration) or is_last_segment:
                segment = channel_data[current_pos:end_pos]
                segments.append(segment)
            
            current_pos = end_pos
        
        return segments

    def _random_permute_windows(self, windows):
        permuted_indices = np.random.permutation(len(windows))
        permuted_windows = windows[permuted_indices]
        return np.concatenate(permuted_windows, axis=1)

    def _segment_epochs(self, psg_file, hyp_file):
        try:
            psg_basename = os.path.basename(psg_file)
            patient_id = re.match(r"(SC\d{4})", psg_basename).group(1)

            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
            annots = mne.read_annotations(hyp_file)
            raw.set_annotations(annots)

            events, _ = mne.events_from_annotations(
                raw,
                event_id=annotation_desc_2_event_id,
                chunk_duration=30.0,
                verbose='ERROR'
            )

            self.sfreq = int(raw.info["sfreq"])  
            self.T = self.sfreq * 30 
            epoch_length = self.T
            n_channels = raw.info["nchan"]
            data = raw.get_data()  # shape: [n_channels, n_samples]

            epochs_data = []      
            epochs_labels = []    

            valid_events = 0

            for event_idx, (onset, _, stage) in enumerate(events):
                start = onset
                end = start + epoch_length
                if end <= data.shape[1] and stage in annotation_desc_2_event_id.values():
                    segment = data[:, start:end]
                    normalized_segment, means, stds = self._zscore_normalize_epoch(segment)
                    epochs_data.append(normalized_segment)
                    epochs_labels.append(stage)
                    valid_events += 1

            if len(epochs_labels) == 0:
                return None, None, None  

            original_data = np.stack(epochs_data)  # shape: [n_epochs, n_channels, T]
            original_labels = np.array(epochs_labels, dtype=np.int64)  # shape: [n_epochs]

            return original_data, original_labels, None  

        except Exception as e:
            print(f"❌ Failed to load or process file {os.path.basename(psg_file)}: {str(e)}")
            return None, None, None

    def preprocess_to_batch_files(self, batch_size=32, augmentation_config=None):
        BATCH_DIR = os.path.join(CACHE_DIR, "batches")          
        TEST_BATCH_DIR = os.path.join(CACHE_DIR, "test_batches")  
        os.makedirs(BATCH_DIR, exist_ok=True)
        os.makedirs(TEST_BATCH_DIR, exist_ok=True)

        batch_metadata, file_metadata = [], []
        test_batch_idx, train_batch_idx = 0, 0
        total_valid_files = total_batches = 0

        augmenter = EEGDataAugmenter(**(augmentation_config or {})) if augmentation_config else None

        insert_counts = augmentation_config["insert_counts"]

        for file_idx, (psg_file, hyp_file) in enumerate(self.file_list):
            psg_basename = os.path.basename(psg_file)
            patient_id = re.match(r"(SC\d{4})", psg_basename).group(1)

            original_data, original_labels, _ = self._segment_epochs(psg_file, hyp_file)
            if original_data is None or original_labels is None:
                file_metadata.append({
                    "file_idx": file_idx,
                    "psg_file": psg_basename,
                    "patient_id": patient_id,
                    "valid": False,
                    "reason": "Parsing failed or no valid epoch exists."
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

            augmented_data, augmented_labels = [], []

            for idx, (epoch_data, label) in enumerate(zip(original_data, original_labels)):
                augmented_data.append(epoch_data)
                augmented_labels.append(label)

                if label in insert_counts:
                    n_insert = insert_counts[label]
                    for _ in range(n_insert):
                        enhanced_epoch = self.augmenter_single_epoch(epoch_data)
                        augmented_data.extend([epoch_data, enhanced_epoch])
                        augmented_labels.extend([label, label])

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

        with open(os.path.join(CACHE_DIR, "batch_metadata.pkl"), "wb") as f:
            pickle.dump(batch_metadata, f)
        with open(os.path.join(CACHE_DIR, "file_metadata.pkl"), "wb") as f:
            pickle.dump(file_metadata, f)

        print("\n✅ Preprocessing completed:")
        print(f"Total valid files: {total_valid_files}")
        print(f"Test batches: {test_batch_idx}, Training batches: {train_batch_idx}, Total: {total_batches}")
        return os.path.join(CACHE_DIR, "batch_metadata.pkl")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("Please use BatchLoadedSleepEDFDataset to load data")


class BatchLoadedSleepEDFDataset(Dataset):
    def __init__(self, batch_metadata=None, batch_metadata_path=None, batch_size=1, 
                 use_augmentation=False, augmentation_config=None):
        if batch_metadata is not None:
            self.batch_metadata = batch_metadata
        elif batch_metadata_path is not None:
            with open(batch_metadata_path, "rb") as f:
                self.batch_metadata = pickle.load(f)
        else:
            raise ValueError("Either batch_metadata or batch_metadata_path parameter must be provided")
        
        self.batch_size = batch_size  
        self.use_augmentation = use_augmentation
        
        if self.use_augmentation:
            if augmentation_config is None:
                self.augmenter = EEGDataAugmenter()
            else:
                self.augmenter = EEGDataAugmenter(**augmentation_config)
        else:
            self.augmenter = None

        patient_ids = set(batch.get("patient_id", "未知") for batch in self.batch_metadata)

    def __len__(self):
        return len(self.batch_metadata)

    def __getitem__(self, idx):
        meta = self.batch_metadata[idx]
        
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


if __name__ == "__main__":
    augmentation_config = {
        "sfreq": 100,
        "T_h": 6,
        "T_l": 6,
        "window_size": 6,
        "insert_counts": {
            1: 0,   
            2: 0,   
            3: 0,  
            4: 0    
        }
    }
    
    dataset = SleepEDFDataset(
        data_path="sleepedf", L=5
    )
    batch_metadata_path = dataset.preprocess_to_batch_files(
        batch_size=1,
        augmentation_config=augmentation_config  
    )
