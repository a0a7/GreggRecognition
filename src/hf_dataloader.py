"""
Hugging Face dataset loader for Gregg-1916 dataset
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
from PIL import Image
import io
from config import CONFIG

class GreggHFDataset(Dataset):
    def __init__(self, split='train', max_H=256, max_W=256, max_label_leng=100, aug_types=1, channels=1):
        self.dataset = load_dataset("a0a7/Gregg-1916", split=split)
        self.H, self.W = max_H, max_W
        self.channels = channels
        self.vocabulary = 'abcdefghijklmnopqrstuvwxyz+#'
        self.dict_c2i = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.max_label_length = max_label_leng
        self.max_context_length = self.max_label_length - 1
        self.aug_types = aug_types
        
        # Build instance indices for variable-length context training
        self.instance_indices = []
        
        for idx, item in enumerate(self.dataset):
            label = item['label']  # Assuming the dataset has a 'label' field
            seq = '+' + label + '#'
            max_context_len = min(len(seq) - 1, self.max_context_length)
            
            for length in range(1, max_context_len + 1):
                for aug in range(self.aug_types):
                    self.instance_indices.append({
                        'dataset_idx': idx,
                        'seq': seq,
                        'aug_type': aug,
                        'context_length': length
                    })
    
    def __len__(self):
        return len(self.instance_indices)
    
    def __getitem__(self, idx):
        instance = self.instance_indices[idx]
        dataset_idx = instance['dataset_idx']
        seq = instance['seq']
        aug_type = instance['aug_type']
        context_length = instance['context_length']
        
        # Get image from HF dataset
        item = self.dataset[dataset_idx]
        image = item['image']  # Assuming the dataset has an 'image' field
        
        # Convert PIL image to numpy array and preprocess
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # Convert to grayscale
        if len(image.shape) == 3:
            image = self.rgb2grey(image)
        
        # Apply augmentation
        img_processed = self.augment_image(image, aug_type)
        img_processed = np.expand_dims(img_processed, axis=0)  # Add channel dimension
        
        # Prepare context and target
        x_context = np.array([self.dict_c2i[char] for char in seq[:context_length]])
        y = self.dict_c2i[seq[context_length]]
        
        return torch.tensor(img_processed, dtype=torch.float32), torch.tensor(x_context, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
    def rgb2grey(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    
    def augment_image(self, image, aug_type):
        """Simple augmentation - can be expanded"""
        image_augmented = np.ones((self.H, self.W))
        h, w = np.shape(image)
        
        # Simple padding augmentation
        stride_0 = max(0, self.H - h)
        stride_1 = max(0, self.W - w)
        
        offset_0 = (aug_type % 3) * (stride_0 // 3) if stride_0 > 0 else 0
        offset_1 = (aug_type % 3) * (stride_1 // 3) if stride_1 > 0 else 0
        
        end_0 = min(offset_0 + h, self.H)
        end_1 = min(offset_1 + w, self.W)
        
        image_augmented[offset_0:end_0, offset_1:end_1] = image[:end_0-offset_0, :end_1-offset_1]
        
        return image_augmented

def create_hf_dataloaders(config):
    """Create data loaders using Hugging Face dataset"""
    
    # Create datasets
    train_dataset = GreggHFDataset(
        split='train',
        max_H=256, max_W=256, 
        aug_types=9, 
        max_label_leng=100, 
        channels=1
    )
    
    # Check if validation split exists, otherwise use a portion of train
    try:
        val_dataset = GreggHFDataset(
            split='validation',
            max_H=256, max_W=256,
            aug_types=1,
            max_label_leng=100,
            channels=1
        )
    except:
        # If no validation split, create one from train
        print("No validation split found, using 10% of training data for validation")
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Check if test split exists
    try:
        test_dataset = GreggHFDataset(
            split='test',
            max_H=256, max_W=256,
            aug_types=1,
            max_label_leng=100,
            channels=1
        )
    except:
        # If no test split, use validation as test
        print("No test split found, using validation split for testing")
        test_dataset = val_dataset
    
    return train_dataset, val_dataset, test_dataset

def collate_fn_hf(batch):
    """Collate function for HF dataset"""
    from torch.nn.utils.rnn import pad_sequence
    
    imgs, labels, targets = zip(*batch)
    imgs = pad_sequence(imgs, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return imgs, labels, targets
