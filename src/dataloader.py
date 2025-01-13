import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from config import CONFIG

def rgb2grey(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

import os
import numpy as np
from config import CONFIG

def data_split():
    config = CONFIG()
    folder = config.data_folder
    val_proportion = float(config.val_proportion)  # Convert to float

    files = os.listdir(folder)
    period = int(np.round(1 / val_proportion))

    # Split  here
    train_files = files[::period]
    val_files = files[1::period]
    test_files = files[2::period]
    max_H, max_W, max_seq_length = 256, 256, 100  # Example values

    return train_files, val_files, test_files, max_H, max_W, max_seq_length

def augmentation_simple(filename, aug_type, max_H, max_W, folder=CONFIG.data_folder):
    image = rgb2grey(mpimg.imread(os.path.join(folder, filename)))
    image_augmented = np.ones((max_H, max_W))
    h, w = np.shape(image)
    stride_0, stride_1 = max_H - h, (max_W - w) // 2
    offset = ((aug_type % 2) * stride_0, (aug_type % 3) * stride_1)
    image_augmented[offset[0]: h + offset[0], offset[1]: w + offset[1]] = image
    return image_augmented

def augmentation_nine(filename, aug_type, max_H, max_W, folder=CONFIG.data_folder):
    image_augmented = np.ones((max_H, max_W))
    image = Image.open(os.path.join(folder, filename)).convert('RGB')
    w_ori, h_ori = image.size
    rotate_ind = aug_type % 3
    scale_ind = aug_type // 3

    image = ImageOps.invert(image)
    if rotate_ind == 1:
        image = image.rotate(2, expand=True)
    elif rotate_ind == 2:
        image = image.rotate(-2, expand=True)
    image = ImageOps.invert(image)

    h, w = image.size
    if scale_ind == 1:
        h, w = int(np.floor(h * 0.98)), int(np.floor(w * 0.98))
        image = image.resize((h, w))
    elif scale_ind == 2:
        h, w = int(np.floor(h * 0.96)), int(np.floor(w * 0.96))
        image = image.resize((h, w))

    image = rgb2grey(np.array(image) / 255)
    h, w = np.shape(image)
    stride_0, stride_1 = (max_H - 10 - h_ori) // 2, (max_W - 10 - w_ori) // 2
    offset = ((aug_type % 3) * stride_0, (aug_type % 3) * stride_1)
    try:
        image_augmented[offset[0]: h + offset[0], offset[1]: w + offset[1]] = image
    except ValueError:
        print(filename)
    return image_augmented

class ShorthandGenerationDataset(Dataset):
    def __init__(self, file_list, max_H, max_W, max_label_leng, aug_types, channels=1):
        self.file_list = file_list
        self.H, self.W = max_H, max_W
        self.channels = channels
        self.vocabulary = 'abcdefghijklmnopqrstuvwxyz+#'
        self.dict_c2i = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.max_label_length = max_label_leng
        self.max_context_length = self.max_label_length - 1
        self.aug_types = aug_types
        self.instance_indices_by_length = {i: [] for i in range(1, self.max_context_length + 1)}

        for file in file_list:
            seq = '+' + file[:-4] + '#'
            max_context_len = len(seq) - 1
            for length in range(1, max_context_len + 1):
                for aug in range(self.aug_types):
                    self.instance_indices_by_length[length].append([seq, aug, length])

        self.total_size = sum(len(self.instance_indices_by_length[i]) for i in range(1, self.max_context_length))

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        context_length = 1
        while sum(len(self.instance_indices_by_length[length]) for length in range(1, context_length + 1)) <= idx:
            context_length += 1

        num_batch_in_length = idx - sum(len(self.instance_indices_by_length[length]) for length in range(1, context_length))
        starting_index = num_batch_in_length

        seq, augmentation_type, instance_context_length = self.instance_indices_by_length[context_length][starting_index]

        file_name = seq[1:-1] + '.png'
        img = augmentation_nine(file_name, augmentation_type, self.H, self.W)
        img = np.expand_dims(img, axis=0)  # Add channel dimension

        x_context = np.array([self.dict_c2i[char] for char in seq[:instance_context_length]])
        y = self.dict_c2i[seq[instance_context_length]]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(x_context, dtype=torch.long), torch.tensor(y, dtype=torch.long)