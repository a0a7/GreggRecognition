# GreggRecognition

This repo houses GreggRecognition, a deep learning solution to handwritten text recognition of Gregg shorthand made with Pytorch. It was developed in the span of 8 months as a part of the Minnetonka Research program and is intended for use in research environments.

## architecture

GreggRecognition uses a two-stage approach:

1. **Feature Extraction**: An 8-layer convolutional neural network with batch normalization and max pooling operations extracts visual features from shorthand images
2. **Sequence Generation**: A Gated Recurrent Unit decoder with embedding layers generates character predictions conditioned on extracted visual features and contextual information

## features

### training/model quality
- nine data augmentation strategies including rotation, scaling, and spatial transformations to improve model robustness, given that existing Gregg shorthand datasets are fairly limited in size
- support for variable-length sequence training with character-level prediction objectives 

### metrics
- Character-level and word-level accuracy metrics, edit distance calculations, and confusion matrix analysis
- tracking of training metrics with automated plot generation using matplotlib. Also configurable checkpointing

## !! compatibility / extensibility !!

Designed for use with the Gregg-1916 dataset introduced by Zhai et al. (2018), though extensible to other shorthand recognition tasks. The system expects image files with corresponding character labels embedded in filenames.

## requirements

- Python 3.7+ 
- supports CUDA optionally

## Usage

### Initial Setup

1. **dependencies**:
   ```bash
   cd src
   pip install -r requirements.txt
   ```

2. **dataset**: Place shorthand images in `src/data/` directory. Images should be named with their corresponding text labels. This model was made for training on the [Gregg-1916 dataset](https://github.com/anonimously/Gregg1916-Recognition/blob/master/data)

### training

```bash
cd src
python main.py
```

The training process will generate training plots showing loss and accuracy curves. It will also save the best model to `models/best_model.pth` and evaluate it on the test set

### config

- `learning_rate`: Initial learning rate (default: varies by config)
- `batch_size`: Batch size for training (default: varies by config)
- `vocabulary_size`: Size of character vocabulary
- `embedding_size`: Dimension of character embeddings
- `RNN_size`: Hidden size of GRU layers
- `drop_out`: Dropout rate for regularization

### using trained model

To load and use a trained model for inference:
```python
import torch
from model import Model
from config import CONFIG

# Load configuration and model
config = CONFIG()
model = Model(max_H=256, max_W=256, config=config)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Use model for prediction on new images
# (see evaluation.py for detailed inference examples)
```