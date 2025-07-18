# GreggRecognition

A novel deep learning approach to handwritten text recognition of Gregg shorthand, developed using the PyTorch framework and trained on the Gregg-1916 dataset comprising greater than 16,000 Gregg shorthand words.

## Overview

GreggRecognition implements a CNN-GRU hybrid architecture for the optical character recognition of Gregg shorthand manuscripts. The system addresses the unique challenges presented by shorthand scripts, including the lack of distinct character features and simplified lexicons that often omit vowels or other defining word characteristics.

## Architecture

The model employs a two-stage approach:

1. **Feature Extraction**: An 8-layer convolutional neural network with batch normalization and max pooling operations extracts visual features from shorthand images
2. **Sequence Generation**: A Gated Recurrent Unit (GRU) decoder with embedding layers generates character predictions conditioned on extracted visual features and contextual information

## Key Features

- **Multi-scale Data Augmentation**: Implements nine augmentation strategies including rotation, scaling, and spatial transformations to improve model robustness
- **Progressive Training**: Supports variable-length sequence training with character-level prediction objectives  
- **Comprehensive Evaluation**: Provides character-level and word-level accuracy metrics, edit distance calculations, and confusion matrix analysis
- **Training Visualization**: Real-time tracking of training metrics with automated plot generation
- **Model Persistence**: Automatic saving of best-performing models with configurable checkpointing

## Dataset Compatibility

Designed for use with the Gregg-1916 dataset introduced by Zhai et al. (2018), though extensible to other shorthand recognition tasks. The system expects image files with corresponding character labels embedded in filenames.

## Applications

This tool serves the preservation of shorthand documents with historical value across domains where time-efficient handwriting was essential, including legal, medical, and administrative records. The digitization of such manuscripts presents benefits for historical and cultural preservation while advancing the development of extensible handwritten text recognition systems.

## Technical Requirements

- PyTorch framework with CUDA support (optional)
- Python 3.7+ with standard scientific computing libraries
- Compatible with both CPU and GPU training environments
