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