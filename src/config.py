import os 

class CONFIG:
    vocabulary_size = 28
    embedding_size = 256
    RNN_size = 512
    drop_out = 0.5
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    val_proportion = '0.1'
    
    learning_rate = 0.001
    batch_size = 32
    weight_decay = 1e-5
    gradient_clip = 1.0
    
    use_mixed_precision = True  # Use automatic mixed precision for faster training
    num_workers = 4  # Number of data loading workers
    pin_memory = True  # Pin memory for faster GPU transfer
    compile_model = True  # Use torch.compile for faster inference (PyTorch 2.0+)
    
    dataset_source = 'local'  # local or huggingface
    hf_dataset_name = 'a0a7/Gregg-1916'