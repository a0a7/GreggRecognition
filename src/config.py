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
    
    use_mixed_precision = True 
    num_workers = 0 if os.name == 'nt' else 4 
    pin_memory = True  # Will be disabled automatically if no GPU
    compile_model = True  # torch.compile 
    prefetch_factor = 2  # how many batches to prefetch
    persistent_workers = False  # enable if not on windows
    
    dataset_source = 'local'  # local or huggingface
    hf_dataset_name = 'a0a7/Gregg-1916'