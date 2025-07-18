import os 

class CONFIG:
    vocabulary_size = 28
    embedding_size = 256
    RNN_size = 512
    drop_out = 0.5
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    val_proportion = '0.1'
    
    # Additional training parameters
    learning_rate = 0.001
    batch_size = 32
    weight_decay = 1e-5
    gradient_clip = 1.0
    
    dataset_source = 'huggingface'  # local or huggingface
    hf_dataset_name = 'a0a7/Gregg-1916'