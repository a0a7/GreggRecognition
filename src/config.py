import os 

class CONFIG:
    vocabulary_size = 10000
    embedding_size = 256
    RNN_size = 512
    drop_out = 0.5
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    val_proportion = '0.1'