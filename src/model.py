import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        return x

class Model(nn.Module):
    def __init__(self, H, W, config):
        super(Model, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * (H // 8) * (W // 8), config.RNN_size)
        self.embedding = nn.Embedding(config.vocabulary_size, config.embedding_size)
        self.gru = nn.GRU(config.embedding_size, config.RNN_size, batch_first=True)
        self.dropout = nn.Dropout(config.drop_out)
        self.fc_out = nn.Linear(config.RNN_size, config.vocabulary_size)

    def forward(self, img, x_context):
        img_f = self.feature_extractor(img)
        img_f = self.flatten(img_f)
        img_f = F.relu(self.fc(img_f))
        
        x_seq_embedding = self.embedding(x_context)
        h_t, _ = self.gru(x_seq_embedding, img_f.unsqueeze(0))
        h_t_dropped = self.dropout(h_t)
        predictions = F.softmax(self.fc_out(h_t_dropped), dim=-1)
        return predictions