import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import Model
from dataloader import ShorthandGenerationDataset, data_split
from config import CONFIG
from tqdm import tqdm
import os

def evaluate_model(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels, targets in test_loader:
            imgs, labels, targets = imgs.to(device), labels.to(device), targets.to(device)
            outputs = model(imgs, labels)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            test_loss += criterion(outputs_flat, targets.view(-1)).item()
            
            _, predicted = torch.max(outputs_flat.data, 1)
            total += targets.view(-1).size(0)
            correct += (predicted == targets.view(-1)).sum().item()
    
    test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return test_loss, accuracy

def collate_fn(batch):
    imgs, labels, additional = zip(*batch)
    imgs = pad_sequence(imgs, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    additional = torch.stack(additional)
    return imgs, labels, additional

config = CONFIG()
train_files, val_files, test_files, max_H, max_W, max_seq_length = data_split()

train_dataset = ShorthandGenerationDataset(train_files, max_H, max_W, aug_types=9, max_label_leng=max_seq_length, channels=1)
val_dataset = ShorthandGenerationDataset(val_files, max_H, max_W, aug_types=1, max_label_leng=max_seq_length, channels=1)
test_dataset = ShorthandGenerationDataset(test_files, max_H, max_W, aug_types=1, max_label_leng=max_seq_length, channels=1)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

model = Model(max_H, max_W, config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

os.makedirs('models', exist_ok=True)
best_val_loss = float('inf')

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for imgs, labels, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs, labels, targets = imgs.to(device), labels.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs, labels)
        
        outputs_flat = outputs.view(-1, outputs.size(-1))
        loss = criterion(outputs_flat, targets.view(-1))
        
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, targets in val_loader:
            imgs, labels, targets = imgs.to(device), labels.to(device), targets.to(device)
            outputs = model(imgs, labels)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            val_loss += criterion(outputs_flat, targets.view(-1)).item()
            
            _, predicted = torch.max(outputs_flat.data, 1)
            total += targets.view(-1).size(0)
            correct += (predicted == targets.view(-1)).sum().item()
    
    val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f'New best model saved with loss: {val_loss:.4f}')

print('Training completed!')
test_loss, test_accuracy = evaluate_model(model, test_loader, device, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')