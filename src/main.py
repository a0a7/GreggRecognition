import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
import torch.nn.utils as utils
from model import Model
from dataloader import ShorthandGenerationDataset, data_split
from hf_dataloader import create_hf_dataloaders, collate_fn_hf
from config import CONFIG
from utils.training_tracker import TrainingTracker
from tqdm import tqdm
import os

def collate_fn(batch):
    imgs, labels, additional = zip(*batch)
    imgs = pad_sequence(imgs, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    additional = torch.stack(additional)
    return imgs, labels, additional

def evaluate_model(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels, targets in test_loader:
            imgs, labels, targets = imgs.to(device), labels.to(device), targets.to(device)
            outputs = model(imgs, labels)
            
            # Handle output shape properly
            if outputs.dim() == 3:
                outputs = outputs[:, -1, :]
            elif outputs.dim() == 2:
                pass
            else:
                outputs = outputs.view(outputs.size(0), -1)
            
            if targets.dim() > 1:
                targets = targets.view(-1)
                
            test_loss += criterion(outputs, targets).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    test_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return test_loss, accuracy

config = CONFIG()

if config.dataset_source == 'huggingface':
    print("Using Hugging Face dataset...")
    train_dataset, val_dataset, test_dataset = create_hf_dataloaders(config)
    max_H, max_W = 256, 256
    collate_fn = collate_fn_hf
else:
    print("Using local dataset...")
    train_files, val_files, test_files, max_H, max_W, max_seq_length = data_split()
    train_dataset = ShorthandGenerationDataset(train_files, max_H, max_W, aug_types=9, max_label_leng=max_seq_length, channels=1)
    val_dataset = ShorthandGenerationDataset(val_files, max_H, max_W, aug_types=1, max_label_leng=max_seq_length, channels=1)
    test_dataset = ShorthandGenerationDataset(test_files, max_H, max_W, aug_types=1, max_label_leng=max_seq_length, channels=1)
    
    def collate_fn(batch):
        imgs, labels, additional = zip(*batch)
        imgs = pad_sequence(imgs, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=0)
        additional = torch.stack(additional)
        return imgs, labels, additional

tracker = TrainingTracker()

# Dynamically adjust pin_memory based on GPU availability
actual_pin_memory = config.pin_memory and torch.cuda.is_available()

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, 
                         num_workers=config.num_workers, pin_memory=actual_pin_memory,
                         prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None, 
                         persistent_workers=config.persistent_workers and config.num_workers > 0)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
                       num_workers=config.num_workers, pin_memory=actual_pin_memory,
                       prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
                       persistent_workers=config.persistent_workers and config.num_workers > 0)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn,
                        num_workers=config.num_workers, pin_memory=actual_pin_memory,
                        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
                        persistent_workers=config.persistent_workers and config.num_workers > 0)

model = Model(max_H, max_W, config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Debug GPU information
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    print(f"CUDA Device Name: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("CUDA not available - using CPU")

# Dynamically adjust pin_memory based on GPU availability
actual_pin_memory = config.pin_memory and torch.cuda.is_available()
print(f"Using pin_memory: {actual_pin_memory}")

print(f"Using device: {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Mixed precision scaler
scaler = GradScaler() if config.use_mixed_precision and device.type == 'cuda' else None
if scaler:
    print("Using mixed precision training")

os.makedirs('models', exist_ok=True)
best_val_loss = float('inf')

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    for imgs, labels, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs, labels, targets = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if scaler:
            with autocast():
                outputs = model(imgs, labels)
                
                # Handle output shape properly
                if outputs.dim() == 3:
                    outputs = outputs[:, -1, :]
                elif outputs.dim() == 2:
                    pass
                else:
                    outputs = outputs.view(outputs.size(0), -1)
                
                if targets.dim() > 1:
                    targets = targets.view(-1)
                    
                loss = criterion(outputs, targets)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            outputs = model(imgs, labels)
            
            # Handle output shape properly
            if outputs.dim() == 3:
                outputs = outputs[:, -1, :]
            elif outputs.dim() == 2:
                pass
            else:
                outputs = outputs.view(outputs.size(0), -1)
            
            if targets.dim() > 1:
                targets = targets.view(-1)
                
            loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping
            utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_train_loss = epoch_loss / num_batches
    
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, targets in val_loader:
            imgs, labels, targets = imgs.to(device), labels.to(device), targets.to(device)
            outputs = model(imgs, labels)
            
            # Handle output shape properly
            if outputs.dim() == 3:
                outputs = outputs[:, -1, :]
            elif outputs.dim() == 2:
                pass
            else:
                outputs = outputs.view(outputs.size(0), -1)
            
            if targets.dim() > 1:
                targets = targets.view(-1)
                
            val_loss += criterion(outputs, targets).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    tracker.update(epoch+1, avg_train_loss, val_loss, accuracy)
    
    scheduler.step()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f'New best model saved with loss: {val_loss:.4f}')

print('Training completed!')
tracker.plot_metrics()
test_loss, test_accuracy = evaluate_model(model, test_loader, device, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')