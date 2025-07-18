import matplotlib.pyplot as plt
import os

class TrainingTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.epochs = []
    
    def update(self, epoch, train_loss, val_loss, val_accuracy):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_accuracy)
    
    def plot_metrics(self, save_path='training_plots'):
        os.makedirs(save_path, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.epochs, self.val_accuracies, 'g-', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/training_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()
