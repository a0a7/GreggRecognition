"""
Comprehensive evaluation module for GreggRecognition model
Provides character-level and word-level metrics, confusion matrices, and analysis tools
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
import editdistance
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, model, device, vocabulary='abcdefghijklmnopqrstuvwxyz+#'):
        self.model = model
        self.device = device
        self.vocabulary = vocabulary
        self.char_to_idx = {char: idx for idx, char in enumerate(vocabulary)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocabulary)}
        
    def evaluate_model(self, dataloader, max_samples=None):
        """
        Comprehensive model evaluation
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_predicted_sequences = []
        all_target_sequences = []
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (imgs, labels, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
                if max_samples and total_samples >= max_samples:
                    break
                    
                imgs, labels, targets = imgs.to(self.device), labels.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(imgs, labels)
                
                # Get predictions
                _, predicted = torch.max(outputs.data, -1)
                
                # Store character-level predictions and targets
                all_predictions.extend(predicted.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                
                # Generate sequence predictions for word-level evaluation
                batch_size = imgs.size(0)
                for i in range(batch_size):
                    # Convert label sequence to string
                    label_seq = labels[i].cpu().numpy()
                    target_char = targets[i].cpu().item()
                    pred_char = predicted[i].cpu().numpy()
                    
                    # Build target sequence
                    target_sequence = ''.join([self.idx_to_char[idx] for idx in label_seq if idx < len(self.vocabulary)])
                    target_sequence += self.idx_to_char[target_char]
                    
                    # Build predicted sequence
                    pred_sequence = ''.join([self.idx_to_char[idx] for idx in label_seq if idx < len(self.vocabulary)])
                    if len(pred_char.shape) > 0:
                        pred_sequence += self.idx_to_char[pred_char[0]]
                    else:
                        pred_sequence += self.idx_to_char[pred_char.item()]
                    
                    all_predicted_sequences.append(pred_sequence)
                    all_target_sequences.append(target_sequence)
                    
                total_samples += batch_size
        
        return self._compute_metrics(all_predictions, all_targets, 
                                   all_predicted_sequences, all_target_sequences)
    
    def _compute_metrics(self, predictions, targets, pred_sequences, target_sequences):
        """
        Compute comprehensive evaluation metrics
        """
        metrics = {}
        
        # Character-level accuracy
        char_accuracy = np.mean(np.array(predictions) == np.array(targets))
        metrics['character_accuracy'] = char_accuracy
        
        # Character-level precision, recall, F1
        char_report = classification_report(targets, predictions, 
                                          target_names=[self.idx_to_char[i] for i in range(len(self.vocabulary))],
                                          output_dict=True, zero_division=0)
        metrics['character_classification_report'] = char_report
        
        # Word-level metrics
        word_accuracy = np.mean([pred == target for pred, target in zip(pred_sequences, target_sequences)])
        metrics['word_accuracy'] = word_accuracy
        
        # Edit distance (Levenshtein distance)
        edit_distances = [editdistance.eval(pred, target) for pred, target in zip(pred_sequences, target_sequences)]
        metrics['average_edit_distance'] = np.mean(edit_distances)
        metrics['normalized_edit_distance'] = np.mean([
            dist / max(len(target), 1) for dist, target in zip(edit_distances, target_sequences)
        ])
        
        # Character frequency analysis
        char_freq_errors = defaultdict(int)
        char_freq_total = defaultdict(int)
        
        for pred, target in zip(predictions, targets):
            char_freq_total[target] += 1
            if pred != target:
                char_freq_errors[target] += 1
        
        char_error_rates = {char: char_freq_errors[char] / char_freq_total[char] 
                           for char in char_freq_total if char_freq_total[char] > 0}
        metrics['character_error_rates'] = char_error_rates
        
        # Confusion matrix data
        metrics['confusion_matrix'] = confusion_matrix(targets, predictions)
        metrics['predictions'] = predictions
        metrics['targets'] = targets
        
        return metrics
    
    def plot_confusion_matrix(self, confusion_mat, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_mat, 
                   xticklabels=list(self.vocabulary), 
                   yticklabels=list(self.vocabulary),
                   annot=True, fmt='d', cmap='Blues')
        plt.title('Character Classification Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_character_error_rates(self, char_error_rates, save_path='character_error_rates.png'):
        """
        Plot character-wise error rates
        """
        chars = list(char_error_rates.keys())
        error_rates = list(char_error_rates.values())
        
        plt.figure(figsize=(15, 6))
        bars = plt.bar(range(len(chars)), error_rates)
        plt.xlabel('Characters')
        plt.ylabel('Error Rate')
        plt.title('Character-wise Error Rates')
        plt.xticks(range(len(chars)), [self.idx_to_char[char] for char in chars])
        
        # Color bars based on error rate
        for i, bar in enumerate(bars):
            if error_rates[i] > 0.5:
                bar.set_color('red')
            elif error_rates[i] > 0.3:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, metrics, save_path='evaluation_report.txt'):
        """
        Generate a comprehensive evaluation report
        """
        report = []
        report.append("=" * 60)
        report.append("GREGG RECOGNITION MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL PERFORMANCE:")
        report.append(f"Character Accuracy: {metrics['character_accuracy']:.4f}")
        report.append(f"Word Accuracy: {metrics['word_accuracy']:.4f}")
        report.append(f"Average Edit Distance: {metrics['average_edit_distance']:.4f}")
        report.append(f"Normalized Edit Distance: {metrics['normalized_edit_distance']:.4f}")
        report.append("")
        
        # Character-level performance
        report.append("CHARACTER-LEVEL PERFORMANCE:")
        char_report = metrics['character_classification_report']
        
        # Macro averages
        if 'macro avg' in char_report:
            macro_avg = char_report['macro avg']
            report.append(f"Macro Average Precision: {macro_avg['precision']:.4f}")
            report.append(f"Macro Average Recall: {macro_avg['recall']:.4f}")
            report.append(f"Macro Average F1-Score: {macro_avg['f1-score']:.4f}")
        
        # Weighted averages
        if 'weighted avg' in char_report:
            weighted_avg = char_report['weighted avg']
            report.append(f"Weighted Average Precision: {weighted_avg['precision']:.4f}")
            report.append(f"Weighted Average Recall: {weighted_avg['recall']:.4f}")
            report.append(f"Weighted Average F1-Score: {weighted_avg['f1-score']:.4f}")
        
        report.append("")
        
        # Most problematic characters
        char_error_rates = metrics['character_error_rates']
        sorted_errors = sorted(char_error_rates.items(), key=lambda x: x[1], reverse=True)
        
        report.append("MOST PROBLEMATIC CHARACTERS:")
        for char_idx, error_rate in sorted_errors[:10]:
            char = self.idx_to_char[char_idx]
            report.append(f"'{char}': {error_rate:.4f} error rate")
        
        report.append("")
        report.append("=" * 60)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        # Print report
        print('\n'.join(report))
    
    def analyze_predictions(self, dataloader, num_samples=10):
        """
        Analyze specific predictions to understand model behavior
        """
        self.model.eval()
        
        samples_analyzed = 0
        
        with torch.no_grad():
            for imgs, labels, targets in dataloader:
                if samples_analyzed >= num_samples:
                    break
                    
                imgs, labels, targets = imgs.to(self.device), labels.to(self.device), targets.to(self.device)
                outputs = self.model(imgs, labels)
                
                # Get prediction probabilities
                probabilities = torch.softmax(outputs, dim=-1)
                _, predicted = torch.max(outputs.data, -1)
                
                batch_size = imgs.size(0)
                for i in range(min(batch_size, num_samples - samples_analyzed)):
                    # Convert sequences to readable format
                    label_seq = [self.idx_to_char[idx.item()] for idx in labels[i] if idx.item() < len(self.vocabulary)]
                    target_char = self.idx_to_char[targets[i].item()]
                    pred_char = self.idx_to_char[predicted[i].item() if predicted[i].dim() == 0 else predicted[i][0].item()]
                    
                    # Get confidence scores
                    if probabilities[i].dim() == 1:
                        confidence = probabilities[i][predicted[i]].item()
                        top_3_probs, top_3_indices = torch.topk(probabilities[i], 3)
                    else:
                        confidence = probabilities[i][0][predicted[i][0]].item()
                        top_3_probs, top_3_indices = torch.topk(probabilities[i][0], 3)
                    
                    print(f"\nSample {samples_analyzed + 1}:")
                    print(f"Context: {''.join(label_seq)}")
                    print(f"Target: '{target_char}'")
                    print(f"Predicted: '{pred_char}' (confidence: {confidence:.4f})")
                    print(f"Correct: {target_char == pred_char}")
                    
                    print("Top 3 predictions:")
                    for j, (prob, idx) in enumerate(zip(top_3_probs, top_3_indices)):
                        char = self.idx_to_char[idx.item()]
                        print(f"  {j+1}. '{char}': {prob.item():.4f}")
                    
                    samples_analyzed += 1
                    
                if samples_analyzed >= num_samples:
                    break

def evaluate_model_comprehensive(model, test_loader, device, save_dir='evaluation_results'):
    """
    Run comprehensive model evaluation and save all results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    evaluator = ModelEvaluator(model, device)
    
    print("Running comprehensive model evaluation...")
    metrics = evaluator.evaluate_model(test_loader)
    
    # Generate and save plots
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'], 
                                   f'{save_dir}/confusion_matrix.png')
    evaluator.plot_character_error_rates(metrics['character_error_rates'], 
                                        f'{save_dir}/character_error_rates.png')
    
    # Generate evaluation report
    evaluator.generate_evaluation_report(metrics, f'{save_dir}/evaluation_report.txt')
    
    # Analyze specific predictions
    print("\nAnalyzing specific predictions:")
    evaluator.analyze_predictions(test_loader, num_samples=5)
    
    return metrics
