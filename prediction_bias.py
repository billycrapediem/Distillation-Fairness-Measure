import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple, Dict, List, Optional

class ModelPredictionBias:
    """
    Calculate Model Prediction Bias with automatic category splitting based on performance.
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        """Reset accumulated statistics"""
        self.gn = np.zeros(self.num_classes)  # Negative bias
        self.gp = np.zeros(self.num_classes)  # Positive bias
        self.class_freq = np.zeros(self.num_classes)  # Class frequency
        self.class_correct = np.zeros(self.num_classes)  # Correct predictions per class
        self.class_total = np.zeros(self.num_classes)  # Total predictions per class
        
    def update(self, logits: Union[np.ndarray, torch.Tensor], 
               labels: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Update bias statistics with a new batch of predictions.
        """
        # Convert to numpy if needed
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
            
        # Convert logits to probabilities using softmax
        probs = self._softmax(logits)
        
        # Update class frequency
        for label in labels:
            self.class_freq[label] += 1
            
        # Calculate predictions
        predictions = np.argmax(probs, axis=1)
        
        # Update correct predictions
        for pred, label in zip(predictions, labels):
            self.class_total[label] += 1
            if pred == label:
                self.class_correct[label] += 1
            
        # Calculate one-hot labels
        labels_onehot = np.zeros((labels.size, self.num_classes))
        labels_onehot[np.arange(labels.size), labels] = 1
        
        # Update Gn and Gp
        self.gn += np.sum(probs * (1 - labels_onehot), axis=0)
        self.gp += np.sum(probs * labels_onehot, axis=0)

    def _get_class_categories(self) -> Dict[str, List[int]]:
        """
        Split classes into Hard/Medium/Easy based on accuracy.
        """
        # Calculate per-class accuracy
        accuracies = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            if self.class_total[i] > 0:
                accuracies[i] = self.class_correct[i] / self.class_total[i]
        
        # Sort classes by accuracy
        sorted_indices = np.argsort(accuracies)
        
        # Calculate splits
        n_classes = len(sorted_indices)
        n_per_category = n_classes // 3
        remainder = n_classes % 3
        
        # Distribute remainder classes from hard to easy
        splits = [n_per_category + (1 if i < remainder else 0) for i in range(3)]
        
        # Create category assignments
        start_idx = 0
        categories = {}
        for category, n in zip(['Hard', 'Medium', 'Easy'], splits):
            end_idx = start_idx + n
            categories[category] = sorted_indices[start_idx:end_idx].tolist()
            start_idx = end_idx
            
        return categories
        
    def compute(self, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
        """
        Compute the final Model Prediction Bias metrics.
        """
        if normalize and np.any(self.class_freq > 0):
            gn = self.gn / np.maximum(self.class_freq.sum(), 1e-8)
            gp = self.gp / np.maximum(self.class_freq.sum(), 1e-8)
            cf = self.class_freq / np.maximum(self.class_freq.sum(), 1e-8)
        else:
            gn, gp, cf = self.gn, self.gp, self.class_freq
            
        # Get automatic category split
        class_categories = self._get_class_categories()
            
        # Calculate category-wise statistics
        category_stats = {}
        for category, class_indices in class_categories.items():
            accuracies = [self.class_correct[i] / max(self.class_total[i], 1) for i in class_indices]
            
            category_stats[category] = {
                'mean_gn': np.mean(gn[class_indices]),
                'std_gn': np.std(gn[class_indices]),
                'mean_gp': np.mean(gp[class_indices]),
                'std_gp': np.std(gp[class_indices]),
                'mean_cf': np.mean(cf[class_indices]),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'num_classes': len(class_indices),
                'class_indices': class_indices
            }
            
        return gn, gp, cf, category_stats
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax function to input array."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def example_usage():
    # Create a dummy dataset
    batch_size = 1000
    num_classes = 10
    
    # Initialize calculator
    mpb = ModelPredictionBias(num_classes=num_classes)
    
    # Generate synthetic data with different difficulties
    logits = np.zeros((batch_size, num_classes))
    labels = np.random.randint(0, num_classes, size=batch_size)
    
    for i in range(batch_size):
        true_class = labels[i]
        # Make some classes easier to predict than others
        difficulty = true_class / num_classes  # Higher class index = more difficult
        logits[i, true_class] = np.random.normal(5 * (1 - difficulty), 1)
        logits[i, :] += np.random.normal(0, 0.5 + difficulty, num_classes)
    
    # Update statistics
    mpb.update(logits, labels)
    
    # Compute final metrics with automatic category analysis
    gn, gp, cf, category_stats = mpb.compute(normalize=True)
    
    # Print results
    print("\nCategory-wise Statistics:")
    for category in ['Easy', 'Medium', 'Hard']:
        stats = category_stats[category]
        print(f"\n{category} Classes:")
        print(f"Class indices: {stats['class_indices']}")
        print(f"Number of classes: {stats['num_classes']}")
        print(f"Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        print(f"Range: [{stats['min_accuracy']:.4f}, {stats['max_accuracy']:.4f}]")
        print(f"Mean Negative Bias (Gn): {stats['mean_gn']:.4f} ± {stats['std_gn']:.4f}")
        print(f"Mean Positive Bias (Gp): {stats['mean_gp']:.4f} ± {stats['std_gp']:.4f}")
        print(f"Mean Class Frequency: {stats['mean_cf']:.4f}")

if __name__ == "__main__":
    example_usage()