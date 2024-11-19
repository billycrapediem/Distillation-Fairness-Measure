import datetime
import os
import time
import warnings

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import wandb
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from collections import defaultdict
import matplotlib.pyplot as plt

def visualize_class_accuracies(class_accuracy, output_file='class_accuracies.png'):
    # Sort the class accuracies
    sorted_accuracies = sorted(class_accuracy.items(), key=lambda x: x[1], reverse=True)
    classes, accuracies = zip(*sorted_accuracies)

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot the line
    ax.plot(range(len(classes)), accuracies, linewidth=2, color='#3498db')
    
    # Fill the area under the line
    ax.fill_between(range(len(classes)), accuracies, alpha=0.3, color='#3498db')

    # Customize the plot
    ax.set_xlabel('Class Index (Sorted by Accuracy)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Per-class Accuracy (Sorted)', fontsize=18, fontweight='bold')
    
    # Set x-axis ticks
    ax.set_xticks(range(0, len(classes), 20))
    ax.set_xticklabels(range(0, len(classes), 20), fontsize=12)
    
    # Set y-axis limits and ticks
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([f'{x:.1f}' for x in [0, 0.2, 0.4, 0.6, 0.8, 1.0]], fontsize=12)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add a line for mean accuracy
    mean_accuracy = sum(accuracies) / len(accuracies)
    ax.axhline(y=mean_accuracy, color='#e74c3c', linestyle='--', linewidth=2, 
               label=f'Mean Accuracy: {mean_accuracy:.3f}')

    # Add legend
    ax.legend(fontsize=12, loc='lower left')

    # Add text annotations for top and bottom classes
    top_class = sorted_accuracies[0]
    bottom_class = sorted_accuracies[-1]
    ax.annotate(f'Top: Class {top_class[0]} ({top_class[1]:.3f})', 
                xy=(0, top_class[1]), xytext=(10, -10),
                textcoords='offset points', ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    ax.annotate(f'Bottom: Class {bottom_class[0]} ({bottom_class[1]:.3f})', 
                xy=(len(classes)-1, bottom_class[1]), xytext=(-10, 10),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Class accuracy plot saved to {output_file}")

    # Print top 5 and bottom 5 classes
    print("\nTop 5 Classes:")
    for class_id, acc in sorted_accuracies[:5]:
        print(f"Class {class_id}: {acc:.3f}")
    print("\nBottom 5 Classes:")
    for class_id, acc in sorted_accuracies[-5:]:
        print(f"Class {class_id}: {acc:.3f}")
        
def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.inference_mode():
        for image, target in data_loader, print_freq, header:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            batch_size = image.shape[0]
            num_processed_samples += batch_size

            # Calculate per-class accuracy
            _, predicted = torch.max(output, 1)
            correct = (predicted == target).squeeze()
            for i in range(batch_size):
                label = target[i].item()
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )
    # Calculate per-class accuracy
    class_accuracy = {class_id: class_correct[class_id] / class_total[class_id] 
                      for class_id in class_total.keys()}
    visualize_class_accuracies(class_accuracy)
