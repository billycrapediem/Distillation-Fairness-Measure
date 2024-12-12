import os
from prediction_bias import ModelPredictionBias
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tiny_imagenet_dataset import TinyImageNet
import numpy as np
import pandas as pd

def plot_bias_analysis(category_stats, mpb: ModelPredictionBias, save_dir: str = "bias_analysis_plots"):
    """Create and save visualization plots for model prediction bias analysis."""
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Accuracy Distribution Plot
    plt.figure(figsize=(10, 7))
    accuracies = []
    categories = []
    
    for category in ['Easy', 'Medium', 'Hard']:
        stats = category_stats[category]
        class_accs = [mpb.class_correct[idx]/mpb.class_total[idx] for idx in stats['class_indices']]
        accuracies.extend(class_accs)
        categories.extend([category] * len(class_accs))
    df = pd.DataFrame({
        'Category': categories,
        'Value': accuracies
    })
    sns.violinplot(data=df, x='Category', y='Value', 
                  color='#03cffc',  # Light blue fill color
                  inner='box',        # Show box plot inside violin
                  linewidth=1.5) 
    plt.title('Distribution by Category', pad=20)
    plt.ylabel('Value')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bias Comparison Plot
    plt.figure(figsize=(10, 7))
    categories = ['Easy', 'Medium', 'Hard']
    mean_gn = [category_stats[cat]['mean_gn'] for cat in categories]
    mean_gp = [category_stats[cat]['mean_gp'] for cat in categories]
    std_gn = [category_stats[cat]['std_gn'] for cat in categories]
    std_gp = [category_stats[cat]['std_gp'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, mean_gn, width, label='Negative Bias (Gn)', 
            yerr=std_gn, capsize=5)
    plt.bar(x + width/2, mean_gp, width, label='Positive Bias (Gp)', 
            yerr=std_gp, capsize=5)
    plt.xlabel('Category')
    plt.ylabel('Bias Value')
    plt.title('Comparison of Positive and Negative Bias')
    plt.xticks(x, categories)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bias_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-Class Accuracy Plot
    plt.figure(figsize=(15, 8))
    all_accuracies = []
    class_indices = []
    category_colors = []
    colors = {'Easy': 'green', 'Medium': 'orange', 'Hard': 'red'}
    
    for category in ['Easy', 'Medium', 'Hard']:
        stats = category_stats[category]
        indices = stats['class_indices']
        accs = [mpb.class_correct[idx]/mpb.class_total[idx] for idx in indices]
        all_accuracies.extend(accs)
        class_indices.extend(indices)
        category_colors.extend([colors[category]] * len(indices))
    
    sorted_indices = np.argsort(all_accuracies)
    sorted_accs = np.array(all_accuracies)[sorted_indices]
    sorted_colors = np.array(category_colors)[sorted_indices]
    
    plt.bar(range(len(sorted_accs)), sorted_accs, color=sorted_colors)
    plt.title('Per-Class Accuracy (Sorted)')
    plt.xlabel('Class Index')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Detailed Class Analysis Plot
    plt.figure(figsize=(15, 10))
    sorted_gn = []
    sorted_gp = []
    for idx in class_indices:
        sorted_gn.append(mpb.gn[idx])
        sorted_gp.append(mpb.gp[idx])
    sorted_gn = np.array(sorted_gn)[sorted_indices]
    sorted_gp = np.array(sorted_gp)[sorted_indices]
    
    plt.plot(range(len(sorted_accs)), sorted_gn, 'r-', label='Negative Bias (Gn)', alpha=0.7)
    plt.plot(range(len(sorted_accs)), sorted_gp, 'g-', label='Positive Bias (Gp)', alpha=0.7)
    plt.plot(range(len(sorted_accs)), sorted_accs, 'b-', label='Accuracy', alpha=0.7)
    plt.title('Comparison of Accuracy, Positive Bias, and Negative Bias per Class')
    plt.xlabel('Class Index (Sorted by Accuracy)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detailed_class_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"All plots have been saved in the '{save_dir}' directory.")

def analyze_model_bias(model_path: str, save_dir: str, mode=False, device: str = 'cuda'):
    """Analyze model prediction bias on CIFAR-100 test set.
    
    Args:
        model_path (str): Path to the model checkpoint
        save_dir (str): Directory to save analysis plots
        mode (bool): Mode flag for different data loading methods
        device (str): Device to run the model on
    
    Returns:
        dict: Dictionary containing bias metrics including CAD
    """
    # Set up data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Load dataset based on mode
    if mode:
        testset = torchvision.datasets.CIFAR100(
            root='/scratch/bzhang44/SRe2L/SRe2L/*small_dataset/data', 
            train=False,
            download=True, 
            transform=transform
        )
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        model = torchvision.models.get_model("resnet18", num_classes=200)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
        model = nn.DataParallel(model).cuda()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        testset = load_data()
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        model = create_model('resnet18', model_path)
    
    model = model.to(device)
    model.eval()
    
    # Initialize bias calculator
    mpb = ModelPredictionBias(num_classes=200)
    
    # Evaluate model
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            mpb.update(outputs, labels)
    
    # Compute standard metrics
    gn, gp, cf, category_stats = mpb.compute(normalize=True)
    
    # Compute CAD
    accuracies = []
    for idx in range(200):  # Assuming 200 classes
        if mpb.class_total[idx] > 0:  # Avoid division by zero
            accuracy = mpb.class_correct[idx] / mpb.class_total[idx]
            accuracies.append(accuracy)
    
    accuracies = torch.tensor(accuracies)
    mean_accuracy = torch.mean(accuracies)
    cad = torch.sqrt(torch.mean((accuracies - mean_accuracy) ** 2))
    
    # Print results
    print(f"\nOverall Metrics:")
    print(f"CAD: {cad:.4f}")
    print(f"Global Negative Bias (Gn): {gn}")
    print(f"Global Positive Bias (Gp): {gp}")
    print(f"Confusion Factor (CF): {cf}")
    
    print("\nCategory-wise Statistics:")
    for category in ['Easy', 'Medium', 'Hard']:
        stats = category_stats[category]
        print(f"\n{category} Classes:")
        print(f"Number of classes: {stats['num_classes']}")
        print(f"Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        print(f"Range: [{stats['min_accuracy']:.4f}, {stats['max_accuracy']:.4f}]")
        print(f"Mean Negative Bias (Gn): {stats['mean_gn']:.4f} ± {stats['std_gn']:.4f}")
        print(f"Mean Positive Bias (Gp): {stats['mean_gp']:.4f} ± {stats['std_gp']:.4f}")
        print("\nClasses in this category:")
        for idx in stats['class_indices']:
            print(f"  {idx}: {mpb.class_correct[idx]/mpb.class_total[idx]:.4f}")
        print()

    # Save visualization
    plot_bias_analysis(category_stats, mpb, save_dir)
    
    # Return metrics dictionary
    x= {
        'cad': cad.item(),
        'gn': gn,
        'gp': gp,
        'cf': cf,
        'category_stats': category_stats,
        'class_accuracies': accuracies.tolist()
    }
    print(x)

def load_data():
    # Data loading code
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])

    print("Loading validation data")
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    dataset_test = TinyImageNet('/scratch/bzhang44/tiny-imagenet/data', split='val', download=False, transform=val_transform)
    print("Creating data loaders")

    return dataset_test

def create_model(model_name, path=None):
        model = torchvision.models.get_model(model_name, weights=None, num_classes=200)
        model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        model.maxpool = nn.Identity()
        if path is not None:
            checkpoint = torch.load(path, map_location="cpu")
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            if "module." in list(checkpoint.keys())[0]:
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        model.to("cuda")
        return model

if __name__ == "__main__":
    "/scratch/bzhang44/Distillation-Fairness-Measure/SRE2L/cifar100/ckpt.pth"
    ckpt_path = "/scratch/bzhang44/tiny-imagenet/save/rn18_50ep/checkpoint_best.pth"#"/scratch/bzhang44/Distillation-Fairness-Measure/SRE2L/Tiny/checkpoint_best.pth"
    save_dir = "./base"  # Specify your desired directory name
    analyze_model_bias(ckpt_path, save_dir=save_dir,mode=False)
