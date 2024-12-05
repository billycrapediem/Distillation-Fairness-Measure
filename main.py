from prediction_bias import ModelPredictionBias
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

def plot_bias_analysis(category_stats, mpb: ModelPredictionBias, class_names):
    """Create visualization plots for model prediction bias analysis."""
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Accuracy Distribution Plot
    plt.subplot(2, 2, 1)
    accuracies = []
    categories = []
    for category in ['Easy', 'Medium', 'Hard']:
        stats = category_stats[category]
        class_accs = [mpb.class_correct[idx]/mpb.class_total[idx] for idx in stats['class_indices']]
        accuracies.extend(class_accs)
        categories.extend([category] * len(class_accs))
    
    sns.boxplot(x=categories, y=accuracies)
    plt.title('Accuracy Distribution by Category')
    plt.ylabel('Accuracy')
    
    # 2. Bias Comparison Plot
    plt.subplot(2, 2, 2)
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
    
    # 3. Per-Class Accuracy Plot
    plt.subplot(2, 1, 2)
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
    sorted_class_names = np.array(class_names)[np.array(class_indices)[sorted_indices]]
    
    plt.bar(range(len(sorted_accs)), sorted_accs, color=sorted_colors)
    plt.title('Per-Class Accuracy (Sorted)')
    plt.xlabel('Class Index')
    plt.ylabel('Accuracy')
    
    # Add class names for some points
    step = len(sorted_accs) // 10  # Show every nth class name
    for i in range(0, len(sorted_accs), step):
        plt.text(i, sorted_accs[i], sorted_class_names[i], 
                rotation=45, ha='right', va='bottom')
    
    plt.tight_layout()
    
    # 4. Create a separate figure for detailed class analysis
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
    plt.show()

def analyze_model_bias(model_path: str, device: str = 'cuda'):
    """Analyze model prediction bias on CIFAR-100 test set."""
    
    # Set up data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load CIFAR-100 test set
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Load model
    model = torchvision.models.get_model("resnet18", num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    model = nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    
    # Initialize bias calculator
    mpb = ModelPredictionBias(num_classes=100)
    
    # Evaluate model
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            mpb.update(outputs, labels)
    
    # Compute metrics
    gn, gp, cf, category_stats = mpb.compute(normalize=True)
    
    # Get class names
    class_names = testset.classes
    print(f"gn: {gn}, gp:{gp}, cf:{cf}")
    # Print results
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
            print(f"  {class_names[idx]}: {mpb.class_correct[idx]/mpb.class_total[idx]:.4f}")
        print()
    plot_bias_analysis(category_stats, mpb, class_names)
        
if __name__ == "__main__":
    ckpt_path = "./save_post_cifar100/ipc50/ckpt.pth"
    analyze_model_bias(ckpt_path)

