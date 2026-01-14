import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from intact import intact_unlearn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def get_cifar10_dataloaders(batch_size=128, data_dir='./data'):
    """Load CIFAR-10 with standard normalization."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset


def create_class_subsets(dataset, classes):
    """Create a subset containing only specified classes."""
    indices = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in classes:
            indices.append(i)
    return Subset(dataset, indices)


def pretrain_model(model, train_loader, criterion, device, epochs=10, lr=0.001):
    """Pretrain the model on all classes."""
    log.info(f"Pretraining model for {epochs} epochs...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
   
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        log.info(f"Pretrain Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.2f}%")
    
    log.info("Pretraining complete!")


def test_model(model, test_loader, device, all_classes):
    """Test model and return per-class accuracy."""
    model.eval()
    
    class_correct = {cls: 0 for cls in all_classes}
    class_total = {cls: 0 for cls in all_classes}
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            _, predicted = torch.max(output, 1)
            
            for i in range(y.size(0)):
                label = y[i].item()
                if label in all_classes:
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
    
    # Calculate per-class accuracy
    per_class_acc = {}
    for cls in all_classes:
        if class_total[cls] > 0:
            acc = 100.0 * class_correct[cls] / class_total[cls]
            per_class_acc[cls] = acc
        else:
            per_class_acc[cls] = 0.0
    
    # Overall accuracy
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    
    return per_class_acc, overall_acc


def main():
    parser = argparse.ArgumentParser(description='InTAct Unlearning Experiment')
    parser.add_argument('--unlearn_classes', nargs='+', type=int, default=[0],
                        help='Class to unlearn (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--pretrain_epochs', type=int, default=10,
                        help='Number of pretraining epochs (default: 10)')
    parser.add_argument('--unlearn_epochs', type=int, default=10,
                        help='Number of unlearning epochs (default: 10)')
    parser.add_argument('--pretrain_lr', type=float, default=0.001,
                        help='Learning rate for pretraining (default: 0.001)')
    parser.add_argument('--unlearn_lr', type=float, default=0.0001,
                        help='Learning rate for unlearning (default: 0.0001)')
    parser.add_argument('--reduced_dim', type=int, default=32,
                        help='Number of principal components (default: 32)')
    parser.add_argument('--lambda_interval', type=float, default=100.0,
                        help='Weight for interval protection loss (default: 100.0)')
    parser.add_argument('--margin_percentile', type=float, default=0.2,
                        help='Margin percentile for interval bounds (default: 0.2)')
    parser.add_argument('--infinity_scale', type=float, default=100.0,
                        help='Scale factor for infinity bounds (default: 100.0)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data (default: ./data)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: auto)')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    log.info(f"Using device: {device}")
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    all_classes = list(range(10))
    unlearn_classes = args.unlearn_classes
    retain_classes = [cls for cls in all_classes if cls not in unlearn_classes]
    
    log.info(f"All classes: {all_classes}")
    log.info(f"Unlearn classes: {unlearn_classes} ({[class_names[c] for c in unlearn_classes]})")
    log.info(f"Retain classes: {retain_classes} ({[class_names[c] for c in retain_classes]})")
    
    # Load data
    log.info("Loading CIFAR-10...")
    train_dataset, test_dataset = get_cifar10_dataloaders(
        batch_size=args.batch_size, data_dir=args.data_dir
    )
    
    # Create dataloaders
    full_train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    
    forget_subset = create_class_subsets(train_dataset, unlearn_classes)
    forget_loader = DataLoader(
        forget_subset, batch_size=args.batch_size, shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    log.info(f"Train set size: {len(train_dataset)}")
    log.info(f"Forget set size: {len(forget_subset)}")
    log.info(f"Test set size: {len(test_dataset)}")
    
    # Create model
    log.info("Creating ResNet18 model...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)  # Replace classifier for CIFAR-10
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Pretrain model
    if args.pretrain_epochs > 0:
        pretrain_model(
            model, full_train_loader, criterion, device,
            epochs=args.pretrain_epochs, lr=args.pretrain_lr
        )
    
    # Test before unlearning
    log.info("\n=== Testing BEFORE unlearning ===")
    per_class_acc_before, overall_acc_before = test_model(
        model, test_loader, device, all_classes
    )
    
    log.info(f"Overall accuracy: {overall_acc_before:.2f}%")
    for cls in all_classes:
        log.info(f"  Class {cls} ({class_names[cls]}): {per_class_acc_before[cls]:.2f}%")
    
    # Unlearning
    log.info("\n=== Starting InTAct Unlearning ===")
    for name, p in model.named_parameters():
        if "fc" not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.unlearn_lr)
    
    model = intact_unlearn(
        model=model,
        forget_loader=forget_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.unlearn_epochs,
        device=device,
        lambda_interval=args.lambda_interval,
        lower_percentile=args.margin_percentile,
        upper_percentile=1.0 - args.margin_percentile,
        reduced_dim=args.reduced_dim,
        infinity_scale=args.infinity_scale
    )
    
    # Test after unlearning
    log.info("\n=== Testing AFTER unlearning ===")
    per_class_acc_after, overall_acc_after = test_model(
        model, test_loader, device, all_classes
    )
    
    log.info(f"Overall accuracy: {overall_acc_after:.2f}%")
    for cls in all_classes:
        log.info(f"  Class {cls} ({class_names[cls]}): {per_class_acc_after[cls]:.2f}%")
    
    # Summary
    log.info("\n=== Unlearning Summary ===")
    unlearn_acc_before = np.mean([per_class_acc_before[cls] for cls in unlearn_classes])
    unlearn_acc_after = np.mean([per_class_acc_after[cls] for cls in unlearn_classes])
    retain_acc_before = np.mean([per_class_acc_before[cls] for cls in retain_classes])
    retain_acc_after = np.mean([per_class_acc_after[cls] for cls in retain_classes])
    
    log.info(f"Forget classes: {unlearn_acc_before:.2f}% -> {unlearn_acc_after:.2f}% "
             f"(Δ {unlearn_acc_after - unlearn_acc_before:.2f}%)")
    log.info(f"Retain classes: {retain_acc_before:.2f}% -> {retain_acc_after:.2f}% "
             f"(Δ {retain_acc_after - retain_acc_before:.2f}%)")
    log.info(f"Overall: {overall_acc_before:.2f}% -> {overall_acc_after:.2f}% "
             f"(Δ {overall_acc_after - overall_acc_before:.2f}%)")


if __name__ == '__main__':
    main()
