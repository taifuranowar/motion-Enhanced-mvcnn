import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, resnet50
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ========== Command Line Arguments ==========
def parse_args():
    parser = argparse.ArgumentParser(description='MVCNN Training')
    
    # Add elevation parameter
    parser.add_argument('--elevations', type=str, default=None,
                        help='Comma-separated list of elevations to use (e.g. "0,30"). If not specified, use all available elevations.')
    
    # Basic configuration
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to the generated MVCNN dataset')
    parser.add_argument('--output-dir', type=str, default='mvcnn_results',
                        help='Output directory for trained model and results')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='CNN backbone architecture')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay for optimizer')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='Use pretrained weights for the backbone')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for fully connected layers')

    # Class selection
    parser.add_argument('--selected-classes', type=str, default=None,
                        help='Comma-separated list of class names to use (e.g. "chair,table,sofa,bed,car")')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes to use for training (selects first N classes)')
    
    # Save/load parameters
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save model every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Add max-views parameter
    parser.add_argument('--max-views', type=int, default=None,
                        help='Maximum number of views to use per model (default: use all available views)')
    
    args = parser.parse_args()
    return args

# ========== Dataset Handling ==========
class MVCNNDataset(Dataset):
    printed_view_count = False  # Class variable to control printing

    """Multi-View CNN Dataset"""
    def __init__(self, dataset_path, split='train', transform=None, selected_classes=None, max_views=None, elevations=None):
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        self.selected_classes = selected_classes
        self.max_views = max_views
        self.elevations = elevations  # List of elevation values to use or None for all
        
        # Parse elevations if provided as string
        if isinstance(self.elevations, str):
            self.elevations = [float(e.strip()) for e in self.elevations.split(',')]
        
        # Path to renders directory
        self.renders_path = os.path.join(dataset_path, 'renders')
        
        # Load dataset metadata
        with open(os.path.join(dataset_path, 'dataset_metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        self.classes = []
        self.class_to_idx = {}
        
        # Build class list and mapping
        for idx, class_data in enumerate(self.metadata['classes']):
            class_name = class_data['class_name']
            if self.selected_classes is not None and class_name not in self.selected_classes:
                continue
            self.classes.append(class_name)
            self.class_to_idx[class_name] = len(self.classes) - 1
        
        self.samples = []
        
        # Build dataset samples list
        for class_data in self.metadata['classes']:
            class_name = class_data['class_name']
            if self.selected_classes is not None and class_name not in self.selected_classes:
                continue
            class_idx = self.class_to_idx[class_name]
            
            # Select models based on split
            if split == 'train':
                models_data = class_data['train_models']
            else:  # test or val
                models_data = class_data['test_models']
            
            for model_data in models_data:
                model_name = model_data['model_name']
                model_path = os.path.join(self.renders_path, class_name, split, model_name)
                
                # Load model metadata to get view information
                with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
                    model_metadata = json.load(f)
                
                # Get view filenames along with their elevation and azimuth
                view_files = []
                view_elevations = []
                view_azimuths = []
                for view in sorted(model_metadata['views'], key=lambda x: x['view_idx']):
                    # Filter by elevation if specified
                    if self.elevations is not None and view['elevation'] not in self.elevations:
                        continue
                    
                    view_files.append(os.path.join(model_path, view['filename']))
                    view_elevations.append(view['elevation'])
                    view_azimuths.append(view['azimuth'])
                
                # Limit the number of views if specified, distributing across elevations
                if self.max_views is not None:
                    unique_elevations = sorted(set(view_elevations))
                    num_unique_elevs = len(unique_elevations)
                    total_views = len(view_files)
                    
                    if num_unique_elevs <= 1 or total_views <= self.max_views:
                        # Only one elevation or fewer views than max, use standard spacing
                        if total_views > self.max_views:
                            indices = [int(i * total_views / self.max_views) for i in range(self.max_views)]
                            view_files = [view_files[i] for i in indices]
                            view_elevations = [view_elevations[i] for i in indices]
                            view_azimuths = [view_azimuths[i] for i in indices]
                    else:
                        # Multiple elevations - distribute views across elevations
                        views_per_elev = self.max_views // num_unique_elevs
                        remaining = self.max_views % num_unique_elevs
                        
                        # If some elevations will get more views
                        elev_view_counts = {e: views_per_elev for e in unique_elevations}
                        for i in range(remaining):
                            elev_view_counts[unique_elevations[i]] += 1
                        
                        # Select views for each elevation
                        selected_indices = []
                        for elev in unique_elevations:
                            # Get indices for this elevation
                            elev_indices = [i for i, e in enumerate(view_elevations) if e == elev]
                            num_views_for_elev = elev_view_counts[elev]
                            
                            if num_views_for_elev > 0 and elev_indices:
                                # Select evenly spaced views from this elevation
                                elev_total = len(elev_indices)
                                elev_selected = [elev_indices[int(i * elev_total / num_views_for_elev)] 
                                                for i in range(num_views_for_elev)]
                                selected_indices.extend(elev_selected)
                        
                        # Use the selected indices
                        selected_indices.sort()  # Maintain original order
                        view_files = [view_files[i] for i in selected_indices]
                        view_elevations = [view_elevations[i] for i in selected_indices]
                        view_azimuths = [view_azimuths[i] for i in selected_indices]
                
                self.samples.append({
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'model_name': model_name,
                    'view_files': view_files,
                    'view_elevations': view_elevations,
                    'view_azimuths': view_azimuths
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        views = []
        # Print the number of views and elevations for this sample only once per session
        if not MVCNNDataset.printed_view_count:
            print(f"Number of views for sample {idx}: {len(sample['view_files'])}")
            if 'view_elevations' in sample:
                unique_elevs = sorted(set(sample['view_elevations']))
                print(f"Elevations used: {unique_elevs}")
                for elev in unique_elevs:
                    count = sum(1 for e in sample['view_elevations'] if e == elev)
                    print(f"  Elevation {elev}°: {count} views")
            MVCNNDataset.printed_view_count = True
        
        for view_file in sample['view_files']:
            img = Image.open(view_file).convert('RGB')
            if self.transform:
                img = self.transform(img)
            views.append(img)
        views = torch.stack(views)
        
        # Convert elevation and azimuth data to tensors
        elevations = torch.tensor(sample['view_elevations'], dtype=torch.float32) if 'view_elevations' in sample else None
        azimuths = torch.tensor(sample['view_azimuths'], dtype=torch.float32) if 'view_azimuths' in sample else None
    
        return {
            'views': views,
            'label': sample['class_idx'],
            'class_name': sample['class_name'],
            'model_name': sample['model_name'],
            'elevations': elevations,
            'azimuths': azimuths
        }

# ========== MVCNN Model ==========
class MVCNN(nn.Module):
    def __init__(self, num_classes, backbone='resnet18', pretrained=True, dropout=0.5):
        super(MVCNN, self).__init__()
        
        # Choose backbone architecture
        if backbone == 'resnet18':
            base_model = resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            base_model = resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Backbone {backbone} not supported")
            
        # Remove the final FC layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Shared feature extractor for all views
        self.shared_features = True
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: [batch, num_views, channels, height, width]
        batch_size = x.shape[0]
        num_views = x.shape[1]
        
        # Reshape for feature extraction (merge batch and views)
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        
        # Extract features
        features = self.features(x)
        features = features.view(features.shape[0], -1)  # Flatten features
        
        # Reshape back to separate batch and views
        features = features.view(batch_size, num_views, -1)
        
        # View pooling (max across views)
        pooled_features = torch.max(features, dim=1)[0]
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits

# ========== Training and Evaluation Functions ==========
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, data in enumerate(dataloader):
        views = data['views'].to(device)
        labels = data['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(views)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch: {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} | ' +
                  f'Acc: {100 * correct / total:.2f}% ({correct}/{total})')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            views = data['views'].to(device)
            labels = data['label'].to(device)
            
            outputs = model(views)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Track per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
            
            # Track predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate loss and accuracy
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    # Calculate per-class accuracy
    class_acc = {}
    for label in class_total:
        class_acc[label] = 100 * class_correct[label] / class_total[label]
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'class_accuracy': class_acc,
        'predictions': all_preds,
        'labels': all_labels
    }

# ========== Main Function ==========
def main():
    args = parse_args()

    # Parse selected classes if provided
    selected_classes = None
    if args.selected_classes is not None:
        selected_classes = [c.strip() for c in args.selected_classes.split(',') if c.strip()]
    else:
        # If not provided, get all classes from metadata
        with open(os.path.join(args.dataset_path, 'dataset_metadata.json'), 'r') as f:
            metadata = json.load(f)
        selected_classes = [c.get('class_name', c.get('class')) for c in metadata['classes']]

    # Apply num-classes if specified
    if args.num_classes is not None:
        selected_classes = selected_classes[:args.num_classes]
    print(f"Selected classes: {selected_classes}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join('mvcnn_results', f'train_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MVCNNDataset(
        args.dataset_path, 
        split='train', 
        transform=train_transform, 
        selected_classes=selected_classes, 
        max_views=args.max_views,
        elevations=args.elevations
    )
    test_dataset = MVCNNDataset(
        args.dataset_path, 
        split='test', 
        transform=test_transform,
        selected_classes=selected_classes, 
        max_views=args.max_views,
        elevations=args.elevations
    )
    
    # Filter selected classes if specified
    if args.selected_classes:
        selected_classes = [c.strip() for c in args.selected_classes.split(',')]
        train_dataset.samples = [s for s in train_dataset.samples if s['class_name'] in selected_classes]
        test_dataset.samples = [s for s in test_dataset.samples if s['class_name'] in selected_classes]
        
        # Update class index mapping
        train_dataset.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        test_dataset.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        
        print(f"Selected classes: {selected_classes}")
        print(f"Filtered dataset: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    
    # Create data loaders
    # Remove device argument from pin_memory
    use_pin_memory = torch.cuda.is_available() and args.device == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_pin_memory)
    
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    print(f"Classes: {len(train_dataset.classes)}")
    
    # Create model
    model = MVCNN(num_classes=len(train_dataset.classes), 
                  backbone=args.backbone, 
                  pretrained=args.use_pretrained, 
                  dropout=args.dropout)
    
    # Move model to device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create scheduler (reduce LR on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training history tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_acc': 0.0,
        'best_epoch': 0
    }
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            history = checkpoint['history']
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print('-' * 50)
        
        # Train one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate model
        print("Evaluating...")
        eval_results = evaluate(model, test_loader, criterion, device)
        val_loss = eval_results['loss']
        val_acc = eval_results['accuracy']
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Check if this is the best model
        if val_acc > history['best_acc']:
            history['best_acc'] = val_acc
            history['best_epoch'] = epoch
            
            # Save best model
            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'val_acc': val_acc,
                'class_accuracy': eval_results['class_accuracy'],
                'num_classes': len(train_dataset.classes),
                'classes': train_dataset.classes,
                'args': vars(args)
            }, best_model_path)
            print(f"New best model saved! Accuracy: {val_acc:.2f}%")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'args': vars(args)
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Final evaluation
    print("\nTraining complete. Running final evaluation...")
    final_results = evaluate(model, test_loader, criterion, device)
    
    # Save confusion matrix
    conf_matrix = confusion_matrix(final_results['labels'], final_results['predictions'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Generate classification report
    class_report = classification_report(
        final_results['labels'], 
        final_results['predictions'],
        target_names=train_dataset.classes,
        output_dict=True
    )
    
    # Save final metrics
    metrics = {
        'final_accuracy': final_results['accuracy'],
        'best_accuracy': history['best_acc'],
        'best_epoch': history['best_epoch'],
        'class_accuracy': final_results['class_accuracy'],
        'train_history': {
            'loss': history['train_loss'],
            'accuracy': history['train_acc']
        },
        'val_history': {
            'loss': history['val_loss'],
            'accuracy': history['val_acc']
        },
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'classes': train_dataset.classes,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_parameters': vars(args)
    }
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'final_acc': final_results['accuracy'],
        'class_accuracy': final_results['class_accuracy'],
        'num_classes': len(train_dataset.classes),
        'classes': train_dataset.classes,
        'args': vars(args)
    }, final_model_path)
    
    # Plot loss and accuracy curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    
    # Print final results
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best validation accuracy: {history['best_acc']:.2f}% (epoch {history['best_epoch']+1})")
    print(f"Final validation accuracy: {final_results['accuracy']:.2f}%")
    print(f"Results saved to: {args.output_dir}")
    
    # Print per-class accuracy for the top 5 and bottom 5 classes
    class_acc = [(train_dataset.classes[i], final_results['class_accuracy'][i]) 
                for i in range(len(train_dataset.classes))]
    class_acc.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 classes:")
    for name, acc in class_acc[:5]:
        print(f"{name}: {acc:.2f}%")
    
    print("\nBottom 5 classes:")
    for name, acc in class_acc[-5:]:
        print(f"{name}: {acc:.2f}%")

if __name__ == '__main__':
    main()