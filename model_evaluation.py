import os
import torch
import json
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# Import models from original scripts
from mvcnn_training import MVCNN
from motion_mvcnn_training import MotionEnhancedMVCNN

# ========== Command Line Arguments ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Model Evaluation on Unseen Views')
    
    # Basic configuration
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to the generated MVCNN dataset')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint (best_model.pth or final_model.pth)')
    parser.add_argument('--model-type', type=str, choices=['mvcnn', 'motion_mvcnn'], required=True,
                        help='Type of model to evaluate')
    parser.add_argument('--motion-mode', type=str, default='static', choices=['static', 'learnable'],
                        help='Motion estimation mode for motion_mvcnn only')
    parser.add_argument('--test-views', type=int, default=None,
                        help='Number of views to use for testing (default: same as training)')
    parser.add_argument('--offset', type=int, default=1,
                        help='Offset for selecting unseen views (default: 1)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--elevation-mode', type=str, default='same', choices=['same', 'different', 'all'],
                        help='Test on same elevations as training, different elevations, or all')
    parser.add_argument('--test-elevations', type=str, default=None,
                        help='Comma-separated list of elevations to test on')
    
    args = parser.parse_args()
    return args

# ========== Unseen Views Dataset ==========
class UnseenViewsDataset(Dataset):
    """Dataset for testing with views that were not used in training"""
    def __init__(self, dataset_path, transform=None, classes=None, trained_views=None, 
                 test_views=None, offset=1, elevation_mode='same', 
                 training_elevations=None, test_elevations=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.classes = classes or []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.trained_views = trained_views
        self.test_views = test_views or trained_views
        self.offset = offset
        self.elevation_mode = elevation_mode
        self.training_elevations = training_elevations
        self.test_elevations = test_elevations
        
        # Path to renders directory
        self.renders_path = os.path.join(dataset_path, 'renders')
        
        # Load dataset metadata
        with open(os.path.join(dataset_path, 'dataset_metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        self.samples = []
        
        # Build dataset samples list
        for class_data in self.metadata['classes']:
            class_name = class_data['class_name']
            if class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]
            
            # Use test models
            models_data = class_data['test_models']
            
            for model_data in models_data:
                model_name = model_data['model_name']
                model_path = os.path.join(self.renders_path, class_name, 'test', model_name)
                
                # Load model metadata to get view information
                with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
                    model_metadata = json.load(f)
                
                # Get all view filenames sorted by view_idx
                all_views = []
                for view in sorted(model_metadata['views'], key=lambda x: x['view_idx']):
                    # Filter by elevation based on mode
                    view_elevation = view['elevation']
                    
                    if self.elevation_mode == 'same' and self.training_elevations:
                        # Only use same elevations as training
                        if view_elevation not in self.training_elevations:
                            continue
                    elif self.elevation_mode == 'different' and self.training_elevations:
                        # Only use elevations not in training
                        if view_elevation in self.training_elevations:
                            continue
                    elif self.test_elevations:
                        # Use specific test elevations
                        if view_elevation not in self.test_elevations:
                            continue
                    
                    all_views.append(os.path.join(model_path, view['filename']))
                
                total_views = len(all_views)
                
                # Determine which views to use based on trained views
                if self.trained_views is not None:
                    # Calculate indices used in training (based on even spacing)
                    train_indices = [int(i * total_views / self.trained_views) for i in range(self.trained_views)]
                    
                    # Select unseen views by choosing views between trained views
                    test_indices = []
                    for i in range(self.test_views):
                        # Calculate midpoint between trained views with offset
                        idx1 = train_indices[i % self.trained_views]
                        idx2 = train_indices[(i + 1) % self.trained_views]
                        if idx1 < idx2:
                            mid_idx = idx1 + (idx2 - idx1) // 2
                        else:  # wrap around
                            mid_idx = (idx1 + (total_views + idx2 - idx1) // 2) % total_views
                        
                        # Apply offset
                        mid_idx = (mid_idx + self.offset) % total_views
                        test_indices.append(mid_idx)
                else:
                    # If no trained_views specified, use evenly spaced views
                    test_indices = [int(i * total_views / self.test_views) for i in range(self.test_views)]
                
                # Get the view files for testing
                view_files = [all_views[i] for i in sorted(test_indices)]
                
                self.samples.append({
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'model_name': model_name,
                    'view_files': view_files,
                    'test_indices': test_indices
                })
        
        print(f"Created unseen views dataset with {len(self.samples)} samples")
        print(f"Using {self.test_views} views per sample with offset {self.offset}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        views = []
        
        for view_file in sample['view_files']:
            img = Image.open(view_file).convert('RGB')
            if self.transform:
                img = self.transform(img)
            views.append(img)
        
        views = torch.stack(views)
        return {
            'views': views,
            'label': sample['class_idx'],
            'class_name': sample['class_name'],
            'model_name': sample['model_name'],
            'test_indices': sample['test_indices']
        }

# ========== Evaluation Function ==========
def evaluate_model(model, dataloader, criterion, device, compute_flow=False):
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
            
            # Handle different model types
            if compute_flow:
                # For motion_mvcnn in static mode, compute flow
                num_views = views.shape[1]
                H, W = views.shape[3], views.shape[4]
                
                # Create motion_vectors for each sample in batch
                batch_motion = []
                for b in range(views.shape[0]):
                    sample_views = views[b]  # (num_views, C, H, W)
                    
                    # Use OpenCV flow calculation logic from motion_mvcnn_training.py
                    import cv2
                    motion_vectors = torch.zeros(num_views, num_views, H, W, 2)
                    for i in range(num_views):
                        img_i = sample_views[i].permute(1,2,0).cpu().numpy()  # (H,W,C)
                        img_i_gray = cv2.cvtColor((img_i*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                        for j in range(num_views):
                            if i != j:
                                img_j = sample_views[j].permute(1,2,0).cpu().numpy()
                                img_j_gray = cv2.cvtColor((img_j*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                                flow = cv2.calcOpticalFlowFarneback(
                                    img_i_gray, img_j_gray, None, 
                                    0.5, 3, 15, 3, 5, 1.2, 0
                                )
                                flow = cv2.resize(flow, (W, H))
                                motion_vectors[i, j] = torch.from_numpy(flow).float()
                    
                    batch_motion.append(motion_vectors)
                
                # Stack motion vectors for the batch
                motion_vectors = torch.stack(batch_motion).to(device)
                outputs = model(views, motion_vectors)
            else:
                # For standard MVCNN or motion_mvcnn in learnable mode
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
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Evaluated {batch_idx+1}/{len(dataloader)} batches...')
    
    # Calculate loss and accuracy
    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Calculate per-class accuracy
    class_acc = {}
    for label in class_total:
        class_acc[label] = 100 * class_correct[label] / class_total[label]
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'class_accuracy': class_acc,
        'predictions': all_preds,
        'labels': all_labels
    }

# ========== Main Function ==========
def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamped test directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join('tests', f"{args.model_type}_{timestamp}")
    os.makedirs(test_dir, exist_ok=True)
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract model parameters
    model_args = checkpoint.get('args', {})
    num_classes = checkpoint.get('num_classes', None) or len(checkpoint.get('classes', []))
    classes = checkpoint.get('classes', [])
    
    # Get number of views used in training
    trained_views = model_args.get('max_views', None)
    
    # Choose test views if not specified
    test_views = args.test_views or trained_views
    
    # Define transforms for testing
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset with unseen views
    test_dataset = UnseenViewsDataset(
        args.dataset_path,
        transform=test_transform,
        classes=classes,
        trained_views=trained_views,
        test_views=test_views,
        offset=args.offset,
        elevation_mode=args.elevation_mode,
        training_elevations=model_args.get('training_elevations', None),
        test_elevations=model_args.get('test_elevations', None)
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and args.device == 'cuda'
    )
    
    # Initialize model based on type
    if args.model_type == 'mvcnn':
        backbone = model_args.get('backbone', 'resnet18')
        dropout = model_args.get('dropout', 0.5)
        model = MVCNN(num_classes=num_classes, backbone=backbone, pretrained=False, dropout=dropout)
    else:  # motion_mvcnn
        motion_threshold = model_args.get('motion_threshold', 0.05)
        model = MotionEnhancedMVCNN(
            num_classes=num_classes, 
            motion_threshold=motion_threshold,
            motion_mode=args.motion_mode
        )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    print("Evaluating model on unseen views...")
    compute_flow = args.model_type == 'motion_mvcnn' and args.motion_mode == 'static'
    results = evaluate_model(model, test_loader, criterion, device, compute_flow=compute_flow)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {args.model_type}")
    if args.model_type == 'motion_mvcnn':
        print(f"Motion mode: {args.motion_mode}")
    print(f"Accuracy on unseen views: {results['accuracy']:.2f}%")
    print(f"Loss: {results['loss']:.4f}")
    
    # Save confusion matrix
    conf_matrix = confusion_matrix(results['labels'], results['predictions'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix ({args.model_type} - Unseen Views)')
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, 'confusion_matrix.png'))
    
    # Generate classification report
    class_report = classification_report(
        results['labels'], 
        results['predictions'],
        target_names=classes,
        output_dict=True
    )
    
    # Prepare test metrics
    test_metrics = {
        'model_type': args.model_type,
        'model_path': args.model_path,
        'accuracy': results['accuracy'],
        'loss': results['loss'],
        'class_accuracy': results['class_accuracy'],
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'classes': classes,
        'trained_views': trained_views,
        'test_views': test_views,
        'view_offset': args.offset,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'test_parameters': vars(args),
        'training_parameters': model_args
    }
    
    if args.model_type == 'motion_mvcnn':
        test_metrics['motion_mode'] = args.motion_mode
    
    # Save test metrics
    metrics_path = os.path.join(test_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Print per-class accuracy for the top 5 and bottom 5 classes
    class_acc = [(classes[int(i)], results['class_accuracy'][int(i)]) 
                for i in results['class_accuracy']]
    class_acc.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 classes:")
    for name, acc in class_acc[:5]:
        print(f"{name}: {acc:.2f}%")
    
    print("\nBottom 5 classes:")
    for name, acc in class_acc[-5:]:
        print(f"{name}: {acc:.2f}%")
    
    print(f"\nResults saved to: {test_dir}")

if __name__ == '__main__':
    main()