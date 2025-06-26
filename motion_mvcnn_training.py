import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import json
import argparse
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ========== Command Line Arguments ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Motion Enhanced MVCNN Training')
    
    # Basic configuration
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='Path to the generated MVCNN dataset')
    parser.add_argument('--output-dir', type=str, default='motion_mvcnn_results',
                        help='Output directory for trained model and results')
    
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
    parser.add_argument('--motion-threshold', type=float, default=0.05,
                        help='Threshold for motion-based view scheduling')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for fully connected layers')

    # Class selection
    parser.add_argument('--selected-classes', type=str, default=None,
                        help='Comma-separated list of class names to use (e.g. "chair,table,sofa,bed,car")')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes to use for training (selects first N classes, optionally from selected-classes)')
    
    # Save/load parameters
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Save model every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    return args

# ========== SimpleFlowNet ==========
class SimpleFlowNet(nn.Module):
    """A lightweight flow estimator for demonstration (not accurate, but differentiable)."""
    def __init__(self, in_channels=6, out_channels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1)
        )

    def forward(self, img1, img2):
        # img1, img2: (B, C, H, W)
        x = torch.cat([img1, img2], dim=1)
        return self.net(x)  # (B, 2, H, W)

# ========== Motion Enhanced MVCNN Model ==========
class MotionEnhancedMVCNN(nn.Module):
    def __init__(self, num_classes=40, motion_threshold=0.05):
        super().__init__()
        self.motion_threshold = motion_threshold

        # 1) Split ResNet-18 into explicit blocks
        resnet = resnet18(pretrained=True)
        self.conv1   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2   # ⇒ coarse features (B,128,H/8,W/8)
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4   # ⇒ fine features   (B,512,H/32,W/32)

        # 2) Occlusion/refinement at both scales
        self.occl_s1 = nn.Sequential(
            nn.Conv2d(128*2 + 2, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),      nn.ReLU()
        )
        self.occl_s2 = nn.Sequential(
            nn.Conv2d(512*2 + 2, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),      nn.ReLU()
        )

        # 3) Final classifier now takes (128+512)-dim global feature
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128+512, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def extract_scales(self, x):
        """Run x through ResNet up to layer2 (coarse) and layer4 (fine)."""
        x  = self.conv1(x)
        x  = self.layer1(x)
        s1 = self.layer2(x)   # (B,128,H/8,W/8)
        x  = self.layer3(s1)
        s2 = self.layer4(x)   # (B,512,H/32,W/32)
        return s1, s2

    def get_reference_views(self, all_views, ref_indices):
        B,N,C,H,W = all_views.shape
        # pull out the reference images
        flat = all_views[:, ref_indices].reshape(-1,C,H,W)
        # get multi-scale feats
        f1,f2 = self.extract_scales(flat)  
        # reshape back: (B, R, C1, h1, w1) & (B, R, C2, h2, w2)
        _,C1,h1,w1 = f1.shape
        _,C2,h2,w2 = f2.shape
        f1 = f1.view(B, -1, C1, h1, w1)
        f2 = f2.view(B, -1, C2, h2, w2)
        return f1, f2

    def warp_features(self, ref_features, motion_vectors):
        """
        ref_features: (B, 1, C, h, w)
        motion_vectors: (B, h, w, 2)
        Returns: (B, 1, C, h, w)
        """
        B, one, C, h, w = ref_features.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=motion_vectors.device),
            torch.arange(w, device=motion_vectors.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).float()  # (h, w, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, h, w, 2)
        wh = torch.tensor([w, h], dtype=torch.float32, device=motion_vectors.device)
        warped_grid = (grid + motion_vectors) * 2.0 / wh - 1.0  # (B, h, w, 2)
        warped = F.grid_sample(
            ref_features.view(B, C, h, w),
            warped_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        return warped.view(B, 1, C, h, w)

    def dynamic_view_schedule(self, motion_vectors):
        """
        Greedy algorithm: start at view 0, 
        then whenever avg motion from last_ref to i > threshold, mark i as new reference.
        Always include the last view.
        Args:
            motion_vectors: (B, N, N, H, W, 2)
        Returns:
            ref_indices: list of ints
        """
        B, N, _, H, W, _ = motion_vectors.shape
        ref_indices = [0]
        last = 0
        mv = motion_vectors
        mag = mv.norm(dim=-1)
        mag = mag.mean(dim=[0,2,3,4])
        for i in range(1, N-1):
            if mag[last, i] > self.motion_threshold:
                ref_indices.append(i)
                last = i
        if ref_indices[-1] != N-1:
            ref_indices.append(N-1)
        return ref_indices

    def estimate_motion(self, views):
        """Estimate optical flow between all pairs of views"""
        B, N, C, H, W = views.shape
        motion_vectors = torch.zeros(B, N, N, H, W, 2, device=views.device)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                # Simple approximation using scaled coordinate differences
                # In a real implementation, use optical flow or learned flow
                flow_x = torch.zeros((B, H, W), device=views.device)
                flow_y = torch.zeros((B, H, W), device=views.device)
                motion_vectors[:, i, j, :, :, 0] = flow_x
                motion_vectors[:, i, j, :, :, 1] = flow_y
        return motion_vectors

    def forward(self, views, motion_vectors=None):
        B, N, C, H, W = views.shape

        # Estimate motion if not provided
        if motion_vectors is None:
            motion_vectors = self.estimate_motion(views)

        # dynamic scheduler stays the same…
        ref_idxs = self.dynamic_view_schedule(motion_vectors)
        pred_idxs= [i for i in range(N) if i not in ref_idxs]

        # 1) extract and stash ref features at both scales
        ref1, ref2 = self.get_reference_views(views, ref_idxs)
        _,R,C1,h1,w1 = ref1.shape
        _,_,C2,h2,w2 = ref2.shape

        all1 = views.new_zeros(B, N, C1, h1, w1)
        all2 = views.new_zeros(B, N, C2, h2, w2)
        for i,vid in enumerate(ref_idxs):
            all1[:,vid] = ref1[:,i]
            all2[:,vid] = ref2[:,i]

        # 2) for each predicted view, warp & refine at both scales
        for vid in pred_idxs:
            bef = max([r for r in ref_idxs if r<vid], default=None)
            aft = min([r for r in ref_idxs if r>vid], default=None)
            warps1, masks1, warps2, masks2 = [], [], [], []
            for r in (bef,aft):
                if r is None: continue

                # downsample motion to coarse / fine grids
                mv     = motion_vectors[:,r,vid]                          # (B,H,W,2)
                mv_s1  = F.interpolate(mv.permute(0,3,1,2), size=(h1,w1),
                                       mode='bilinear', align_corners=True
                                      ).permute(0,2,3,1)
                mv_s2  = F.interpolate(mv.permute(0,3,1,2), size=(h2,w2),
                                       mode='bilinear', align_corners=True
                                      ).permute(0,2,3,1)

                # warp
                ref_f1 = all1[:,r].unsqueeze(1)    # (B,1,C1,h1,w1)
                ref_f2 = all2[:,r].unsqueeze(1)    # (B,1,C2,h2,w2)
                w1_   = self.warp_features(ref_f1, mv_s1)
                w2_   = self.warp_features(ref_f2, mv_s2)

                # simple occlusion masks
                occ1  = (mv_s1.norm(dim=-1)<self.motion_threshold).unsqueeze(1)
                occ2  = (mv_s2.norm(dim=-1)<self.motion_threshold).unsqueeze(1)

                warps1.append(w1_); masks1.append(occ1)
                warps2.append(w2_); masks2.append(occ2)

            if len(warps1)==2:
                # refine coarse
                f1_cat = torch.cat(warps1,dim=1).view(B,2*C1,h1,w1)
                m1_cat = torch.cat(masks1,dim=1)
                all1[:,vid] = self.occl_s1(torch.cat([f1_cat,m1_cat],dim=1))

                # refine fine
                f2_cat = torch.cat(warps2,dim=1).view(B,2*C2,h2,w2)
                m2_cat = torch.cat(masks2,dim=1)
                all2[:,vid] = self.occl_s2(torch.cat([f2_cat,m2_cat],dim=1))

        # 3) fuse scales & classify
        up1    = F.interpolate(all1, size=(h2,w2),
                               mode='bilinear', align_corners=True)
        fused  = torch.cat([up1, all2], dim=2)    # (B,N,C1+C2,h2,w2)
        global_feat,_ = fused.max(dim=1)          # (B,C1+C2,h2,w2)
        return self.classifier(global_feat)

# ========== Dataset Handling ==========
class MotionMVCNNDataset(Dataset):
    """Motion-Enhanced Multi-View CNN Dataset"""
    def __init__(self, dataset_path, split='train', transform=None, selected_classes=None, compute_flow=True):
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        self.selected_classes = selected_classes
        self.compute_flow = compute_flow
        
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
                
                # Get view filenames sorted by view_idx
                view_files = []
                for view in sorted(model_metadata['views'], key=lambda x: x['view_idx']):
                    view_files.append(os.path.join(model_path, view['filename']))
                
                self.samples.append({
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'model_name': model_name,
                    'view_files': view_files
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load all views for this model
        views = []
        raw_views = []  # Keep originals for flow computation
        for view_file in sample['view_files']:
            img = Image.open(view_file).convert('RGB')
            raw_views.append(np.array(img))
            if self.transform:
                img = self.transform(img)
            views.append(img)
        
        # Stack views into a tensor [num_views, channels, height, width]
        views = torch.stack(views)
        
        # Create empty motion vectors tensor
        num_views = len(views)
        
        # Structure for motion vectors: [num_views, num_views, H, W, 2]
        motion_vectors = torch.zeros(num_views, num_views, views.shape[2], views.shape[3], 2)
        
        # Compute optical flow between views if requested
        if self.compute_flow:
            for i in range(num_views):
                img_i = raw_views[i]
                img_i_gray = cv2.cvtColor(img_i, cv2.COLOR_RGB2GRAY)
                
                for j in range(num_views):
                    if i != j:
                        img_j = raw_views[j]
                        img_j_gray = cv2.cvtColor(img_j, cv2.COLOR_RGB2GRAY)
                        
                        # Calculate optical flow
                        flow = cv2.calcOpticalFlowFarneback(
                            img_i_gray, img_j_gray, None, 
                            0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        
                        # Resize flow to match transformed image size
                        flow = cv2.resize(flow, (views.shape[2], views.shape[3]))
                        motion_vectors[i, j] = torch.from_numpy(flow).float()
        
        return {
            'views': views,
            'motion_vectors': motion_vectors,
            'label': sample['class_idx'],
            'class_name': sample['class_name'],
            'model_name': sample['model_name']
        }

# ========== Training and Evaluation Functions ==========
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, data in enumerate(dataloader):
        views = data['views'].to(device)
        motion_vectors = data['motion_vectors'].to(device)
        labels = data['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(views, motion_vectors)
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
            motion_vectors = data['motion_vectors'].to(device)
            labels = data['label'].to(device)
            
            outputs = model(views, motion_vectors)
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
        selected_classes = [c['class_name'] for c in metadata['classes']]

    # Apply num-classes if specified
    if args.num_classes is not None:
        selected_classes = selected_classes[:args.num_classes]
    print(f"Selected classes: {selected_classes}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    train_dataset = MotionMVCNNDataset(args.dataset_path, split='train', transform=train_transform, 
                                      selected_classes=selected_classes)
    test_dataset = MotionMVCNNDataset(args.dataset_path, split='test', transform=test_transform,
                                     selected_classes=selected_classes)
    
    # Create data loaders
    use_pin_memory = torch.cuda.is_available() and args.device == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_pin_memory)
    
    print(f"Dataset loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    print(f"Classes: {len(train_dataset.classes)}")
    
    # Create model
    model = MotionEnhancedMVCNN(num_classes=len(train_dataset.classes), 
                              motion_threshold=args.motion_threshold)
    
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

if __name__ == '__main__':
    main()