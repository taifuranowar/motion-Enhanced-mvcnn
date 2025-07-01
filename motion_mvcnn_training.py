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
    
    # Add elevation parameter
    parser.add_argument('--elevations', type=str, default=None,
                        help='Comma-separated list of elevations to use (e.g. "0,30"). If not specified, use all available elevations.')
    
    # Add max-views parameter
    parser.add_argument('--max-views', type=int, default=None,
                        help='Maximum number of views to use per model (default: use all available views)')
    
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
    parser.add_argument('--motion-mode', type=str, default='static', choices=['static', 'learnable'],
                        help='Motion estimation mode: static (OpenCV) or learnable (end-to-end flow-net)')
    
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
    def __init__(self, num_classes=40, motion_threshold=0.05, motion_mode='static'):
        super().__init__()
        self.motion_threshold = motion_threshold
        self.motion_mode = motion_mode

        resnet = resnet18(pretrained=True)
        self.conv1   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2   # (B,128,H/8,W/8)
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4   # (B,512,H/32,W/32)

        # Occlusion/refinement at both scales (now expects 2D+2 input)
        self.occl_s1 = nn.Sequential(
            nn.Conv2d(128*2 + 2, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),      nn.ReLU()
        )
        self.occl_s2 = nn.Sequential(
            nn.Conv2d(512*2 + 2, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),      nn.ReLU()
        )

        # Visibility heads for soft occlusion masks (coarse and fine)
        self.visibility_head_s1 = nn.Sequential(
            nn.Conv2d(128*2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1), nn.Sigmoid()
        )
        self.visibility_head_s2 = nn.Sequential(
            nn.Conv2d(512*2, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 2, 3, padding=1), nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128+512, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        if self.motion_mode == 'learnable':
            self.flow_net = SimpleFlowNet()
        else:
            self.flow_net = None

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

    def dynamic_view_schedule(self, motion_vectors, elevations=None, azimuths=None):
        """
        Enhanced algorithm: start at view 0, 
        then whenever avg motion from last_ref to i > threshold, mark i as new reference.
        Also ensure coverage of all elevations when elevation data is available.
        Always include the last view.
        """
        B, N, _, H, W, _ = motion_vectors.shape
        ref_indices = [0]
        last = 0
        mv = motion_vectors
        mag = mv.norm(dim=-1)            # (B,N,N,H,W)
        mag = mag.mean(dim=[0,3,4])      # (N,N)  mean over batch, H, W

        # If elevation data is provided, use it to enhance scheduling
        if elevations is not None and elevations.numel() > 0:
            elevs = elevations[0].cpu().numpy()  # First item in batch
            unique_elevs = sorted(set(elevs))
            
            # First pass: ensure we have at least one reference view per elevation
            if len(unique_elevs) > 1:  # Only needed for multiple elevations
                for elev in unique_elevs:
                    elev_indices = [i for i, e in enumerate(elevs) if e == elev]
                    # Check if we already have a ref view at this elevation
                    if not any(idx in ref_indices for idx in elev_indices):
                        # Find best view at this elevation (max motion or center view)
                        if len(elev_indices) > 0:
                            best_idx = elev_indices[len(elev_indices) // 2]  # Middle view as default
                            ref_indices.append(best_idx)

        # Second pass: add reference views based on motion threshold (existing logic)
        for i in range(1, N-1):
            if i not in ref_indices and mag[last, i] > self.motion_threshold:
                ref_indices.append(i)
                last = i
                
        # Always include the last view if not already included
        if N-1 not in ref_indices:
            ref_indices.append(N-1)
        
        # Sort indices to maintain original order
        return sorted(ref_indices)

    def estimate_motion(self, views, needed_pairs=None):
        """Estimate optical flow between all pairs of views using learnable flow-net if in learnable mode"""
        B, N, C, H, W = views.shape
        device = views.device
        if needed_pairs is None:
            needed_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        motion_vectors = {}
        for i, j in needed_pairs:
            img_i = views[:, i]
            img_j = views[:, j]
            flow = self.flow_net(img_i, img_j)  # (B, 2, H, W)
            flow = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
            motion_vectors[(i, j)] = flow
        return motion_vectors

    def forward(self, views, motion_vectors=None, elevations=None, azimuths=None):
        B, N, C, H, W = views.shape
        if self.motion_mode == 'learnable':
            # First, use zero motion to get initial ref schedule
            zero_motion = torch.zeros(B, N, N, H, W, 2, device=views.device)
            ref_idxs = self.dynamic_view_schedule(zero_motion, elevations, azimuths)
            pred_idxs = [i for i in range(N) if i not in ref_idxs]
            # Only compute needed pairs for warping
            needed_pairs = []
            for vid in pred_idxs:
                bef = max([r for r in ref_idxs if r < vid], default=None)
                aft = min([r for r in ref_idxs if r > vid], default=None)
                for r in (bef, aft):
                    if r is not None:
                        needed_pairs.append((r, vid))
            # Compute only needed flows
            motion_dict = self.estimate_motion(views, needed_pairs)
            # Now build a motion_vectors tensor for dynamic_view_schedule
            motion_vectors = torch.zeros(B, N, N, H, W, 2, device=views.device)
            for (i, j), flow in motion_dict.items():
                motion_vectors[:, i, j] = flow
        elif motion_vectors is None:
            motion_vectors = self.estimate_motion(views)
            ref_idxs = self.dynamic_view_schedule(motion_vectors, elevations, azimuths)
            pred_idxs = [i for i in range(N) if i not in ref_idxs]
        else:
            ref_idxs = self.dynamic_view_schedule(motion_vectors, elevations, azimuths)
            pred_idxs = [i for i in range(N) if i not in ref_idxs]
        ref1, ref2 = self.get_reference_views(views, ref_idxs)
        _, R, C1, h1, w1 = ref1.shape
        _, _, C2, h2, w2 = ref2.shape
        # Instead of in-place assignment, use lists and stack at the end for autograd safety
        all1_list = [None] * N
        all2_list = [None] * N
        for i, vid in enumerate(ref_idxs):
            all1_list[vid] = ref1[:, i]
            all2_list[vid] = ref2[:, i]
        for vid in pred_idxs:
            bef = max([r for r in ref_idxs if r < vid], default=None)
            aft = min([r for r in ref_idxs if r > vid], default=None)
            warps1, warps2 = [], []
            for r in (bef, aft):
                if r is None:
                    continue
                if self.motion_mode == 'learnable':
                    mv = motion_dict[(r, vid)]
                else:
                    mv = motion_vectors[:, r, vid]
                mv_s1 = F.interpolate(mv.permute(0, 3, 1, 2), size=(h1, w1), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
                mv_s2 = F.interpolate(mv.permute(0, 3, 1, 2), size=(h2, w2), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
                ref_f1 = all1_list[r].unsqueeze(1)
                ref_f2 = all2_list[r].unsqueeze(1)
                w1_ = self.warp_features(ref_f1, mv_s1)
                w2_ = self.warp_features(ref_f2, mv_s2)
                warps1.append(w1_)
                warps2.append(w2_)
            if len(warps1) == 2:
                # Coarse scale: soft mask fusion
                f1_cat = torch.cat(warps1, dim=1).view(B, 2 * C1, h1, w1)
                vis1 = self.visibility_head_s1(f1_cat)  # (B,2,h1,w1)
                combine1 = torch.cat([f1_cat, vis1], dim=1)
                all1_list[vid] = self.occl_s1(combine1)
                # Fine scale: soft mask fusion
                f2_cat = torch.cat(warps2, dim=1).view(B, 2 * C2, h2, w2)
                vis2 = self.visibility_head_s2(f2_cat)  # (B,2,h2,w2)
                combine2 = torch.cat([f2_cat, vis2], dim=1)
                all2_list[vid] = self.occl_s2(combine2)
        all1 = torch.stack(all1_list, dim=1)
        all2 = torch.stack(all2_list, dim=1)
        B, N, C1, h1, w1 = all1.shape
        _, _, C2, h2, w2 = all2.shape
        all1_reshape = all1.view(B * N, C1, h1, w1)
        up1 = F.interpolate(all1_reshape, size=(h2, w2), mode='bilinear', align_corners=True)
        up1 = up1.view(B, N, C1, h2, w2)
        fused = torch.cat([up1, all2], dim=2)
        global_feat, _ = fused.max(dim=1)
        return self.classifier(global_feat)

# ========== Dataset Handling ==========
class MotionMVCNNDataset(Dataset):
    """Motion-Enhanced Multi-View CNN Dataset"""
    printed_view_count = False  # Class variable to control printing

    def __init__(self, dataset_path, split='train', transform=None, selected_classes=None, compute_flow=True, max_views=None, elevations=None):
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        self.selected_classes = selected_classes
        self.compute_flow = compute_flow
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
            class_name = class_data.get('class_name', class_data.get('class'))
            if self.selected_classes is not None and class_name not in self.selected_classes:
                continue
            self.classes.append(class_name)
            self.class_to_idx[class_name] = len(self.classes) - 1
        
        self.samples = []
        
        # Build dataset samples list
        for class_data in self.metadata['classes']:
            class_name = class_data.get('class_name', class_data.get('class'))
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
        if not MotionMVCNNDataset.printed_view_count:
            print(f"Number of views for sample {idx}: {len(sample['view_files'])}")
            if 'view_elevations' in sample:
                unique_elevs = sorted(set(sample['view_elevations']))
                print(f"Elevations used: {unique_elevs}")
                for elev in unique_elevs:
                    count = sum(1 for e in sample['view_elevations'] if e == elev)
                    print(f"  Elevation {elev}°: {count} views")
            MotionMVCNNDataset.printed_view_count = True
            
        for view_file in sample['view_files']:
            img = Image.open(view_file).convert('RGB')
            if self.transform:
                img = self.transform(img)
            views.append(img)
        views = torch.stack(views)
        
        # Convert elevation and azimuth data to tensors
        elevations = torch.tensor(sample['view_elevations'], dtype=torch.float32) if 'view_elevations' in sample else None
        azimuths = torch.tensor(sample['view_azimuths'], dtype=torch.float32) if 'view_azimuths' in sample else None
        
        # Only return motion_vectors if compute_flow is True (static mode)
        if self.compute_flow:
            num_views = views.shape[0]
            H, W = views.shape[2], views.shape[3]
            motion_vectors = torch.zeros(num_views, num_views, H, W, 2)
            for i in range(num_views):
                img_i = views[i].permute(1,2,0).numpy()  # (H,W,C)
                img_i_gray = cv2.cvtColor((img_i*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                for j in range(num_views):
                    if i != j:
                        img_j = views[j].permute(1,2,0).numpy()
                        img_j_gray = cv2.cvtColor((img_j*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                        flow = cv2.calcOpticalFlowFarneback(
                            img_i_gray, img_j_gray, None, 
                            0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        flow = cv2.resize(flow, (W, H))
                        motion_vectors[i, j] = torch.from_numpy(flow).float()
            return {
                'views': views,
                'motion_vectors': motion_vectors,
                'label': sample['class_idx'],
                'class_name': sample['class_name'],
                'model_name': sample['model_name'],
                'elevations': elevations,
                'azimuths': azimuths
            }
        else:
            return {
                'views': views,
                'label': sample['class_idx'],
                'class_name': sample['class_name'],
                'model_name': sample['model_name'],
                'elevations': elevations,
                'azimuths': azimuths
            }

# ========== Training and Evaluation Functions ==========
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, data in enumerate(dataloader):
        views = data['views'].to(device)
        labels = data['label'].to(device)
        elevations = data.get('elevations', None)
        azimuths = data.get('azimuths', None)
        if elevations is not None:
            elevations = elevations.to(device)
        if azimuths is not None:
            azimuths = azimuths.to(device)
        if 'motion_vectors' in data:
            motion_vectors = data['motion_vectors'].to(device)
            outputs = model(views, motion_vectors, elevations, azimuths)
        else:
            outputs = model(views, None, elevations, azimuths)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
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
            if 'motion_vectors' in data:
                motion_vectors = data['motion_vectors'].to(device)
                outputs = model(views, motion_vectors)
            else:
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
        selected_classes = [c['class_name'] for c in metadata['classes']]

    # Apply num-classes if specified
    if args.num_classes is not None:
        selected_classes = selected_classes[:args.num_classes]
    print(f"Selected classes: {selected_classes}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = os.path.join('motion_mvcnn_results', f'train_{timestamp}')
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
    compute_flow = args.motion_mode == 'static'
    train_dataset = MotionMVCNNDataset(
        args.dataset_path, 
        split='train', 
        transform=train_transform, 
        selected_classes=selected_classes, 
        compute_flow=compute_flow, 
        max_views=args.max_views,
        elevations=args.elevations
    )
    test_dataset = MotionMVCNNDataset(
        args.dataset_path, 
        split='test', 
        transform=test_transform,
        selected_classes=selected_classes, 
        compute_flow=compute_flow, 
        max_views=args.max_views,
        elevations=args.elevations
    )
    
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
                              motion_threshold=args.motion_threshold,
                              motion_mode=args.motion_mode)
    
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