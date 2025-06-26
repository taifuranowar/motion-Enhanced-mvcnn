import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import numpy as np
import cv2

# Configuration
NUM_VIEWS = 12
REF_VIEW_STRIDE = 4  # Not used in dynamic scheduling, kept for compatibility
FEATURE_DIM = 512
NUM_CLASSES = 40  # ModelNet40 classes
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

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

class MotionEnhancedMVCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, motion_threshold=0.05):
        super().__init__()
        self.motion_threshold = motion_threshold

        # Base feature extractor (shared weights)
        self.base_cnn = resnet18(pretrained=True)
        self.base_cnn = nn.Sequential(*list(self.base_cnn.children())[:-2])  # Remove avgpool and fc

        # Lightweight flow estimator (learnable, end-to-end)
        self.flow_net = SimpleFlowNet(in_channels=6, out_channels=2)

        # Occlusion handling and refinement module
        self.occlusion_handler = nn.Sequential(
            nn.Conv2d(FEATURE_DIM * 2 + 2, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, FEATURE_DIM, 3, padding=1),
            nn.ReLU()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

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

        # Precompute mean motion mags from each last->i
        mv = motion_vectors  # alias
        mag = mv.norm(dim=-1)            # (B,N,N,H,W)
        mag = mag.mean(dim=[0,2,3,4])    # (N,)  mean over B, dest-view index, H, W

        for i in range(1, N-1):
            if mag[last, i] > self.motion_threshold:
                ref_indices.append(i)
                last = i

        # always include final view
        if ref_indices[-1] != N-1:
            ref_indices.append(N-1)

        return ref_indices

    def get_reference_views(self, all_views, ref_indices):
        """
        Args:
            all_views: (B,N,C,H,W)
            ref_indices: list of ints
        Returns:
            ref_feats: (B, len(ref_indices), FEATURE_DIM, h, w)
        """
        B, N, C, H, W = all_views.shape
        refs = all_views[:, ref_indices]              # (B, R, C, H, W)
        flat  = refs.reshape(-1, C, H, W)             # (B*R, C, H, W)
        feats = self.base_cnn(flat)                   # (B*R, D, h, w)
        D, h, w = feats.shape[1:]
        return feats.view(B, -1, D, h, w)

    def estimate_motion(self, views):
        """
        Estimate motion fields between all pairs of views using the learnable flow_net.
        Args:
            views: (B, N, C, H, W)
        Returns:
            motion_vectors: (B, N, N, H, W, 2)
        """
        B, N, C, H, W = views.shape
        device = views.device
        motion_vectors = torch.zeros(B, N, N, H, W, 2, device=device)
        for i in range(N):
            img_i = views[:, i]  # (B, C, H, W)
            for j in range(N):
                if i == j:
                    continue
                img_j = views[:, j]
                flow = self.flow_net(img_i, img_j)  # (B, 2, H, W)
                motion_vectors[:, i, j] = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
        return motion_vectors

    def warp_features(self, ref_features, motion_vectors):
        """Warp reference features using motion vectors"""
        batch_size, num_refs, feat_dim, feat_h, feat_w = ref_features.shape

        # Create sampling grid from motion vectors
        B, H, W, _ = motion_vectors.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=motion_vectors.device), torch.arange(W, device=motion_vectors.device), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float()  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)

        # Apply motion vectors and normalize to [-1, 1]
        wh = torch.tensor([W, H], dtype=torch.float32, device=motion_vectors.device)
        warped_grid = (grid + motion_vectors) * 2.0 / wh - 1.0  # (B, H, W, 2)

        # grid_sample expects (B, C, H, W) and grid (B, H, W, 2)
        warped_features = F.grid_sample(
            ref_features.view(-1, feat_dim, feat_h, feat_w),
            warped_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        return warped_features.view(batch_size, -1, feat_dim, feat_h, feat_w)

    def forward(self, views, motion_vectors=None):
        """
        views:            (B, N, C, H, W)
        motion_vectors:   (B, N, N, H, W, 2) or None
        """
        B, N, C, H, W = views.shape

        # If motion_vectors not provided, estimate them with the learnable flow_net
        if motion_vectors is None:
            motion_vectors = self.estimate_motion(views)  # (B, N, N, H, W, 2)

        # 1) pick refs dynamically
        ref_indices  = self.dynamic_view_schedule(motion_vectors)
        pred_indices = [i for i in range(N) if i not in ref_indices]

        # 2) extract ref features
        ref_feats = self.get_reference_views(views, ref_indices)
        _, R, D, fh, fw = ref_feats.shape

        # 3) build feature table
        all_feats = views.new_zeros(B, N, D, fh, fw)
        for ridx, vid in enumerate(ref_indices):
            all_feats[:, vid] = ref_feats[:, ridx]

        # 4) warp for the predicted views
        for vid in pred_indices:
            # find nearest before/after refs
            before = max([r for r in ref_indices if r < vid], default=None)
            after  = min([r for r in ref_indices if r > vid], default=None)

            warped_list, mask_list = [], []
            for r in (before, after):
                if r is None: continue
                mv      = motion_vectors[:, r, vid]      # (B,H,W,2)
                ref_f   = all_feats[:, r].unsqueeze(1)   # (B,1,D,fh,fw)
                warped  = self.warp_features(ref_f, mv)  # (B,1,D,fh,fw)

                # simple occlusion mask
                occ_m   = (mv.norm(dim=-1) < 20.0).unsqueeze(1).float()  # (B,1,fh,fw)
                warped_list.append(warped)
                mask_list.append(occ_m)

            if len(warped_list) == 2:
                feats_cat = torch.cat(warped_list, dim=1)  # (B,2,D,fh,fw)
                masks_cat = torch.cat(mask_list,  dim=1)   # (B,2,fh,fw)
                combine   = torch.cat([
                    feats_cat.view(B, -1, fh, fw), masks_cat
                ], dim=1)                               # (B,2D+2,fh,fw)
                refined   = self.occlusion_handler(combine)
                all_feats[:, vid] = refined

        # 5) pool & classify
        global_feat, _ = all_feats.max(dim=1)
        return self.classifier(global_feat)

class ModelNetMotionDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.samples = []  # List of (obj_id, class_id)
        
        # Populate samples (implementation depends on dataset structure)
        # For demo, leave as empty or implement as needed.
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        obj_id, class_id = self.samples[idx]
        
        # Load multi-view images
        views = []
        for i in range(NUM_VIEWS):
            img_path = f"{self.root_dir}/{obj_id}/view_{i:02d}.png"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            views.append(img)
        views = np.stack(views)
        
        # Load precomputed motion vectors (for demo: using zeros)
        motion_vectors = np.zeros((NUM_VIEWS, NUM_VIEWS, views.shape[2], views.shape[3], 2), dtype=np.float32)
        
        # Example: If you want to compute optical flow, you need two images (not implemented here)
        # for i in range(NUM_VIEWS):
        #     for j in range(NUM_VIEWS):
        #         if i != j:
        #             flow = cv2.calcOpticalFlowFarneback(
        #                 cv2.cvtColor(views[i].transpose(1,2,0), cv2.COLOR_RGB2GRAY),
        #                 cv2.cvtColor(views[j].transpose(1,2,0), cv2.COLOR_RGB2GRAY),
        #                 None, 0.5, 3, 15, 3, 5, 1.2, 0
        #             )
        #             motion_vectors[i, j] = flow
        
        return {
            'views': torch.tensor(views, dtype=torch.float32),
            'motion_vectors': torch.tensor(motion_vectors, dtype=torch.float32),
            'label': class_id
        }

def main():
    # Initialize dataset and dataloader
    train_dataset = ModelNetMotionDataset('path/to/modelnet_motion', 'train')
    val_dataset = ModelNetMotionDataset('path/to/modelnet_motion', 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MotionEnhancedMVCNN().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            views = batch['views'].to(device)
            motion_vectors = batch['motion_vectors'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(views, motion_vectors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                views = batch['views'].to(device)
                motion_vectors = batch['motion_vectors'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(views, motion_vectors)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Print statistics
        train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        val_acc = 100. * correct / total if total > 0 else 0
        
        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
    
    print("Training complete")

if __name__ == "__main__":
    main()