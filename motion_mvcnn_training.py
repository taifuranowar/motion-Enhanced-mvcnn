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

    def forward(self, views, motion_vectors):
        B,N,_,H,W = views.shape

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