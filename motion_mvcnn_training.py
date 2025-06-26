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
REF_VIEW_STRIDE = 4  # Process every 4th view as reference
FEATURE_DIM = 512
NUM_CLASSES = 40  # ModelNet40 classes
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

class MotionEnhancedMVCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        # Base feature extractor (shared weights)
        self.base_cnn = resnet18(pretrained=True)
        self.base_cnn = nn.Sequential(*list(self.base_cnn.children())[:-2])  # Remove avgpool and fc
        
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
    
    def get_reference_views(self, all_views):
        """Process reference views through full CNN"""
        batch_size, num_views, C, H, W = all_views.shape
        ref_views = all_views[:, ::REF_VIEW_STRIDE, :, :, :]
        ref_views = ref_views.reshape(-1, C, H, W)
        ref_features = self.base_cnn(ref_views)
        # Calculate output spatial size after base_cnn
        feat_h, feat_w = ref_features.shape[-2], ref_features.shape[-1]
        ref_features = ref_features.view(batch_size, -1, FEATURE_DIM, feat_h, feat_w)
        return ref_features
    
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
    
    def forward(self, views, motion_vectors):
        """
        Args:
            views: (B, N, C, H, W) tensor of input views
            motion_vectors: (B, N, N, H, W, 2) tensor of motion vectors
        """
        B, N, C, H, W = views.shape
        
        # Identify reference and predicted views
        ref_indices = list(range(0, N, REF_VIEW_STRIDE))
        pred_indices = [i for i in range(N) if i not in ref_indices]
        
        # Process reference views
        ref_features = self.get_reference_views(views)  # (B, num_refs, F, h, w)
        feat_h, feat_w = ref_features.shape[-2], ref_features.shape[-1]
        
        # Initialize feature tensor
        all_features = torch.zeros(B, N, FEATURE_DIM, feat_h, feat_w, 
                                  device=views.device)
        all_features[:, ref_indices] = ref_features
        
        # Process predicted views using motion warping
        for idx in pred_indices:
            # Find nearest reference views
            ref_before = max([i for i in ref_indices if i < idx], default=None)
            ref_after = min([i for i in ref_indices if i > idx], default=None)
            
            # Bidirectional feature warping
            warped_features = []
            occlusion_masks = []
            
            for ref_idx in [ref_before, ref_after]:
                if ref_idx is not None:
                    # Get motion vectors from current view to reference view
                    mv = motion_vectors[:, idx, ref_idx]  # (B, H, W, 2)
                    
                    # Warp reference features
                    ref_feat = all_features[:, ref_idx].unsqueeze(1)  # (B, 1, F, h, w)
                    warped = self.warp_features(ref_feat, mv)  # (B, 1, F, h, w)
                    
                    # Compute occlusion mask (simple threshold for demo)
                    occlusion_mask = (torch.norm(mv, dim=-1) < 20.0)
                    occlusion_mask = occlusion_mask.unsqueeze(1).float()  # (B, 1, H, W)
                    
                    warped_features.append(warped)
                    occlusion_masks.append(occlusion_mask)
            
            # Combine warped features and handle occlusions
            if len(warped_features) == 2:
                # Concatenate features and occlusion masks
                features_cat = torch.cat(warped_features, dim=1)  # (B, 2, F, h, w)
                masks_cat = torch.cat(occlusion_masks, dim=1)      # (B, 2, H, W)
                
                # Process through occlusion handler
                combined = torch.cat([
                    features_cat.view(B, -1, feat_h, feat_w), 
                    masks_cat
                ], dim=1)
                
                refined = self.occlusion_handler(combined)
                all_features[:, idx] = refined.squeeze(1) if refined.dim() == 5 else refined
        
        # View aggregation (max pooling)
        global_feature, _ = torch.max(all_features, dim=1)
        
        # Classification
        return self.classifier(global_feature)

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