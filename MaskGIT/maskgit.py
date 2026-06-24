import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class MaskGIT(nn.Module):
    def __init__(self, base_model, mlm_layer, img_size=32):
        super().__init__()
        self.base_model = base_model
        self.mlm_layer = mlm_layer
        self.img_size = img_size
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def cosine_schedule(self, t):
        return math.cos((math.pi / 2) * t)

    def forward_step(self, images):
        B, C, H, W = images.shape
        device = images.device
        
        t = torch.rand(B, 1, 1, 1, device=device)
        mask_ratio = self.cosine_schedule(t)
        rand_matrix = torch.rand(B, 1, H, W, device=device)
        bool_mask = rand_matrix < mask_ratio
        norm_images = (images.float() / 127.5) - 1.0 
        
        masked_images = norm_images.clone()
        masked_images = masked_images * (~bool_mask).float() 
        unet_input = torch.cat([masked_images, bool_mask.float()], dim=1)
        
        features = self.base_model(unet_input)
        logits = self.mlm_layer(features)
        
        bool_mask_expanded = bool_mask.expand(-1, 3, -1, -1) 
        
        loss = self.criterion(logits, images.long())
        masked_loss = (loss * bool_mask_expanded.float()).sum() / (bool_mask_expanded.float().sum() + 1e-5)
        
        return masked_loss

    @torch.no_grad()
    def deploy(self, num_samples, T=10, device='cpu'):
        self.eval()
        B, C, H, W = num_samples, 3, self.img_size, self.img_size
        
        current_img = torch.zeros(B, C, H, W, device=device, dtype=torch.long)
        bool_mask = torch.ones(B, 1, H, W, device=device, dtype=torch.bool)
        
        for step in range(T):
            ratio = step / T
            next_ratio = self.cosine_schedule((step + 1) / T) 
            norm_images = (current_img.float() / 127.5) - 1.0
            norm_images = norm_images * (~bool_mask).float()
            unet_input = torch.cat([norm_images, bool_mask.float()], dim=1)
            
            features = self.base_model(unet_input)
            logits = self.mlm_layer(features)
            
            probs = F.softmax(logits, dim=2)
            max_probs, preds = torch.max(probs, dim=2) 
            
            current_img = torch.where(bool_mask.expand(-1, 3, -1, -1), preds, current_img)
            avg_confidence = max_probs.mean(dim=1, keepdim=True) 
            
            avg_confidence[~bool_mask] = 100.0 
            flat_conf = avg_confidence.view(B, -1)
            k = int(next_ratio * H * W)
            
            if k > 0:
                threshold_val = torch.kthvalue(flat_conf, k, dim=1).values.view(B, 1, 1, 1)
                bool_mask = avg_confidence <= threshold_val
            else:
                bool_mask = torch.zeros_like(bool_mask)
                
        return current_img