import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# =====================================================================
# 1. DIRECTORY PLACEHOLDERS
# =====================================================================
# Update these paths to point to your saved 128x128 images
GENERATED_IMAGES_DIR = "/path/to/your/saved/generated_128x128_images"
GROUND_TRUTH_DIR = "/path/to/your/saved/ground_truth_128x128_images"

# =====================================================================
# 2. EVALUATION CLASS
# =====================================================================
class MetricsScorekeeper:
    def __init__(self, device: str = 'cuda', data_range: float = 1.0):
        self.device = torch.device(device)
        self.data_range = data_range
        
        # Initialize standard metrics
        self.psnr = PeakSignalNoiseRatio(data_range=self.data_range).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=self.data_range).to(self.device)
        
        # Initialize perceptual metric (LPIPS with VGG backbone)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg', 
            normalize=True
        ).to(self.device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Move to GPU/CPU
        preds = preds.to(self.device)
        target = target.to(self.device)
        
        # Clamp predictions to valid [0, 1] range to avoid math errors
        preds = torch.clamp(preds, 0.0, self.data_range)

        # Accumulate metrics
        self.psnr.update(preds, target)
        self.ssim.update(preds, target)
        self.lpips.update(preds, target)

    def compute(self):
        return {
            "PSNR": self.psnr.compute().item(),
            "SSIM": self.ssim.compute().item(),
            "LPIPS": self.lpips.compute().item()
        }

# =====================================================================
# 3. MAIN EXECUTION LOOP
# =====================================================================
def run_evaluation_from_disk():
    # Automatically use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running evaluation on: {device}")
    
    scorekeeper = MetricsScorekeeper(device=device)
    
    # Get a sorted list of filenames from the generated folder
    image_filenames = sorted(os.listdir(GENERATED_IMAGES_DIR))
    
    # Filter for valid image extensions (ignores hidden files)
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_filenames = [f for f in image_filenames if f.lower().endswith(valid_exts)]
    
    if len(image_filenames) == 0:
        print("No images found! Please check your GENERATED_IMAGES_DIR path.")
        return

    print(f"Found {len(image_filenames)} images to evaluate.")

    for filename in tqdm(image_filenames, desc="Evaluating Images"):
        gen_path = os.path.join(GENERATED_IMAGES_DIR, filename)
        gt_path = os.path.join(GROUND_TRUTH_DIR, filename)
        
        # Safety check: Ensure the matching ground truth image exists
        if not os.path.exists(gt_path):
            print(f"\nWarning: Ground truth not found for {filename}. Skipping.")
            continue
            
        # Load images and convert to RGB
        gen_img_pil = Image.open(gen_path).convert("RGB")
        gt_img_pil = Image.open(gt_path).convert("RGB")
        
        # Convert to PyTorch Tensors: Scales 0-255 to 0.0-1.0 and changes shape to (C, H, W)
        gen_tensor = TF.to_tensor(gen_img_pil).unsqueeze(0) # Adds batch dim: (1, C, H, W)
        gt_tensor = TF.to_tensor(gt_img_pil).unsqueeze(0)
        
        # Feed directly to scorekeeper
        scorekeeper.update(preds=gen_tensor, target=gt_tensor)

    # Calculate and print final results
    final_scores = scorekeeper.compute()
    
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    print(f"PSNR:  {final_scores['PSNR']:.4f} dB (Higher is better)")
    print(f"SSIM:  {final_scores['SSIM']:.4f} (Higher is better)")
    print(f"LPIPS: {final_scores['LPIPS']:.4f} (Lower is better)")
    print("="*30)

# =====================================================================
# 4. RUN
# =====================================================================
if __name__ == "__main__":
    run_evaluation_from_disk()
