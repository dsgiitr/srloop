import torch
from src.model import ElasticTransformerMLM

if __name__ == "__main__":
    # 1. Setup a highly custom configuration
    VOCAB_SIZE = 1024  # Size of the visual codebook dictionary

    model = ElasticTransformerMLM(
        img_size=128,       # Adjustable Resolution (e.g., 128x128)
        patch_size=8,       # Adjustable Patch Splitting (128/8 = 256 tokens)
        embed_dim=256,      # Inner vector dimension
        num_layers_N=3,     # N = 3 unique layers inside the block
        vocabulary_size=VOCAB_SIZE
    )

    # Total parameter count verification
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Physical Model Parameters: {total_params / 1e6:.2f} Million")

    # Create mock batch of images [Batch size=2, Channels=3, H=128, W=128]
    dummy_images = torch.randn(2, 3, 128, 128)

    # 2. Emulate an On-Device Scenario (Low latency budget: L = 2 loops)
    # Effective Depth D = 3 layers * 2 loops = 6 total mathematical transformations
    low_budget_logits = model(dummy_images, num_loops_L=2)
    print(f"Low-Budget Output Shape (L=2): {low_budget_logits.shape}")

    # 3. Emulate a Cloud-Server Scenario (High fidelity budget: L = 10 loops)
    # Effective Depth D = 3 layers * 10 loops = 30 total mathematical transformations
    high_budget_logits = model(dummy_images, num_loops_L=10)
    print(f"High-Budget Output Shape (L=10): {high_budget_logits.shape}")

    # Confirming the tensors output format matches expectations for Cross-Entropy Loss
    assert low_budget_logits.shape == high_budget_logits.shape == (2, 256, VOCAB_SIZE)
    print("Success! The architecture is completely decoupled and functionally elastic.")