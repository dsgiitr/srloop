import sys
import os

# Add DiT and ELT to path for importing
sys.path.append(os.path.abspath("dit-model-implementation/dit"))
from models import DiT_models
from models.elt import ELT
from configs.config import get_config

def main():
    print("=== Verification of ELT Parameter Sharing ===\n")
    
    # 1. Instantiate baseline DiT
    dit_model = DiT_models["DiT-XS/2"](input_size=4, num_classes=10)
    dit_params = sum(p.numel() for p in dit_model.parameters())
    print(f"DiT-XS/2 Total Parameters: {dit_params:,}")

    # 2. Instantiate ELT (6 unique blocks, 3 loops)
    config = get_config()
    elt_model = ELT(config)
    elt_params = sum(p.numel() for p in elt_model.parameters())
    print(f"ELT (6 blocks, 3 loops) Total Parameters: {elt_params:,}")
    
    # 3. Assert true parameter sharing
    assert dit_params == elt_params, f"Mismatch! DiT: {dit_params}, ELT: {elt_params}"
    print("\n✅ Verification Passed: ELT perfectly matches DiT parameter count!")
    
    # Verify module references
    print(f"\nELT Unique Blocks Count: {len(elt_model.blocks)}")
    assert len(elt_model.blocks) == 6, "ELT should only instantiate 6 unique block parameters."
    
    print("\n✅ True parameter sharing confirmed. The loops reuse the same 6 blocks.")

if __name__ == "__main__":
    main()
