# Elastic Looped Transformer for Visual Generation(MLM Setup)

Paper Referred: ELT: Elastic Looped Transformers for Visual Generation

## Overview:
Traditional generative transformers stack a fixed, massive sequence of unique neural layers. ELT decouples physical parameter size from computational depth by stacking a small group of $N$ layers into a composite block ($g_\Theta$) and cycling data through it recursively for $L$ runtime loops ($g_\Theta^L$). This provides massive parameter savings and supports **Any-Time Inference** adjustments dynamically.