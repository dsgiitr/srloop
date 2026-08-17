import kagglehub

# Download latest version
path = kagglehub.dataset_download("greatgamedota/ffhq-face-data-set")

print("Path to dataset files:", path)