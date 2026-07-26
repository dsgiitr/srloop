import zipfile
import os

zip_path = r'C:\Users\Parth\Downloads\elt_model_kaggle.zip'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == 'create_zip.py' or file.endswith('.zip'):
                continue
            file_path = os.path.join(root, file)
            # Force forward slashes for cross-platform compatibility
            arcname = os.path.relpath(file_path, '.').replace('\\', '/')
            zipf.write(file_path, arcname)

print("Zip created successfully with forward slashes!")
