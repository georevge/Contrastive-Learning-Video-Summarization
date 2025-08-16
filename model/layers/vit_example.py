from transformers import AutoImageProcessor, ViTModel, ViTConfig
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

# Initializing a ViT vit-base-patch16-224 style configuration
##sqrt = math.sqrt(512)
#configuration = ViTConfig(image_size=(45,512), patch_size=512, num_channels=1)

# Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
#model = ViTModel(configuration)

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

inputs1 = image_processor(image, return_tensors="pt")
#inputs = {'pixel_values': torch.rand(1,1,45, 512)}
with torch.no_grad():
    outputs = model(**inputs1)
k=0