import os
import torch
from torchvision import transforms
from PIL import Image
from unet_imagetoimage import UNet

# Directories
input_dir = "images"
output_dir = "augmented_images_unet"
os.makedirs(output_dir, exist_ok=True)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model.load_state_dict(torch.load("unet_image_to_image.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((1600, 1200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[-1.0], std=[2.0]),
    transforms.ToPILImage()
])

for image_name in os.listdir(input_dir):
    if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
        # Load and preprocess the image
        image_path = os.path.join(input_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor)

        output_image = output_tensor.squeeze(0).cpu()
        output_image = inverse_transform(output_image)
        output_image.save(os.path.join(output_dir, image_name))

print(f"Image generation complete. Results saved to '{output_dir}'.")
