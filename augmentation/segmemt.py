import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import os
from PIL import Image
import matplotlib.pyplot as plt

# Function to transform the input image for the model
def transform_image(image_path):
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    return input_batch

# Function to perform occlusion segmentation
def segment_occlusion(image_path, model):
    input_batch = transform_image(image_path)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions

# Load a pre-trained DeepLabv3 model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# Segment occlusion
image_path = os.path.join('data', 'images', 'image1.jpg')
output_predictions = segment_occlusion(image_path, model)

# Display the segmentation mask
plt.imshow(output_predictions.cpu().numpy())
plt.show()
