# Assuming you have already installed the RePaint package and its dependencies
from repaint import RePaint
import torch
from PIL import Image
import numpy as np

# Load your pretrained RePaint model
# Please ensure that you adjust the path to where you have saved your pretrained RePaint model.
repaint_model = RePaint(pretrained=True, device='cuda' if torch.cuda.is_available() else 'cpu')

def inpaint_image_with_mask(image, mask):
    # Convert the PIL Image to a NumPy array
    image_np = np.array(image)

    # Assuming the mask is a binary mask from the segmentation step
    # The mask should be a binary (0 or 1) numpy array where occluded regions are marked with 1

    # Use RePaint to inpaint the image based on the mask
    inpainted_image = repaint_model.inpaint(image_np, mask)

    # Convert the inpainted image from NumPy array back to PIL Image for further processing or visualization
    inpainted_image_pil = Image.fromarray(inpainted_image)

    return inpainted_image_pil

# Load your image and mask
original_image = Image.open('path_to_your_image.jpg')
segmentation_mask = Image.open('path_to_your_mask.png')  # This should be the mask obtained from DeepLabv3

# Perform inpainting
inpainted_image = inpaint_image_with_mask(original_image, segmentation_mask)

# Save or display the inpainted image
inpainted_image.save('inpainted_image.png')
inpainted_image.show()
