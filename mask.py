import torch
import numpy as np
import cv2
import sys
import os
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor

# ---- Paths (update if needed) ----
base_path = "models/VACE-Annotators"
gdino_config = os.path.join(base_path, "gdino/GroundingDINO_SwinT_OGC.py")
gdino_weights = os.path.join(base_path, "gdino/groundingdino_swint_ogc.pth")
sam_checkpoint = os.path.join(base_path, "sam/sam_vit_b_01ec64.pth")
sam_model_type = "vit_b"  # Use "vit_b" since you have that variant
image_path = "assets/images/tom_cruise.jpg"  # <-- update this
text_prompt = "a person"  # or "a man", "a car", etc.

# ---- Load image ----
image_source, image = load_image(image_path)

# ---- Load GroundingDINO ----
model = load_model(gdino_config, gdino_weights)

# ---- Predict boxes using text prompt ----
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=text_prompt,
    box_threshold=0.3,
    text_threshold=0.25
)

# Convert to pixel coordinates
H, W = image.shape[:2]
boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
boxes_xyxy = boxes_xyxy.cpu().numpy().astype(int)

# ---- Load SAM and run predictor ----
sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
predictor.set_image(image_source)

masks = []
for box in boxes_xyxy:
    mask, _, _ = predictor.predict(box=box, multimask_output=False)
    masks.append(mask[0])

# ---- Visualize segmentation result ----
segmented_image = image_source.copy()
for mask in masks:
    segmented_image[mask] = [0, 255, 0]  # highlight in green

output_path = "processed/segmented_output.jpg"
cv2.imwrite(output_path, segmented_image)
print(f"Saved segmentation to: {output_path}")
