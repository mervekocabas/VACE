import cv2
import os
from pathlib import Path

import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

import pandas as pd
import numpy as np
import re
from typing import List, Tuple
import imageio.v3 as iio

# 1. Prepare pipeline
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    #redirect_common_files=False,
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
    skip_download = True,
)

pipe.enable_vram_management()

vae_stride = (4, 8, 8)
patch_size = (1, 2, 2)
SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '1024*1024': (1024, 1024),
    '720p': (1280, 720),
    '480p': (480, 832)
}

def frames_to_video(frame_dir: Path, output_video_path: Path, fps: int = 16, crf: int = 23):
    frame_paths = sorted(frame_dir.glob("frame_*.jpg"))
    if not frame_paths:
        print(f"[!] No frames found in {frame_dir}")
        return

    frames = []
    for frame_path in frame_paths:
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"[!] Failed to read: {frame_path}")
            continue
        # Convert BGR (OpenCV) to RGB for imageio
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img_rgb)

    if not frames:
        print("[!] No valid frames to write.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    with iio.imopen(output_video_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        writer._video_stream.options = {"crf": str(crf)}
        for frame in frames:
            writer.write_frame(np.ascontiguousarray(frame, dtype=np.uint8))
            
    video_np = np.stack(frames)  # Shape: (num_frames, H, W, C)
    video_np = np.transpose(video_np, (0, 3, 1, 2))  # Now: (num_frames, C, H, W)

    # Convert to torch tensor (optional)
    video_tensor = torch.from_numpy(video_np).float()
    return video_tensor

def save_video_frames(video_frames, output_dir):
    frame_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    for idx, frame in enumerate(video_frames):
        filename = f"frame_{idx:06d}.jpg"
        path = os.path.join(frame_dir, filename)
        frame.save(path)

    print(f"Saved {len(video_frames)} frames to {frame_dir}")
    
    video_dir = os.path.join(output_dir, "out_video.mp4")
    with iio.imopen(video_dir, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=16)
        writer._video_stream.options = {"crf": str(23)}
        for frame in video_frames:
            writer.write_frame(np.ascontiguousarray(frame, dtype=np.uint8))
            
# Path to the src_frames directory
src_frames_dir = Path("results/diffsynth_finvers/20221010_3_1000_batch01hand/seq_000166/chunk_1_plus_35/src_frames/ye")

# Get all frame paths
frame_files = sorted(src_frames_dir.glob("frame_*.jpg"))
if not frame_files:
    print("[!] No frames found.")
    exit()

# Infer resolution from first frame
with Image.open(frame_files[0]) as img:
    width_frame, height_frame = img.size

# Swap if necessary
if height_frame > width_frame:
    height_frame, width_frame = 832, 480
else:
    height_frame, width_frame = 480, 832

# Convert frames to video
output_video_path = Path("tmp_input_video.mp4")
video_tensor = frames_to_video(src_frames_dir, output_video_path, fps=16)

# Wrap in VideoData
control_video = VideoData(output_video_path, height=height_frame, width=width_frame)

# Optional prompt
prompt = "A static, medium-wide camera view captures three figures on a slightly damp grass field under an overcast late afternoon sky, distant trees blurring the background. The tall, thin figure on the left, wearing a faded blue hoodie and short dark hair, begins turning towards the center of the frame. Simultaneously, the medium-built person in the center, in black workout leggings and a bright yellow t-shirt with hair in a ponytail, pivots towards the right, extending their arms wide before pulling them in as their left leg lifts and bends sharply upwards. At the same time, the shorter, stocky figure on the right, in grey sweatpants and a white tank top with a buzz cut, rises from a crouched position, lowering their extended arms."

# Run inference
video = pipe(
    prompt=prompt,
    vace_video=control_video,
    seed=2025,
    tiled=True,
    height=height_frame,
    width=width_frame,
    sigma_shift=16.0,
    negative_prompt = "deformed, disfigured, mutated, bad anatomy, unrealistic body, extra limbs, extra fingers, fused fingers, missing fingers, poorly drawn hands, malformed hands, unnatural pose, distorted face, ugly face, poorly drawn face, blurry face, low detail face, bad eyes, crossed eyes, asymmetrical eyes, low resolution, bad quality, low quality, jpeg artifacts, watermark, signature, text, caption, blurry, grainy, noisy, overexposed, underexposed, bad lighting, duplicated limbs, extra arms, extra legs, broken limb, clone face, lopsided, twisted, unnatural skin texture, skin blemish, unnatural colors, zombie, monster, doll, mannequin, uncanny valley",
)

# Save result
output_dir = Path("results/manual_inference_output")
output_dir.mkdir(parents=True, exist_ok=True)
save_video_frames(video, output_dir)
