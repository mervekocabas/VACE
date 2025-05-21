#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import time
import argparse
import cv2
import numpy as np
from PIL import Image

# Import necessary modules from VACE
from vace.annotators.pose import PoseBodyAnnotator
from vace.annotators.utils import read_video_frames, save_one_video

def main():
    # Configuration for pose task
    task_cfg = {
        "NAME": "PoseBodyHandVideoKeypointExtractor",
        "DETECTION_MODEL": "models/VACE-Annotators/pose/yolox_l.onnx",
        "POSE_MODEL": "models/VACE-Annotators/pose/dw-ll_ucoco_384.onnx",
        "RESIZE_SIZE": 1024,
        "USE_BODY": True,
        "USE_FACE": False,
        "USE_HAND": True
    }

    # Input parameters
    input_params = {
        'video': 'assets/videos/test.mp4',
        'frames': None  # Will be populated from video
    }

    # Output parameters
    output_params = ['frames']

    # Create save directory
    pre_save_dir = os.path.join('processed', 'pose', time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
    os.makedirs(pre_save_dir, exist_ok=True)

    # Read video frames
    print(f"Reading video: {input_params['video']}")
    frames, fps, width, height, num_frames = read_video_frames(
        input_params['video'], 
        use_type='cv2', 
        info=True
    )
    assert frames is not None, "Video read error"
    input_params['frames'] = frames

    # Initialize pose extractor
    print("Initializing pose extractor...")
    pose_extractor = PoseBodyAnnotator(
        cfg=task_cfg,
        device=f'cuda:{os.getenv("RANK", 0)}'
    )

    # Process frames
    print("Processing frames...")
    results = pose_extractor.forward(**input_params)

    # Save results
    if isinstance(results, dict):
        frames = results['frames']
    else:
        frames = results

    if frames is not None:
        save_path = os.path.join(pre_save_dir, 'src_video-pose.mp4')
        save_one_video(save_path, frames, fps=fps)
        print(f"Saved processed video to: {save_path}")

if __name__ == "__main__":
    main() 