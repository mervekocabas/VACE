import cv2
import os
from pathlib import Path

import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

import pandas as pd
import subprocess
import numpy as np
import re
from typing import List, Tuple
import shutil
import imageio.v3 as iio

# 1. Prepare pipeline
'''
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)
'''
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
    skip_download = True,
)

pipe.enable_vram_management()
    
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

def concatenate_chunks_to_sequence_output():
    base_result_dir = Path("results/fps_change")
    final_output_dir = Path("results/bedlam_framebyframe_results")
    final_output_dir.mkdir(parents=True, exist_ok=True)

    for scene_path in base_result_dir.iterdir():
        if not scene_path.is_dir():
            continue

        for seq_path in scene_path.iterdir():
            if not seq_path.is_dir():
                continue

            output_img_dir = final_output_dir / scene_path.name / seq_path.name / "img"
            output_img_dir.mkdir(parents=True, exist_ok=True)

            all_frame_files = []
            chunk_dirs = sorted(seq_path.glob("chunk_*"), key=lambda x: int(x.name.split('_')[1]))
            
            for chunk_dir in chunk_dirs:
                chunk_name = chunk_dir.name
                frames_dir = chunk_dir / "frames"
                if not frames_dir.exists():
                    continue

                frame_files = sorted(frames_dir.glob("frame_*.jpg"))

                if chunk_name == "chunk_0":
                    # keep all frames
                    all_frame_files.extend(frame_files)
                elif "plus" in chunk_name:
                    # Extract x from plus_x
                    match = re.match(r"chunk_\d+_plus_(\d+)", chunk_name)
                    x = int(match.group(1)) if match else 5  # fallback to 5 if no match

                    # Remove last 5 frames from previous chunk before adding this chunk's frames
                    if len(all_frame_files) > 0:
                        all_frame_files = all_frame_files[:-5]

                    # Skip first x frames of this plus chunk
                    frame_files = frame_files[x:]
                    all_frame_files.extend(frame_files)
                else:
                    # Normal chunks except chunk_0: skip first 5 frames
                    frame_files = frame_files[5:]
                    all_frame_files.extend(frame_files)

            # Symlink or copy into final folder with continuous frame numbering
            for i, frame_path in enumerate(all_frame_files):
                target_path = output_img_dir / f"frame_{i:06d}.jpg"
                if not target_path.exists():
                    try:
                        target_path.symlink_to(frame_path.resolve())
                    except FileExistsError:
                        pass  # Skip if symlink already exists

            print(f"[✓] Combined {len(all_frame_files)} frames → {output_img_dir}")

def get_frame_chunks(frame_files: List[Path], chunk_size: int = 81, overlap: int = 5) -> List[Tuple[str, List[Path], List[Path]]]:
    """
    Splits frame files into chunks of `chunk_size` with `overlap` frames between them.
    Ensures no frames are lost and every frame is included in at least one chunk.
    The final chunk is padded from earlier frames if it's too short.
    Returns a list of tuples: (chunk_name, frames_to_process, original_frames_used)
    """
    chunks = []
    num_frames = len(frame_files)
    stride = chunk_size - overlap
    start = 0
    chunk_idx = 0

    while start < num_frames:
        end = start + chunk_size
        if end <= num_frames:
            chunk_frames = frame_files[start:end]
            chunk_name = f"chunk_{chunk_idx}"
            chunks.append((chunk_name, chunk_frames, chunk_frames))
        else:
            # Not enough frames left; pad with earlier frames if needed
            remaining_frames = frame_files[start:]
            needed = chunk_size - len(remaining_frames)
            if needed > 0 and start >= needed:
                extra = frame_files[start - needed:start]
                chunk_frames = extra + remaining_frames
                chunk_name = f"chunk_{chunk_idx}_plus_{needed}"
                chunks.append((chunk_name, chunk_frames, extra + remaining_frames))
            else:
                # If not enough to pad, just use remaining (e.g. for very short sequences)
                chunk_frames = remaining_frames
                chunk_name = f"chunk_{chunk_idx}"
                chunks.append((chunk_name, chunk_frames, chunk_frames))
            break

        start += stride
        chunk_idx += 1

    return chunks

def parse_video_name(video_name: str) -> Tuple[str, str]:
    """Extract scene_name and seq_number from video_name (scene_name_seq_number.mp4)"""
    match = re.match(r"^(.+)_seq_(\d+)\.mp4$", video_name)
    if match:
        return match.group(1), match.group(2)
    return None, None

def run_inference(idx: int, video_name: str, prompt: str):
    # Parse scene_name and seq_number from video_name
    scene_name, seq_number = parse_video_name(video_name)
    if not scene_name or not seq_number:
        print(f"[!] Invalid video name format: {video_name}")
        return

    # Build path to frame directory
    frame_dir = Path("./vace_bedlam_100_dataset/bedlam_100_videos_face_hand_vids_dwpose_framebyframe") / scene_name / f"seq_{seq_number.zfill(6)}"
    
    if not frame_dir.exists():
        print(f"[!] Missing frame directory: {frame_dir}")
        return

    # Get all frame files in this sequence
    frame_files = sorted(frame_dir.glob('frame_*.jpg'))
    
    if not frame_files:
        print(f"[!] No frames found in {frame_dir}")
        return
        
    print(f"[{idx}] Processing: {video_name} => {len(frame_files)} frames in {frame_dir}")

    # Get all chunks at once and store them
    chunks = get_frame_chunks(frame_files)
    
    # Process each chunk
    for chunk_idx, (chunk_name, frame_chunk, original_frames) in enumerate(chunks):
        chunk_size = len(frame_chunk)
        print(f"  Processing {chunk_name} with {chunk_size} frames (original frames: {len(original_frames)})")
        
        # Create temp directory for this chunk
        temp_dir = Path("./vace_bedlam_100_dataset/bedlam_100_videos_face_hand_vids_dwpose_framebyframe") / scene_name / f"seq_{seq_number.zfill(6)}" / f"temp_{scene_name}_seq{seq_number}_{chunk_name}"
        temp_dir.mkdir(exist_ok=True)

        frames_to_replace = 5
        offset = 0  # Use

        # Check for "plus_X" in chunk name
        match = re.match(r"chunk_\d+_plus_(\d+)", chunk_name)
        if match:
            offset = int(match.group(1))  # start of padding range

        if chunk_idx != 0:
            prev_chunk_name = chunks[chunk_idx - 1][0]
            prev_output_dir = Path(f"results/fps_change/{scene_name}/seq_{seq_number}/{prev_chunk_name}/frames")
            
            if prev_output_dir.exists():
                prev_frames = sorted(prev_output_dir.glob("frame_*.jpg"))
                
                
                if 'plus' in chunk_name: 
                    # Get 5 frames starting from -offset to -offset+5
                    start = -offset
                    end = start + frames_to_replace
                    prev_overlap_frames = prev_frames[start:end]
                else:
                    prev_overlap_frames = prev_frames[:-5]
                
                for i, frame_path in enumerate(prev_overlap_frames):
                    (temp_dir / f"frame_{i:06d}.jpg").symlink_to(frame_path.resolve())
                
                # Skip the first 5 overlapping frames from the current chunk
                frame_chunk = frame_chunk[frames_to_replace:]
            else:
                print(f"[!] Previous chunk frames not found at {prev_output_dir}")

        # Now add the remaining 76 new frames
        for i, frame_path in enumerate(frame_chunk, start=5 if chunk_idx != 0 else 0):
            (temp_dir / f"frame_{i:06d}.jpg").symlink_to(frame_path.resolve())
        
        # Create output directory with chunk name
        output_dir = Path(f"results/fps_change/{scene_name}/seq_{seq_number}/{chunk_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy frames into output_dir/frames/
        output_frames_dir = output_dir / "frames"
        output_frames_dir.mkdir(parents=True, exist_ok=True)
                
        video_output_path = output_dir / f"src_{chunk_name}.mp4"
        frames_to_video(temp_dir, video_output_path, fps=16)
        control_video = VideoData(video_output_path, height=480, width=832)
        
        # 4. Run inference
        video = pipe(
            prompt=prompt,
            vace_video=control_video,
            seed=1, tiled=True,
        )
        
        import ipdb;ipdb.set_trace()
                
        # Run inference
        '''
        cmd = [
            "torchrun", "--nproc_per_node=8", "vace/vace_wan_inference.py",
            "--dit_fsdp",
            "--t5_fsdp",
            "--ulysses_size", "4",
            "--ring_size", "2",
            "--ckpt_dir", "models/VACE-Wan2.1-1.3B-Preview",
            "--frames_dir", str(temp_dir),
            "--prompt", prompt,
            "--save_dir", str(output_dir)
        ]
        
        
        cmd = [
            "python", "vace/vace_wan_inference.py",
            "--ckpt_dir", "models/VACE-Wan2.1-1.3B-Preview",
            "--frames_dir", str(temp_dir),
            "--prompt", prompt,
            "--save_dir", str(output_dir)
        ]
                
        env = {"PYTHONPATH": "/lustre/home/mkocabas/projects/VACE", **os.environ}
                
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {chunk_name}: {e}")
        '''
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"[!] Failed to delete temp directory {temp_dir}: {e}")

if __name__ == "__main__":
    csv_path = "./vace_bedlam_100_dataset/final_metadata_2.csv"
    df = pd.read_csv(csv_path, delimiter=';')

    for idx, row in df.iterrows():
        run_inference(idx, row["file_name"], row["text"])
    
    # After all inferences are done, run post-processing
    concatenate_chunks_to_sequence_output()