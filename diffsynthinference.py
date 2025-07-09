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

from vace.models.utils.preprocessor import VaceVideoProcessor
import einops

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

vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(vae_stride, patch_size)]),
            min_area=480 * 832,
            max_area=480 * 832,
            min_fps=16,
            max_fps=16,
            zero_start=True,
            seq_len=32760,
            keep_last=True)

def concatenate_videos(video_a, video_b):
    frames_a = [video_a[i] for i in range(len(video_a))]
    frames_b = [video_b[i] for i in range(len(video_b))]
    return frames_a + frames_b

def prepare_source(src_video, src_mask, num_frames, image_size, device):
        area = image_size[0] * image_size[1]
        vid_proc.set_area(area)
        if area == 720*1280:
            vid_proc.set_seq_len(75600)
        elif area == 480*832:
            vid_proc.set_seq_len(32760)
        else:
            raise NotImplementedError(f'image_size {image_size} is not supported')

        image_size = (image_size[1], image_size[0])
        image_sizes = []
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = vid_proc.load_video_pair(sub_src_video, sub_src_mask)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i], _, _, _ = vid_proc.load_video(sub_src_video)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(src_video[i].shape[2:])

        return src_video, src_mask
    
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
    
def save_video(video_frames, output_dir):    
    with iio.imopen(output_dir, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=16)
        writer._video_stream.options = {"crf": str(23)}
        for frame in video_frames:
            writer.write_frame(np.ascontiguousarray(frame, dtype=np.uint8))
            
def save_black_white_video_from_tensor(mask_tensor, output_path, fps=16):
    """
    Save a binary (0 or 1) mask tensor of shape (T, 1, H, W) as a black & white video.
    """
    assert mask_tensor.ndim == 4, "Expected shape (T, 1, H, W)"
    mask_tensor = mask_tensor.squeeze(1)  # -> (T, H, W)

    # Convert to uint8 [0, 255]
    mask_numpy = (mask_tensor.detach().cpu().numpy() > 0).astype(np.uint8) * 255

    with iio.imopen(str(output_path), "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        writer._video_stream.options = {"crf": "23"}

        for frame in mask_numpy:
            frame_rgb = np.stack([frame] * 3, axis=-1)  # (H, W, 3)
            writer.write_frame(np.ascontiguousarray(frame_rgb, dtype=np.uint8))
        
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

def get_frame_chunks(frame_files: List[Path], chunk_size: int = 81, overlap: int = 5) -> List[Tuple[str, List[Path], List[int]]]:
    """
    Custom chunking logic:
    - If total frames <= 81: one chunk
    - Else:
        chunk_0: 81 input frames
        chunk_i: 5 generated + 76 input frames
        Final chunk:
          * If >=76 input frames: as usual
          * If <76: pad using previous input frames and adjust gen_frame_indices
    Returns: list of (chunk_name, input_frame_paths, gen_frame_indices)
    """
    total_frames = len(frame_files)
    chunks = []

    if total_frames <= chunk_size:
        chunks.append(("chunk_0", frame_files[:chunk_size], []))
        return chunks

    chunk_idx = 0
    start = 0

    # First chunk: normal 81 input frames
    chunks.append(("chunk_0", frame_files[start:start + chunk_size], []))
    start += chunk_size - overlap  # move 76 frames forward

    while start < total_frames:
        end = start + 76
        if end <= total_frames:
            chunk_name = f"chunk_{chunk_idx+1}"
            input_frames = frame_files[start:end]
            gen_frame_indices = list(range(-5, 0))  # use last 5 from generated
            chunks.append((chunk_name, input_frames, gen_frame_indices))
        else:
            # Final chunk: check how many are left
            remaining = total_frames - start
            if remaining >= 76:
                chunk_name = f"chunk_{chunk_idx+1}"
                input_frames = frame_files[start:start+76]
                gen_frame_indices = list(range(-5, 0))
                chunks.append((chunk_name, input_frames, gen_frame_indices))
            else:
                # Need to pad
                needed = 76 - remaining
                if start - needed >= 0:
                    padding = frame_files[start - needed:start]
                    input_frames = padding + frame_files[start:]
                    chunk_name = f"chunk_{chunk_idx+1}_plus_{needed}"
                    # Instead of last 5 from generated, go before padding
                    gen_frame_indices = list(range(-(needed + 5), -needed))
                    chunks.append((chunk_name, input_frames, gen_frame_indices))
                else:
                    print("[!] Not enough input frames to pad final chunk")
            break

        start += 76
        chunk_idx += 1

    return chunks

def parse_video_name(video_name: str) -> Tuple[str, str]:
    """Extract scene_name and seq_number from video_name (scene_name_seq_number.mp4)"""
    match = re.match(r"^(.+)_seq_(\d+)\.mp4$", video_name)
    if match:
        return match.group(1), match.group(2)
    return None, None

def run_inference(idx: int, video_name: str, prompt: str):
    gen = 0 
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

    # Get height and width from the first frame (0th index)
    with Image.open(frame_files[0]) as img:
        width_frame, height_frame = img.size  # PIL returns (width, height)

    # Swap dimensions if portrait mode (height > width)
    if height_frame > width_frame:
        height_frame, width_frame = 832, 480  # Portrait resolution
    else:
        height_frame, width_frame = 480, 832  # Landscape resolution
        
    # Get all chunks at once and store them
    chunks = get_frame_chunks(frame_files)
    
    # Process each chunk
    for chunk_idx, (chunk_name, frame_chunk, original_frames) in enumerate(chunks):
        output_dir = Path(f"results/fps_change/{scene_name}/seq_{seq_number}/{chunk_name}")
        output_video = output_dir / "out_video.mp4"
        if output_video.exists():
            print(f"[✓] Skipping {chunk_name} — output video already exists.")
            continue
        
        chunk_size = len(frame_chunk)
        print(f"  Processing {chunk_name} with {chunk_size} frames (original frames: {len(original_frames)})")
        
        # Create temp directory for this chunk
        temp_dir = Path("./vace_bedlam_100_dataset/bedlam_100_videos_face_hand_vids_dwpose_framebyframe") / scene_name / f"seq_{seq_number.zfill(6)}" / f"temp_{scene_name}_seq{seq_number}_{chunk_name}"
        temp_dir.mkdir(exist_ok=True)

        frames_to_replace = 5
        offset = 5  # Use

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
                    # Get the 5 frames before the padding begins
                    prev_overlap_frames = prev_frames[-(offset + 5):-offset]
                else:
                    # Take last 5 frames of previous chunk
                    prev_overlap_frames = prev_frames[-5:]
                
                for i, frame_path in enumerate(prev_overlap_frames):
                    (temp_dir / f"frame_{i:06d}.jpg").symlink_to(frame_path.resolve())
                
                # Skip the first 5 overlapping frames from the current chunk
                frame_chunk = frame_chunk[frames_to_replace:]
            else:
                print(f"[!] Previous chunk frames not found at {prev_output_dir}")
                
            gen_temp_dir = temp_dir / "generated_frames"
            input_temp_dir = temp_dir / "input_frames"
            gen_temp_dir.mkdir(exist_ok=True)
            input_temp_dir.mkdir(exist_ok=True)
            
            # Store generated frames in gen_temp_dir
            for i, frame_path in enumerate(prev_overlap_frames):
                (gen_temp_dir / f"frame_{i:06d}.jpg").symlink_to(frame_path.resolve())
                
            # Store input frames in input_temp_dir (skipping first 5 overlapping frames)
            for i, frame_path in enumerate(frame_chunk[frames_to_replace:]):
                (input_temp_dir / f"frame_{i:06d}.jpg").symlink_to(frame_path.resolve())

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
        
        if 'plus' in chunk_name:
            gen = 1
            
        
        if gen:
            src_video = frames_to_video(input_temp_dir, video_output_path, fps=16)
            video_output_path_gen = output_dir / f"src_{chunk_name}_gen.mp4"
            src_video_gen = frames_to_video(gen_temp_dir, video_output_path_gen, fps=16)
        else:
            src_video = frames_to_video(temp_dir, video_output_path, fps=16)
        
        '''
        mask_output_path = output_dir / f"src_mask_{chunk_name}.mp4"
        src_mask = torch.zeros((src_video.shape[0], 1, src_video.shape[2], src_video.shape[3]))
        save_black_white_video_from_tensor(src_mask, mask_output_path, fps=16)

        src_video, src_mask = prepare_source([str(video_output_path)],
                                                             [str(mask_output_path)],
                                                             81, SIZE_CONFIGS['480p'], device="cuda")
        frames_tensor = src_video[0]  # shape: (3, 81, 848, 464)
        frames_mask = src_mask[0]

        # Rearrange to (81, 848, 464, 3)
        frames_tensor = frames_tensor.permute(1, 2, 3, 0)  # (F, H, W, C)
        frames_mask = frames_mask.permute(1, 2, 3, 0) 

        # Convert to list of numpy arrays
        src_convid = [frame.cpu().numpy() for frame in frames_tensor]  
        mask_convid = [frame.cpu().numpy() for frame in frames_mask]  
        output_dir_c = output_dir / f"src_test_{chunk_name}.mp4"
        video_np = frames_tensor.cpu().numpy()
        mask_np = frames_mask.cpu().numpy()
        save_video(video_np, output_dir_c)
        mask_output_path_2 = output_dir / f"src_mask_{chunk_name}2.mp4"
        save_video(mask_np * 255, mask_output_path_2)
        '''
            
        control_video = VideoData(video_output_path, height=height_frame, width=width_frame)
        if gen:
            control_video_gen = VideoData(video_output_path_gen, height=height_frame, width=width_frame)
            control_video = concatenate_videos(control_video_gen, control_video)
        
        if chunk_idx == 0:
            vace_video_mask = [torch.ones((height_frame, width_frame , 1), dtype=torch.float32) for _ in range(len(control_video))]
        else:
            vace_video_mask = [] 
            for i in range(len(control_video)):
                if i < 5:
                    mask = torch.ones((height_frame, width_frame, 1), dtype=torch.float32)
                else:
                    mask = torch.zeros((height_frame, width_frame, 1), dtype=torch.float32)
                vace_video_mask.append(mask)
                
        # 4. Run inference
        video = pipe(
            prompt=prompt,
            vace_video=control_video,
            #vace_video_mask = vace_video_mask,
            seed=2025, tiled=True,
            height = height_frame,
            width = width_frame,
            sigma_shift = 16.0,
            #sample_solver='unipc',
        )
        
        import ipdb; ipdb.set_trace()
          
        save_video_frames(video, output_dir)
                
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