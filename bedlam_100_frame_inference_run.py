import pandas as pd
import subprocess
from pathlib import Path
import os
import numpy as np
import re

def concatenate_chunks_to_sequence_output():
    base_result_dir = Path("results/fps_change")
    final_output_dir = Path("results/bedlam_framebyframe_results")
    final_output_dir.mkdir(exist_ok=True)

    for scene_path in base_result_dir.iterdir():
        if not scene_path.is_dir():
            continue

        for seq_path in scene_path.iterdir():
            if not seq_path.is_dir():
                continue

            output_img_dir = final_output_dir / scene_path.name / seq_path.name / "img"
            output_img_dir.mkdir(parents=True, exist_ok=True)

            all_frame_files = []
            for chunk_dir in sorted(seq_path.glob("chunk_*")):
                chunk_id = int(chunk_dir.name.split("_")[-1])
                frames_dir = chunk_dir / "frames"
                if not frames_dir.exists():
                    continue

                # Get frame files sorted
                frame_files = sorted(frames_dir.glob("frame_*.jpg"))

                # Skip first 5 frames if not chunk 0
                if chunk_id != 0:
                    frame_files = frame_files[5:]

                all_frame_files.extend(frame_files)

            # Symlink or copy into final folder with continuous frame numbering
            for i, frame_path in enumerate(all_frame_files):
                target_path = output_img_dir / f"frame_{i:06d}.jpg"
                target_path.symlink_to(frame_path.resolve())  # or use shutil.copy2 if you prefer copying

            print(f"[✓] Combined {len(all_frame_files)} frames → {output_img_dir}")
            
def get_frame_chunks(frame_files, chunk_size=81, overlap=5):
    """Split frame files into chunks of 81 frames with 5-frame overlap"""
    import ipdb; ipdb.set_trace()
    num_frames = len(frame_files)
    step = chunk_size - overlap
    for i in range(0, num_frames, step):
        yield frame_files[i:i + chunk_size]

def parse_video_name(video_name):
    """Extract scene_name and seq_number from video_name (scene_name_seq_number.mp4)"""
    match = re.match(r"^(.+)_seq_(\d+)\.mp4$", video_name)
    if match:
        return match.group(1), match.group(2)
    return None, None

def run_inference(idx, video_name, prompt):
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

    # Process in chunks of 81 frames
    import ipdb; ipdb.set_trace()
    for chunk_idx, frame_chunk in enumerate(get_frame_chunks(frame_files)):
        chunk_size = len(frame_chunk)
        print(f"  Processing chunk {chunk_idx+1} with {chunk_size} frames")
        
        # Create temp directory for this chunk
        temp_dir = Path(f"temp_{scene_name}_seq{seq_number}_chunk{chunk_idx}")
        temp_dir.mkdir(exist_ok=True)

        # === NEW: Include last 5 frames from previous generated chunk ===
        offset = 0
        if chunk_idx != 0:
            prev_output_dir = Path(f"results/fps_change/{scene_name}/seq_{seq_number}/chunk_{chunk_idx - 1}/frames")
            if prev_output_dir.exists():
                prev_frames = sorted(prev_output_dir.glob("frame_*.jpg"))[-5:]
                for i, frame_path in enumerate(prev_frames):
                    (temp_dir / f"frame_{i:06d}.jpg").symlink_to(frame_path.resolve())
                offset = 5
            else:
                print(f"[!] Previous chunk frames not found at {prev_output_dir}")

        # Symlink current chunk's frames after the previous 5
        for i, frame_path in enumerate(frame_chunk):
            (temp_dir / f"frame_{i + offset:06d}.jpg").symlink_to(frame_path.resolve())
        
        # Run inference
        cmd = [
            "torchrun", "--nproc_per_node=8", "vace/vace_wan_inference.py",
            "--dit_fsdp",
            "--t5_fsdp",
            "--ulysses_size", "4",
            "--ring_size", "2",
            "--ckpt_dir", "models/VACE-Wan2.1-1.3B-Preview",
            "--frames_dir", str(temp_dir),
            "--prompt", prompt,
            "--save_dir", f"results/fps_change/{scene_name}/seq_{seq_number}/chunk_{chunk_idx}"
        ]

        env = {"PYTHONPATH": "/lustre/home/mkocabas/projects/VACE", **os.environ}
        
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing chunk {chunk_idx}: {e}")
        finally:
            # Clean up temp directory
            for f in temp_dir.iterdir():
                f.unlink()
            temp_dir.rmdir()

if __name__ == "__main__":
    csv_path = "./vace_bedlam_100_dataset/final_metadata_1.csv"
    df = pd.read_csv(csv_path, delimiter=';')

    for idx, row in df.iterrows():
        run_inference(idx, row["file_name"], row["text"])
    
    # After all inferences are done, run post-processing
    concatenate_chunks_to_sequence_output()
    