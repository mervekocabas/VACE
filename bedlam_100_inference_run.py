'''
import pandas as pd
import subprocess
from pathlib import Path
import os
from multiprocessing import Pool, cpu_count

def run_inference(args):
    idx, file_name, prompt = args
    video_dir = Path("./vace_bedlam_100_dataset/bedlam_100_videos_2dpose_processed")
    src_video = video_dir / file_name

    if not src_video.exists():
        print(f"[!] Missing video: {src_video}")
        return

    print(f"[{idx}] Running inference on: {file_name}")

    cmd = [
        "torchrun", "--nproc_per_node=4", "vace/vace_wan_inference.py",
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size", "2",
        "--ring_size", "2",
        "--ckpt_dir", "models/VACE-Wan2.1-14B",
        "--src_video", str(src_video),
        "--prompt", prompt,
    ]

    env = {"PYTHONPATH": "/lustre/home/mkocabas/projects/VACE", **os.environ}
    
    # Run inference
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    csv_path = "./vace_bedlam_100_dataset/final_metadata.csv"
    df = pd.read_csv(csv_path)

    # Create input list for Pool
    inputs = [(idx, row["file_name"], row["text"]) for idx, row in df.iterrows()]

    # Run inferences with 2 concurrent processes
    with Pool(processes=2) as pool:
        pool.map(run_inference, inputs)
'''


import pandas as pd
import subprocess
from pathlib import Path
import os

def run_inference(idx, file_name, prompt):
    video_dir = Path("./vace_bedlam_100_dataset/bedlam_100_videos_2dpose_processed")
    src_video = video_dir / file_name

    if not src_video.exists():
        print(f"[!] Missing video: {src_video}")
        return

    print(f"[{idx}] Running inference on: {file_name}")

    cmd = [
        "torchrun", "--nproc_per_node=4", "vace/vace_wan_inference.py",
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size", "2",
        "--ring_size", "2",
        "--size", "720p",
        "--model_name", "vace-14B",
        "--ckpt_dir", "models/VACE-Wan2.1-14B",
        "--src_video", str(src_video),
        "--prompt", prompt,
    ]

    env = {"PYTHONPATH": "/lustre/home/mkocabas/projects/VACE", **os.environ}
    
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    csv_path = "./vace_bedlam_100_dataset/final_metadata.csv"
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        run_inference(idx, row["file_name"], row["text"])
        
'''     
import pandas as pd
import subprocess
from pathlib import Path
import os

def run_inference(idx, file_name):
    video_dir = Path("./vace_bedlam_100_dataset/bedlam_100_videos")
    src_video = video_dir / file_name

    if not src_video.exists():
        print(f"[!] Missing video: {src_video}")
        return

    print(f"[{idx}] Running inference on: {file_name}")

    cmd = [
        "torchrun", "--nproc_per_node=4", "vace/vace_wan_inference.py",
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size", "2",
        "--ring_size", "2",
        "--ckpt_dir", "models/VACE-Wan2.1-1.3B-Preview",
        "--src_video", str(src_video),
        "--prompt", "xxx",
    ]

    env = {"PYTHONPATH": "/lustre/home/mkocabas/projects/VACE", **os.environ}
    
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    csv_path = "./vace_bedlam_100_dataset/final_metadata.csv"
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        run_inference(idx, row["file_name"])
'''