import pandas as pd
import subprocess
from pathlib import Path
import os

def run_inference(idx, file_name):
    #video_dir = Path("./results/vace-14B-v2prompt-face-body-videos")
    #video_dir = Path("./results/seedvr_output")
    video_dir = Path("./vace_bedlam_100_dataset/bedlam_100_videos")
    
    src_video = video_dir / file_name

    if not src_video.exists():
        print(f"[!] Missing video: {src_video}")
        return

    print(f"[{idx}] Running pose extract on: {file_name}")

    cmd = [
        "python",
        "vace/vace_preproccess.py",
        "--task", "pose",
        "--video", str(src_video)
    ]

    env = {"PYTHONPATH": "/lustre/home/mkocabas/projects/VACE", **os.environ}
    
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    csv_path = "./vace_bedlam_100_dataset/parallel_2.csv"
    df = pd.read_csv(csv_path, delimiter=';')

    for idx, row in df.iterrows():
        file_name = row["file_name"]  # e.g., '20221020_3-8_250_xyz_seq_000170.mp4'
        #base_name = Path(file_name).stem  # Removes .mp4 → '20221020_3-8_250_xyz_seq_000170'
        #new_name = base_name + "_out_video.mp4"
        #run_inference(idx, new_name)
        run_inference(idx, file_name)
