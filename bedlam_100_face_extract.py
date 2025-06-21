import pandas as pd
import subprocess
from pathlib import Path
import os

def run_inference(idx, file_name):
    #video_dir = Path("./results/vace-14B-v2prompt-face-body-videos")
    video_dir = Path("./results/bedlam_src_vids_16fps")
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
    csv_path = "./vace_bedlam_100_dataset/final_metadata.csv"
    df = pd.read_csv(csv_path, delimiter=';')

    for idx, row in df.iterrows():
        run_inference(idx, row["file_name"])
