"""把 Day 0 的 parquet + JPG 转成 LeRobot 数据集格式（v0.3.3）。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK = "corridor navigation: forward, left turn, right turn, stop"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/parquet/episode_000000.parquet")
    ap.add_argument("--root", default="data/lerobot/bag4")
    ap.add_argument("--repo-id", default="weason955/wheeltec_bag4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(REPO_ROOT / args.parquet).sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].to_numpy()

    # 按 0.1 秒绝对时间网格找最近的帧
    grid = np.arange(0.0, ts[-1] + 1e-9, 1.0 / args.fps)
    right = np.clip(np.searchsorted(ts, grid), 0, len(ts) - 1)
    left = np.clip(right - 1, 0, len(ts) - 1)
    pick = np.where(np.abs(ts[left] - grid) <= np.abs(ts[right] - grid), left, right)
    drift = np.abs(ts[pick] - grid)

    print(f"源帧数      {len(ts)}")
    print(f"原始时长    {ts[-1]:.3f} s  (平均 {len(ts) / ts[-1]:.2f} Hz)")
    print(f"网格点数    {len(grid)}  @ {args.fps} Hz")
    print(f"时间偏差    中位数 {np.median(drift) * 1000:.1f} ms, 最大 {drift.max() * 1000:.1f} ms")
    print(f"偏差>60ms   {(drift > 0.06).sum()} 个点")
    print(f"state 范围  v [{df.state_v.min():.3f}, {df.state_v.max():.3f}]  w [{df.state_w.min():.3f}, {df.state_w.max():.3f}]")
    print(f"action 范围 v [{df.action_v.min():.3f}, {df.action_v.max():.3f}]  w [{df.action_w.min():.3f}, {df.action_w.max():.3f}]")
    still = ((df.state_v.abs() < 1e-3) & (df.state_w.abs() < 1e-3)).mean()
    print(f"静止帧占比  {still * 100:.1f}%  (原始数据)")

    if args.dry_run:
        print("dry-run, 未写文件")
        return

    first = np.asarray(Image.open(REPO_ROOT / df.loc[pick[0], "image_path"]).convert("RGB"), dtype=np.uint8)
    h, w, c = first.shape
    print(f"图像形状    {first.shape}")

    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": ["v", "w"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["v", "w"]},
        "observation.images.front": {
            "dtype": "image",
            "shape": (h, w, c),
            "names": ["height", "width", "channels"],
        },
    }

    root = REPO_ROOT / args.root
    if root.exists():
        raise SystemExit(f"{root} 已存在，先 rm -rf 再跑")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=root,
        robot_type="wheeltec_v550",
        use_videos=False,
    )

    for n, i in enumerate(pick):
        row = df.loc[i]
        img = np.asarray(Image.open(REPO_ROOT / row["image_path"]).convert("RGB"), dtype=np.uint8)
        frame = {
            "observation.images.front": img,
            "observation.state": np.array([row["state_v"], row["state_w"]], dtype=np.float32),
            "action": np.array([row["action_v"], row["action_w"]], dtype=np.float32),
        }
        dataset.add_frame(frame, task=args.task)
        if (n + 1) % 200 == 0:
            print(f"  写入 {n + 1}/{len(pick)}")

    dataset.save_episode()
    print(f"完成: {root}")
    print(f"frames={dataset.num_frames} episodes={dataset.num_episodes}")


if __name__ == "__main__":
    main()
