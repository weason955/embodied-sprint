from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

PROJECT = Path.home() / "embodied-sprint"
PARQUET = PROJECT / "data" / "parquet" / "episode_000000.parquet"
OUT = PROJECT / "data" / "preview"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PARQUET)
print("frames:", len(df))
print("duration:", round(df["timestamp"].max(), 2), "s")

# 1) 动作分布
still = ((df["action_v"].abs() < 1e-6) & (df["action_w"].abs() < 1e-6)).sum()
print(f"still frames: {still} ({still / len(df) * 100:.1f}%)")
print("v > 0:", int((df["action_v"] > 0).sum()))
print("w < 0:", int((df["action_w"] < 0).sum()))
print("w > 0:", int((df["action_w"] > 0).sum()))

# 2) 大断档
dt = df["timestamp"].diff()
gaps = df.loc[dt > 0.2, ["frame_index", "timestamp"]].copy()
gaps["gap_s"] = dt[dt > 0.2].values
print("\ngaps > 0.2s:")
print(gaps.to_string(index=False))

# 3) 信号曲线
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(df["timestamp"], df["action_v"], label="action_v (cmd)", linewidth=1)
axes[0].plot(df["timestamp"], df["state_v"], label="state_v (odom)", alpha=0.7, linewidth=1)
axes[0].set_ylabel("v (m/s)")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["timestamp"], df["action_w"], label="action_w (cmd)", linewidth=1)
axes[1].plot(df["timestamp"], df["state_w"], label="state_w (odom)", alpha=0.7, linewidth=1)
axes[1].set_ylabel("w (rad/s)")
axes[1].set_xlabel("time (s)")
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

fig.suptitle("episode_000000: command vs odom")
fig.tight_layout()
signal_path = OUT / "episode_000000_signals.png"
fig.savefig(signal_path, dpi=150)
plt.close(fig)
print("\nsaved:", signal_path)

# 4) 8 帧拼图
n = len(df)
idxs = [int(i * (n - 1) / 7) for i in range(8)]
tw, th = 320, 180
cols, rows = 2, 4
pad = 44
sheet = Image.new("RGB", (tw * cols, (th + pad) * rows), "white")
draw = ImageDraw.Draw(sheet)

for k, idx in enumerate(idxs):
    row = df.iloc[idx]
    img = Image.open(PROJECT / row["image_path"]).convert("RGB").resize((tw, th))
    x = (k % cols) * tw
    y = (k // cols) * (th + pad)
    sheet.paste(img, (x, y))
    text = (
        f"f{int(row['frame_index']):03d}  t={row['timestamp']:.2f}s\n"
        f"a_v={row['action_v']:.2f}  a_w={row['action_w']:.2f}\n"
        f"s_v={row['state_v']:.2f}  s_w={row['state_w']:.2f}"
    )
    draw.multiline_text((x + 4, y + th + 2), text, fill="black")

sheet_path = OUT / "episode_000000_sheet.jpg"
sheet.save(sheet_path, quality=90)
print("saved:", sheet_path)
