import io
import math
from pathlib import Path

import pandas as pd
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

PROJECT = Path.home() / "embodied-sprint"
BAG = Path.home() / "bag4"
OUT = PROJECT / "data"
IMG_DIR = OUT / "images" / "episode_000000"
PARQUET = OUT / "parquet" / "episode_000000.parquet"
RESIZE = (640, 360)          # 想保留 1280x720 就改成 None
TOL = 0.06                   # 图像(30Hz) 对 odom(20Hz) 的容差，半周期 25ms，留点余量
CMD_TOL = 0.10               # cmd_vel 挂到 odom 的容差
MARGIN = 0.2                 # 图像窗口前后多保留一点 odom/cmd


def quat_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET.parent.mkdir(parents=True, exist_ok=True)

    # 清掉上一次生成的图，避免新旧混在一起
    for old in IMG_DIR.glob("*.jpg"):
        old.unlink()

    images = []
    odom_rows = []
    cmd_rows = []

    store = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([BAG], default_typestore=store) as reader:
        print("topics:")
        for c in reader.connections:
            print(f"  {c.topic:20s} {c.msgtype}")

        # ---- /image ----
        conns = [c for c in reader.connections if c.topic == "/image"]
        for i, (conn, t, raw) in enumerate(reader.messages(connections=conns)):
            msg = reader.deserialize(raw, conn.msgtype)
            if msg.encoding != "jpeg":
                raise SystemExit(f"expected jpeg, got {msg.encoding}")
            try:
                img = Image.open(io.BytesIO(bytes(msg.data))).convert("RGB")
            except Exception as e:
                print(f"skip image {i}: {e}")
                continue
            if RESIZE is not None:
                resample = getattr(Image, "Resampling", Image).BILINEAR
                img = img.resize(RESIZE, resample)
            path = IMG_DIR / f"frame_{i:06d}.jpg"
            img.save(path, quality=90)
            images.append({
                "bag_time": t / 1e9,
                "hdr_time": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "image_path": str(path.relative_to(PROJECT)),
            })
        print(f"images: {len(images)}")

        # ---- /odom ----
        conns = [c for c in reader.connections if c.topic == "/odom"]
        for conn, t, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            q = msg.pose.pose.orientation
            odom_rows.append({
                "bag_time": t / 1e9,
                "hdr_time": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "state_v": msg.twist.twist.linear.x,
                "state_w": msg.twist.twist.angular.z,
                "x": msg.pose.pose.position.x,
                "y": msg.pose.pose.position.y,
                "yaw": quat_to_yaw(q.x, q.y, q.z, q.w),
            })
        print(f"odom: {len(odom_rows)}")

        # ---- /cmd_vel ----
        conns = [c for c in reader.connections if c.topic == "/cmd_vel"]
        for conn, t, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            cmd_rows.append({
                "bag_time": t / 1e9,
                "action_v": msg.linear.x,
                "action_w": msg.angular.z,
            })
        print(f"cmd_vel: {len(cmd_rows)}")

    img_df = pd.DataFrame(images).sort_values("bag_time").reset_index(drop=True)
    odom_df = pd.DataFrame(odom_rows).sort_values("bag_time").reset_index(drop=True)
    cmd_df = pd.DataFrame(cmd_rows).sort_values("bag_time").reset_index(drop=True)

    if img_df.empty or odom_df.empty or cmd_df.empty:
        raise SystemExit("有一路数据为空，检查话题名")

    # ---- 统一到同一个墙钟 ----
    # 图像 header 是相机的单调时钟（约 2833s），bag 是墙钟（约 1.795e9）。
    # 两者差值在整段录制里只差几毫秒（hdr span == bag span），所以取
    # (bag - hdr) 的最小值作为两台时钟的固定偏移，把 header 换算成墙钟。
    off_img = (img_df["bag_time"] - img_df["hdr_time"]).min()
    img_df["t"] = img_df["hdr_time"] + off_img

    # odom 的 header 本身就是墙钟，而且比 bag 时间平滑，直接用它。
    odom_df["t"] = odom_df["hdr_time"]

    # cmd_vel 没有 header，只能先用 bag 时间；稍后按 bag 时间挂到 odom 上。
    cmd_df["t"] = cmd_df["bag_time"]

    print("\nimage bag-hdr offset (s):")
    print((img_df["bag_time"] - img_df["hdr_time"]).describe())
    print(f"\nused offset: {off_img:.3f}")

    print("\nodom header dt:")
    print(odom_df["t"].diff().describe())
    print("\ncmd bag dt:")
    print(cmd_df["t"].diff().describe())

    t0 = float(img_df["t"].min())
    t1 = float(img_df["t"].max())
    print(f"\nimage window (wall clock): {t0:.3f} .. {t1:.3f}  ({t1 - t0:.2f}s)")
    print(f"odom range: {odom_df['t'].min():.3f} .. {odom_df['t'].max():.3f}")
    print(f"cmd range:  {cmd_df['t'].min():.3f} .. {cmd_df['t'].max():.3f}")

    # ---- 裁剪到图像时间窗 ----
    odom_df = odom_df[
        (odom_df["t"] >= t0 - MARGIN) & (odom_df["t"] <= t1 + MARGIN)
    ].reset_index(drop=True)
    cmd_df = cmd_df[
        (cmd_df["t"] >= t0 - MARGIN) & (cmd_df["t"] <= t1 + MARGIN)
    ].reset_index(drop=True)

    # ---- 第一步：cmd_vel 按 bag 时间挂到 odom ----
    # 两路都是 20Hz，并且都被录包器缓冲成同样的 burst，按 bag 时间最近邻最稳。
    odom_df = odom_df.sort_values("bag_time").reset_index(drop=True)
    cmd_df = cmd_df.sort_values("bag_time").reset_index(drop=True)
    odom_df = pd.merge_asof(
        odom_df,
        cmd_df[["bag_time", "action_v", "action_w"]],
        on="bag_time", direction="nearest", tolerance=CMD_TOL,
    )
    n_odom = len(odom_df)
    n_cmd_missing = int(odom_df["action_v"].isna().sum())
    print(f"\nodom rows without cmd match: {n_cmd_missing} / {n_odom}")
    odom_df = odom_df.dropna(subset=["action_v"]).reset_index(drop=True)

    # ---- 第二步：图像按墙钟 header 挂到 odom ----
    img_df = img_df.sort_values("t").reset_index(drop=True)
    odom_df = odom_df.sort_values("t").reset_index(drop=True)
    df = pd.merge_asof(
        img_df,
        odom_df[["t", "state_v", "state_w", "x", "y", "yaw",
                 "action_v", "action_w"]],
        on="t", direction="nearest", tolerance=TOL,
    )

    n0 = len(df)
    df = df.dropna(subset=["state_v", "action_v"]).reset_index(drop=True)
    print(f"aligned: {len(df)} / {n0} images")
    if df.empty:
        raise SystemExit("对齐后为空：检查时间戳/容差")

    df["episode_index"] = 0
    df["frame_index"] = range(len(df))
    df["timestamp"] = df["t"] - t0

    cols = [
        "episode_index", "frame_index", "timestamp",
        "bag_time", "hdr_time", "image_path",
        "state_v", "state_w", "action_v", "action_w", "x", "y", "yaw",
    ]
    df = df[cols]
    df.to_parquet(PARQUET, index=False)

    print(f"\nsaved: {PARQUET}")
    print(df.head(8).to_string(index=False))

    dt = df["timestamp"].diff().dropna()
    print("\nimage dt (s):")
    print(dt.describe())

    print("\nstate/action stats:")
    print(df[["state_v", "state_w", "action_v", "action_w"]].describe().loc[["min", "50%", "max"]])


if __name__ == "__main__":
    main()