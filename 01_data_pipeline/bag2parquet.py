"""把 ROS2 bag 转成 parquet 的最小示例。

当前只处理 /odom 和 /scan：
- /odom：一行一个里程计消息
- /scan：取最近距离和有效点数
- 以 odom 的 header_time 为基准，用最近邻对齐 scan

相机和 /cmd_vel 以后按同样的模式加。
"""
import math
from pathlib import Path

import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

BAG = Path.home() / "bag1"
OUT = Path.home() / "embodied-sprint" / "data" / "parquet"
OUT.mkdir(parents=True, exist_ok=True)


def quat_to_yaw(x, y, z, w):
    """四元数转偏航角（绕 z 轴）。"""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def read_odom(reader):
    rows = []
    conns = [c for c in reader.connections if c.topic == "/odom"]
    for conn, bag_ns, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        q = msg.pose.pose.orientation
        stamp = msg.header.stamp
        rows.append({
            "bag_time": bag_ns / 1e9,
            "header_time": stamp.sec + stamp.nanosec * 1e-9,
            "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y,
            "yaw": quat_to_yaw(q.x, q.y, q.z, q.w),
            "v": msg.twist.twist.linear.x,
            "w": msg.twist.twist.angular.z,
        })
    return pd.DataFrame(rows)


def read_scan(reader):
    rows = []
    conns = [c for c in reader.connections if c.topic == "/scan"]
    for conn, bag_ns, raw in reader.messages(connections=conns):
        msg = reader.deserialize(raw, conn.msgtype)
        stamp = msg.header.stamp
        valid = [float(r) for r in msg.ranges if math.isfinite(r) and r > 0.0]
        rows.append({
            "bag_time": bag_ns / 1e9,
            "header_time": stamp.sec + stamp.nanosec * 1e-9,
            "scan_min": min(valid) if valid else float("nan"),
            "scan_count": len(valid),
        })
    return pd.DataFrame(rows)


def main():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([BAG], default_typestore=typestore) as reader:
        print("topics:")
        for c in reader.connections:
            print(f"  {c.topic:20s} {c.msgtype}")

        odom = read_odom(reader)
        scan = read_scan(reader)

    print(f"odom rows: {len(odom)}, scan rows: {len(scan)}")
    if odom.empty or scan.empty:
        raise SystemExit("odom 或 scan 为空，检查话题名")

    odom = odom.sort_values("header_time").reset_index(drop=True)
    scan = scan.sort_values("header_time").reset_index(drop=True)

    merged = pd.merge_asof(
        odom,
        scan[["header_time", "scan_min", "scan_count"]],
        on="header_time",
        direction="nearest",
        tolerance=0.10,
    )

    out = OUT / "odom_scan.parquet"
    merged.to_parquet(out, index=False)

    print(f"saved: {out}")
    print(f"shape: {merged.shape}")
    print(merged.head(5).to_string(index=False))

    offset = odom["header_time"] - odom["bag_time"]
    print("\nheader_time - bag_time (s):")
    print(offset.describe())


if __name__ == "__main__":
    main()
