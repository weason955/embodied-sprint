# embodied-sprint

具身智能 / 世界模型实习冲刺（2026-11 至 2026-12）。

## 内容

- 真机数据采集与格式转换（RDK X3 + ROS2 → LeRobot）
- 模仿学习基线复现与标准基准评测
- 动作条件视频预测小实验
- 论文笔记与实验归因

## 进度

- [x] Day 0：环境、git、GitHub 仓库
- [ ] Day 1：环境与代码能力
- [ ] Day 2：世界模型阅读
- [ ] Day 3：真机数据管线
- [ ] Day 4：训练与评测
- [ ] Day 5：真机微调
- [ ] Day 6：整理与简历
- [ ] Day 7：投递

## 数据管线：ROS2 bag → parquet

输入：`~/bag4`，103.5s，`/image`（JPEG 1280x720，约 30Hz）、`/odom`（20Hz）、`/cmd_vel`（20Hz）。

处理流程：

1. 用 `rosbags` 读取 ROS2 Humble bag；
2. 图像是 JPEG-in-`sensor_msgs/Image`（`encoding == "jpeg"`），按 JPEG 字节解码，不能按 `step` reshape；
3. 图像缩放到 640x360，保存为 jpg；
4. state = `/odom` 的 twist `[v, ω]`；action = `/cmd_vel` 的 `[v, ω]`；
5. 时间轴修正：
   - 相机 header 是单调时钟（约 2833s），bag 是墙钟（约 1.795e9），两者只差一个固定偏移；
   - 用 `min(bag_time - header_time)` 把图像换算到墙钟；
   - odom 的 header 本身就是墙钟，直接使用；
   - cmd_vel 没有 header，按 bag 时间最近邻挂到 odom；
   - 图像按墙钟最近邻挂到 odom，容差 60ms。

结果：2884/2884 帧全部对齐，时长 103.34s，平均 27.9Hz；图像 dt 中位数 32ms，最大 300ms；真实掉帧 21 处（0.2–0.3s，早期约每 3.4s 一次），不是时间戳换算造成的。

已知限制：

- 只有 1 条 episode，够做管线和世界模型 proof-of-concept，不够训通用策略；
- 相机存在周期性 0.2–0.3s 掉帧，后续按 gap 切段，不做图像插值；
- state 用 odom twist，不用全局 x/y/yaw（会漂移）；
- 仍然帧占 60.6%，10Hz 重采样后再做一次仍帧下采样。
