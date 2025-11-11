# 读取 ROS2 mcap（rosbag2）文件，提取图像、关节状态、目标动作，并按图像时间对齐。
# 可选将结果写入 LeRobotDataset，便于下游训练。

import os
import glob
import bisect
import pdb
import yaml
import cv2
import tyro
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# mcap-ros2 支持
from mcap_ros2.reader import read_ros2_messages
# 可选：写入 LeRobot
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except Exception:
    LeRobotDataset = None


# ========= 话题配置（来自你的 metadata.yaml） =========
@dataclass
class TopicConfig:
    # 选择一个参考相机（以它的时间线对齐其它信号）
    ref_image: str = "/hdas/camera_wrist_right/color/image_rect_raw/compressed"

    # 可同时采集的其它图像（会随 ref 对齐；不存在则自动跳过）
    extra_images: List[str] = (
        "/hdas/camera_wrist_left/color/image_rect_raw/compressed",
        "/hdas/camera_head/left_raw/image_raw_color/compressed",
        "/hdas/camera_head/right_raw/image_raw_color/compressed",
        "/hdas/camera_wrist_right/aligned_depth_to_color/image_raw",
        "/hdas/camera_wrist_left/aligned_depth_to_color/image_raw",
    ) # type: ignore

    # 关节反馈（状态），用于 state
    feedback_joint_topics: List[str] = (
        "/hdas/feedback_arm_left",
        "/hdas/feedback_arm_right",
        "/hdas/feedback_gripper_left",
        "/hdas/feedback_gripper_right",
        "/hdas/feedback_torso",
        "/hdas/feedback_chassis",
    ) # type: ignore

    # 将使用 “目标” 作为动作 proxy（避免依赖你的自定义 hdas_msg）
    # 动作 = 左臂关节目标 + 右臂关节目标 + 左右夹爪目标 + 底盘速度目标(vx, vy, wz)
    action_joint_targets: List[str] = (
        "/motion_target/target_joint_state_arm_left",
        "/motion_target/target_joint_state_arm_right",
        "/motion_target/target_position_gripper_left",
        "/motion_target/target_position_gripper_right",
    ) # type: ignore
    action_chassis_speed: str = "/motion_target/target_speed_chassis"  # geometry_msgs/TwistStamped


# ========== 工具函数 ==========
def find_mcap_file(bag_path: str) -> str:
    # bag_path 可以是目录（包含 metadata.yaml 和 *.mcap），也可以直接是 .mcap 文件
    if os.path.isdir(bag_path):
        meta = os.path.join(bag_path, "metadata.yaml")
        if os.path.exists(meta):
            with open(meta, "r") as f:
                info = yaml.safe_load(f)
            rels = info.get("rosbag2_bagfile_information", {}).get("relative_file_paths", [])
            if rels:
                candidate = os.path.join(bag_path, rels[0])
                if os.path.exists(candidate):
                    return candidate
        # 兜底：直接找 *.mcap
        mcaps = glob.glob(os.path.join(bag_path, "*.mcap"))
        if not mcaps:
            raise FileNotFoundError(f"目录 {bag_path} 下未找到 .mcap")
        return mcaps[0]
    else:
        if not bag_path.endswith(".mcap"):
            raise ValueError("请提供目录（含 metadata.yaml）或 .mcap 文件路径")
        return bag_path


def image_from_ros2(msg) -> np.ndarray:
    # 支持 CompressedImage 与 Image
    # CompressedImage: .format, .data
    if hasattr(msg, "format") and hasattr(msg, "data") and not hasattr(msg, "encoding"):
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("OpenCV 解码失败")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img

    # Image: .encoding, .data, .width, .height
    enc = getattr(msg, "encoding", "rgb8")
    if enc in ("rgb8", "bgr8", "mono8"):
        dtype = np.uint8
    elif enc in ("mono16", "16UC1"):
        dtype = np.uint16
    else:
        dtype = np.uint8

    arr = np.frombuffer(msg.data, dtype=dtype)
    if enc in ("rgb8", "bgr8"):
        img = arr.reshape((msg.height, msg.width, 3))
        if enc == "bgr8":
            img = img[:, :, ::-1]
    elif enc in ("mono8", "mono16", "16UC1"):
        img = arr.reshape((msg.height, msg.width))
    else:
        ch = max(1, (len(arr) // (msg.height * msg.width)))
        img = arr.reshape((msg.height, msg.width, ch))
    # 若是深度图，保持原 dtype；彩色图返回 RGB
    return img


def to_uint8_rgb(img: np.ndarray, size=(256, 256)) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if img.dtype != np.uint8:
        # 简单归一化到 0..255（如深度请按需求处理，这里仅示例）
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def nearest_idx(ts_list: List[int], t_ref: int) -> int:
    i = bisect.bisect_left(ts_list, t_ref)
    if i == 0:
        return 0
    if i == len(ts_list):
        return len(ts_list) - 1
    return i if abs(ts_list[i] - t_ref) < abs(t_ref - ts_list[i - 1]) else i - 1


# ========== 主流程 ==========
def extract_streams(mcap_path: str, cfg: TopicConfig):
    pdb.set_trace()
    include_topics = [cfg.ref_image]
    include_topics += list(cfg.extra_images)
    include_topics += list(cfg.feedback_joint_topics)
    include_topics += list(cfg.action_joint_targets)
    include_topics += [cfg.action_chassis_speed]
    include_topics = sorted(set([t for t in include_topics if t]))

    image_streams: Dict[str, List[Tuple[int, np.ndarray]]] = {t: [] for t in include_topics}
    joint_streams: Dict[str, List[Tuple[int, Dict[str, float]]]] = {}
    action_joint_streams: Dict[str, List[Tuple[int, Dict[str, float]]]] = {}
    twist_stream: List[Tuple[int, np.ndarray]] = []

    def iter_msgs(stream):
        # 优先使用新签名过滤；老版本无 topics 参数则全量读取再手动过滤
        try:
            it = read_ros2_messages(stream, topics=include_topics)
            filter_needed = False
        except TypeError:
            it = read_ros2_messages(stream)
            filter_needed = True

        for item in it:
            # 兼容：对象 or 三元组
            if isinstance(item, tuple) and len(item) == 3:
                topic, msg, t = item
            else:
                topic = getattr(item, "topic", None)
                msg = getattr(item, "ros_msg", None) or getattr(item, "message", None)
                # 时间戳字段名在不同版本可能为 log_time 或 publish_time（纳秒）
                t = getattr(item, "log_time", None)
                if t is None:
                    t = getattr(item, "publish_time", None)
            if filter_needed and topic not in include_topics:
                continue
            yield topic, msg, t

    with open(mcap_path, "rb") as f:
        for topic, msg, t in iter_msgs(f):
            # 图像
            if topic in [cfg.ref_image, *cfg.extra_images]:
                try:
                    img = image_from_ros2(msg)
                    image_streams[topic].append((t, img))
                except Exception as e:
                    print(f"[warn] 解码图像失败 {topic} @ {t}: {e}")
                continue

            # 关节反馈 JointState
            if topic in cfg.feedback_joint_topics:
                names = list(getattr(msg, "name", []))
                pos = list(getattr(msg, "position", []))
                jd = {n: float(p) for n, p in zip(names, pos)}
                joint_streams.setdefault(topic, []).append((t, jd))
                continue

            # 动作（目标）JointState
            if topic in cfg.action_joint_targets:
                names = list(getattr(msg, "name", []))
                pos = list(getattr(msg, "position", []))
                jd = {n: float(p) for n, p in zip(names, pos)}
                action_joint_streams.setdefault(topic, []).append((t, jd))
                continue

            # 底盘目标速度 TwistStamped
            if topic == cfg.action_chassis_speed:
                tw = getattr(msg, "twist", None)
                if tw is not None:
                    v = tw.linear
                    w = tw.angular
                    vec = np.array([float(v.x), float(v.y), float(w.z)], dtype=np.float32)
                    twist_stream.append((t, vec))
                continue

    # 排序
    for k in image_streams:
        image_streams[k].sort(key=lambda x: x[0])
    for k in joint_streams:
        joint_streams[k].sort(key=lambda x: x[0])
    for k in action_joint_streams:
        action_joint_streams[k].sort(key=lambda x: x[0])
    twist_stream.sort(key=lambda x: x[0])

    return image_streams, joint_streams, action_joint_streams, twist_stream


def build_state_order(joint_streams: Dict[str, List[Tuple[int, Dict[str, float]]]]) -> List[str]:
    # 将所有反馈关节名合并成一个稳定顺序（按首次出现顺序）
    seen = set()
    order: List[str] = []
    for topic in sorted(joint_streams.keys()):
        for _, jd in joint_streams[topic]:
            for n in jd.keys():
                if n not in seen:
                    seen.add(n)
                    order.append(n)
        # 只用该 topic 的第一批即可获得名字
    return order


def build_action_order(action_joint_streams: Dict[str, List[Tuple[int, Dict[str, float]]]], cfg: TopicConfig) -> List[str]:
    order: List[str] = []
    seen = set()

    def add_prefixed(topic: str, jd: Dict[str, float], prefix: str):
        nonlocal order, seen
        for n in jd.keys():
            key = f"{prefix}:{n}"
            if key not in seen:
                seen.add(key)
                order.append(key)

    # 约定前缀，确保左右手臂、夹爪不冲突
    if "/motion_target/target_joint_state_arm_left" in action_joint_streams:
        jd = action_joint_streams["/motion_target/target_joint_state_arm_left"][0][1]
        add_prefixed("/motion_target/target_joint_state_arm_left", jd, "L")
    if "/motion_target/target_joint_state_arm_right" in action_joint_streams:
        jd = action_joint_streams["/motion_target/target_joint_state_arm_right"][0][1]
        add_prefixed("/motion_target/target_joint_state_arm_right", jd, "R")
    if "/motion_target/target_position_gripper_left" in action_joint_streams:
        jd = action_joint_streams["/motion_target/target_position_gripper_left"][0][1]
        add_prefixed("/motion_target/target_position_gripper_left", jd, "GL")
    if "/motion_target/target_position_gripper_right" in action_joint_streams:
        jd = action_joint_streams["/motion_target/target_position_gripper_right"][0][1]
        add_prefixed("/motion_target/target_position_gripper_right", jd, "GR")

    # 底盘 3 维
    order += ["CH:vx", "CH:vy", "CH:wz"]
    return order


def vectorize_state(joint_streams: Dict[str, List[Tuple[int, Dict[str, float]]]], names_order: List[str], t_ref: int, hold_ms: int = 200) -> Optional[np.ndarray]:
    # 最近邻 + 超过阈值则置 None
    hold_ns = hold_ms * 1_000_000
    merged: Dict[str, float] = {}

    for topic, seq in joint_streams.items():
        if not seq:
            continue
        ts = [t for t, _ in seq]
        i = nearest_idx(ts, t_ref)
        t_sel, jd = seq[i]
        if abs(t_sel - t_ref) <= hold_ns:
            merged.update(jd)

    if not merged:
        return None

    vec = np.zeros((len(names_order),), dtype=np.float32)
    for i, n in enumerate(names_order):
        if n in merged:
            vec[i] = merged[n]
    return vec


def vectorize_action(action_joint_streams: Dict[str, List[Tuple[int, Dict[str, float]]]], twist_stream: List[Tuple[int, np.ndarray]], action_order: List[str], t_ref: int, hold_ms: int = 200) -> Optional[np.ndarray]:
    hold_ns = hold_ms * 1_000_000
    merged: Dict[str, float] = {}

    def pull(topic: str, prefix: str):
        if topic not in action_joint_streams or not action_joint_streams[topic]:
            return
        ts = [t for t, _ in action_joint_streams[topic]]
        i = nearest_idx(ts, t_ref)
        t_sel, jd = action_joint_streams[topic][i]
        if abs(t_sel - t_ref) <= hold_ns:
            for n, v in jd.items():
                merged[f"{prefix}:{n}"] = v

    pull("/motion_target/target_joint_state_arm_left", "L")
    pull("/motion_target/target_joint_state_arm_right", "R")
    pull("/motion_target/target_position_gripper_left", "GL")
    pull("/motion_target/target_position_gripper_right", "GR")

    # chassis
    if twist_stream:
        ts = [t for t, _ in twist_stream]
        i = nearest_idx(ts, t_ref)
        t_sel, vec = twist_stream[i]
        if abs(t_sel - t_ref) <= hold_ns:
            merged["CH:vx"] = float(vec[0])
            merged["CH:vy"] = float(vec[1])
            merged["CH:wz"] = float(vec[2])

    if not merged:
        return None

    out = np.zeros((len(action_order),), dtype=np.float32)
    for i, k in enumerate(action_order):
        out[i] = merged.get(k, 0.0)
    return out


def align_frames(image_streams, joint_streams, action_joint_streams, twist_stream, cfg: TopicConfig, max_diff_ms: int = 50):
    ref = image_streams.get(cfg.ref_image, [])
    if not ref:
        raise RuntimeError(f"参考相机 {cfg.ref_image} 没有数据")
    ref_ts = [t for t, _ in ref]

    # 预计算维度与顺序
    state_order = build_state_order(joint_streams)
    action_order = build_action_order(action_joint_streams, cfg)

    frames = []
    max_diff_ns = max_diff_ms * 1_000_000

    # 其它图像 topic 的时间索引表
    other_imgs = {t: image_streams.get(t, []) for t in cfg.extra_images}
    other_ts = {t: [x for x, _ in seq] for t, seq in other_imgs.items()}

    for t_img, img in ref:
        # 对齐其它图像
        images: Dict[str, np.ndarray] = {cfg.ref_image: img}
        for tpc, seq in other_imgs.items():
            if not seq:
                continue
            idx = nearest_idx(other_ts[tpc], t_img)
            t_sel, im2 = seq[idx]
            if abs(t_sel - t_img) <= max_diff_ns:
                images[tpc] = im2

        # 对齐 state / action
        state = vectorize_state(joint_streams, state_order, t_img, hold_ms=200)
        action = vectorize_action(action_joint_streams, twist_stream, action_order, t_img, hold_ms=200)

        frames.append(
            dict(
                t_ns=t_img,
                images=images,
                state=state,          # np.ndarray 或 None
                action=action,        # np.ndarray 或 None
                state_order=state_order,
                action_order=action_order,
            )
        )
    return frames


def save_frames_to_lerobot(frames, image_size=(256, 256), repo_id: str = "username/ros2_mcap_dataset", fps: int = 10, push_to_hub: bool = False):
    if LeRobotDataset is None:
        raise RuntimeError("未安装 lerobot，请先安装：uv pip install lerobot")

    # 从首帧推断维度（如缺失则置 0）
    first = next((f for f in frames if f.get("state") is not None or f.get("action") is not None), frames[0])
    state_dim = 0 if first.get("state") is None else len(first["state"])
    action_dim = 0 if first.get("action") is None else len(first["action"])

    # 标准化只保留两路图像：右腕彩色 -> image；左腕彩色 -> wrist_image（若不存在则跳过）
    # 可按需修改为 head/深度等
    right = "/hdas/camera_wrist_right/color/image_rect_raw/compressed"
    left = "/hdas/camera_wrist_left/color/image_rect_raw/compressed"

    features = {
        "image": {"dtype": "image", "shape": (image_size[1], image_size[0], 3), "names": ["H", "W", "C"]},
    }
    if left in (frames[0]["images"].keys()):
        features["wrist_image"] = {"dtype": "image", "shape": (image_size[1], image_size[0], 3), "names": ["H", "W", "C"]}
    if state_dim > 0:
        features["state"] = {"dtype": "float32", "shape": (state_dim,), "names": ["state"]}
    if action_dim > 0:
        features["actions"] = {"dtype": "float32", "shape": (action_dim,), "names": ["actions"]}

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="generic",
        fps=fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=2,
    )

    for f in frames:
        im_right = f["images"].get(right)
        if im_right is None:
            continue
        rec: Dict[str, Any] = {}
        rec["image"] = to_uint8_rgb(im_right, size=image_size)
        if left in f["images"]:
            rec["wrist_image"] = to_uint8_rgb(f["images"][left], size=image_size)

        if f.get("state") is not None and "state" in features:
            rec["state"] = f["state"].astype(np.float32)
        if f.get("action") is not None and "actions" in features:
            rec["actions"] = f["action"].astype(np.float32)

        ds.add_frame(rec)

    ds.save_episode()
    if push_to_hub:
        ds.push_to_hub(tags=["ros2", "mcap"], private=False, push_videos=True, license="apache-2.0")
    return ds


def main(
    bag_path: str,
    ref_image: str = "/hdas/camera_wrist_right/color/image_rect_raw/compressed",
    fps: int = 10,
    max_align_diff_ms: int = 50,
    save_lerobot: bool = False,
    repo_id: str = "antuo/ros2_mcap_dataset",
    push_to_hub: bool = False,
):
    cfg = TopicConfig(ref_image=ref_image)
    mcap_file = find_mcap_file(bag_path)
    print(f"[info] 读取 {mcap_file}")

    image_streams, joint_streams, action_joint_streams, twist_stream = extract_streams(mcap_file, cfg)
    pdb.set_trace()
    frames = align_frames(image_streams, joint_streams, action_joint_streams, twist_stream, cfg, max_diff_ms=max_align_diff_ms)

    print(f"[info] 对齐帧数: {len(frames)}")
    # 简单打印一个样本
    for f in frames[:1]:
        im = f["images"].get(cfg.ref_image)
        print(f"[sample] t={f['t_ns']} image={None if im is None else im.shape} state_dim={0 if f['state'] is None else len(f['state'])} action_dim={0 if f['action'] is None else len(f['action'])}")
        print(f"[state_order] N={len(f['state_order'])}")
        print(f"[action_order] N={len(f['action_order'])}")

    if save_lerobot:
        save_frames_to_lerobot(frames, image_size=(256, 256), repo_id=repo_id, fps=fps, push_to_hub=push_to_hub)
        print(f"[info] 已保存到 LeRobot 数据集: {repo_id}")


if __name__ == "__main__":
    tyro.cli(main)