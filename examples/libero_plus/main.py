import collections
import dataclasses
import json
import logging
import math
import pathlib
import sys
from typing import Optional

# Add LIBERO-plus to Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "third_party" / "LIBERO-plus"))

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO-plus environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, libero_mix
    )
    # Optional category filter. If provided, only tasks with this category will be evaluated.
    # Options: Background Textures, Camera Viewpoints, Language Instructions, Light Conditions,
    #          Objects Layout, Robot Initial States, Sensor Noise, or None to evaluate all tasks
    category: Optional[str] = None
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1  # Number of rollouts per task (LIBERO-plus requires 1)

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero_plus/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero_plus(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    video_out_path = args.video_out_path + "_" + args.task_suite_name + "_" + args.category if args.category else args.video_out_path + "_" + args.task_suite_name
    # Initialize LIBERO-plus task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Number of tasks: {num_tasks_in_suite}")

    # Load task classification and filter tasks by suite and category
    task_classification_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "third_party"
        / "LIBERO-plus"
        / "libero"
        / "libero"
        / "benchmark"
        / "task_classification.json"
    )
    
    with open(task_classification_path, "r") as f:
        task_classification = json.load(f)
    
    # Filter tasks based on task suite and optional category
    filtered_task_ids = []
    if args.task_suite_name in task_classification:
        tasks = task_classification[args.task_suite_name]
        for task in tasks:
            # If category is specified, only include tasks matching that category
            if args.category is None or task["category"] == args.category:
                # Note: task["id"] is 1-indexed, but we need 0-indexed for array access
                # We'll store the task info and use it later
                filtered_task_ids.append({
                    "id": task["id"] - 1,  # Convert to 0-indexed
                    "name": task["name"],
                    "category": task["category"],
                })
    
    if not filtered_task_ids:
        if args.category:
            raise ValueError(
                f"No tasks found for suite '{args.task_suite_name}' with category '{args.category}'. "
                f"Available categories: Background Textures, Camera Viewpoints, Language Instructions, "
                f"Light Conditions, Objects Layout, Robot Initial States, Sensor Noise"
            )
        else:
            raise ValueError(
                f"No tasks found for suite '{args.task_suite_name}' in task_classification.json"
            )
    
    logging.info(f"Filtered {len(filtered_task_ids)} tasks for evaluation")
    if args.category:
        logging.info(f"Category filter: {args.category}")
    else:
        logging.info("No category filter applied - evaluating all tasks in suite")
    
    # Log some example filtered tasks for verification
    if filtered_task_ids:
        logging.info(f"Example filtered tasks (first 5):")
        for task_info in filtered_task_ids[:5]:
            logging.info(f"  - Task ID {task_info['id']}: {task_info['name'][:60]}... (Category: {task_info['category']})")
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Determine max_steps based on task suite
    # LIBERO-plus extends the original suites with perturbation tasks
    if args.task_suite_name == "libero_spatial":
        max_steps = 220 * 3 # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280 * 3 # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300 * 3 # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520 * 3 # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400 * 3 # longest training demo has 373 steps
    elif args.task_suite_name == "libero_mix":
        # libero_mix is a combination of tasks, use a conservative max_steps
        max_steps = 520 * 3
    else:
        raise ValueError(
            f"Unknown task suite: {args.task_suite_name}. "
            f"Available options: libero_spatial, libero_object, libero_goal, libero_10, libero_90, libero_mix"
        )

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation using filtered task IDs
    total_episodes, total_successes = 0, 0
    for task_info in tqdm.tqdm(filtered_task_ids, desc="Tasks"):
        task_id = task_info["id"]
        
        # Validate task_id is within range
        if task_id < 0 or task_id >= num_tasks_in_suite:
            logging.warning(f"Skipping task_id {task_id} (out of range [0, {num_tasks_in_suite}))")
            continue
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO-plus initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO-plus environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task {task_id} episodes", leave=False):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            # Handle case where initial_states might have different shapes for different perturbations
            # LIBERO-plus returns torch tensors, convert to numpy if needed
            import torch
            if isinstance(initial_states, torch.Tensor):
                initial_states = initial_states.numpy()
            
            if len(initial_states.shape) > 1 and initial_states.shape[0] > episode_idx:
                init_state = initial_states[episode_idx]
            else:
                # Fallback to first initial state if not enough states available
                init_state = initial_states[0] if len(initial_states.shape) > 1 else initial_states
            obs = env.set_init_state(init_state)

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            # reduce video saving frequency
            if t % 10 == 0:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_task{task_id}_ep{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results for this task
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")
    # 将测试结果保存到文件
    with open(pathlib.Path(video_out_path) / "test_results.txt", "w") as f:
        f.write(f"Total success rate: {float(total_successes) / float(total_episodes)}\n")
        f.write(f"Total episodes: {total_episodes}\n")
        f.write(f"Total successes: {total_successes}\n")
    


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO-plus environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero_plus)

