# LIBERO-plus Benchmark

This example runs the LIBERO-plus benchmark: https://github.com/sylvestf/LIBERO-plus

LIBERO-plus is an extended benchmark that systematically exposes vulnerabilities of Vision-Language-Action (VLA) models through comprehensive robustness evaluation across seven perturbation dimensions:
1. **Objects Layout** - Confounding objects and target object displacement
2. **Camera Viewpoints** - Position, orientation, and field-of-view changes
3. **Robot Initial States** - Manipulator initial pose variations
4. **Language Instructions** - LLM-based instruction rewriting
5. **Light Conditions** - Intensity, direction, color, and shadow variations
6. **Background Textures** - Scene and surface appearance changes
7. **Sensor Noise** - Photometric distortions and image degradation

## Prerequisites

This example requires:
1. LIBERO-plus submodule to be initialized
2. LIBERO-plus assets to be downloaded and extracted
3. LIBERO-plus configuration file to be set up

### Initialize LIBERO-plus

Make sure the LIBERO-plus submodule is initialized:

```bash
git submodule update --init --recursive
```

### Download LIBERO-plus Assets

Download the assets from [LIBERO-plus HuggingFace](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main) and extract `assets.zip` to:
```
third_party/LIBERO-plus/libero/libero/assets/
```

The extracted directory structure should look like:
```
third_party/LIBERO-plus/libero/libero/assets/
├── articulated_objects/
├── new_objects/
├── scenes/
├── stable_hope_objects/
├── stable_scanned_objects/
├── textures/
├── turbosquid_objects/
├── serving_region.xml
├── wall_frames.stl
└── wall.xml
```

### Configure LIBERO-plus

LIBERO-plus uses a configuration file at `~/.libero/plus-config.yaml`. The first time you run the evaluation, it will prompt you to create this file. Alternatively, you can create it manually:

```bash
mkdir -p ~/.libero
cat > ~/.libero/plus-config.yaml <<EOF
benchmark_root: /path/to/openpi/third_party/LIBERO-plus/libero/libero
bddl_files: /path/to/openpi/third_party/LIBERO-plus/libero/libero/bddl_files
init_states: /path/to/openpi/third_party/LIBERO-plus/libero/libero/init_files
datasets: /path/to/openpi/third_party/LIBERO-plus/libero/datasets
assets: /path/to/openpi/third_party/LIBERO-plus/libero/libero/assets
EOF
```

Replace `/path/to/openpi` with the actual path to your openpi repository.

## Installation

### With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./checkpoints/pi05_libero/<exp_name>/<step>" \
docker compose -f examples/libero_plus/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./checkpoints/pi05_libero/<exp_name>/<step>" \
docker compose -f examples/libero_plus/compose.yml up --build
```

You can customize the loaded checkpoint and task suite:

```bash
# To load a custom checkpoint:
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run a specific task suite:
export CLIENT_ARGS="--args.task-suite-name libero_spatial"
```

### Without Docker

Terminal window 1 (Policy Server):

```bash
# Run the policy server with pi05_libero checkpoint
uv run scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir ./checkpoints/pi05_libero/<exp_name>/<step>
```

Terminal window 2 (Evaluation Client):

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero_plus/.venv
source examples/libero_plus/.venv/bin/activate

# Install dependencies
uv pip sync examples/libero/requirements.txt third_party/LIBERO-plus/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/LIBERO-plus

# Set Python path to include LIBERO-plus
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/LIBERO-plus

# Run the evaluation
python examples/libero_plus/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero_plus/main.py
```

## Usage

### Command Line Arguments

The evaluation script supports the following arguments:

- `--host`: Policy server host (default: "0.0.0.0")
- `--port`: Policy server port (default: 8000)
- `--resize-size`: Image resize size (default: 224)
- `--replan-steps`: Number of steps to replan (default: 5)
- `--task-suite-name`: Task suite to evaluate (default: "libero_spatial")
  - Options: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`, `libero_mix`
- `--num-steps-wait`: Number of steps to wait for objects to stabilize (default: 10)
- `--num-trials-per-task`: Number of rollouts per task (default: 1, as required by LIBERO-plus)
- `--video-out-path`: Path to save videos (default: "data/libero_plus/videos")
- `--seed`: Random seed (default: 7)

### Examples

Evaluate on libero_spatial suite:

```bash
python examples/libero_plus/main.py --task-suite-name libero_spatial
```

Evaluate on libero_mix suite (includes perturbation tasks):

```bash
python examples/libero_plus/main.py --task-suite-name libero_mix
```

Evaluate with custom checkpoint path:

```bash
# First, start the server with your checkpoint
uv run scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir ./checkpoints/pi05_libero/my_experiment/30000

# Then run the evaluation
python examples/libero_plus/main.py --task-suite-name libero_spatial
```

## Task Suites

LIBERO-plus extends the original LIBERO benchmark with additional perturbation tasks. Available task suites:

- **libero_spatial**: Spatial reasoning tasks (original LIBERO suite)
- **libero_object**: Object manipulation tasks (original LIBERO suite)
- **libero_goal**: Goal-conditioned tasks (original LIBERO suite)
- **libero_10**: 10-task suite (original LIBERO suite)
- **libero_90**: 90-task suite (original LIBERO suite)
- **libero_mix**: Mixed suite with perturbation tasks (LIBERO-plus extension)

Each suite contains tasks with various perturbations across the seven dimensions mentioned above.

## Results

The evaluation script will:
1. Run each task in the selected suite
2. Log success rates per task and overall
3. Save video recordings of each episode to `data/libero_plus/videos/`

Results are logged to the console and include:
- Per-task success rate
- Overall success rate
- Total number of episodes and successes

## Notes

- **num_trials_per_task**: LIBERO-plus requires `num_trials_per_task=1` (unlike original LIBERO which uses 50). This is set as the default.
- **Checkpoint**: This example is configured to use the `pi05_libero` config, which should be fine-tuned on the LIBERO dataset.
- **Performance**: LIBERO-plus evaluation can take a long time depending on the task suite size. The `libero_mix` suite contains thousands of tasks.

## Troubleshooting

### Import Errors

If you encounter import errors related to LIBERO-plus, make sure:
1. The LIBERO-plus submodule is initialized
2. LIBERO-plus is installed: `pip install -e third_party/LIBERO-plus`
3. The Python path includes LIBERO-plus: `export PYTHONPATH=$PYTHONPATH:$PWD/third_party/LIBERO-plus`

### Configuration File Issues

If LIBERO-plus cannot find the configuration file:
1. Check that `~/.libero/plus-config.yaml` exists
2. Verify all paths in the config file are correct and absolute
3. Ensure the assets directory contains all required files

### EGL Errors

If you encounter EGL errors, try using GLX instead:
```bash
MUJOCO_GL=glx python examples/libero_plus/main.py
```

## Citation

If you use LIBERO-plus in your research, please cite:

```bibtex
@article{fei25libero-plus,
    title={LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models},
    author={Senyu Fei and Siyin Wang and Junhao Shi and Zihao Dai and Jikun Cai and Pengfang Qian and Li Ji and Xinzhe He and Shiduo Zhang and Zhaoye Fei and Jinlan Fu and Jingjing Gong and Xipeng Qiu},
    journal = {arXiv preprint arXiv:2510.13626},
    year={2025},
}
```

