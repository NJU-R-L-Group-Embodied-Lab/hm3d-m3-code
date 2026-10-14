from typing import List, Optional, Union

import habitat_baselines.config.default
from habitat.config.default import Config as CN
from habitat.config.default import CONFIG_FILE_SEPARATOR
from habitat_extensions.config.default import get_extended_config as get_task_config


# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/zs_vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.TRAINER_NAME = "ZS-Evaluator"
_C.ENV_NAME = "VLNCEZeroShotEnv"
_C.SIMULATOR_GPU_IDS = [0]
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"
_C.BLIP2_MODEL_DIR = "data/blip2/blip2_model.pt"
_C.BLIP2_VIS_PROCESSORS_DIR = "data/blip2/blip2_vis_processors.pt"
_C.BLIP2_TEXT_PROCESSORS_DIR = "data/blip2/blip2_text_processors.pt"
_C.VQA_MODEL_DIR = "data/vqa/vqa_model.pt"
_C.VQA_VIS_PROCESSORS_DIR = "data/vqa/vqa_vis_processors.pt"
_C.VQA_TEXT_PROCESSORS_DIR = "data/vqa/vqa_text_processors.pt"
_C.KEYBOARD_CONTROL = 0
_C.SAVE_RGB_IMAGES = False
_C.RGB_SAVE_DIR = "saved_rgb_images"
_C.USE_IPLANNER = False
_C.DEBUG_LOGGING = False
_C.DEBUG_LOG_DIR = "logs/debug_logs"
_C.TASK_TYPE = "VLN"  # one of {"VLN", "OBJNAV", "EQA"}

# -----------------------------------------------------------------------------
# MAP CONFIG
# -----------------------------------------------------------------------------
_C.MAP = CN()
_C.MAP.GROUNDING_DINO_CONFIG_PATH = "data/grounded_sam/GroundingDINO_SwinT_OGC.py"
_C.MAP.GROUNDING_DINO_CHECKPOINT_PATH = "data/grounded_sam/groundingdino_swint_ogc.pth"
_C.MAP.SAM_CHECKPOINT_PATH = "data/grounded_sam/sam_vit_h_4b8939.pth"
_C.MAP.RepViTSAM_CHECKPOINT_PATH = "data/grounded_sam/repvit_sam.pt"
_C.MAP.SAM_ENCODER_VERSION = "vit_h"
_C.MAP.BOX_THRESHOLD = 0.25
_C.MAP.TEXT_THRESHOLD = 0.25
_C.MAP.FRAME_WIDTH = 160
_C.MAP.FRAME_HEIGHT = 120
_C.MAP.MAP_RESOLUTION = 5
_C.MAP.MAP_SIZE_CM = 2400
_C.MAP.GLOBAL_DOWNSCALING = 2
_C.MAP.VISION_RANGE = 100
_C.MAP.DU_SCALE = 1
_C.MAP.CAT_PRED_THRESHOLD = 5.0
_C.MAP.EXP_PRED_THRESHOLD = 1.0
_C.MAP.MAP_PRED_THRESHOLD = 1.0
_C.MAP.MAX_SEM_CATEGORIES = 16
_C.MAP.CENTER_RESET_STEPS = 25
_C.MAP.MIN_Z = 2 # a lager min_z could lost some information on the floor, 2cm is ok
_C.MAP.VISUALIZE = False
_C.MAP.PRINT_IMAGES = False
_C.MAP.REPVITSAM = 0


# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.SPLIT = "val_unseen"  # The split to evaluate on
_C.EVAL.USE_CKPT_CONFIG = True
_C.EVAL.EPISODE_COUNT = 5000
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.USE_TODO_LIST = True
_C.EVAL.BACKTRACK_SECOND_CHANCE = True
# When True, episodes are dispatched dynamically via a shared mp.Queue so
# fast workers can grab the next pending episode instead of being stuck with
# a fixed slice. Each worker rebuilds its habitat env per episode (~5-15s
# overhead per ep) but reuses its trainer/model state across episodes.
_C.EVAL.DYNAMIC_QUEUE = False
# When True (and USE_TODO_LIST=False), use the *_NO_TODO prompt templates so
# the LA prompt has no "Current TODO List" line, the LA output JSON has no
# reasoning_todo / todo_updates fields, and the VA prompt also drops the TODO
# segment ("pure FT"; the multi-agent LaViRA runner sets it). When USE_TODO_LIST=False and this
# flag is False, we fall back to FT-Unified (TT template + TODO rendered as
# "(none)"; output still contains empty todo_updates).
_C.EVAL.NO_TODO_PROMPT = False
# When True, the LA navigation history is the dense "every N steps a forward-view
# RGB" stream (continuous). When False, the LA history is the discrete waypoint
# stream used by the original single-agent LaViRA: per completed waypoint, two images
# (init_image = panorama-init at arrival; dir_image = view after the chosen turn)
# plus text labels including "Navigate to: <description>". Far fewer images per
# call (~2 per waypoint vs ~N/3 per episode).
_C.EVAL.USE_CONTINUOUS_HISTORY = False

# LA / VA model overrides. Empty string → fall back to hard-coded defaults
# inside VLMReasoningAgent.__init__.
_C.EVAL.LA_MODEL_NAME = ""
_C.EVAL.VA_MODEL_NAME = ""
_C.EVAL.LA_BASE_URL   = ""
_C.EVAL.LA_API_KEY    = ""
_C.EVAL.VA_BASE_URL   = ""
_C.EVAL.VA_API_KEY    = ""
# Optional: enable thinking on the LA path with an explicit budget. Default off
# matches the previous force-disable behavior in api.py.
_C.EVAL.LA_ENABLE_THINKING  = False
_C.EVAL.LA_THINKING_BUDGET  = 0   # tokens; 0 → no budget cap from our side


# -----------------------------------------------------------------------------
# MULTI-AGENT CONFIG (P1+)
# -----------------------------------------------------------------------------
# Knobs consumed by the multi-agent ObjectNav pipeline (Sim-MA-v1 + the new
# MA_Evaluator_mp). Single-agent runs ignore this block entirely.
_C.MULTI_AGENT = CN()
_C.MULTI_AGENT.NUM_AGENTS = 1
_C.MULTI_AGENT.MIN_SPAWN_DISTANCE = 3.0
_C.MULTI_AGENT.MAX_SPAWN_TRIES = 50


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        del config[k]
        config.register_deprecated_key(k)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = CN()
    config.merge_from_other_cfg(habitat_baselines.config.default._C)
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(
                    config.BASE_TASK_CONFIG_PATH
                )
                prev_task_config = config.BASE_TASK_CONFIG_PATH

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    assert config.TASK_TYPE in {"VLN", "OBJNAV", "EQA"}, (
        f"Unknown TASK_TYPE: {config.TASK_TYPE!r}. Expected one of VLN, OBJNAV, EQA."
    )

    config.freeze()
    return config
