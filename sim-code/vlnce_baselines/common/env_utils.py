import sys
import random
import os
import gzip
import json
from typing import List, Optional, Type, Union, Tuple

import habitat
from habitat import logger
from habitat import Config, Env, RLEnv, VectorEnv, make_dataset

random.seed(0)

def make_env_fn(config: Config, env_class: Type[Union[Env, RLEnv]], dataset: Optional[habitat.Dataset] = None) -> Union[Env, RLEnv]:
    """
    Constructs an environment instance for the given config and env_class.
    If dataset is provided, it is passed directly to the environment constructor.
    """
    if dataset is not None:
        env = env_class(config=config, dataset=dataset)
    else:
        env = env_class(config=config)
    return env


def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
    specific_episodes_allowed: Optional[List[Tuple[str, str]]] = None,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param auto_reset_done: Whether or not to automatically reset the env on done
    :return: VectorEnv object created according to specification.
    """

    num_envs_per_gpu = config.NUM_ENVIRONMENTS
    if isinstance(config.SIMULATOR_GPU_IDS, list):
        gpus = config.SIMULATOR_GPU_IDS
    else:
        gpus = [config.SIMULATOR_GPU_IDS]
    num_gpus = len(gpus)
    num_envs = num_gpus * num_envs_per_gpu

    if episodes_allowed is not None:
        config.defrost()
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = episodes_allowed
        config.freeze()

    if specific_episodes_allowed is not None:
        # Load the full dataset to filter it manually
        print(specific_episodes_allowed)
        logger.info(f"Loading full dataset for manual filtering...")
        print(config.TASK_CONFIG.DATASET.TYPE)
        print(config.TASK_CONFIG.DATASET)
        full_dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
        logger.info(f"Full dataset loaded. Total episodes: {len(full_dataset.episodes)}")
        
        # Filter episodes
        filtered_episodes = []
        # specific_episodes_allowed is a list of (id, scene_id)
        # We construct a set for faster lookup, handling potential scene_id mismatch (basename vs full path)
        target_episodes = []
        for eid, sid in specific_episodes_allowed:
            target_episodes.append({'id': str(eid), 'scene_id': sid})
            
        logger.info(f"Targeting {len(target_episodes)} specific episodes.")
        
        for ep in full_dataset.episodes:
            for target in target_episodes:
                if str(ep.episode_id) == target['id']:
                    # Check scene_id match
                    if ep.scene_id.endswith(target['scene_id']) or \
                       os.path.basename(ep.scene_id) == os.path.basename(target['scene_id']):
                        filtered_episodes.append(ep)
                        break
        
        if len(filtered_episodes) == 0:
            logger.error(f"No episodes matched specific_episodes_allowed! Target examples: {target_episodes[:3]}")
            if len(full_dataset.episodes) > 0:
                 sample_ep = full_dataset.episodes[0]
                 logger.error(f"Sample dataset episode: ID={sample_ep.episode_id}, Scene={sample_ep.scene_id}")
        else:
            logger.info(f"Filtered dataset from {len(full_dataset.episodes)} to {len(filtered_episodes)} episodes based on specific assignment.")
            
        # Update dataset episodes
        full_dataset.episodes = filtered_episodes
    else:
        full_dataset = None

    configs = []
    env_classes = [env_class for _ in range(num_envs)]
    
    # We don't need to load dataset here for scene check if we are in specific mode
    if specific_episodes_allowed is None:
        dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
        scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
        if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
            scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)
        logger.info(f"SPLTI: {config.TASK_CONFIG.DATASET.SPLIT}, NUMBER OF SCENES: {len(scenes)}")
    else:
        scenes = ["*"] # Dummy scenes
        logger.info(f"SPLTI: {config.TASK_CONFIG.DATASET.SPLIT}, Specific Episodes Mode")

    if num_envs > 1 and specific_episodes_allowed is None:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multi-process logic relies on being able"
                " to split scenes uniquely between processes"
            )

        if len(scenes) < num_envs and len(scenes) != 1:
            raise RuntimeError(
                "reduce the number of GPUs or envs as there"
                " aren't enough number of scenes"
            )

        random.shuffle(scenes)

    if len(scenes) == 1:
        scene_splits = [[scenes[0]] for _ in range(num_envs)]
    else:
        scene_splits = [[] for _ in range(num_envs)]
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)

        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_gpus):
        for j in range(num_envs_per_gpu):
            proc_config = config.clone()
            proc_config.defrost()
            proc_id = (i * num_envs_per_gpu) + j

            task_config = proc_config.TASK_CONFIG
            task_config.SEED += proc_id
            if len(scenes) > 0:
                task_config.DATASET.CONTENT_SCENES = scene_splits[proc_id]

            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpus[i]

            task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

            # Debug logging for config
            logger.info(f"Proc {proc_id} Config Check:")
            logger.info(f"  DATA_PATH: {task_config.DATASET.DATA_PATH}")
            logger.info(f"  CONTENT_SCENES: {task_config.DATASET.CONTENT_SCENES}")
            # logger.info(f"  EPISODES_ALLOWED: {task_config.DATASET.EPISODES_ALLOWED}")

            proc_config.freeze()
            configs.append(proc_config) 
            
    # Prepare datasets list
    datasets = [None] * num_envs
    if full_dataset is not None:
        # Pass the filtered dataset object directly to each env
        datasets = [full_dataset for _ in range(num_envs)]

    is_debug = True if sys.gettrace() else False
    env_entry = habitat.ThreadedVectorEnv
    envs = env_entry(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes, datasets)), 
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs
