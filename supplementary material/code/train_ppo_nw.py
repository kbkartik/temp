import os
import json
import time
from collections import defaultdict
from typing import Tuple

import gym
import gym3
import numpy as np
import torch as th
from torch import nn
from gym_minigrid.wrappers import ImgObsWrapper

from minirl.algos.ppo.agent import PPOWorker
from minirl.buffer import Buffer
from minirl.envs.gym3_wrapper import ObsTransposeWrapper

import logger
import minigrid_env
from wrapper import ModifiedEpisodeStatsWrapper, RecordRoomWrapper, CheapReseedWrapper


def make_gym_env(**env_kwargs):
    env = gym.make(**env_kwargs)
    env = CheapReseedWrapper(env, int(1e7), np.random.randint(int(1e7)))
    env = ImgObsWrapper(env)
    env = RecordRoomWrapper(env)
    return env


def make_gym3_env(**kwargs):
    env = gym3.vectorize_gym(**kwargs)
    env = ObsTransposeWrapper(env, axes=(2, 0, 1))
    env = ModifiedEpisodeStatsWrapper(env)
    return env


def train(config):
    # Setup logger
    env_name = config["worker_kwargs"]["env_kwargs"]["env_kwargs"]["id"]
    task_name = "-".join(env_name.split("-")[1:-1])
    run_dir = os.path.join(
        config["run_cfg"]["log_dir"], task_name, f"run_{config['run_cfg']['run_id']}",
    )
    logger.configure(dir=run_dir, format_strs=["csv", "stdout"])
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)

    # Modify worker to add episodic intrinsic rewards
    class ModifiedPPOWorker(PPOWorker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ep_cnts = [dict() for _ in range(self.env.num)]

        def collect_batch(self) -> Tuple[dict, np.ndarray, np.ndarray]:
            """
            Additionally, collect next obs
            """
            # Update episodic unique states set before collecting experience
            reward, obs, first = self.env.observe()
            state_keys = [tuple(x) for x in obs.reshape(obs.shape[0], -1).tolist()]
            for env_idx, (key, ep_cnt) in enumerate(zip(state_keys, self.ep_cnts)):
                if first[env_idx]:
                    ep_cnt.clear()
                ep_cnt[key] = 1 + ep_cnt.get(key, 0)

            # Rollout
            batch = defaultdict(list)
            for _ in range(self.n_steps):
                reward, obs, first = self.env.observe()
                action, value, logpacs = self.policy.step(
                    obs[None, ...], first[None, ...]
                )
                batch["obs"].append(obs)
                batch["first"].append(first)
                batch["action"].append(action.squeeze(0))
                batch["value"].append(value.squeeze(0))
                batch["logpac"].append(logpacs.squeeze(0))
                self.env.act(action.squeeze(0))
                reward, next_obs, next_first = self.env.observe()
                # Record episodic visitation count and calculate curiosity
                ep_curiosity = np.zeros(shape=(self.env.num,), dtype=np.float32)
                state_keys = [
                    tuple(x) for x in next_obs.reshape(next_obs.shape[0], -1).tolist()
                ]
                for env_idx, (key, ep_cnt) in enumerate(zip(state_keys, self.ep_cnts)):
                    if next_first[env_idx]:
                        ep_cnt.clear()
                        ep_cnt[key] = 1
                    else:
                        ep_cnt[key] = 1 + ep_cnt.get(key, 0)
                        if config["ep_curiosity"] == "visit":
                            ep_curiosity[env_idx] = ep_cnt[key] == 1
                        elif config["ep_curiosity"] == "count":
                            ep_curiosity[env_idx] = 1 / np.sqrt(ep_cnt[key])
                # Add into batch
                batch["ep_curiosity"].append(ep_curiosity)
                batch["reward"].append(reward)
                batch["next_obs"].append(next_obs)
            # Concatenate
            batch["reward"] = np.asarray(batch["reward"], dtype=np.float32)
            batch["obs"] = np.asarray(batch["obs"], dtype=obs.dtype)
            batch["next_obs"] = np.asarray(batch["next_obs"], dtype=next_obs.dtype)
            batch["first"] = np.asarray(batch["first"], dtype=np.bool)
            batch["ep_curiosity"] = np.asarray(batch["ep_curiosity"], dtype=np.float32)
            batch["action"] = np.asarray(batch["action"])
            batch["value"] = np.asarray(batch["value"], dtype=np.float32)
            batch["logpac"] = np.asarray(batch["logpac"], dtype=np.float32)
            return batch, next_obs, next_first

        def process_batch(
            self, batch: dict, last_obs: np.ndarray, last_first: np.ndarray
        ):
            """
            Add episodic intrinsic reward
            """
            if config["ep_curiosity"] in ("visit", "count"):
                intrinsic_rewards = batch["ep_curiosity"]
                if config['intrinsic_only']:
                    batch["reward"] = intrinsic_rewards * config["intrinsic_reward_coef"]
                else:
                    batch["reward"] += intrinsic_rewards * config["intrinsic_reward_coef"]
            super().process_batch(batch=batch, last_obs=last_obs, last_first=last_first)
            return batch

    worker = ModifiedPPOWorker(**config["worker_kwargs"])

    # Create buffer
    buffer_size = worker.env.num * worker.n_steps
    buffer = Buffer(max_size=buffer_size, sequence_length=worker.n_steps)

    # Training
    n_iters = int(config["run_cfg"]["n_timesteps"] / worker.env.num / worker.n_steps)
    for i in range(n_iters):
        t_start = time.perf_counter()
        # Collect data
        worker.collect(scheduler_step=i, buffer=buffer)
        # Learn on data
        stats_dict = worker.learn(scheduler_step=i, buffer=buffer)
        # Logging
        ret = worker.env.callmethod("get_ep_stat_mean", "r")
        finish = worker.env.callmethod("get_ep_stat_mean", "finish")
        avg_explored_rooms = worker.env.callmethod("get_ep_stat_mean", "avg_rooms")
        logger.logkv("time", time.perf_counter() - t_start)
        logger.logkv("iter", i + 1)
        logger.logkv("return", ret)
        logger.logkv("success", finish)
        logger.logkv("avg_rooms", avg_explored_rooms)
        for key, value in stats_dict.items():
            logger.logkv(key, value)
        logger.dumpkvs()

        if ((i + 1) * buffer_size % int(1e6)) <= buffer_size:
            th.save(
                worker.policy.state_dict(), os.path.join(run_dir, f"policy_{i + 1}.pt")
            )

    # Save model
    th.save(worker.policy.state_dict(), os.path.join(run_dir, "policy_final.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--n_timesteps", type=int, default=int(4e7))
    parser.add_argument("--env_id", type=str, default="MiniGrid-KeyCorridorS4R3-v0")
    parser.add_argument("--n_envs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_steps", type=float, default=128)
    parser.add_argument("--n_epochs", type=float, default=4)
    parser.add_argument("--n_minibatches", type=int, default=8)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--intrinsic_only", action="store_true")
    parser.add_argument("--intrinsic_reward_coef", type=float, default=1e-3)
    parser.add_argument(
        "--ep_curiosity", type=str, choices=("visit", "count", "none"), default="none"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    config = {
        # Run
        "run_cfg": {
            "run_id": args.run_id,
            "log_dir": f"./exps/ppo_nw_{args.ep_curiosity}/",
            "n_timesteps": args.n_timesteps,
        },
        # Agent
        "worker_kwargs": {
            "env_fn": make_gym3_env,
            "env_kwargs": {
                "env_fn": make_gym_env,
                "num": args.n_envs,
                "env_kwargs": {"id": args.env_id},
                "use_subproc": False,
            },
            "policy_fn": "minirl.algos.ppo.policy.PPODiscretePolicy",
            "policy_kwargs": {
                "extractor_fn": "cnn",
                "extractor_kwargs": {
                    "input_shape": (3, 7, 7),
                    "conv_kwargs": (
                        {
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        },
                        {
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        },
                        {
                            "out_channels": 32,
                            "kernel_size": 3,
                            "stride": 2,
                            "padding": 1,
                        },
                    ),
                    "activation": nn.ELU,
                    "hiddens": (512,),
                    "final_activation": nn.ReLU,
                },
                "n_actions": 7,
            },
            "optimizer_fn": "torch.optim.Adam",
            "optimizer_kwargs": {"lr": args.lr},
            "n_steps": args.n_steps,
            "n_epochs": args.n_epochs,
            "n_minibatches": args.n_minibatches,
            "discount_gamma": args.discount_gamma,
            "gae_lambda": args.gae_lambda,
            "normalize_adv": True,
            "clip_range": args.clip_range,
            "entropy_coef": args.entropy_coef,
            "device": args.device,
        },
        # Episodic curiosity
        "ep_curiosity": args.ep_curiosity,
        "intrinsic_only": args.intrinsic_only,
        "intrinsic_reward_coef": args.intrinsic_reward_coef,
    }

    train(config)
