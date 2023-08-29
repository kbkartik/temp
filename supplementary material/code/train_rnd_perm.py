import os
import json
import time
from collections import defaultdict
from typing import Tuple, Union

import gym
import gym3
import numpy as np
import torch as th
from torch import nn
from torch.distributed.rpc import RRef
from gym_minigrid.wrappers import ImgObsWrapper

from minirl.algos.ppo.agent import PPOWorker
from minirl.algos.ppo.policy import PPODiscretePolicy
from minirl.buffer import Buffer
from minirl.envs.gym3_wrapper import ObsTransposeWrapper
from minirl.utils import explained_variance

import logger
import minigrid_env
from network import StateEmbeddingNet
from wrapper import ModifiedEpisodeStatsWrapper


def make_gym_env(**env_kwargs):
    env = gym.make(**env_kwargs)
    env = ImgObsWrapper(env)
    return env


def make_gym3_env(**kwargs):
    env = gym3.vectorize_gym(**kwargs)
    env = ObsTransposeWrapper(env, axes=(2, 0, 1))
    env = ModifiedEpisodeStatsWrapper(env)
    return env


class RNDPPODiscretePolicy(PPODiscretePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_net = StateEmbeddingNet(
            input_shape=kwargs["extractor_kwargs"]["input_shape"]
        )
        self.predictor_net = StateEmbeddingNet(
            input_shape=kwargs["extractor_kwargs"]["input_shape"]
        )

    def loss(self, *args, next_obs, **kwargs):
        pg_loss, vf_loss, entropy, extra_out = super().loss(*args, **kwargs)

        next_obs = th.as_tensor(next_obs).to(self.device).float()
        firsts = th.as_tensor(kwargs["firsts"]).to(self.device)

        with th.no_grad():
            next_state_emb, _ = self.target_net.extract_features(next_obs, firsts)
        pred_next_state_emb, _ = self.predictor_net.extract_features(next_obs, firsts)

        rnd_loss = th.norm(pred_next_state_emb - next_state_emb, dim=2, p=2).mean()
        return pg_loss, vf_loss, entropy, rnd_loss, extra_out


def train(config):
    # Setup logger
    env_name = config["worker_kwargs"]["env_kwargs"]["env_kwargs"]["id"]
    task_name = "-".join(env_name.split("-")[1:-1])
    run_dir = os.path.join(
        config["run_cfg"]["log_dir"],
        task_name,
        f"run_{config['run_cfg']['run_id']}",
    )
    logger.configure(dir=run_dir, format_strs=["csv", "stdout"])
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)

    # Modify worker to add RND intrinsic rewards
    class RNDPPOWorker(PPOWorker):
        def __init__(self, *args, rnd_loss_coef, **kwargs):
            super().__init__(*args, **kwargs)
            self.ep_cnts = [dict() for _ in range(self.env.num)]
            self.rnd_loss_coef = rnd_loss_coef

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
                # Calculate RND intrinsic reward
                with th.no_grad():
                    next_state_emb, _ = self.policy.target_net.extract_features(
                        th.as_tensor(obs[None, ...]).float().to(self.device),
                        th.as_tensor(first[None, ...]).float().to(self.device),
                    )
                    pred_next_state_emb, _ = self.policy.predictor_net.extract_features(
                        th.as_tensor(next_obs[None, ...]).float().to(self.device),
                        th.as_tensor(next_first[None, ...]).float().to(self.device),
                    )
                rnd_rew = th.norm(next_state_emb - pred_next_state_emb, dim=2, p=2)
                rnd_rew = rnd_rew.cpu().numpy().squeeze(0)
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
                batch["rnd_rew"].append(rnd_rew)
                batch["ep_curiosity"].append(ep_curiosity)
                batch["reward"].append(reward)
                batch["next_obs"].append(next_obs)
            # Concatenate
            batch["reward"] = np.asarray(batch["reward"], dtype=np.float32)
            batch["obs"] = np.asarray(batch["obs"], dtype=obs.dtype)
            batch["next_obs"] = np.asarray(batch["next_obs"], dtype=next_obs.dtype)
            batch["first"] = np.asarray(batch["first"], dtype=np.bool)
            batch["ep_curiosity"] = np.asarray(batch["ep_curiosity"], dtype=np.float32)
            batch["rnd_rew"] = np.asarray(batch["rnd_rew"], dtype=np.float32)
            batch["action"] = np.asarray(batch["action"])
            batch["value"] = np.asarray(batch["value"], dtype=np.float32)
            batch["logpac"] = np.asarray(batch["logpac"], dtype=np.float32)
            return batch, next_obs, next_first

        def process_batch(
            self, batch: dict, last_obs: np.ndarray, last_first: np.ndarray
        ):
            """
            Add RND intrinsic reward
            """
            intrinsic_rewards = batch["rnd_rew"]
            to_shuffle = intrinsic_rewards[intrinsic_rewards != 0]
            np.random.shuffle(to_shuffle)
            intrinsic_rewards[intrinsic_rewards != 0] = to_shuffle
            if config["ep_curiosity"] in ("visit", "count"):
                intrinsic_rewards *= batch["ep_curiosity"]
            if config["intrinsic_only"]:
                batch["reward"] = intrinsic_rewards * config["intrinsic_reward_coef"]
            else:
                batch["reward"] += intrinsic_rewards * config["intrinsic_reward_coef"]
            super().process_batch(batch=batch, last_obs=last_obs, last_first=last_first)
            return batch

        def learn(self, scheduler_step: int, buffer: Union[Buffer, RRef]):
            # Retrieve data from buffer
            if isinstance(buffer, RRef):
                batch = buffer.rpc_sync().get_all()
            else:
                batch = buffer.get_all()
            # Build a dict to save training statistics
            stats_dict = defaultdict(list)
            # Minibatch training
            B, T = batch["obs"].shape[:2]
            if self.policy.is_recurrent:
                batch_size = B
                indices = np.arange(B)
            else:
                batch_size = B * T
                indices = np.mgrid[0:B, 0:T].reshape(2, batch_size).T
            minibatch_size = batch_size // self.n_minibatches
            assert minibatch_size > 1
            # Get current clip range
            cur_clip_range = self.clip_range.value(step=scheduler_step)
            cur_vf_clip_range = self.vf_clip_range.value(step=scheduler_step)
            # Train for n_epochs
            for _ in range(self.n_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    if self.policy.is_recurrent:
                        sub_indices = indices[start:end]
                        rnn_states = batch["rnn_states"][sub_indices].swapaxes(0, 1)
                    else:
                        sub_indices = indices[start:end]
                        sub_indices = tuple(sub_indices.T) + (None,)
                        rnn_states = None
                    self.optimizer.zero_grad()
                    (
                        pg_loss,
                        vf_loss,
                        entropy,
                        rnd_loss,
                        extra_out,
                    ) = self.policy.loss(
                        obs=batch["obs"][sub_indices].swapaxes(0, 1),
                        next_obs=batch["next_obs"][sub_indices].swapaxes(0, 1),
                        advs=batch["adv"][sub_indices].swapaxes(0, 1),
                        firsts=batch["first"][sub_indices].swapaxes(0, 1),
                        actions=batch["action"][sub_indices].swapaxes(0, 1),
                        old_values=batch["value"][sub_indices].swapaxes(0, 1),
                        old_logpacs=batch["logpac"][sub_indices].swapaxes(0, 1),
                        rnn_states=rnn_states,
                        clip_range=cur_clip_range,
                        vf_clip_range=cur_vf_clip_range,
                        normalize_adv=self.normalize_adv,
                    )
                    total_loss = (
                        pg_loss
                        + self.vf_loss_coef * vf_loss
                        - self.entropy_coef * entropy
                        + self.rnd_loss_coef * rnd_loss
                    )
                    total_loss.backward()
                    self.pre_optim_step_hook()
                    self.optimizer.step()
                    # Saving statistics
                    stats_dict["policy_loss"].append(pg_loss.item())
                    stats_dict["value_loss"].append(vf_loss.item())
                    stats_dict["rnd_loss"].append(rnd_loss.item())
                    stats_dict["entropy"].append(entropy.item())
                    stats_dict["total_loss"].append(total_loss.item())
                    for key in extra_out:
                        stats_dict[key].append(extra_out[key].item())
            # Compute mean
            for key in stats_dict:
                stats_dict[key] = np.mean(stats_dict[key])
            # Compute explained variance
            stats_dict["explained_variance"] = explained_variance(
                y_pred=batch["value"], y_true=batch["value"] + batch["adv"]
            )
            return stats_dict

    # Create worker
    worker = RNDPPOWorker(**config["worker_kwargs"])

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
        logger.logkv("time", time.perf_counter() - t_start)
        logger.logkv("iter", i + 1)
        logger.logkv("return", ret)
        logger.logkv("success", finish)
        for key, value in stats_dict.items():
            logger.logkv(key, value)
        logger.dumpkvs()

    # Save model
    th.save(worker.policy.state_dict(), os.path.join(run_dir, "policy.pt"))


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
            "log_dir": f"./exps/rnd_perm_{args.ep_curiosity}/",
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
            "policy_fn": "__main__.RNDPPODiscretePolicy",
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
            "rnd_loss_coef": 0.1,
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
        # RND
        "ep_curiosity": args.ep_curiosity,
        "intrinsic_only": args.intrinsic_only,
        "intrinsic_reward_coef": args.intrinsic_reward_coef,
    }

    train(config)
