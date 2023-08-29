# Code for submission "Revisiting Lifelong and Episodic Curiosity for Exploration in Procedurally Generated Environments"

To run the code, first create a anaconda environment by
```bash
conda env create -f environment.yml
conda activate curiosity
```

Then install the package that implements RL algorithms
```bash
cd minirl
pip install -e .
```

For training the agent with episodic intrinsic curiosity only, please use
```bash
python train_ppo.py --env_id MiniGrid-KeyCorridorS4R3-v0 --n_timesteps 40000000 --ep_curiosity visit --intrinsic_reward_coef 0.005
```

To use different lifelong curiosities, please replace `train_ppo.py` with other files such as `train_icm.py`.

To specify different episodic curiosities, please choose value from `[visit, count, none]` for the argument `--ep_curiosity`.

For training with permuted lifelong curiosity, please use python files whose name ends with `perm.py`.

For training without extrinsic reward, please use python files whose name ends with `nw.py`.
