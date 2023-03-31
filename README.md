# LUGL

This is a repo for a simple framework that I used to train lightGBM RL agents, with a procedure I call "local updates, global learning". In short, it's normal RL; the Q-value function is a table, and the updates happen either online or via policy iteration. Whenever a state has not been visited before, it gets a Q-value of 0. This roughly translates into doing "local" updates. Every n iterations (where n=5000) in my experiments, the table is approximated via a normal supervised learning procedure via LightGBM (through normal supervised learning via X inputs y (i.e. Q) outputs ). If the agent created can beat the previous agent (starting with a random agent), the whole table is thrown away, and the LightGBM is used to provide initial Q-values (instead of 0s). This keeps happendng until you are bored with training. 

There are multiple agents in the repo, and the whole framework is based on https://github.com/deepmind/open_spiel: 

- DQN, normal DQN agent (copied from open_spiel)
- LUGLDLightGBM, LightGBM agent, does deterministic Q-learning updates
- LUGLLightGBM, LightGBM agent, normal Q-learning updates
- LUGLPILightGBM, LightGBM agent, does approximate policy iteration
- LUGLVLightGBM, LightGBM agent, uses V values instead of Q-values (i.e. doing one step lookaheads)

There are two more agents, which I have not trained properly yet

- LUGLVDL, which uses linear regression with features extracted from the leafs of a decision tree ( a well known method from kaggle winners)
- LUGLVPM, which uses a fancy NN termed polynomial (does not need an activation, it approximates polynomial regression). I've rediscovered this before realising the ideas has been floating since 2020

The agents perform better vs DQN, but this could simply be due to hyperparameter tuning. 

## Example results

- `hex.png` ![hex](/reports/figures/hex.png)
- `othello.png` ![othello](/reports/figures/othello.png)
- `tic_tac_toe.png` ![tic-tac-toe](/reports/figures/tic_tac_toe.png)
- `connect_four.pdf` ![connect-4](/reports/figures/connect_four.png)


```bibtex
@article{ssamot,
  author    = {Spyridon Samothrakis et. al. },
  title     = {Local updates, global learning (LUGL): Reinforcement Learning with non-neural non-neural methods},
  journal   = {To be submitted},
  url    = {https://github.com/ssamot/luglrl},
  year      = {2023}
}
```



# Makefile

This Makefile provides commands for training and running tournaments of different agents on different games, and generating figures based on the results of those tournaments.

## Commands

### `train_agents`

This command trains all agents listed in the "agents" file on all games listed in the "games" file. The trained agents are saved every `n` games (currently set to 5000) in the "models" directory.


### `run_tournaments`

This command runs a tournament of all trained agents found in the "models" directory on every game listed in the "games" file. The results of the tournaments are saved in the "data/interim/tournaments" directory.

### `calculate_glicko`

This command uses the saved results from the tournaments in the "data/interim/tournaments" directory to generate figures in the "report/figures" directory. The figures generated include:


