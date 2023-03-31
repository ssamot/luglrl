# LUGL

This is the repo for a feature 




### `run_tournaments`

This command runs a tournament of all trained agents found in the "models" directory on every game listed in the "games" file. The results of the tournaments are saved in the "data/interim/tournaments" directory.

# Makefile

This Makefile provides commands for training and running tournaments of different agents on different games, and generating figures based on the results of those tournaments.

## Commands

### `train_agents`

This command trains all agents listed in the "agents" file on all games listed in the "games" file. The trained agents are saved every `n` games (currently set to 5000) in the "models" directory.




### `calculate_glicko`

This command uses the saved results from the tournaments in the "data/interim/tournaments" directory to generate figures in the "report/figures" directory. The figures generated include:

### Example results

- `hex.png` ([link](./report/figures/hex.png))
- `othello.png` ([link](./report/figures/othello.png))
- `tic_tac_toe.png` ([link](./report/figures/tic_tac_toe.png))
- `connect_four.pdf` ([link](./report/figures/connect_four.pdf))




@article{ssamot,
  author    = {Spyridon Samothrakis et. al. },
  title     = {Local updates, global learning (LUGL): training
non-neural methods incrementally},
  journal   = {To be submitted},
  volume    = {notyet},
  year      = {2023}
}
