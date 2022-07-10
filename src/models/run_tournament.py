import joblib

import pandas as pd
from pathlib import Path
import glob
import os
import random
from tqdm import tqdm
from open_spiel.python.algorithms import random_agent
import click




def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    #wins = np.zeros(2)
    wins = [[],[]]
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [trained_agents[0], random_agents[1]]
        else:
            cur_agents = [random_agents[0], trained_agents[1]]
        for _ in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = cur_agents[player_id].step(time_step,
                                                          is_evaluation=True)
                time_step = env.step([agent_output.action])
           # print(time_step.rewards[player_pos])
            #if time_step.rewards[player_pos] > 0:
            wins[player_pos].append(time_step.rewards[player_pos])
            #else:
            #    wins[player_pos].append(0)
    return wins


@click.command()
@click.argument('game_name', type = click.STRING)
@click.argument('save_path', type=click.Path())
@click.argument('n_games', type=click.INT, default = 1000)
def main(game_name, save_path, n_games):
        game = game_name
        num_players = 2
        from open_spiel.python import rl_environment

        env = rl_environment.Environment(game)
        num_actions = env.action_spec()["num_actions"]

        num_actions = env.action_spec()["num_actions"]

        # random agents for evaluation
        random_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
        ]

        project_dir = Path(__file__).resolve().parents[2]



        agents = {"Random":random_agents}
        for file in glob.glob(f"{project_dir}/models/agents/{save_path}/*.agent"):
            name = Path(file).name
            stem = name.split(".")[0]
            print(stem)
            agents[stem] = joblib.load(file)



        map_frame = {"player_1":[], "player_2":[], "score":[], "position":[]}

        for i in tqdm(range(n_games)):
            agent_name_1, agent_1 = random.choice(list(agents.items()))
            agent_name_2, agent_2 = agent_name_1, agent_1
            while(agent_name_1==agent_name_2):
                agent_name_2, agent_2 = random.choice(list(agents.items()))

            win_rates_vs_random = eval_against_random_bots(env, agent_1, agent_2,
                                                         10)
            for pos in [0, 1]:
                for win in range(len(win_rates_vs_random[pos])):
                    map_frame["player_1"].append(agent_name_1)
                    map_frame["player_2"].append(agent_name_2)
                    map_frame["score"].append(win_rates_vs_random[pos][win])
                    map_frame["position"].append(pos)


        df = pd.DataFrame(data=map_frame)
        df.to_csv(f"{project_dir}/data/interim/{save_path}.csv")

if __name__ == '__main__':
    main()
