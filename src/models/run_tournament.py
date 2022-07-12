import joblib

import pandas as pd
from pathlib import Path
import glob
import os
import random
from tqdm import tqdm
from open_spiel.python.algorithms import random_agent
import click
from tqdm import trange
from utils import evaluate







@click.command()
@click.argument('game_name', type = click.STRING)
@click.argument('n_games', type=click.INT, default = 1000)
def main(game_name, n_games):
        game = game_name
        num_players = 2
        from open_spiel.python import rl_environment

        env = rl_environment.Environment(game)


        num_actions = env.action_spec()["num_actions"]

        # random games for evaluation
        random_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
        ]

        project_dir = Path(__file__).resolve().parents[2]


        print("Loading agents...")
        agents = {"Random:-10000":random_agents}
        files = list(glob.glob(f"{project_dir}/models/games/{game_name}/*.agent"))
        with trange(len(files)) as t:
            for i in t:
                file = files[i]
                name = Path(file).name
                stem = name.split(".")[0]
                t.set_description(f"Loading {stem}")
                agents[stem] = joblib.load(file)


        if(len(agents) == 1):
            exit("No agents to play against!")



        map_frame = {"player_1":[], "player_2":[], "score":[], "position":[]}

        for i in tqdm(range(n_games)):
            agent_name_1, agent_1 = random.choice(list(agents.items()))
            agent_name_2, agent_2 = agent_name_1, agent_1
            while(agent_name_1==agent_name_2):
                agent_name_2, agent_2 = random.choice(list(agents.items()))

            win_rates_vs_random = evaluate(env, agent_1, agent_2,
                                           10)

            for pos in [0, 1]:
                for win in range(len(win_rates_vs_random[pos])):
                    map_frame["player_1"].append(agent_name_1)
                    map_frame["player_2"].append(agent_name_2)
                    map_frame["score"].append(win_rates_vs_random[pos][win])
                    map_frame["position"].append(pos)


        df = pd.DataFrame(data=map_frame)
        df.to_csv(f"{project_dir}/data/interim/{game_name}.csv")

if __name__ == '__main__':
    main()
