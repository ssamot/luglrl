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
from open_spiel.python import rl_environment
import importlib



def run_tournament(game, n_games, agents):
    map_frame = {"player_1": [], "player_2": [], "score": [], "position": []}

    env = rl_environment.Environment(game)

    for i in tqdm(range(n_games)):
        agent_name_1, agent_1 = random.choice(list(agents.items()))
        agent_name_2, agent_2 = agent_name_1, agent_1
        while (agent_name_1 == agent_name_2):
            agent_name_2, agent_2 = random.choice(list(agents.items()))

        win_rates_vs_random = evaluate(env, agent_1, agent_2,
                                       1)

        for pos in [0, 1]:
            for win in range(len(win_rates_vs_random[pos])):
                map_frame["player_1"].append(agent_name_1)
                map_frame["player_2"].append(agent_name_2)
                map_frame["score"].append(win_rates_vs_random[pos][win])
                map_frame["position"].append(pos)

    df = pd.DataFrame(data=map_frame)
    return df


@click.command()
@click.argument('game_name', type = click.STRING)
@click.argument('n_games', type=click.INT, default = 1000)
def main(game_name, n_games):
        game = game_name
        num_players = 2

        env = rl_environment.Environment(game)

        num_actions = env.action_spec()["num_actions"]
        state_size = env.observation_spec()["info_state"][0]

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
                if (os.path.isdir(file)):
                    with open(file + "/classname.txt") as f:
                        agent_class = f.readline()


                    module_name, class_name = agent_class.rsplit(".", 1)
                    AgentClass = getattr(importlib.import_module(module_name),
                                         class_name)

                    my_agents = [
                        AgentClass(player_id=idx,
                                   state_representation_size=state_size,
                                   num_actions=num_actions)
                        for idx in range(num_players)
                    ]

                    try:
                        for player, agent in enumerate(my_agents):
                            print(file + f"/{player}")
                            agent.restore(file + f"/{player}")
                            print("............................")
                    except ValueError:
                        continue
                    agents[stem] = my_agents
                else:
                    agents[stem] = joblib.load(file)


        if(len(agents) == 1):
            exit("No agents to play against!")

        df = run_tournament(game, n_games,agents)


        df.to_csv(f"{project_dir}/data/interim/{game_name}.csv")

if __name__ == '__main__':
    main()
