# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import click
import numpy as np
import joblib
from open_spiel.python import rl_environment

from tqdm import trange
from pathlib import Path
import importlib
from utils import evaluate
from copy import deepcopy
from utils import evaluate, bcolors
from open_spiel.python.algorithms import random_agent
from scipy.stats import ttest_1samp
from visualization.calculate_glicko2_scores import calculate_glicko_scores
from run_tournament import run_tournament

@click.command()
@click.argument('game_name', type = click.STRING)
@click.argument('agent_class', type = click.STRING)
@click.argument('comparison_point', type = click.INT, default = 5000)
@click.argument('training_episodes', type=click.INT, default = 100001)
def main(game_name, agent_class, comparison_point,  training_episodes):
    project_dir = Path(__file__).resolve().parents[2]
    game = game_name
    num_players = 2

    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    state_size = env.observation_spec()["info_state"][0]

    module_name, class_name = agent_class.rsplit(".", 1)
    AgentClass = getattr(importlib.import_module(module_name), class_name)

    agents = [
        AgentClass(player_id=idx, state_representation_size = state_size, num_actions=num_actions)
        for idx in range(num_players)
    ]

    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]


    archive = {"Random:-10000":random_agents}

    with trange(training_episodes) as t:
        for cur_episode in t:
            if cur_episode % int(1e4) == 0 and cur_episode!=0:
                filename = f"{project_dir}/models/games/{game_name}/{class_name}:{cur_episode}.agent"
                try:
                    joblib.dump(agents, filename, compress = 3)
                except (TypeError, IsADirectoryError):
                    for agent in agents:
                        agent.save(filename)



            if (cur_episode + 1) % (int(comparison_point) ) == 0  :
                current_agent = f"{class_name}:{cur_episode}"

                print(bcolors.OKBLUE + f"Training" + bcolors.ENDC)

                try:
                    models = [agents[0].model, agents[1].model]
                    for agent in agents:
                        agent.train_supervised()

                    archive[current_agent] = deepcopy(agents)

                    df = run_tournament(game_name, len(archive)*3, archive)
                    glicko_df = calculate_glicko_scores(df)
                    glicko_df = glicko_df.sort_values(by=['glicko2'])
                    is_latest_best = (glicko_df.iloc[-1][
                                          "n_games"] == cur_episode)


                    if(is_latest_best):
                        print(bcolors.OKGREEN + f"Latest agent is elite" + bcolors.ENDC)
                        for agent in agents:
                            agent._reset_dict()
                    else:
                        print(
                            bcolors.FAIL + f"Still note there" + bcolors.ENDC)
                        for m, agent in enumerate(agents):
                            agent.model = models[m]
                        del archive[current_agent]

                except AttributeError:
                    import traceback
                    print(traceback.format_exc())
                    print("Agents do not support supervised training")

            t.set_description(f"Game {cur_episode}")


            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                time_step = env.step([agent_output.action])

            # Episode is over, step all games with final info state.
            for agent in agents:
                agent.step(time_step)



if __name__ == "__main__":
    main()