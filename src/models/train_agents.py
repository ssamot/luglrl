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

    module_name, class_name = agent_class.rsplit(".", 1)
    AgentClass = getattr(importlib.import_module(module_name), class_name)

    agents = [
        AgentClass(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]

    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
        for idx in range(num_players)
    ]


    previous_agents = None
    agent_before_previous_agent = None

    with trange(training_episodes) as t:
        for cur_episode in t:
            if cur_episode % int(1e4) == 0:
                filename = f"{project_dir}/models/games/{game_name}/{class_name}:{cur_episode}.agent"
                joblib.dump(agents, filename, compress = 3)

            if cur_episode % (int(comparison_point) + 1) == 0:
                print("Evaluating current and previous")
                if(agent_before_previous_agent is None):
                    agent_before_previous_agent = deepcopy(agents)
                    print("before")
                elif(agent_before_previous_agent is not None and previous_agents is None):
                    previous_agents = deepcopy(agents)
                    print("previous")
                else:
                    #play games
                    def get_diff():
                        scores_past = evaluate(env, deepcopy(previous_agents),
                                                   deepcopy(agent_before_previous_agent), 1000,
                                                   0.1)

                        scores_past = np.array(scores_past)

                        scores_before_previous = evaluate(env, deepcopy(agents),
                                          deepcopy(agent_before_previous_agent), 1000, 0.1)

                        scores_before_previous = np.array(scores_before_previous)

                        scores_before_previous_means = scores_before_previous.mean(axis=1)

                        scores_past_mean = scores_past.mean(axis=1)

                        diff = scores_before_previous_means - scores_past_mean

                        return diff

                    diff = get_diff()

                    if diff.min() > 0.0:
                        print(bcolors.OKGREEN + f"We have improvement --  {diff}" + bcolors.ENDC)

                        try:
                            models = [agents[0].model, agents[1].model]
                            for agent in agents:
                                agent.train_supervised()
                            diff = get_diff()
                            if(np.min(diff) > 0.01):
                                print(bcolors.OKGREEN + f"reseting with --  {diff}" + bcolors.ENDC)
                                for agent in agents:
                                    agent._reset_dict()
                                agent_before_previous_agent = previous_agents
                                previous_agents = deepcopy(agents)
                            else:
                                print(
                                    bcolors.FAIL + f"Horrible with --  {diff}" + bcolors.ENDC)
                                for m, agent in enumerate(agents):
                                    agent.model = models[m]

                        except AttributeError:
                            print("Agents do not support supervised training")
                    else:
                        print(bcolors.FAIL + f"No improvement {diff}" + bcolors.ENDC)


                    #print(scores_past_mean, scores_before_previous_means)
                    # exit()
                    #
                    # #p1_scores = ttest_1samp(scores[0], 0)
                    # #p2_scores = ttest_1samp(scores[0], 0)
                    # #print(p1_scores, p2_scores)
                    # if(scores.mean() > 0.0):
                    #     print("Newer is better")
                    #     previous_agents = deepcopy(agents)
                    #     agents[0].test = "flag"
                    #exit()

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