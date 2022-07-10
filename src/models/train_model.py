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


@click.command()
@click.argument('game_name', type = click.STRING)
@click.argument('agent_class', type = click.STRING)
@click.argument('save_prefix', type = click.STRING)
@click.argument('save_path', type=click.Path())
@click.argument('training_episodes', type=click.INT, default = 100001)
def main(game_name, agent_class, save_prefix, save_path, training_episodes):
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



    with trange(training_episodes) as t:
        for cur_episode in t:
            #print(agents)
            if cur_episode % int(1e4) == 0:
                filename = f"{project_dir}/models/agents/{save_path}/{save_prefix}_{cur_episode}.agent"
                joblib.dump(agents, filename)


            t.set_description(
                'Epoch %i' % (cur_episode))


            time_step = env.reset()
            while not time_step.last():
                state = time_step.observations["info_state"]
                state = np.array(state)
                #print(state.shape)
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                time_step = env.step([agent_output.action])

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)



if __name__ == "__main__":
    main()