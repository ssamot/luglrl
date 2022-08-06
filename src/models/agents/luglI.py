# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tabular Q-learning agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from river.tree import HoeffdingAdaptiveTreeRegressor, HoeffdingTreeRegressor
from river.ensemble import AdaptiveRandomForestRegressor, BaggingRegressor


N_BOOSTRAPS = 20





class LUGLDecisionTreeHoeffding(rl_agent.AbstractAgent):
    """Tabular Q-Learning agent.

    See open_spiel/python/examples/tic_tac_toe_qlearner.py for an usage example.
    """

    def __init__(self,
                 player_id,
                 state_representation_size,  ## ignored
                 num_actions,
                 step_size=0.1,
                 epsilon_schedule=rl_tools.ConstantSchedule(0.1),
                 discount_factor=1.0,
                 centralized=False,

                 ):
        """Initialize the Q-Learning agent."""
        self._player_id = player_id
        self._num_actions = num_actions
        self._step_size = step_size
        self._epsilon_schedule = epsilon_schedule
        self._epsilon = epsilon_schedule.value
        self._discount_factor = discount_factor
        self._centralized = centralized

        self._prev_info_state = None
        self._last_loss_value = None
        self._n_games = 0
        self.episode_length = 0
        self.state_representation_size = state_representation_size
        self.model = None
        self.new_model()


    def new_model(self):
        # self.nm = HoeffdingTreeRegressor(leaf_prediction="model",
        #                                      min_samples_split=200,
        #                                       nominal_attributes=[
        #                                           self.state_representation_size])

        #self.nm = AdaptiveRandomForestRegressor()
        self.nm = BaggingRegressor(HoeffdingTreeRegressor(

            min_samples_split=200,
                                                   nominal_attributes=[
                                                       self.state_representation_size]

        ))


    def get_state_action(self, info_state, action):
        return (tuple(info_state), tuple([action]))

    def infoactionToMap(self, infoaction):
        action = infoaction[1][0]
        state = infoaction[0]
        map = {i:state[i] for i in range(len(state))}
        map[len(map) + 1] = action
        return map


    def get_Q(self, infoaction):
        if(self.model is None):
            return 0.0
        x = self.infoactionToMap(infoaction)
        return self.model.predict_one(x)

    def train_int(self, infoaction, new_value):
        #print(infoaction, new_value)
        #print("==============")
        x = self.infoactionToMap(infoaction)
        #print(x)

        self.nm.learn_one(x, new_value)




    def _epsilon_greedy(self, info_state, legal_actions, epsilon):
        """Returns a valid epsilon-greedy action and valid action probs.

        If the agent has not been to `info_state`, a valid random action is chosen.

        Args:
          info_state: hashable representation of the information state.
          legal_actions: list of actions at `info_state`.
          epsilon: float, prob of taking an exploratory action.

        Returns:
          A valid epsilon-greedy action and valid action probabilities.
        """

        # q_values[info_state][a]  transformed to q_values[tuple(info_state) + tuple(a)]
        probs = np.zeros(self._num_actions)

        q_values = [self.get_Q(self.get_state_action(info_state,
                                                         a)) + np.random.random() * 0.0001
                    for a in legal_actions]
        greedy_q = np.argmax(q_values)
        # print(q_values)

        probs[legal_actions] = epsilon / len(legal_actions)
        probs[legal_actions[greedy_q]] += (1 - epsilon)
        # print(probs, np.sum(probs))
        action = np.random.choice(range(self._num_actions), p=probs)
        return action, probs

    def step(self, time_step, is_evaluation=False):
        """Returns the action to be taken and updates the Q-values if needed.

        Args:
          time_step: an instance of rl_environment.TimeStep.
          is_evaluation: bool, whether this is a training or evaluation call.

        Returns:
          A `rl_agent.StepOutput` containing the action probs and chosen action.
        """
        if self._centralized:
            info_state = tuple(time_step.observations["info_state"])
        else:
            info_state = tuple(
                time_step.observations["info_state"][self._player_id])
        legal_actions = time_step.observations["legal_actions"][self._player_id]

        # Prevent undefined errors if this agent never plays until terminal step
        action, probs = None, None

        # Act step: don't act at terminal states.
        if not time_step.last():
            epsilon = 0.0 if is_evaluation else self._epsilon
            action, probs = self._epsilon_greedy(
                info_state, legal_actions, epsilon=epsilon)
        # print(time_step.rewards, time_step.last())
        # Learn step: don't learn during evaluation or at first agent steps.
        if self._prev_info_state and not is_evaluation:
            # print("training")
            target = time_step.rewards[self._player_id]
            state_actions = [self.get_state_action(info_state, a) for a in
                             legal_actions]
            if not time_step.last():  # Q values are zero for terminal.
                target += self._discount_factor * max(
                    [self.get_Q(state_action)for state_action in
                     state_actions])
            # print(target)
            prev = self.get_state_action(self._prev_info_state,
                                         self._prev_action)
            #prev_q_value = self.get_Q(prev)
            #self._last_loss_value = target - prev_q_value
            # print(target, prev_q_value)

            #newQ = prev_q_value + (self._step_size * self._last_loss_value)
            self.train_int(prev, target)


            self._epsilon = self._epsilon_schedule.step()
            self.episode_length += 1
            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                self._n_games += 1
                self.episode_length = 0
                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action # print(time_step.rewards, "rewards")
                # exit()
        return rl_agent.StepOutput(action=action, probs=probs)

    @property
    def loss(self):
        return self._last_loss_value


    def get_phi(self, key):
        state_features, action = list(key[0]), list(key[1])
        total_features = state_features + action
        phi = np.array(total_features)
        return phi

    def _default_value(self, key):
        if (self.model is None):
            return 0.0
        else:
            state_features, action = list(key[0]), list(key[1])
            total_features = state_features + action
            phi = np.array(total_features)[np.newaxis, :]
            value = self.model.predict(phi)

            return value[0]

    def train_supervised(self):
        self.model = self.nm
        #print(self.model.summary)

    def _reset_dict(self):
        self.new_model()

