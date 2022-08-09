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
from lightgbm import LGBMRegressor
from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoLarsCV, LassoLars
from sklearn.metrics import mean_squared_error
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import SGDRegressor
import category_encoders

class keydefaultdict(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class LUGL_lightGBM(rl_agent.AbstractAgent):
    """Tabular Q-Learning agent.

    See open_spiel/python/examples/tic_tac_toe_qlearner.py for an usage example.
    """

    def __init__(self,
                 player_id,
                 num_actions,
                 step_size=0.1,
                 epsilon_schedule=rl_tools.ConstantSchedule(0.2),
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
        self._reset_dict()
        self._n_games = 0
        self.episode_length = 0
        self.model = None
        self.updated = set()

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_q_values']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._reset_dict()

    def _reset_dict(self):
        self._q_values = keydefaultdict(self._default_value)
        self.updated = set()

    def get_state_action(self, info_state, action):
        return tuple(info_state), tuple([action])

    def _default_value(self, key):
        if(self.model is None):
            return 0.0 #np.random.random()*0.0001
        else:
            state_features, action = list(key[0]), list(key[1])

            total_features = state_features + action
            phi = np.array(total_features)[np.newaxis,:]
            value = self.model.predict(phi)

            return value[0]

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


        greedy_q = max([self._q_values[tuple(info_state),tuple([a])]  for a in legal_actions])
        #print(greedy_q)
        greedy_actions = [
            a for a in legal_actions if
            self._q_values[tuple(info_state),tuple([a])] == greedy_q
        ]
        #print(greedy_actions)
        probs[legal_actions] = epsilon / len(legal_actions)
        probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)

        action = np.random.choice(range(self._num_actions), p=probs)
        #print(len(self._q_values))
        return action, probs

    def process_data_andregressor(self, X, y):
        clf = LGBMRegressor()
        return X,y, clf

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
            if not time_step.last():  # Q values are zero for terminal.
                target += self._discount_factor * max(
                    [self._q_values[self.get_state_action(info_state, a)] for a
                     in legal_actions])
            # print(target)
            prev = self.get_state_action(self._prev_info_state,
                                         self._prev_action)
            prev_q_value = self._q_values[prev]
            self._last_loss_value = target - prev_q_value
            # print(target, prev_q_value)
            self._q_values[prev] += (
                    self._step_size * self._last_loss_value)

            self.updated.add(prev)

            self._epsilon = self._epsilon_schedule.step()

            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                return



        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
        return rl_agent.StepOutput(action=action, probs=probs)

    @property
    def loss(self):
        return self._last_loss_value

    def train_supervised(self):

        print("About to start training")
        all_features = []
        all_Qs = []
        for key, q_value in self._q_values.items():
            if key in self.updated:
                state_features, action = list(
                    key[0]), key[1]

               # action_features = list(
               #     np.zeros(shape=self._num_actions))
                #action_features[action[0]] = 1
                total_features = state_features + list(action)
                all_Qs.append(q_value)
                all_features.append(total_features)

            # print(len(state_features), len(action_features), len(total_features))
        X = np.array(all_features)
        y = np.array(all_Qs)
        print(X.shape, y.shape)
        clf = LGBMRegressor()

        clf.fit(X,y, categorical_feature = range(X.shape[1]))
        self.model = clf
        mse = metrics.mean_squared_error(y, self.model.predict(X))
        r2 = metrics.explained_variance_score(y, self.model.predict(X))

        print(mse, r2)

