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
from sklearn import linear_model
from sklearn import metrics
from lightgbm.sklearn import LGBMRegressor





def child_U(all_childs):
    C = 1
    parent_visits = np.sum(all_childs)
    all_U = [C* np.sqrt(np.log(parent_visits)/child_visits)
             for child_visits in all_childs]
    return all_U


def Q_MC(r, old_mean, visits):
    new_mean = old_mean + ((r-old_mean)/visits)
    return new_mean

class keydefaultdict(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def one():
    return 1



class LUGLLightGBM(rl_agent.AbstractAgent):
    """Tabular Q-Learning agent.

    See open_spiel/python/examples/tic_tac_toe_qlearner.py for an usage example.
    """

    def __init__(self,
                 player_id,
                 state_representation_size,  ## ignored
                 num_actions,
                 discount_factor=1.0,
                 centralized=False,

                 ):
        """Initialize the Q-Learning agent."""
        self._player_id = player_id
        self._num_actions = num_actions
        self._discount_factor = discount_factor
        self._centralized = centralized

        self._prev_info_state = None
        self._last_loss_value = None
        self._reset_dict()
        self._n_games = 0
        self.episode_length = 0
        self.model = None

        self.episode_length_mean = 0

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_q_values']
        del state['states_visited']
        del state['_visits']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._reset_dict()


    def _reset_dict(self):
        self._q_values = keydefaultdict(self._default_value)
        self._visits = collections.defaultdict(one)
        self.states_visited = []

    def _default_value(self, key):
        if(self.model is None):
            return np.random.random() * 0.001
        else:
            state_features, action = list(key[0]), list(key[1])[0]
            action_features = list(np.zeros(shape=self._num_actions))
            if (action is not None):
                action_features[action] = 1
            total_features = state_features + action_features
            phi = np.array(total_features)[np.newaxis,:]
            value = self.model.predict(phi)
            #print(value)

            #print(value)
            return value[0]

    def get_state(self, info_state, action):
        return tuple(info_state), tuple([action])

    def _UCB(self, info_state, legal_actions, is_evaluation):

        probs = np.zeros(self._num_actions)
        child_visits = np.array([self._visits[self.get_state(info_state, action)] for
                  action in legal_actions])
        #never_visited = (child_visits == 0)
        # if(np.sum(never_visited) > 0):
        #     # [f(x) if condition else g(x) for x in sequence]
        #     all_Qs = [np.random.random() if nv else 0 for nv in never_visited]
        #
        # else:
        all_Qs = [self._q_values[self.get_state(info_state, action)] for action
                      in legal_actions]
        all_Qs = np.array(all_Qs)
        # break ties!
        all_Qs+= (np.random.random(size = all_Qs.shape) * 0.00000001)

        if(is_evaluation):
            best_legal_action = np.argmax(np.array(all_Qs))
        else:
            all_Us = child_U(child_visits)
            best_legal_action = np.argmax(np.array(all_Qs) + np.array(all_Us))

        action = legal_actions[best_legal_action]
        probs[action] = 1


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
            action, probs = self._UCB(
                info_state, legal_actions, is_evaluation)
            #print(action)
            sa = self.get_state(info_state, action)

            self.states_visited.append(sa)
            self._visits[sa] +=1
            self.episode_length += 1
        else:
            if not is_evaluation:
                target = time_step.rewards[self._player_id]
                self._n_games += 1
                #print(self.episode_length_mean)
                self.episode_length_mean = Q_MC(self.episode_length, self.episode_length_mean, 1000)
                self.episode_length = 0

                ## backpropagate
                for state in self.states_visited:
                    p = self._visits[state]
                    q = self._q_values[state]
                    self._q_values[state] = Q_MC(target,q,p)
                self.states_visited = []
                return

        return rl_agent.StepOutput(action=action, probs=probs)

    @property
    def loss(self):
        return self._last_loss_value





class LUGLUCBDecisionTree(LUGLLightGBM):

    def _default_value(self, key):
        if (self.model is None):
            return 0.0
        else:
            state_features, action = list(key[0]), key[1][0]
            #print(action, "action")
            phi = np.array(state_features)[np.newaxis, :]
            if(self.model[action] is None):
                return 0.0
            else:
                value = self.model[action].predict(phi)
                return value[0]

    def train_supervised(self):

        print("About to start training")

        print(len(self._q_values))

        #q_values = self.replay()
        q_values = self._q_values

        data_per_action = [[] for _ in range(self._num_actions)]
        Qs_per_action = [[] for _ in range(self._num_actions)]

        for key, q_value in q_values.items():
            state_features, action = list(
                key[0]), key[1][0]

            data_per_action[action].append(state_features)
            Qs_per_action[action].append(q_value)



        self.model = []
        total = 0
        for action in range(len(data_per_action)):
            X = np.array(data_per_action[action])
            y = np.array(Qs_per_action[action])
            total +=X.shape[0]
            print(X.shape, y.shape)
            if(X.shape[0] > 10):
                from sklearn.model_selection import train_test_split
                # X_train, X_test, y_train, y_test = train_test_split(X, y,
                #                                                     random_state=0)

                from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
                from sklearn.metrics import mean_squared_error
                #model = linear_model.LinearRegression()
                from sklearn.model_selection import GridSearchCV


                clf = DecisionTreeRegressor(max_depth=3)
                clf.fit(X, y)
                model = clf


                mse = metrics.mean_squared_error(y, model.predict(X))
                r2 = metrics.explained_variance_score(y, model.predict(X))
                print(mse, r2)
                self.model.append(model)
            else:
                self.model.append(None)


        print("Total", total)





class LUGLUCBLightGBM(LUGLLightGBM):

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

        print("About to start training")
        all_features = []
        all_Qs = []

        # q_values = self.replay()
        q_values = self._q_values

        for key, q_value in q_values.items():
            state_features, action = list(
                key[0]), key[1]

            total_features = state_features + list(action)
            all_Qs.append(q_value)
            all_features.append(total_features)


        X = np.array(all_features)
        y = np.array(all_Qs)

        print(X.shape, y.shape)
        from lightgbm.sklearn import LGBMRegressor
        clf = LGBMRegressor(n_jobs=6, n_estimators=1000)
        clf.fit(X, y, categorical_feature=range(X.shape[1]))
        self.model = clf
        mse = metrics.mean_squared_error(y, self.model.predict(X))
        r2 = metrics.explained_variance_score(y, self.model.predict(X))

        print(mse, r2)
