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
from sklearn import metrics
from category_encoders import TargetEncoder
from sklearn import linear_model
from copy import deepcopy




def replay(buffer,discount_factor, q_values, step_size):
    mean_loss = []
    for elements in buffer:
        all_targets = []
        for prev, state_actions, r, t in elements:
            target = r
            if not t:  # Q values are zero for terminal.
                target += discount_factor * np.max(
                    [q_values[state_action] for state_action
                     in state_actions])
            all_targets.append(target)
        target = np.mean(all_targets)
        print(len(all_targets))
        prev_q_value = q_values[prev]
        loss = target - prev_q_value
        # print(loss)
        q_values[prev] += (
                step_size * loss)
        mean_loss.append(loss)
    print("Replay loss", np.mean(mean_loss))


def init_replay(buffer,discount_factor, q_values, step_size):
    mean_loss = []
    for prev in buffer.keys():
        all_targets = []
        for state_actions, r, t  in buffer[prev]:
            target = r
            if not t:  # Q values are zero for terminal.
                target += discount_factor * np.max(
                    [q_values[state_action] for state_action
                     in state_actions])
            all_targets.append(target)
        target = np.mean(all_targets)
        #print(len(all_targets))
        prev_q_value = q_values[prev]
        loss = target - prev_q_value
        # print(loss)
        q_values[prev] = target
        mean_loss.append(loss)

    return np.mean(mean_loss)

class keydefaultdict(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class LUGLNeuralNetwork(rl_agent.AbstractAgent):
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
        self._reset_dict()
        self._n_games = 0
        self.episode_length = 0
        self.model = None

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_q_values']
        del state['_buffer']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._reset_dict()

    def _reset_dict(self):
        self._q_values = keydefaultdict(self._default_value)
        self._buffer = {}

    def _default_value(self, key):
        if(self.model is None):

            return 0.0
        else:
            state_features, action = list(key[0]), list(key[1])[0]
            action_features = list(np.zeros(shape=self._num_actions))
            if (action is not None):
                action_features[action] = 1
            total_features = state_features + action_features
            phi = np.array(total_features)[np.newaxis,:]
            value = self.model(phi)

            return value[0][0]

    def get_state_action(self, info_state, action):
        return (tuple(info_state), tuple([action]))

    def replay(self):
        cloned_q = deepcopy(self._q_values)
        for _ in range(1):
            loss = init_replay(self._buffer, self._discount_factor,
                               cloned_q, self._step_size)
            print("Replay loss", loss)
            if(loss == 0.0):
                break;
        #print(cloned_q.values())
        #exit()
        return cloned_q



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

        q_values = [self._q_values[self.get_state_action(info_state,a)] + np.random.random()*0.0001 for a in legal_actions]
        greedy_q = np.argmax(q_values)
        #print(q_values)

        probs[legal_actions] = epsilon / len(legal_actions)
        probs[legal_actions[greedy_q]] +=(1 - epsilon)
        #print(probs, np.sum(probs))
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
        #print(time_step.rewards, time_step.last())
        # Learn step: don't learn during evaluation or at first agent steps.
        if self._prev_info_state and not is_evaluation:
            #print("training")
            target = time_step.rewards[self._player_id]
            state_actions = [self.get_state_action(info_state,a) for a in legal_actions]
            if not time_step.last():  # Q values are zero for terminal.
                target += self._discount_factor * max(
                    [self._q_values[state_action] for state_action in state_actions])
            #print(target)
            prev = self.get_state_action(self._prev_info_state,self._prev_action)
            prev_q_value = self._q_values[prev]
            self._last_loss_value = target - prev_q_value
            #print(target, prev_q_value)
            self._q_values[prev] += (
                    self._step_size * self._last_loss_value)

            if(prev not in self._buffer):
                self._buffer[prev] = []

            self._buffer[prev].append([
                [a for a in state_actions],
                time_step.rewards[self._player_id],
                time_step.last()
            ])




            self._epsilon = self._epsilon_schedule.step()
            self.episode_length+=1
            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                self._n_games +=1
                #print(self.episode_length)
                self.episode_length = 0
                #print(time_step.rewards, "rewards")
                #exit()
                return


        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
        return rl_agent.StepOutput(action=action, probs=probs)

    @property
    def loss(self):
        return self._last_loss_value


    def build_model(self, y_axis):

        # imports here, because we can't be loading tensorflow in subclasses
        from keras.models import Sequential
        from keras.layers import Dense
        import tensorflow as tf
        import keras
        n_threads = 6
        tf.config.threading.set_inter_op_parallelism_threads(n_threads)
        tf.config.threading.set_intra_op_parallelism_threads(n_threads)

        input_shape = (y_axis,)
        model = Sequential()
        model.add(Dense(y_axis, input_shape=input_shape, activation='relu'))
        model.add(Dense(1, activation='linear'))

        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.Adam(), jit_compile=False,

                      )
        return model

    def train_supervised(self):

        print("About to start training")
        all_features = []
        all_Qs = []

        for key, q_value in self._q_values.items():
            if (q_value != 0):
                state_features, action = list(
                    key[0]), key[1]

                action_features = list(
                     np.zeros(shape=self._num_actions))
                action_features[action[0]] = 1
                total_features = state_features + action_features
                all_Qs.append(q_value)
                all_features.append(total_features)

            # print(len(state_features), len(action_features), len(total_features))
        X = np.array(all_features)
        y = np.array(all_Qs)
        import os
        N = 4
        os.environ["OMP_NUM_THREADS"] = f"{N}"
        os.environ['TF_NUM_INTEROP_THREADS'] = f"{N}"
        os.environ['TF_NUM_INTRAOP_THREADS'] = f"{N}"

        self.model = self.build_model(X.shape[1])
        print(X.shape, y.shape)
        import keras
        callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                    patience=10)
        self.model.fit(X, y, epochs=10000, verbose=False, callbacks = [callback])
        mse = metrics.mean_squared_error(y, self.model.predict(X,
                                                               verbose=False))
        r2 = metrics.explained_variance_score(y, self.model.predict(X,
                                                                    verbose=False))

        print(mse, r2)



class LUGLLightGBM(LUGLNeuralNetwork):

    def get_phi(self,  key):
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


        q_values = self.replay()

        for key, q_value in q_values.items():
            if(q_value!=0):
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
        clf.fit(X,y, categorical_feature = range(X.shape[1]))
        self.model = clf
        mse = metrics.mean_squared_error(y, self.model.predict(X))
        r2 = metrics.explained_variance_score(y, self.model.predict(X))

        print(mse, r2)





class LUGLDecisionTree(LUGLNeuralNetwork):

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

        q_values = self.replay()

        data_per_action = [[] for _ in range(self._num_actions)]
        Qs_per_action = [[] for _ in range(self._num_actions)]

        for key, q_value in q_values.items():
            if(q_value!=0):
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

                from sklearn.tree import DecisionTreeRegressor
                from sklearn.metrics import mean_squared_error
                #model = linear_model.LinearRegression()
                from sklearn.model_selection import GridSearchCV

                # model = GridSearchCV(DecisionTreeRegressor(),
                #                    param_grid={
                #                        "max_depth": params },
                #                    n_jobs=-1, cv=10,
                #                    scoring="neg_mean_squared_error")

                model = DecisionTreeRegressor(min_samples_leaf=3)

                #from lightgbm.sklearn import LGBMRegressor
                #model = LGBMRegressor(n_jobs=6, n_estimators=100)
                # model.fit(X_train, y_train)
                #
                # path = model.cost_complexity_pruning_path(X_train, y_train)
                # ccp_alphas = path.ccp_alphas
                # impurities = path.impurities
                #
                # clfs = []
                # for ccp_alpha in ccp_alphas:
                #     clf = DecisionTreeRegressor(random_state=0,
                #                                  ccp_alpha=ccp_alpha)
                #     clf.fit(X_train, y_train)
                #     clfs.append(clf)
                #
                # ccp_alphas = ccp_alphas[:-1]
                #
                # best = 10000
                # best_clf = -1
                # for i, clf in enumerate(clfs):
                #
                #     y_test_hat = clf.predict(X_test)
                #     score = mean_squared_error(y_test, y_test_hat)
                #     if(score < best):
                #         best  = score
                #         best_clf = i
                #
                # print("Optimum alpha", ccp_alphas[best_clf])

                #model = DecisionTreeRegressor(ccp_alpha=ccp_alphas[best_clf])
                model.fit(X,y)

                mse = metrics.mean_squared_error(y, model.predict(X))
                r2 = metrics.explained_variance_score(y, model.predict(X))
                self.model.append(model)
            else:
                self.model.append(None)

            print(mse, r2)
        print("Total", total)



class LUGLLinear(LUGLNeuralNetwork):

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

        q_values = self.replay()

        data_per_action = [[] for _ in range(self._num_actions)]
        Qs_per_action = [[] for _ in range(self._num_actions)]
        print(len(self._q_values), "Qvals")
        for key, q_value in q_values.items():
            if(q_value!=0):
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

                #from sklearn.tree import DecisionTreeRegressor
                model = linear_model.LinearRegression()

                #from lightgbm.sklearn import LGBMRegressor
                #model = LGBMRegressor(n_jobs=6, n_estimators=100)
                model.fit(X,y)
                #print(model.best_params_)

                mse = metrics.mean_squared_error(y, model.predict(X))
                r2 = metrics.explained_variance_score(y, model.predict(X))
                self.model.append(model)
            else:
                self.model.append(None)

            print(mse, r2)
        print("Total", total)