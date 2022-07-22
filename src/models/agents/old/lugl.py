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
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._reset_dict()

    def _reset_dict(self):
        self._q_values = keydefaultdict(self._default_value)

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
            if not time_step.last():  # Q values are zero for terminal.
                target += self._discount_factor * max(
                    [self._q_values[self.get_state_action(info_state,a)] for a in legal_actions])
            #print(target)
            prev = self.get_state_action(self._prev_info_state,self._prev_action)
            prev_q_value = self._q_values[prev]
            self._last_loss_value = target - prev_q_value
            #print(target, prev_q_value)
            self._q_values[prev] += (
                    self._step_size * self._last_loss_value)




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
        #value = model.predict(phi)
        #print(value)

        #return value[0]

    # def _epsilon_greedy(self, info_state, legal_actions, epsilon):
    #     probs = np.zeros(self._num_actions)
    #
    #     if(self.model is None):
    #         q_values = [self._q_values[self.get_state_action(info_state,a)] + np.random.random()*0.0001 for a in legal_actions]
    #     else:
    #         state_actions = np.array([self.get_phi(self.get_state_action(info_state, a))
    #                                   for a in legal_actions])
    #         #state_actions = np.squeeze(state_actions)
    #         #print(state_actions.shape, "shape")
    #         q_values = self.model.predict(state_actions)
    #         #print(q_values.shape, "shape")
    #         #exit()
    #
    #     greedy_q = np.argmax(q_values)
    #
    #     probs[legal_actions] = epsilon / len(legal_actions)
    #     probs[legal_actions[greedy_q]] +=(1 - epsilon)
    #     #print(probs, np.sum(probs))
    #     action = np.random.choice(range(self._num_actions), p=probs)
    #     return action, probs

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
        for key, q_value in self._q_values.items():
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





class LUGLLinear(LUGLNeuralNetwork):

    def _default_value(self, key):
        if (self.model is None):
            return 0.0
        else:
            state_features, action = list(key[0]), key[1][0]
            #print(action, "action")
            phi = np.array(state_features)[np.newaxis, :]
            value = self.model[action].predict(phi)
            #print("value", value)
            return value[0]

    def train_supervised(self):

        print("About to start training")

        data_per_action = [[] for _ in range(self._num_actions)]
        Qs_per_action = [[] for _ in range(self._num_actions)]
        print(len(self._q_values), "Qvals")
        for key, q_value in self._q_values.items():
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
            # model = linear_model.LinearRegression()
            from lightgbm.sklearn import LGBMRegressor
            model = LGBMRegressor(n_jobs=6, n_estimators=100)
            model.fit(X,y)
            mse = metrics.mean_squared_error(y, model.predict(X))
            r2 = metrics.explained_variance_score(y, model.predict(X))
            self.model.append(model)

            print(mse, r2)
        print("Total", total)

import random

class LUGLExtraTrees(LUGLLightGBM):


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
            if not time_step.last():  # Q values are zero for terminal.
                target += self._discount_factor * max(
                    [self._q_values[self.get_state_action(info_state,a)] for a in legal_actions])
            #print(target)
            prev = self.get_state_action(self._prev_info_state,self._prev_action)
            prev_q_value = self._q_values[prev]
            self._last_loss_value = target - prev_q_value
            #print(target, prev_q_value)
            self._q_values[prev] += (
                    self._step_size * self._last_loss_value)




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


    def _reset_dict(self):
        self._q_values = {}



    def _epsilon_greedy(self, info_state, legal_actions, epsilon):
        probs = np.zeros(self._num_actions)

        action_states = [self.get_state_action(info_state, a) for a in legal_actions]

        if (self.model is None or action_states[0] in self._q_values):
            q_values = []
            for a in action_states:
                if(a not in self._q_values):
                    self._q_values[a] = 0
                q_values.append(self._q_values[a])

            q_values = [self._q_values[a] for a  in action_states]
        else:
            #estimator = random.choice(self.model.estimators_)


            state_actions = np.array(
                [self.get_phi(a)
                 for a in action_states])
            # state_actions = np.squeeze(state_actions)
            # print(state_actions.shape, "shape")
            q_values = self.model.predict(state_actions)
            # print(q_values.shape, "shape")
            # exit()
        for i, a in enumerate(action_states):
            self._q_values[a] = q_values[i]
        greedy_q = np.argmax(q_values + np.random.random(size= len(q_values)) * 0.0001)

        probs[legal_actions] = epsilon / len(legal_actions)
        probs[legal_actions[greedy_q]] += 1 - epsilon
        # print(probs, np.sum(probs))
        action = np.random.choice(range(self._num_actions), p=probs)
        return action, probs

    def _default_value(self, key):
        if(self.model is None):

            return 0.0
        else:
            state_features, action = list(key[0]), list(key[1])
            total_features = state_features + action
            phi = np.array(total_features)[np.newaxis,:]

            value = self.model.predict(phi)
            #print(value)

            return value[0]

    def train_supervised(self):

        print("About to start training")
        all_features = []
        all_Qs = []
        for key, q_value in self._q_values.items():
            if(q_value!=0):
                state_features, action = list(
                    key[0]), key[1]

                total_features = state_features + list(action)
                all_Qs.append(q_value)
                all_features.append(total_features)

        X = np.array(all_features)
        y = np.array(all_Qs)
        #print(X.shape, y.shape)
        #exit()
        #self.encoder = TargetEncoder(cols=[X.shape[1]-1])
        #self.encoder.fit(X, y)
        #X_enc = self.encoder.transform(X)
        #print(X_enc)

        print(X.shape, y.shape)
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.tree import DecisionTreeRegressor
        #clf = ExtraTreesRegressor(n_jobs=12, n_estimators=100,  bootstrap=True)
        from lightgbm.sklearn import LGBMRegressor
        clf = LGBMRegressor(n_jobs=6, n_estimators=100)
        #clf = DecisionTreeRegressor()
        clf.fit(X,y)
        self.model = clf
        mse = metrics.mean_squared_error(y, self.model.predict(X))
        r2 = metrics.explained_variance_score(y, self.model.predict(X))

        print(mse, r2)
