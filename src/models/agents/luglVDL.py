import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from sklearn import metrics
from agents.utils import ReplayBuffer
from agents.utils import LimitedSizeDict
from collections import OrderedDict
import keras
from keras.layers import Dense, Input
from models.agents.keras_helper import NNWeightHelper
import cma
from open_spiel.python.algorithms import minimax, mcts
from sklearn.pipeline import Pipeline

class DCLF(rl_agent.AbstractAgent):
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
        self._q_values = {}
        self._n_games = 0
        self.episode_length = 0
        self.model = None
        self.maximum_size = int(1e5)
        self.batch_size = 32
        self.state_representation_size = state_representation_size
        self._buffer = ReplayBuffer(self.maximum_size)
        self._q_values = OrderedDict()
        self._tbr = OrderedDict()

        self._reset_dict()




    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_buffer']
        del state['_q_values']
        del state['_tbr']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._buffer = ReplayBuffer(self.maximum_size)
        self._q_values = {}
        self._tbr = {}
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
        #info_state = np.array(info_state, dtype= "i8")
        if (info_state not in self._q_values):
            self._q_values[info_state] = np.zeros(shape=self._num_actions)

            if(self.model is not None):
                self._q_values[info_state][legal_actions] = self.get_model_qs(
                    info_state, legal_actions)

        q_values = self._q_values[info_state][legal_actions]
        #q_values += np.random.random(
        #        size=legal_actions) * 0.0001
        #greedy_action = np.argmax(q_values)
        greedy_q = max(q_values)

        greedy_actions = [
            a for a in legal_actions if
            self._q_values[info_state][a] == greedy_q
        ]
        probs[legal_actions] = epsilon / len(legal_actions)
        probs[greedy_actions] += (1 - epsilon) / len(greedy_actions)
        action = np.random.choice(range(self._num_actions), p=probs)

        #print(probs)


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
            # # print("training")
            rewards = time_step.rewards[self._player_id]
            #print(self._prev_action, rewards)

            self._buffer.add([self._prev_info_state, self._prev_action,
                              info_state,
                              legal_actions, rewards, time_step.last()])


            target = rewards

            if (not time_step.last()):
                feature_qs = self._q_values[info_state][
                    legal_actions]

                target += self._discount_factor * max(feature_qs)

            # print(target, prev_q_value)

            #prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
            #loss = target - prev_q_value

            #d_target = np.digitize(target, bins=bins)



            self._q_values[self._prev_info_state][self._prev_action] = target


            sa = (self._prev_info_state, self._prev_action)
            if (sa not in self._tbr):
                self._tbr[sa] = []
            v = self._tbr[sa]
            v.append([target, np.array(self._prev_info_next_state, dtype = np.uint8)])

            if (len(v) > 20):
                #print(v)
                v.pop(0)
            #print(target)


            self._epsilon = self._epsilon_schedule.step()
            self.episode_length += 1
            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                self._n_games += 1
                # print(self.episode_length)
                self.episode_length = 0
                # print(time_step.rewards, "rewards")
                # exit()
                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
            #print(self.state)
            child_state = self.state.clone()
            child_state.apply_action(action)
            #print(child_state)
            obs = child_state.observation_tensor(self._player_id)

            #obs = [ord(c) for c in str(child_state).split(":")[-1]]
            #print(len(obs))
            #print(child_state)
            self._prev_info_next_state = tuple(obs)

        return rl_agent.StepOutput(action=action, probs=probs)


    @property
    def loss(self):
        return self._last_loss_value


    def _reset_dict(self):

        buffer_states = []

        for i,(
                prev_info_state, prev_action, l_info_state, l_legal_actions,
                rewards,
                last) in enumerate(self._buffer._data):

            buffer_states.append(prev_info_state)
            buffer_states.append(l_info_state)

        buffer_states = set(buffer_states)

        for key in self._q_values.copy().keys():

            if(len(self._q_values) > self.maximum_size):
                if(key not in buffer_states):
                    for action in range(self._num_actions):
                        if((key,action) in self._tbr):
                            del self._tbr[(key,action)]
                    if(key in self._q_values):
                        #print("deleting")
                        del self._q_values[key]
                        #print(len(self._q_values))
            else:
                break


        #print(len(self._tbr), "tbr")




        #self._q_values = {}




class LUGLVDL(DCLF):


    def get_model_qs(self, state_features, legal_actions):

        actions = []
        terminal = []
        for i, action in enumerate(legal_actions):
            child_state = self.state.clone()
            child_state.apply_action(action)
            if child_state.is_terminal():
                terminal.append([i, child_state.player_return(self._player_id)])
            obs = child_state.observation_tensor(self._player_id)
            #print(str(child_state))

            #obs = [ord(c) for c in str(child_state).split(":")[-1]]

            actions.append(obs)

        q_values = self.model.predict(actions)

        for t in terminal:
            q_values[t[0]] = t[1]


        return q_values





    def train_supervised(self):

        print("About to start training")
        all_features = []
        all_Qs = []

        # make targets!
        #for self

        for (state, action), q_fstates, in self._tbr.items():
            #for action, Q in enumerate(self._q_values[state]):
                Q = [qf[0] for qf in q_fstates]
                f_state = q_fstates[0][1]
                all_features.append(list(f_state))
                all_Qs.append(np.mean(Q))

        X = np.array(all_features)
        y = np.array(all_Qs)

        #print(X)
        print(X.shape, y.shape)

        tree_encoder = DecisionTreeLeafEncoder(max_leaf_nodes=10000)
        #tree_encoder = RandomForestLeafEncoder(max_leaf_nodes=20,n_jobs=4, n_estimators=10)
        from sklearn.linear_model import RidgeCV, Ridge, LarsCV, ElasticNetCV, OrthogonalMatchingPursuitCV

        clf = Pipeline(
            steps=[
                ('tree_encoder',tree_encoder),
                ('regressor', Ridge(normalize=False))
            ]
        )

        #clf = DecisionTreeRegressor(max_leaf_nodes=2000)
        self.model = clf
        clf.fit(X,y)
        #print(clf.steps[-1][-1].coef_)
        mse = metrics.mean_squared_error(y, self.model.predict(X))
        r2 = metrics.explained_variance_score(y, self.model.predict(X))

        print(mse, r2)





from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class DecisionTreeLeafEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, max_leaf_nodes=None):
        self.max_leaf_nodes = max_leaf_nodes
        self.tree_ = None
        self.one_hot_ = None

    def fit(self, X, y=None):
        self.tree_ = DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes).fit(X, y)
        leaf_nodes = self.tree_.apply(X)
        self.one_hot_ = OneHotEncoder(categories='auto', sparse=False).fit(
            leaf_nodes.reshape(-1, 1))
        return self

    def transform(self, X, y=None):
        leaf_nodes = self.tree_.apply(X)
        leaf_encodings = self.one_hot_.transform(leaf_nodes.reshape(-1, 1))
        #print("decision tree features", X.shape)
        original_features = np.array(X)
        return np.hstack([original_features, leaf_encodings])


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class RandomForestLeafEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_estimators=100, max_leaf_nodes=2, n_jobs=None):
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.n_jobs = n_jobs
        self.forest_ = None
        self.one_hot_ = None

    def fit(self, X, y=None):
        self.forest_ = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_leaf_nodes=self.max_leaf_nodes,
            n_jobs=self.n_jobs
        ).fit(X, y)
        leaf_nodes = self.forest_.apply(X)
        self.one_hot_ = OneHotEncoder(categories='auto', sparse=False).fit(
            leaf_nodes)
        return self

    def transform(self, X, y=None):
        leaf_nodes = self.forest_.apply(X)
        leaf_encodings = self.one_hot_.transform(leaf_nodes)
        original_features = np.array(X)
        return np.hstack([original_features, leaf_encodings])

#
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.tree import DecisionTreeClassifier
# import numpy as np
#
#
# class TreeLeafEncoder(BaseEstimator, TransformerMixin):
#     """
#     A sklearn transformer that uses the leafs of a decision tree as features,
#     after target encoding them, without using pandas.
#     """
#
#     def __init__(self, max_leaf_nodes=3, random_state=None):
#         self.max_leaf_nodes = max_leaf_nodes
#         self.random_state = random_state
#         self.tree = None
#         self.target_mean = None
#
#     def fit(self, X, y):
#         # Fit a decision tree classifier
#         self.tree = DecisionTreeRegressor(max_leaf_nodes=self.max_leaf_nodes,
#                                            random_state=self.random_state)
#         self.tree.fit(X, y)
#
#         # Target encode the leaf nodes
#         leaf_indices = self.tree.apply(X)
#         self.target_mean = {}
#         for i in range(leaf_indices.shape[0]):
#             col = leaf_indices[ i]
#             for j in range(self.tree.tree_.n_node_samples.shape[0]):
#                 if self.tree.tree_.n_node_samples[j] != 0:
#                     self.target_mean[(i, j)] = y[col == j].mean()
#         return self
#
#     def transform(self, X):
#         # Target encode the leaf nodes of the fitted decision tree
#         leaf_indices = self.tree.apply(X)
#         X_transformed = []
#         for i in range(leaf_indices.shape[0]):
#             col = leaf_indices[i]
#             encoded_col = []
#             for j in range(self.tree.tree_.n_node_samples.shape[0]):
#                 if self.tree.tree_.n_node_samples[j] != 0:
#                     encoded_col.append(col == j)
#             encoded_col = np.column_stack(
#                 [self.target_mean.get((i, j), np.nan) for j in
#                  range(self.tree.tree_.n_node_samples.shape[0]) if
#                  self.tree.tree_.n_node_samples[j] != 0])
#             X_transformed.append(encoded_col)
#         return np.hstack(X_transformed)
