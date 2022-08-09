import collections
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from sklearn import metrics
from category_encoders import TargetEncoder
from sklearn import linear_model
from copy import deepcopy
from tqdm import tqdm



N_BOOSTRAPS = 20

def replay(buffer, discount_factor, q_values, step_size):
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


def init_replay(buffer, discount_factor, q_values, step_size):
    mean_loss = []
    for prev in tqdm(buffer.keys()):
        all_targets = []
        for state_actions, r, t in buffer[prev]:
            target = r
            if not t:  # Q values are zero for terminal.
                x = [q_values[state_action] for state_action
                     in state_actions]
                #print(x)
                target += discount_factor * np.max(
                    x)
            all_targets.append(target)
        #print(len(all_targets))
        target = np.mean(all_targets)
        # print(len(all_targets))
        prev_q_value = q_values[prev]
        loss = target - prev_q_value
        # print(loss)
        q_values[prev] = target
        mean_loss.append(loss)

    return np.mean(mean_loss)


class keydefaultdict(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
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
        if (self.model is None):
            return [0] * self._num_actions
        else:
            state_features, action = list(key[0]), list(key[1])[0]
            action_features = list(np.zeros(shape=self._num_actions))
            if (action is not None):
                action_features[action] = 1
            total_features = state_features + action_features
            phi = np.array(total_features)[np.newaxis, :]
            value = self.model(phi)

            return value[0][0]

    def get_state_action(self, info_state, action):
        return (tuple(info_state + [action]), tuple([action]))

    def replay(self):
        cloned_q = (self._q_values)
        for _ in range(1):
            loss = init_replay(self._buffer, self._discount_factor,
                               cloned_q, self._step_size)
            print("Replay loss", loss)
            if(loss == 0.0):
                break;
        # #print(cloned_q.values())
        # #exit()
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

        #if(self.model is None):
        # q_values = [self._q_values[self.state(info_state,
        #                                                      a)] + np.random.random() * 0.0001
        #                 for a in legal_actions]
        # # else:
        #     sa = np.array([list(info_state) + [a] for a in legal_actions])
        #     q_values = self.model.predict(sa)

        q_values = self.get_q_values(info_state)
        greedy_q = np.argmax(q_values)
        # print(q_values)

        probs[legal_actions] = epsilon / len(legal_actions)
        probs[legal_actions[greedy_q]] += (1 - epsilon)
        # print(probs, np.sum(probs))
        action = np.random.choice(range(self._num_actions), p=probs)
        return action, probs, q_values

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
            action, probs, q_values = self._epsilon_greedy(
                info_state, legal_actions, epsilon=epsilon)
        # print(time_step.rewards, time_step.last())
        # Learn step: don't learn during evaluation or at first agent steps.
        if self._prev_info_state and not is_evaluation:
            # print("training")
            target = time_step.rewards[self._player_id]
            state_actions = [self.get_state_action(info_state, a) for a in
                             legal_actions]
            if not time_step.last():  # Q values are zero for terminal.
                target += self._discount_factor * max(q_values)
            # print(target)
            prev = self.get_state_action(self._prev_info_state,
                                         self._prev_action)
            prev_q_value = self._q_values[prev]
            self._last_loss_value = target - prev_q_value
            # print(target, prev_q_value)

            self._q_values[prev] += (
                    self._step_size * self._last_loss_value)

            if (prev not in self._buffer):
                self._buffer[prev] = []

            self._buffer[prev].append([
                [a for a in state_actions],
                time_step.rewards[self._player_id],
                time_step.last()
            ])

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




class LUGLLightGBM(LUGLNeuralNetwork):

    def get_state_action(self, info_state, action):

        return (tuple(info_state), tuple([action]))


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

        q_values = self.replay()
        #q_values = self._q_values

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
        clf.fit(X, y, categorical_feature=[X.shape[1]-1])
        self.model = clf
        mse = metrics.mean_squared_error(y, self.model.predict(X))
        r2 = metrics.explained_variance_score(y, self.model.predict(X))

        print(mse, r2)


    # def cleanup(self):
    #     keys = []
    #
    #     for key in tqdm(self._buffer.keys()):
    #         if(len(self._buffer[key]) > 3):
    #             keys.append(key)
    #     print("CLEANING UP", len(keys))
    #     for key in keys:
    #         del self._buffer[key]
    #         del self._q_values[key]



class LUGLDecisionTree(LUGLLightGBM):

    def get_q_values(self, state_features, legal_actions):

        if (self.model is None):
            q_values = [(0.0 + np.random.random() * 0.0001)
                        for a in legal_actions]
            # print(q_values)
        else:
            q_values = [
                self.model[a].predict(np.array(state_features)[np.newaxis, :])[
                    0]
                for a in legal_actions]
        return q_values

    def train_supervised(self):

        print("About to start training")

        print(len(self._q_values))

        q_values = self.replay()
        #q_values = self._q_values

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
            total += X.shape[0]
            print(X.shape, y.shape)
            if (X.shape[0] > 10):
                from sklearn.model_selection import train_test_split
                # X_train, X_test, y_train, y_test = train_test_split(X, y,
                #                                                     random_state=0)

                from sklearn.tree import DecisionTreeRegressor, \
                    DecisionTreeClassifier
                from sklearn.metrics import mean_squared_error
                from sklearn.ensemble import ExtraTreesRegressor
                # model = linear_model.LinearRegression()
                from sklearn.model_selection import GridSearchCV
                # ex = ExtraTreesRegressor(n_estimators=100, n_jobs=100, bootstrap=True)
                # ex.fit(X,y)

                dt = DecisionTreeRegressor()
                # dt.fit(X,ex.predict(X))
                # #path = dt.cost_complexity_pruning_path(X, y)
                #
                params = {"min_samples_split": [20, 100, 200, 300, 500, 2000, 3000]}
                result = GridSearchCV(dt, param_grid=params,
                                      scoring="neg_mean_squared_error",
                                      n_jobs=100, cv=5)

                model = dt
                result.fit(X, y)
                model = result.best_estimator_
                print(result.best_params_)

                mse = metrics.mean_squared_error(y, model.predict(X))
                r2 = metrics.explained_variance_score(y, model.predict(X))
                print(mse, r2)
                self.model.append(model)
            else:
                self.model.append(None)

        print("Total", total)


class LUGLLinear(LUGLNeuralNetwork):

    def transform(self, y):
        return (y + 1) / 2

    def inverse_transform(self, y):
        return np.clip(y, -1, 1)
        # return (y * 2) - 1

    def _default_value(self, key):
        if (self.model is None):
            return 0.0
        else:
            state_features, action = list(key[0]), key[1][0]
            # print(action, "action")
            phi = np.array(state_features)[np.newaxis, :]
            if (self.model[action] is None):
                return 0.0
            else:
                value = self.model[action].predict(phi)

                #value = self.model[action].bin.inverse_transform(value[:, np.newaxis])

                return self.inverse_transform(value[0])

    def train_supervised(self):

        print("About to start training")

        q_values = self.replay()
        #q_values = self._q_values

        data_per_action = [[] for _ in range(self._num_actions)]
        Qs_per_action = [[] for _ in range(self._num_actions)]
        print(len(self._q_values), "Qvals")
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
            total += X.shape[0]
            print(X.shape, y.shape)
            self.bin = []
            if (X.shape[0] > 10):

                # from lightgbm.sklearn import LGBMRegressor
                # model = LGBMRegressor(n_jobs=6, n_estimators=100)
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline

                from sklearn.pipeline import Pipeline
                from sklearn import random_projection
                clf = linear_model.LinearRegression()
                #from sklearn.kernel_approximation import RBFSampler
                from sklearn.kernel_approximation import PolynomialCountSketch
                from sklearn.neighbors import KNeighborsTransformer, NeighborhoodComponentsAnalysis
                from sklearn.decomposition import PCA
                from sklearn.discriminant_analysis import \
                    LinearDiscriminantAnalysis
                from sklearn.preprocessing import KBinsDiscretizer

                #bin = KBinsDiscretizer(encode="ordinal", n_bins=20)

                #bin.fit(y[:, np.newaxis])
                #y = bin.transform(y[:, np.newaxis]).T[0]
                #n_classes = bin.n_bins_[0]
                #print(X.shape, y.shape, bin.n_bins_)
                #transf = NeighborhoodComponentsAnalysis(
                #    n_components=10,)
                #transf = RBFSampler()
                transf = PolynomialCountSketch()
                #transf = NeighborhoodComponentsAnalysis(n_components=10)
               # exit()
                model = Pipeline([('projection',transf) , ('clf', clf)])

                # model = clf
                # y = self.transform(y)
                # print(y.max(), y.min())

                model.fit(X, y)
                #model.bin = bin
                # print(model.best_params_)

                mse = metrics.mean_squared_error(y, model.predict(X))
                r2 = metrics.explained_variance_score(y, model.predict(X))
                self.model.append(model)
            else:
                self.model.append(None)

            print(mse, r2)
        print("Total", total)


class LUGLRandomForest(LUGLNeuralNetwork):

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
            action, probs, q_values = self._epsilon_greedy(
                info_state, legal_actions, epsilon=epsilon, is_evaluation = is_evaluation)
        # print(time_step.rewards, time_step.last())
        # Learn step: don't learn during evaluation or at first agent steps.
        if self._prev_info_state and not is_evaluation:
            # print("training")
            target = time_step.rewards[self._player_id]
            state_actions = [self.get_state_action(info_state, a) for a in
                             legal_actions]
            if not time_step.last():  # Q values are zero for terminal.
                target += self._discount_factor * max(q_values)
            # print(target)
            prev = self.get_state_action(self._prev_info_state,
                                         self._prev_action)
            prev_q_value = self._q_values[prev]
            self._last_loss_value = target - prev_q_value
            # print(target, prev_q_value)

            self._q_values[prev] += (
                    self._step_size * self._last_loss_value)

            if (prev not in self._buffer):
                self._buffer[prev] = []

            self._buffer[prev].append([
                [a for a in state_actions],
                time_step.rewards[self._player_id],
                time_step.last()
            ])

            self._epsilon = self._epsilon_schedule.step()
            self.episode_length += 1
            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                self._n_games += 1
                # print(self.episode_length)
                self.episode_length = 0
                # print(time_step.rewards, "rewards")
                # exit()
                if(self.model is not None):
                    self.change_est()


                return

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            self._prev_info_state = info_state
            self._prev_action = action
        return rl_agent.StepOutput(action=action, probs=probs)

    def change_est(self):
        self.ests = []
        for amodel in self.model:
            if(amodel is not None):
                est = np.random.randint(0, amodel.n_estimators)
                self.ests.append(est)
            else:
                self.ests.append(0)

    def _epsilon_greedy(self, info_state, legal_actions, epsilon, is_evaluation):
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

        #if(self.model is None):


        if(not is_evaluation):
            if(self.model is not None):
                #print(self.ests)
                q_values = [self.model[a].estimators_[self.ests[a]].predict(np.array(info_state)[np.newaxis, :] )
                            for a in legal_actions]
                #print(np.array(q_values).T[0])
                q_values = np.array(q_values).T[0]
                greedy_q = np.argmax(q_values)
                epsilon = 0.01
                probs[legal_actions] = epsilon / len(legal_actions)
                probs[legal_actions[greedy_q]] += (1 - epsilon)
            # print(q_values)
            else:
                q_values = [self._q_values[self.get_state_action(info_state,
                                                                 a)]
                            for a in legal_actions]
                greedy_q = np.argmax(q_values)
                probs[legal_actions] = epsilon / len(legal_actions)
                probs[legal_actions[greedy_q]] += (1 - epsilon)
        else:
            q_values = [self._q_values[self.get_state_action(info_state,
                                                             a)]
                        for a in legal_actions]
            greedy_q = np.argmax(q_values)
            probs[legal_actions[greedy_q]] = 1
        # print(probs, np.sum(probs))
        action = np.random.choice(range(self._num_actions), p=probs)
        return action, probs, q_values


    def _default_value(self, key):
        if (self.model is None):
            return 0.0
        else:
            state_features, action = list(key[0]), key[1][0]
            # print(action, "action")
            phi = np.array(state_features)[np.newaxis, :]
            if (self.model[action] is None):
                return 0.0
            else:

                value = self.model[action].predict(phi)
                return value[0]

    def train_supervised(self):


        print("About to start training")

        print(len(self._q_values))

        q_values = self.replay()
        #q_values = self._q_values

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
            total += X.shape[0]
            print(X.shape, y.shape)
            if (X.shape[0] > 10):
                from sklearn.model_selection import train_test_split
                # X_train, X_test, y_train, y_test = train_test_split(X, y,
                #                                                     random_state=0)

                from sklearn.tree import DecisionTreeRegressor, \
                    DecisionTreeClassifier
                from sklearn.metrics import mean_squared_error
                # model = linear_model.LinearRegression()
                from sklearn.model_selection import GridSearchCV

                # model = GridSearchCV(DecisionTreeRegressor(),
                #                    param_grid={
                #                        "max_depth": params },
                #                    n_jobs=-1, cv=10,
                #                    scoring="neg_mean_squared_error")

                from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

                model = RandomForestRegressor(max_depth=3, n_jobs=6)
                model.fit(X,y)


                #from sklearn.manifold import LocallyLinearEmbedding
                # from sklearn.tree import DecisionTreeRegressor
                # from sklearn.kernel_approximation import RBFSampler
                # # clf = ExtraTreesRegressor(min_samples_leaf=3, bootstrap=True, n_jobs=20)
                #
                # from lightgbm.sklearn import LGBMRegressor
                # from sklearn.pipeline import Pipeline
                # # clf = LGBMRegressor(n_jobs=12, n_estimators=100,boosting_type="rf",bagging_freq = 1, bagging_fraction = 0.7)
                #
                # model = ExtraTreesRegressor(bootstrap=True, n_jobs=20,
                #                             n_estimators=1000)
                # pipe = Pipeline([('scaler', RBFSampler()), ('svc', model)])
                #
                # pipe.fit(X, y)
                # y_hat = pipe.predict(X)
                #
                # model = DecisionTreeRegressor()
                # pipe = Pipeline(
                #     [('scaler', RBFSampler()), ('svc', model)])
                # pipe = model
                # model.fit(X, y_hat)
                # print("Max depth", model.tree_.max_depth, )

                mse = metrics.mean_squared_error(y, model.predict(X))
                r2 = metrics.explained_variance_score(y, model.predict(X))
                print(mse, r2)
                self.model.append(model)
            else:
                self.model.append(None)

        self.change_est()
        print("Total", total)

        # import warnings
        # warnings.filterwarnings('error')


from scipy.spatial.distance import cdist


class LUGLLinearTemplate(LUGLNeuralNetwork):


    def transform(self, y):
        return (y + 1) / 2

    def inverse_transform(self, y):
        return np.clip(y, -1, 1)
        # return (y * 2) - 1

    def _default_value(self, key):
        if (self.model is None):
            return 0.0
        else:
            state_features, action = list(key[0]), key[1][0]
            # print(action, "action")
            phi = np.array(state_features)[np.newaxis, :]
            if (self.model[action] is None):
                return 0.0
            else:

                X_dst = cdist(phi, self.model[action].latest_best)

                max_column = X_dst.max(axis=-1)[:, np.newaxis]
                min_column = X_dst.min(axis=-1)[:, np.newaxis]

                # print(max_column.shape, min_column.shape)
                # exit()
                X_dst = np.concatenate(
                    [X_dst, max_column, min_column],
                    axis=-1)


                value = self.model[action].predict(X_dst)

                #value = self.model[action].bin.inverse_transform(value[:, np.newaxis])

                return self.inverse_transform(value[0])

    def train_supervised(self):

        print("About to start training")

        # q_values = self.replay()
        q_values = self._q_values

        data_per_action = [[] for _ in range(self._num_actions)]
        Qs_per_action = [[] for _ in range(self._num_actions)]
        print(len(self._q_values), "Qvals")
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
            total += X.shape[0]
            print(X.shape, y.shape)
            self.bin = []
            if (X.shape[0] > 10):

                # from lightgbm.sklearn import LGBMRegressor
                # model = LGBMRegressor(n_jobs=6, n_estimators=100)
                from sklearn.preprocessing import StandardScaler
                from sklearn.pipeline import Pipeline

                from sklearn.pipeline import Pipeline
                from sklearn import random_projection
                clf = linear_model.LinearRegression()
                from sklearn.tree import DecisionTreeRegressor
                #model = DecisionTreeRegressor(max_depth=3)
                model = clf

                maximum_distances = 10
                n_samples = 500

                best = [[] for _ in range(maximum_distances)]
                best_score = [[-10000] for _ in range(maximum_distances)]

                r = np.array(list(range(0, len(X))))

                for dist in range(maximum_distances-1, maximum_distances):
                    for _ in range(n_samples):
                        n_distances = dist
                        if (n_distances > len(X)):
                            break
                        sampled_dst = np.random.choice(r, size=n_distances,
                                                   replace=False)
                        limit = 3000
                        if(len(X) > limit):
                            subsampled = np.random.choice(r, size=limit,
                                                       replace=False)
                        else:
                            subsampled = r
                        X_dst = cdist(X[subsampled], X[sampled_dst])
                        clf = linear_model.LinearRegression()
                        max_column = X_dst.max(axis = -1)[:, np.newaxis]
                        min_column = X_dst.min(axis = -1)[:, np.newaxis]
                        #max_column_a = np.argmax(X_dst, axis=-1)[:, np.newaxis]
                        #min_column_a = np.argmin(X_dst, axis=-1)[:, np.newaxis]

                        #print(max_column.shape, min_column.shape)
                        #exit()
                        X_dst = np.concatenate([X_dst, max_column, min_column], axis = -1)
                        #X_dst = np.concatenate([X, X_dst], axis = -1)
                        #print(X_dst.shape, X.shape)
                        clf.fit(X_dst, y[subsampled])
                        score = clf.score(X_dst, y[subsampled])
                        if (score > best_score[n_distances]):
                            best_score[n_distances] = score
                            best[n_distances] = sampled_dst

                print(best_score)
                print(best)

                latest_best = best[-1]
                #exit()
                X_dst = cdist(X, X[latest_best])
                max_column = X_dst.max(axis=-1)[:, np.newaxis]
                min_column = X_dst.min(axis=-1)[:, np.newaxis]
                #max_column_a = np.argmax(X_dst, axis=-1)[:, np.newaxis]
                #min_column_a = np.argmin(X_dst, axis=-1)[:, np.newaxis]

                # print(max_column.shape, min_column.shape)
                # exit()
                X_dst = np.concatenate(
                    [X_dst, max_column, min_column],
                    axis=-1)

                model.fit(X_dst, y)
                model.latest_best = X[latest_best]
                #model.bin = bin
                # print(model.best_params_)

                mse = metrics.mean_squared_error(y, model.predict(X_dst))
                r2 = metrics.explained_variance_score(y, model.predict(X_dst))
                self.model.append(model)
            else:
                self.model.append(None)

            print(mse, r2)
        print("Total", total)


class LUGLLightGBMCompressed(LUGLLightGBM):
    def get_state_action(self, info_state, action):
        X = np.array([info_state])

        if (not hasattr(self, 'projection')):
            n_components = 30
            # print(info_state)
            # exit()
            # from sklearn import random_projection
            # self.projection = random_projection.GaussianRandomProjection(n_components=10)
            self.projection = np.random.choice([0, 1],
                                               size=(X.shape[1], n_components))
            # print(X.shape)
            # exit()
            # print("sdfasdfsapfdoadfpouOPAUDSOFIU")
            # self.projection.fit(X)
        info_state_new = tuple(np.dot(X, self.projection)[0])
        # print(info_state_new)
        # exit()

        return (tuple(info_state_new), tuple([action]))



    def train_supervised(self):

            print("About to start training")
            all_features = []
            all_Qs = []

            q_values = self.replay()
            # q_values = self._q_values

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
            clf.fit(X, y, categorical_feature=[X.shape[1] - 1])
            self.model = clf
            mse = metrics.mean_squared_error(y, self.model.predict(X))
            r2 = metrics.explained_variance_score(y, self.model.predict(X))

            print(mse, r2)