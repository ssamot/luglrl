import collections
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from sklearn import metrics
from category_encoders import TargetEncoder
from sklearn import linear_model
from copy import deepcopy
from tqdm import tqdm
from open_spiel.python.algorithms.dqn import ReplayBuffer


N_BOOSTRAPS = 20



class keydefaultdict(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class DCLF(rl_agent.AbstractAgent):
    """Tabular Q-Learning agent.

    See open_spiel/python/examples/tic_tac_toe_qlearner.py for an usage example.
    """

    def __init__(self,
                 player_id,
                 state_representation_size,  ## ignored
                 num_actions,
                 step_size=0.0,
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
        self._q_values = None
        self._n_games = 0
        self.episode_length = 0
        self.model = None
        self.maximum_size = int(1e5)
        self._buffer = ReplayBuffer(self.maximum_size)


    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_buffer']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._buffer = ReplayBuffer(self.maximum_size)



    def _default_value(self, key):
        if (self.model is None):

            return 0.0
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

        q_values = self.get_q_values(info_state, legal_actions)

        greedy_q = np.argmax(q_values)

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
            # # print("training")
            rewards = time_step.rewards[self._player_id]
            #print(self._prev_action, rewards)


            self._buffer.add([self._prev_info_state,self._prev_action,
                              info_state,
                              legal_actions, rewards, time_step.last()])

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


    def _reset_dict(self):
        pass


class LightGBM(DCLF):

    def get_q_values(self, state_features, legal_actions):

        if (self.model is None):
            q_values = [(0.0 + np.random.random() * 0.0001)
                        for a in legal_actions]
            # print(q_values)
        else:
            feature_actions = [list(state_features) + [a] for a in legal_actions]
            #print(feature_actions)
            q_values = self.model.predict(feature_actions)
            #print(q_values.shape)

        return q_values





    def train_supervised(self):

        print("About to start training")
        all_features = []
        all_Qs = []

        # make targets!
        for (prev_info_state, prev_action, info_state, legal_actions, rewards,
             last) in self._buffer:
            target = rewards

            if (not last):
                q = self.get_q_values(info_state, legal_actions)
                target += self._discount_factor * max(q)

            all_features.append(list(prev_info_state) + [prev_action])
            all_Qs.append(target)

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




class DecisionTree(LightGBM):




    def get_q_values(self, state_features, legal_actions):

        if(self.model is None):
            q_values = [(0.0 + np.random.random() * 0.0001)
                        for a in legal_actions]
            #print(q_values)
        else:
            q_values = [self.model[a].predict(np.array(state_features)[np.newaxis, :])[0]
                    for a in legal_actions]
        return q_values


    def train_supervised(self):

        print("About to start training")


        data_per_action = [[] for _ in range(self._num_actions)]
        Qs_per_action = [[] for _ in range(self._num_actions)]


        # make targets!
        for (prev_info_state,prev_action,  info_state, legal_actions, rewards, last) in self._buffer:
            target = rewards
            q = self.get_q_values(info_state, legal_actions)
            #print(rewards)
            if(not last):
                target += self._discount_factor * max(q)

            data_per_action[prev_action].append(prev_info_state)
            Qs_per_action[prev_action].append(target)

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
                params = {"min_samples_split": [2, 20,  100, 200, 300]}
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
