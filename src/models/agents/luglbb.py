import collections
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from sklearn import metrics
from tqdm import tqdm




def init_replay(buffer, discount_factor, q_values, num_actions):
    mean_loss = []

    for prev_state in tqdm(buffer.keys()):

        for action in range(num_actions):
            all_targets = []
            if(len(buffer[prev_state][action]) > 0):
                for infostate, legal_actions, r, t in buffer[prev_state][action]:
                    target = r
                    if not t:  # Q values are zero for terminal.
                        x = [q_values[infostate][a] for a in legal_actions]
                        print(x, "sadfasdfasfdafd", r)
                        target += discount_factor * np.max(
                            x)
                    #print(target,r)
                    all_targets.append(target)

            target = np.mean(all_targets)
            # print(len(all_targets))
            prev_q_value = q_values[prev_state][action]
            loss = target - prev_q_value
            # print(loss)
            q_values[prev_state][action] = target
            mean_loss.append(loss)

    return np.mean(mean_loss)

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
        self._buffer = {}
        self.batch_size = 32



    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_buffer']
        del state['_q_values']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._buffer = {}
        self._q_values = {}


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

        if(self.model is None):
            if(info_state not in self._q_values):
                greedy_q =  np.random.choice(len(legal_actions))
            else:
                q_values = self._q_values[info_state][legal_actions]
                greedy_q = np.argmax(q_values)
        else:
            q_values = self.get_model_qs(info_state, legal_actions)
            greedy_q = np.argmax(q_values)


        probs[legal_actions] = epsilon / len(legal_actions)
        probs[legal_actions[greedy_q]] += (1 - epsilon)
        # print(probs, np.sum(probs))
        action = np.random.choice(range(self._num_actions), p=probs)
        return action, probs


    def replay(self):
        cloned_q = (self._q_values)
        for _ in range(1):
            loss = init_replay(self._buffer, self._discount_factor,
                               cloned_q, self._num_actions)
            print("Replay loss", loss)
            if(loss == 0.0):
                break;
        # #print(cloned_q.values())
        # #exit()
        return cloned_q


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


            # self._buffer.add([self._prev_info_state,self._prev_action,
            #                   info_state,
            #                   legal_actions, rewards, time_step.last()])

            if (self._prev_info_state not in self._buffer):
                self._buffer[self._prev_info_state] = [[] for _ in range(self._num_actions)]

            self._buffer[self._prev_info_state][self._prev_action].append([
                info_state, legal_actions,
                time_step.rewards[self._player_id],
                time_step.last()
            ])

            if (info_state not in self._q_values):
                 self._q_values[info_state] = np.random.random(
                    size=self._num_actions) * 0.0001

            if (self._prev_info_state not in self._q_values):
                self._q_values[self._prev_info_state] = np.random.random(
                    size=self._num_actions) * 0.0001
            # sample and calculate q-values






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
        # self._q_values = {}
        # self._buffer = ReplayBuffer(self.maximum_size)


class LUGLLightGBM(DCLF):

    def get_model_qs(self, state_features, legal_actions):


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
        #for self
        q_values = self.replay()

        for state in q_values.keys():
            for action, Q in enumerate(self._q_values[state]):
                all_features.append(list(state) + [action])
                all_Qs.append(Q)

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




class DecisionTree(LUGLLightGBM):




    def get_model_qs(self, state_features, legal_actions):


        q_values = [self.model[a].predict(np.array(state_features)[np.newaxis, :])[0]
                    for a in legal_actions]
        return q_values


    def train_supervised(self):

        print("About to start training")


        data_per_action = [[] for _ in range(self._num_actions)]
        Qs_per_action = [[] for _ in range(self._num_actions)]


        # make targets!
        for state in self._q_values.keys():
            for action, Q in enumerate(self._q_values[state]):
                data_per_action[action].append(state)
                Qs_per_action[action].append(Q)

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

