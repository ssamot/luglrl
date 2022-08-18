import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from sklearn import metrics
from agents.utils import ReplayBuffer
from agents.utils import LimitedSizeDict
from collections import OrderedDict


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
            self._q_values[info_state] = np.random.random(
                size=self._num_actions) * 0.0001

            if(self.model is not None):
                self._q_values[info_state][legal_actions] = self.get_model_qs(
                    info_state, legal_actions)

        q_values = self._q_values[info_state][legal_actions]
        greedy_q = np.argmax(q_values)



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

            prev_q_value = self._q_values[self._prev_info_state][self._prev_action]
            loss = target - prev_q_value

            self._q_values[self._prev_info_state][self._prev_action] += (
                    self._step_size * loss)

            self._tbr[(self._prev_info_state, self._prev_action)] = target

            self.update()






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


    def update(self):
        if (len(self._buffer) > self.batch_size):
            for (
                    prev_info_state, prev_action, l_info_state, l_legal_actions,
                    rewards,
                    last) in self._buffer.sample(self.batch_size):
                target = rewards

                if (not last):

                    feature_qs = self._q_values[l_info_state][
                            l_legal_actions]

                    target += self._discount_factor * max(feature_qs)

                prev_q_value = self._q_values[prev_info_state][prev_action]
                loss = target - prev_q_value

                self._q_values[prev_info_state][prev_action] += (
                        self._step_size * loss)

                self._tbr[(prev_info_state, prev_action)] = target
                # print(loss, target, self._q_values[prev_info_state][prev_action])

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






        #self._q_values = {}




class LUGLBLightGBM(DCLF):

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

        for (state, action), Q in self._tbr.items():
            #for action, Q in enumerate(self._q_values[state]):
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


class LUGLBDecisionTree(DCLF):

    def get_model_qs(self, state_features, legal_actions):


        oh_action = np.zeros(shape=(len(legal_actions), self._num_actions  ))
        #oh_action[legal_actions,list(range(self._num_actions))] = 1
        #print(oh_action)
        for i, a in enumerate(legal_actions): oh_action[i,a] = 1


        feature_actions = [list(state_features) + list(a) for a in oh_action ]
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

        for (state, action), Q in self._tbr.items():
            #for action, Q in enumerate(self._q_values[state]):
                oh_action = np.zeros(shape=self._num_actions)
                oh_action[action] = 1
                all_features.append(list(state) + list(oh_action))
                all_Qs.append(Q)

        X = np.array(all_features)
        y = np.array(all_Qs)

        print(X.shape, y.shape)
        from sklearn.tree import DecisionTreeRegressor

        clf = DecisionTreeRegressor(min_samples_leaf=3)
        clf.fit(X,y)
        self.model = clf
        mse = metrics.mean_squared_error(y, self.model.predict(X))
        r2 = metrics.explained_variance_score(y, self.model.predict(X))

        print(mse, r2)
