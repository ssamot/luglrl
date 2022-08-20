import collections
import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from sklearn import metrics
from agents.luglb import LUGLBDecisionTree, LUGLBLightGBM
from agents.utils import ReplayBuffer


def child_U(all_childs):
    C = 0.01
    parent_visits = np.sum(all_childs)
    all_U = np.array([C* np.sqrt(np.log(parent_visits)/child_visits)
             for child_visits in all_childs])
    return all_U



class UCB(rl_agent.AbstractAgent):
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
        self.episode_length = 0
        self.model = None
        self.maximum_size = int(1e5)
        self.batch_size = 32
        self._reset_dict()





    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_buffer']
        del state['_q_values']
        del state['_tbr']
        del state['_N']
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._reset_dict()


    def get_state_action(self, info_state, action):
        return (tuple(info_state), tuple([action]))


    def _ucb(self, info_state, legal_actions, is_evaluation):
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

        if (info_state not in self._q_values):
            self._q_values[info_state] = np.random.random(
                size=self._num_actions) * 0.0001
            self._N[info_state] = np.ones(shape = (self._num_actions))
            #print(self._N[info_state] , "sadfasdfsdf")

            if(self.model is not None):
                self._q_values[info_state][legal_actions] = self.get_model_qs(
                    info_state, legal_actions)

        q_values = self._q_values[info_state][legal_actions]

        greedy_q = np.argmax(q_values)

        if (not is_evaluation):
            child_visits = self._N[info_state][legal_actions]
            #print(child_visits)
            all_Us = child_U(child_visits)
            #print(all_Us, "sdfsdfsdffds")
            greedy_q = np.argmax(q_values +all_Us)

        action = legal_actions[greedy_q]
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
            epsilon = 0.0 if is_evaluation else self._epsilon
            action, probs = self._ucb(
                info_state, legal_actions, is_evaluation)
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
            self._N[self._prev_info_state][self._prev_action] +=1

            self._tbr[(self._prev_info_state, self._prev_action)] = target

            #self.update()






            self._epsilon = self._epsilon_schedule.step()
            self.episode_length += 1
            if time_step.last():  # prepare for the next episode.
                self._prev_info_state = None
                #self._n_games += 1
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
                target = (rewards + 1) / 2.0

                if (not last):

                    feature_qs = self._q_values[l_info_state][
                            l_legal_actions]

                    target += self._discount_factor * max(feature_qs)

                # print(target, prev_q_value)


                prev_q_value = self._q_values[prev_info_state][prev_action]
                loss = target - prev_q_value

                self._q_values[prev_info_state][prev_action] += (
                        self._step_size * loss)

                self._tbr[(prev_info_state, prev_action)] = \
                    self._q_values[prev_info_state][prev_action]
                # print(loss, target, self._q_values[prev_info_state][prev_action])

    @property
    def loss(self):
        return self._last_loss_value


    def _reset_dict(self):
        #pass
        self._q_values = {}
        self._buffer = ReplayBuffer(self.maximum_size)
        self._tbr = {}
        self._N = {}


class LUGLUCBLightGBM(UCB):
    get_model_qs = LUGLBLightGBM.get_model_qs
    train_supervised = LUGLBLightGBM.train_supervised


class LUGLUCBDecisionTree(UCB):
    get_model_qs = LUGLBDecisionTree.get_model_qs
    train_supervised = LUGLBDecisionTree.train_supervised

