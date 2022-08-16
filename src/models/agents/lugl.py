import numpy as np

from open_spiel.python import rl_agent
from open_spiel.python import rl_tools
from sklearn import metrics
from agents.utils import ReplayBuffer
from agents.utils import LimitedSizeDict
from collections import OrderedDict
from agents.luglb import LUGLBDecisionTree, LUGLBLightGBM, DCLF

class DCLFS(DCLF):


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




class LUGLLightGBM(DCLFS):
    get_model_qs = LUGLBLightGBM.get_model_qs
    train_supervised = LUGLBLightGBM.train_supervised


class LUGLDecisionTree(DCLFS):
    get_model_qs = LUGLBDecisionTree.get_model_qs
    train_supervised = LUGLBDecisionTree.train_supervised

