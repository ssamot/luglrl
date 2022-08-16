
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from open_spiel.python import rl_agent


class RandomAgent(rl_agent.AbstractAgent):
  """Random agent class."""

  def __init__(self, player_id, num_actions, name="random_agent"):
    assert num_actions > 0
    self._player_id = player_id
    self._num_actions = num_actions

  def step(self, df, is_evaluation=False):
    time_step = df[0]
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    # Pick a random legal action.
    cur_legal_actions = time_step.observations["legal_actions"][self._player_id]
    action = np.random.choice(cur_legal_actions)
    probs = np.zeros(self._num_actions)
    probs[cur_legal_actions] = 1.0 / len(cur_legal_actions)

    return rl_agent.StepOutput(action=action, probs=probs)
