import random

class ReplayBuffer(object):
  """ReplayBuffer of fixed size with a FIFO replacement policy.

  Stored transitions can be sampled uniformly.

  The underlying datastructure is a ring buffer, allowing 0(1) adding and
  sampling.
  """

  def __init__(self, replay_buffer_capacity):
    self._replay_buffer_capacity = replay_buffer_capacity
    self._data = []
    self._next_entry_index = 0

  def add(self, element):
    """Adds `element` to the buffer.

    If the buffer is full, the oldest element will be replaced.

    Args:
      element: data to be added to the buffer.
    """
    if len(self._data) < self._replay_buffer_capacity:
      self._data.append(element)
    else:
      self._data[self._next_entry_index] = element
      self._next_entry_index += 1
      self._next_entry_index %= self._replay_buffer_capacity

  def sample(self, num_samples):
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.

    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    if len(self._data) < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(
          num_samples, len(self._data)))
    return random.sample(self._data, num_samples)

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)
