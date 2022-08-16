from open_spiel.python.algorithms import dqn

import os

n_threads = 6
# reduce number of threads
os.environ['TF_NUM_INTEROP_THREADS'] = f"{n_threads}"
os.environ['TF_NUM_INTRAOP_THREADS'] = f"{n_threads}"

import tensorflow.compat.v1 as tf
import os

class DQN():
    def __init__(self, player_id,
                 state_representation_size,  ## ignored
                 num_actions,):



        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=n_threads,
            inter_op_parallelism_threads=n_threads)

        self.graph = tf.Graph()
        self.session = tf.Session(config=session_conf, graph = self.graph)



        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        with self.graph.as_default():
            self.agent = dqn.DQN(
                session=self.session,
                player_id=player_id,
                state_representation_size=state_representation_size,
                num_actions=num_actions,
                hidden_layers_sizes=[3*state_representation_size,3*state_representation_size],
                replay_buffer_capacity=int(1e5),
                batch_size=32)
            self.agent._savers = [
                ("q_network", tf.train.Saver(self.agent._q_network.variables, max_to_keep=None)),
                ("target_q_network",
                 tf.train.Saver(self.agent._target_q_network.variables, max_to_keep=None))]

    def fullname(self):
        klass = self.__class__
        module = klass.__module__
        if module == 'builtins':
            return klass.__qualname__  # avoid outputs like 'builtins.str'
        return module + '.' + klass.__qualname__

    def step(self, time_step, is_evaluation=False):
        return self.agent.step(time_step, is_evaluation)

    def save(self, checkpoint_dir):
        """Saves the q network and the target q-network.

        Note that this does not save the experience replay buffers and should
        only be used to restore the agent's policy, not resume training.

        Args:
          checkpoint_dir: directory where checkpoints will be saved.
        """
        id_checkpoint_dir = checkpoint_dir + f"/{self.agent.player_id}"
        classname_file = checkpoint_dir + "/classname.txt"
        try:
            if(not os.path.isdir(checkpoint_dir)):
                os.remove(checkpoint_dir)
                os.mkdir(checkpoint_dir)
                os.mkdir(id_checkpoint_dir)

        except FileExistsError:
            print("Directory exists, ignoring mkdir...")

        file = open(classname_file, 'w')
        file.write(self.fullname())
        file.close()

        for name, saver in self.agent._savers:

            path = saver.save(
                self.agent._session,
                self.agent._full_checkpoint_name(id_checkpoint_dir, name),
                latest_filename=self.agent._latest_checkpoint_filename(name), )
            print("Saved to path: %s", path)


    def restore(self, checkpoint_dir):
        """Restores the q network and the target q-network.

        Note that this does not restore the experience replay buffers and should
        only be used to restore the agent's policy, not resume training.

        Args:
          checkpoint_dir: directory from which checkpoints will be restored.
        """
        for name, saver in self.agent._savers:
            full_checkpoint_dir = self.agent._full_checkpoint_name(checkpoint_dir,
                                                             name)
            print("Restoring checkpoint: %s", full_checkpoint_dir)
            saver.restore(self.agent._session, full_checkpoint_dir)



