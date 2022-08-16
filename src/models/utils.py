import numpy as np

def evaluate(env, agent_1, agent_2, num_episodes, epsilon = 0.0):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    #wins = np.zeros(2)
    wins = [[],[]]
    for player_pos in range(2):
        if player_pos == 0:
            cur_agents = [agent_1[0], agent_2[1]]
        else:
            cur_agents = [agent_2[0], agent_1[1]]
        for _ in range(num_episodes):
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                legal_actions = time_step.observations["legal_actions"][player_id]

                cur_agents[player_id].state = env.get_state
                agent_output = cur_agents[player_id].step(time_step,
                                                          is_evaluation=True)
                action = agent_output.action
                #print(legal_actions, agent_output.action)
                if(np.random.random() < epsilon):
                    action = np.random.choice(legal_actions)


                time_step = env.step([action])

            wins[player_pos].append(time_step.rewards[player_pos])

    return wins


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
