from test_algos import CognitionAlgorithms
import numpy as np
import csv


def refresh_Q_table(lxy_old, dia_old, lxy_action, dia_action, lxy_new, dia_new, reward_history, sample_count):

    current_state = [lxy_old, dia_old]
    next_state = [lxy_new, dia_new]
    action = [lxy_action, dia_action]

    # update Q-table
    algo = CognitionAlgorithms()
    Q_table, reward = algo.update_Q_table(current_state, action, next_state, reward_history)

    # dump file for debugging: csv file
    csv_file = open('agent_history.csv', mode='a')
    csv_file.write(f'final: t#{sample_count}, {current_state}, {action}, {next_state}, {reward} \n')

    return None


if __name__ == "__main__":
    reward_history = [99.1216]
    lxy_old, dia_old = 50, 43
    lxy_new, dia_new = 50, 43

    for i in range(5):
        sample_count = i+1
        algo = CognitionAlgorithms()
        next_state, action = algo.Q_learning_vanilla(reward_history, lxy_old, dia_old)
        lxy_new, dia_new = next_state
        lxy_action, dia_action = action
        refresh_Q_table(lxy_old, dia_old, lxy_action, dia_action, lxy_new, dia_new, reward_history, sample_count)
        reward_history.append(100 - np.random.rand())
        lxy_old, dia_old = lxy_new, dia_new
