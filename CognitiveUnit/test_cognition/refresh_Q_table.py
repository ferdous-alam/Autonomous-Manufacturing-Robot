from test_algos import *
import numpy as np
import csv


def refresh_Q_table(lxy_old, dia_old, lxy_action, dia_action, lxy_new, dia_new, reward_history, sample_count):

    current_state = [lxy_old, dia_old]
    next_state = [lxy_new, dia_new]
    action = [lxy_action, dia_action]
    action_space = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                         [-1, 1], [-1, -1], [1, -1], [0, 0]]
    # update Q-table
    Q_table, reward = update_Q_table(current_state, action, next_state, reward_history)

    # dump file for debugging: csv file
    csv_file = open('agent_history.csv', mode='a')
    csv_file.write(f'final: t#{sample_count}, {current_state}, {action}, {next_state}, {reward} \n')

    return None


if __name__ == "__main__":
    reward_history = [99.1216]
    lxy_old, dia_old = 50, 43
    lxy_new, dia_new = 50, 43
    action_space = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                         [-1, 1], [-1, -1], [1, -1], [0, 0]]

    for i in range(15):
        sample_count = i+1
        current_state = [lxy_old, dia_old]
        next_state, action = implement_policy(reward_history, current_state)
        lxy_new, dia_new = next_state
        lxy_action, dia_action = action
        action_idx = action_space.index(action)

        Q_table_old = np.load('Q_table.npy')
        print(f'Q_old: {Q_table_old[lxy_old, dia_old, action_idx]}')

        # update Q table
        refresh_Q_table(lxy_old, dia_old, lxy_action, dia_action, lxy_new, dia_new, reward_history, sample_count)
        reward_history.append(100 - np.random.rand())

        Q_table_new = np.load('Q_table.npy')
        print(f'Q_new: {Q_table_new[lxy_old, dia_old, action_idx]}')
        print('------------------------')
        lxy_old, dia_old = lxy_new, dia_new
