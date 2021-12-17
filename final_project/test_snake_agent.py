from agent import SnakeAgent
from snake import Snake
import pandas as pd
import matplotlib.pyplot as plt
import turtle
import numpy as np

# plots the training reward curves for each state space representation
def plot_training_reward_curves(df_base_state, df_no_body_knowledge, df_coordinates, df_no_direction):
    episodes = [i + 1 for i in range(len(df_base_state))]
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, df_base_state['10 episode average reward'], label='base state')
    plt.plot(episodes, df_no_body_knowledge['10 episode average reward'], label='no body knowledge')
    plt.plot(episodes, df_coordinates['10 episode average reward'], label='coordinate representation')
    plt.plot(episodes, df_no_direction['10 episode average reward'], label='no direction')
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.title('10 Episode Average Training Reward')
    plt.legend()
    plt.show()
# plots the training score curves for each state space representation
def plot_training_score_curves(df_base_state, df_no_body_knowledge, df_coordinates, df_no_direction):
    episodes = [i + 1 for i in range(len(df_base_state))]
    plt.figure(figsize=(12, 8))
    plt.plot(episodes, df_base_state['10 episode average score'], label='base state')
    plt.plot(episodes, df_no_body_knowledge['10 episode average score'], label='no body knowledge')
    plt.plot(episodes, df_coordinates['10 episode average score'], label='coordinate representation')
    plt.plot(episodes, df_no_direction['10 episode average score'], label='no direction')
    plt.ylabel('score')
    plt.xlabel('episode')
    plt.legend()
    plt.title('10 Episode Average Training Score')
    plt.show()

# plots a single traning reward curve for a specific state space representation
def plot_single_training_reward_curve(state_type):
    df = pd.read_csv("../rewards/rewards_{}.csv".format(state_type))
    df[['reward', 'average reward', '10 episode average reward']].plot()
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.title('{} reward training curve'.format(state_type))
    plt.legend()
    plt.show()

# plots a single traning score curve for a specific state space representation
def plot_single_training_score_curve(state_type):
    df = pd.read_csv("../rewards/rewards_{}.csv".format(state_type))
    df[['score', 'average score', '10 episode average score']].plot()
    plt.ylabel('score')
    plt.xlabel('episode')
    plt.title('{} score training curve'.format(state_type))
    plt.legend()
    plt.show()

# plots the 10 episode testing scores of each state space representation
def plot_testing_scores(test_results, episodes):
    x = [i + 1 for i in range(episodes)]
    for key in test_results:
        plt.plot(x, test_results[key][1], label=key)
        plt.ylabel('score')
        plt.xlabel('episode')
        plt.legend()
    plt.show()

# plots the 10 episode testing rewards of each state space representation
def plot_testing_rewards(test_results, episodes):
    x = [i + 1 for i in range(episodes)]
    for key in test_results:
        plt.plot(x, test_results[key][0], label=key)
        plt.ylabel('cumulative reward')
        plt.xlabel('episode')
        plt.legend()
    plt.show()

if __name__ == '__main__':
    params = dict()
    params['name'] = None
    params['epsilon'] = 0
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = 0
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [128, 128, 128]

    testing_results = {}

    env_infos = {'State Representation: base state': {'state_space': 'base_state'},
                 'State Representation: only walls': {'state_space': 'no_body_knowledge'},
                 'State Representation: coordinates': {'state_space': 'coordinates'},
                 'State Representation: no direction': {'state_space': 'no_direction'}}

    for key in env_infos:
        params['name'] = key
        env_info = env_infos[key]
        env = Snake(env_info=env_info)
        agent = SnakeAgent(env, params)
        testing_results[env_info['state_space']] = agent.test(episodes=10)
        turtle.Screen().clear()

    plot_testing_scores(testing_results, 10)

    print('AVG score obtained using base state representation is', np.mean(testing_results['base_state'][1]))
    print('AVG score obtained using no body knowledge representation is', np.mean(testing_results['no_body_knowledge'][1]))
    print('AVG score obtained using coordinates representation is', np.mean(testing_results['coordinates'][1]))
    print('AVG score obtained using no direction representation is', np.mean(testing_results['no_direction'][1]))
    print()

    # getting training data from csv files
    df_base_state = pd.read_csv("../rewards/rewards_base_state.csv")
    df_no_body_knowledge = pd.read_csv("../rewards/rewards_no_body_knowledge.csv")
    df_coordinates = pd.read_csv("../rewards/rewards_coordinates.csv")
    df_no_direction = pd.read_csv("../rewards/rewards_no_direction.csv")

    # plotting training rewards
    plot_training_reward_curves(df_base_state, df_no_body_knowledge, df_coordinates, df_no_direction)
    plot_training_score_curves(df_base_state, df_no_body_knowledge, df_coordinates, df_no_direction)
    print('Max score obtained using base state representation is', max(df_base_state['score']))
    print('Max score obtained using no body knowledge representation is', max(df_no_body_knowledge['score']))
    print('Max score obtained using coordinates representation is', max(df_coordinates['score']))
    print('Max score obtained using no direction representation is', max(df_no_direction['score']))

    # individual training score plots
    for key in env_infos:
        state_type = env_infos[key]['state_space']
        plot_single_training_reward_curve(state_type)


    # individual score reward plots
    for key in env_infos:
        state_type = env_infos[key]['state_space']
        plot_single_training_score_curve(state_type)
