from snake import Snake
from plot_script import plot_result
import turtle
from agent import SnakeAgent
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    params = dict()
    params['name'] = None
    params['epsilon'] = 1
    params['gamma'] = .95
    params['batch_size'] = 500
    params['epsilon_min'] = .01
    params['epsilon_decay'] = .995
    params['learning_rate'] = 0.00025
    params['layer_sizes'] = [128, 128, 128]

    results = dict()
    env_infos = {'State Representation: only walls': {'state_space': 'no_body_knowledge'},
                 'State Representation: base state': {'state_space': 'base_state'},
                 'State Representation: coordinates': {'state_space': 'coordinates'},
                 'State Representation: no direction': {'state_space': 'no_direction'}}

    # train our agent over each different state representation
    for key in env_infos.keys():
        params['name'] = key
        env_info = env_infos[key]
        print(env_info)
        env = Snake(env_info=env_info)
        agent = SnakeAgent(env, params)
        results['name'] = agent.train(episodes=150)
        turtle.Screen().clear()