from NaRLA import *
import numpy as np
from ursina import *
import random, sys, time, os, pickle, math, json, gc
from collections import Counter, deque

default_args = {'num_layers': 1,
                'num_neurons': 20,
                'update_type': 'sync',
                'reward_type': 'all'}

ep_reward = 0.0
state_size = 10
action_size = 4
args = default_args
network = NaRLA(args, state_size, num_outputs=action_size)
network.print_layers()
state_dicts = network.capture_model_state()
with open("state_dict.pkl", "wb") as f:
    f.write(pickle.dumps(state_dicts))
new_state_dicts = {}
with open("state_dict.pkl", "rb") as f:
    new_state_dicts = pickle.load(f)
network.set_model_state(new_state_dicts)

for step in range(10):
    state = np.random.uniform(size=(state_size))
    print(state)
    action = network.forward(state)
    print(action)
    reward = np.random.uniform(-1, 1)
    ep_reward += reward
    network.distribute_task_reward(reward)

network.end_episode(ep_reward)
