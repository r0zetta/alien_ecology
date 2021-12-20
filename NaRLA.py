import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from copy import deepcopy
from torch.distributions import Categorical

CUDA = 0
default_args = {'reward_type': 'task', 'env': 'cartpole'}

class Layer:
    def __init__(self, args, in_shape=32, out_shape=32, ID=0):
        self.args      = args
        self.ID        = ID         # LAYER ID
        self.in_shape  = in_shape   # NUMBER OF INCOMING SIGNALS
        self.out_shape = out_shape  # NUMBER OF OUTGOING SIGNALS - ALSO NUM NEURONS
        self.neurons   = []
        self.loss      = []
        self.fig       = None
        self.rewards   = {'task'       : [],
                          'sparsity'   : [],
                          'firing'     : [],
                          'trace'      : []}
        # LAST LAYER HAS NO PREDICTION REWARD
        if self.ID != self.args['num_layers']:
            self.rewards['prediction'] = []

        self._build_layer()

    def _get_all_rewards(self):
        all_rewards = np.array([ v for v in self.rewards.itervalues() ])
        return np.mean(all_rewards, axis=0)

    def store(self, done):
        for neuron in self.neurons:
            neuron.store(done)

    def end_episode(self):
        for neuron in self.neurons:
            neuron.end_episode()

    def _get_neuron_activity(self):
        zeros = []; ones = []
        for neuron in self.neurons:
            zeros.append((len(neuron.old_actions) - sum(neuron.old_actions))/ float(len(neuron.old_actions)))
            ones.append(sum(neuron.old_actions) / float(len(neuron.old_actions)))
        return zeros, ones

    def _build_layer(self):
        # ALL NEURONS ARE BINARY EXCEPT OUTPUT
        num_outputs = self.args['num_outputs'] if self.ID == self.args['num_layers'] else 2

        for ID in range(self.out_shape):
            neuron = PG(args=self.args,
                        in_shape=self.in_shape,
                        ID=ID,
                        num_outputs=num_outputs)

            self.neurons.append(neuron)

    def forward(self, X):
        output = []
        for i, neuron in enumerate(self.neurons):
            output.append(neuron.act(X[:,i]))

        return torch.Tensor(output).unsqueeze(0)

    def reward_neurons(self, R, type):
        # SEND SCALAR TASK REWARD TO ALL NEURONS
        if type == 'task':
            R = [R] * len(self.neurons)

        # SEND REWARDS BASED ON ACTIVITY TO NEURONS
        for r,neuron in zip(R, self.neurons):
            # if type == 'firing': r = 0.
            neuron.store_reward(r, type)

    def learn(self, loss=[]):
        rewards = {
            'task'       : [],
            'sparsity'   : [],
            'firing'     : [],
            'prediction' : [],
            'trace'      : [],
        }

        for neuron in self.neurons:
            for k in self.rewards:
                rewards[k].append(neuron.rewards[k][-1])
            # ONLY NEURONS IN THE LAST LAYER WILL LEARN
            if self.args['train_last'] and self.ID != self.args['num_layers']: continue

            # STORE LOSS FROM TRAINING
            loss.append(neuron.learn())

        # AVERAGE LAYER LOSS
        self.loss.append(np.mean(loss))
        # AVERAGE REWARDS OVER ALL NEURONS
        for k in self.rewards:
            self.rewards[k].append(np.mean(rewards[k]))

class NaRLA:
    def __init__(self, args, input_space, num_outputs=2, sparsity=0.2):
        self.args            = args
        self.layers          = []
        self.neurons         = []
        self.sparsity_goal   = sparsity
        self.connections     = []
        self.episode_rewards = []
        self.fig             = None
        self.bio_then_all    = False

        #if len(input_space) == 1:
            #layer_descr = [input_space[0], 1]
        layer_descr = [input_space, 1]

        if self.args['reward_type'] == 'bio_then_all':
            self.bio_then_all = True

        self.args['num_outputs'] = num_outputs

        # ADD NEURONS FOR EACH ADDITIONAL LAYER
        for i in range(args['num_layers']):
            layer_descr.insert(1, self.args['num_neurons'])

        self._build_network(layer_descr)

    def end_episode(self, R):
        # STORE REWARD ACHIEVED IN EPISODE
        self.episode_rewards.append(R)

        # END EPISODE FOR LAYERS
        for layer in self.layers:
            layer.end_episode()

    def _build_network(self, layer_descr):
        #print(layer_descr)
        for i in range(len(layer_descr) - 1):
            # FIRST LAYER IS FULLY CONNECTED
            if i == 0:
                connections = torch.ones(layer_descr[i], layer_descr[i+1])
            # LATER LAYERS HAVE LOCAL CONNECTIVITY
            else:
                connections = torch.zeros(layer_descr[i], layer_descr[i+1])

                # SET COL CORRESPONDING TO NEURONs INPUT CONNECTIVITY
                if layer_descr[i+1] == 1:
                    connections[:,:] = 1.
                else:
                    w = layer_descr[i+1] / float(layer_descr[i])
                    info = np.ceil(np.arange(0,layer_descr[i+1], w)).astype(np.int)
                    for r,c in enumerate(info):
                        if r + 1 == len(info):
                            connections[r,c:] = 1.
                        else:
                            connections[r, c:info[r+1]+1] = 1.

            self.connections.append(connections)

            # CREATE NEW LAYER
            self.layers.append(Layer(args=self.args,
                                     in_shape=layer_descr[i],
                                     out_shape=layer_descr[i+1],
                                     ID=i))

            # RECORD ALL NEURONS IN NETWORK
            self.neurons.extend(self.layers[-1].neurons)

    def set_model_state(self, states):
        for li, layer in enumerate(self.layers):
            for ni, pg in enumerate(layer.neurons):
                pg.neuron.load_state_dict(states[layer.ID][ni])

    def capture_model_state(self):
        states = {}
        for li, layer in enumerate(self.layers):
            states[layer.ID] = {}
            for ni, pg in enumerate(layer.neurons):
                states[layer.ID][ni] = pg.neuron.state_dict()
        return states

    def print_layers(self):
        for li, layer in enumerate(self.layers):
            print("Layer: " + str(layer.ID))
            for ni, pg in enumerate(layer.neurons):
                print("Neuron: " + str(ni))
                pg.neuron.print_w()

    def store(self, done):
        for layer in self.layers:
            layer.store(done)

    def distribute_task_reward(self, R):
        # TASK LEVEL REWARD DISTRIBUTED ACROSS NETWORK
        for layer in self.layers:
            layer.reward_neurons(R, type='task')

    def learn(self):
        # SAMPLE ONE NEURON TO LEARN
        if self.args['update_type'] == 'async':
            neuron = sample(self.neurons, 1)[0]
            neuron.learn()

        # EACH LAYER IN THE NETWORK LEARNS
        else:
            for layer in self.layers:
                 layer.learn()

    def forward(self, X):
        X = torch.Tensor(X).unsqueeze(0)

        for i, layer in enumerate(self.layers):
            # BROADCAST INPUT ACROSS NEURON CONNECTIONS
            X = X.t() * self.connections[i]

            # PASS INPUT THROUGH NEURONS IN LAYER
            O = layer.forward(X)

            # CAN'T REWARD THE ORIGINAL INPUT
            if i != 0:
                output = O.clone()
                output[output == 0] = -1.

                prediction_reward   = (X * output).sum(dim=-1)

                # REWARD PREVIOUS LAYER FOR PREDICTING CORRECTLY
                self.layers[i-1].reward_neurons(prediction_reward, type='prediction')

            # DON'T REWARD THE LAST LAYER - SINGLE NEURON
            if i == self.args['num_layers']:
                layer.reward_neurons([int(O.squeeze())], type='firing')
                layer.reward_neurons([0.], type='sparsity')
            else:
                # REWARD LAYER FOR SPARSITY
                sparsity = O.sum() / float(len(O.reshape(-1)))
                sparsity_error = sparsity - self.sparsity_goal

                output = O.clone().reshape(-1)
                output[output == 0] = -1.

                if sparsity_error > 0:
                    sparsity_rewards = -sparsity_error * output
                elif sparsity_error < 0:
                    sparsity_rewards =  sparsity_error * output
                elif sparsity_error == 0:
                    sparsity_rewards = output.abs() * self.sparsity_goal

                layer.reward_neurons(sparsity_rewards, type='sparsity')

                # REWARD LAYER FOR FIRING
                layer.reward_neurons(O.reshape(-1), type='firing')

            # OUTPUT OF PREV LAYER IS INPUT TO NEXT
            X = O

        # LAST LAYER CAN'T HAVE A PREDICTION REWARD
        self.layers[-1].reward_neurons([0.] * self.layers[-1].out_shape, type='prediction')
        return int(O.squeeze())


class Agent(nn.Module):
    def __init__(self, in_shape, softmax=True, num_layers=2, num_outputs=2):
        super(Agent, self).__init__()

        layers = []
        out_shape = 16
        self.neuron_desc = []
        for i in range(num_layers):
            if i + 1 == num_layers:
                out_shape = num_outputs

            layers.append(nn.Linear(in_shape, out_shape))
            self.neuron_desc.append([in_shape, out_shape])

            # NOT LAST LAYER ADD ACTIVATION FUNCTION
            if i + 1 != num_layers:
                layers.append(nn.LeakyReLU(.3))
            in_shape = out_shape

        if softmax:
            layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def print_w(self):
        for i, l in enumerate(self.neuron_desc):
            print("layer"+str(i)+" weights: ", l)
        print()


class Neuron:
    def __init__(self):
        pass

    def _process_state(self, state):
        # MAKING SURE SHAPES AND TENSORS MATCH UP
        if state.dtype == np.int or state.dtype == np.float:
            state = torch.from_numpy(state).float()
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        if CUDA: return state.cuda()
        return state

    def store_reward(self, r, type):
        self.episode_rewards[type].append(r)
        if type == 'firing':
            # CALCULATE TRACE BASED ON PREVIOUS FIRING
            x = abs(np.mean(self.episode_rewards['firing']) - .5)
            R = -1746463 + (0.9256812 - -1746463)/(1 + (x/236.1357)**2.160334)
            self.episode_rewards['trace'].append(R)

    def _reset_episode_storage(self):
        if hasattr(self, 'actions'):
            if self.actions != []:
                self.old_actions = self.actions
        self.states = []
        self.actions = []
        self.episode_rewards = {'task'       : [],
                                'sparsity'   : [],
                                'firing'     : [],
                                'prediction' : [],
                                'trace'      : []}

    def _reset_reward_storage(self):
        self.rewards = {'task'       : [],
                        'sparsity'   : [],
                        'firing'     : [],
                        'prediction' : [],
                        'trace'      : []}
        self._reset_episode_storage()
        self.returns = []

    def learn(self):
        return 0

    def store(self, done):
        pass

    def calculate_reward(self, idx):
        # SUM REWARDS FROM DIFFERENT FEATURES
        if self.args['reward_type'] == 'all':
            reward = sum(v[idx] for v in self.episode_rewards.values())
        elif self.args['reward_type'] == 'bio' or self.args['reward_type'] == 'bio_then_all':
            reward = sum(self.episode_rewards[k][idx] for k in ['prediction', 'sparsity', 'firing', 'trace'])
        else:
            reward = self.episode_rewards['task'][idx]
        return reward

    # SAVE METRICS FROM THE EPISODE
    def end_episode(self):
        for k, v in self.episode_rewards.items():
            self.rewards[k].append(np.mean(v))
        self._reset_episode_storage()

class PG(Neuron):
    def __init__(self, in_shape, num_outputs=2, ID=0, args=default_args):
        self.ID              = ID
        self.args            = args
        self.saved_log_probs = []
        self.gamma           = 0.99

        self.neuron          = Agent(in_shape, num_outputs=num_outputs)
        if CUDA: self.neuron.cuda()
        self.opt = optim.Adam(self.neuron.parameters(), lr=1e-2)

        self._reset_reward_storage()

    def act(self, state):
        # CONVERT TO TENSOR - CHECK DIMS
        state = self._process_state(state)

        # ACTION PROBABILITIES
        probs    = self.neuron(state)
        cat_dist = Categorical(probs)

        # SAMPLE ACTION FROM DISTRIBUTION
        action   = cat_dist.sample()
        # STORE ACTION
        self.actions.append(action.detach().item())
        # STORE LOG PROBS
        self.saved_log_probs.append(cat_dist.log_prob(action))
        return action.detach().item()

    def learn(self):
        policy_loss = []

        for log_prob, R in zip(self.saved_log_probs, self.returns):
            # WEIGHT LOG PROBABILITIES BY DISCOUNTED REWARD
            policy_loss.append(-log_prob * R)

        self.opt.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward(retain_graph=False)
        self.opt.step()

        del(self.returns)
        self.returns = []
        del(self.saved_log_probs)
        self.saved_log_probs = []
        # self._reset_reward_storage()

        return policy_loss.detach().item()

    def _discount_rewards(self):
        R       = 0
        returns = []
        eps     = np.finfo(np.float32).eps.item()

        # DISCOUNTING REWARDS BY GAMMA
        for r in self._iter_rewards():
            R = r + self.gamma * R
            returns.insert(0, R)

        # STANDARDIZE REWARDS
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        return returns

    def _iter_rewards(self):
        for i in reversed(range(len(self.episode_rewards['task']))):
            # SUM REWARDS FROM DIFFERENT FEATURES
            yield self.calculate_reward(idx=i)

    def end_episode(self):
        # DISCOUNT REWARDS FOR THE EPISODE
        self.returns.extend(self._discount_rewards())

        # SAVE METRICS FROM THE EPISODE
        for k,v in self.episode_rewards.items():
            # self.rewards[k].extend(v)
            self.rewards[k].append(np.mean(v))

        # RESET EPISODE STORAGE
        self._reset_episode_storage()
