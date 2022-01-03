from ursina import *
import random, sys, time, os, pickle, math, json
import numpy as np
from collections import Counter, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Net(nn.Module):
    def __init__(self, weights, l):
        super(Net, self).__init__()
        self.l = l
        self.weights = weights
        self.inp_blocks = {}
        for entry in self.weights:
            for item in entry:
                if len(item) > 3:
                    block = item[0]
                    depth = item[1]
                    order = item[2]
                    s1 = item[3]
                    s2 = item[4]
                    w = item[5]
                    if block not in self.inp_blocks:
                        self.inp_blocks[block] = {}
                    if depth not in self.inp_blocks[block]:
                        self.inp_blocks[block][depth] = {}
                    if order not in self.inp_blocks[block][depth]:
                        self.inp_blocks[block][depth][order] = []
                    self.inp_blocks[block][depth][order].append([s1, s2, w])
        self.block_inputs = []
        for entry in self.weights:
            if len(entry[0]) > 3:
                self.block_inputs.append(entry[0][3])
        #print("block inputs", self.block_inputs)
        self.num_actions = self.weights[-2][0][1]
        #print("actions", self.num_actions)
        self.num_blocks = len(self.inp_blocks)
        #print("num blocks", self.num_blocks)
        self.prev_states = len(self.inp_blocks[0])
        #print("prev states", self.prev_states)
        self.out_cat = self.num_blocks * self.num_actions
        #print('out_cat', self.out_cat)
        self.out_hidden = self.weights[-3][0][1]
        #print('out_hidden', self.out_hidden)
        self.neurons = nn.ModuleList()
        for bi, block in self.inp_blocks.items():
            for di, dblock in block.items():
                for oi, oblock in dblock.items():
                    for item in oblock:
                        s1 = item[0]
                        s2 = item[1]
                        weights = item[2]
                        fc = nn.Linear(s1, s2, bias=False)
                        fc.weight.data = weights
                        self.neurons.append(fc)
        self.cat_hidden = nn.Linear(self.out_cat, self.out_hidden, bias=False)
        self.cat_hidden.weight.data = self.weights[-3][0][2]
        self.action = nn.Linear(self.out_hidden, self.num_actions, bias=False)
        self.action.weight.data = self.weights[-2][0][2]
        self.value = nn.Linear(self.out_hidden, 1, bias=False)
        self.value.weight.data = self.weights[-1][0][2]

# num_prev_states implemented as follows:
# each prev_state is supplied such that blocks are merged with another block
# state1 state2 state3   state4 state5 state6   state7  state8  state9
# block1 block2 block3   block7 block8 block9   block13 block14 block15
#    block4  block5         block10 block11         block17  block18
#        block6                 block12                  block19
#        out_cat                out_cat                  out_cat

    def forward(self, x):
        block_out = torch.empty((self.num_blocks, self.num_actions))
        neuron_index = 0
        input_index = 0
        for bi, block in self.inp_blocks.items():
            binps = self.block_inputs[bi]
            i1 = input_index
            i2 = input_index + (self.block_inputs[bi]*self.prev_states)
            x1 = x[0][i1:i2]
            prev_outs = []
            out_index = 0
            last_out = None
            for di, dblock in block.items():
                if di == 0:
                    for oi, oblock in dblock.items():
                        inp = x1[oi*binps:(oi*binps)+binps]
                        out = F.tanh(self.neurons[neuron_index](inp))
                        neuron_index += 1
                        out = F.tanh(self.neurons[neuron_index](out))
                        last_out = out
                        prev_outs.append(out)
                        neuron_index += 1
                else:
                    for oi, oblock in dblock.items():
                        inp = torch.cat((prev_outs[out_index], prev_outs[out_index+1]))
                        out_index += 1
                        out = F.tanh(self.neurons[neuron_index](inp))
                        neuron_index += 1
                        out = F.tanh(self.neurons[neuron_index](out))
                        last_out = out
                        prev_outs.append(out)
                        neuron_index += 1
            block_out[bi] = last_out
            input_index += self.block_inputs[bi]*self.prev_states
        rc = torch.ravel(torch.tensor(block_out))
        rc = F.tanh(self.cat_hidden(rc))
        a = F.softmax(F.relu(self.action(rc)), dim=-1)
        v = F.relu(self.value(rc))
        return a, v

    def get_param_count(self, item):
        count = 1
        for c in item.shape:
            count = count * c
        return count

    def print_w(self):
        total_params = 0
        genome = []
        neuron_index = 0
        while neuron_index < len(self.neurons):
            entry = []
            print("Block: " + str(index))
            d1 = self.neurons[neuron_index].weight.data.detach().numpy()
            neuron_index += 1
            print("fc0 weights: ", d1.shape)
            total_params += self.get_param_count(d1)
            d1 = np.ravel(d1)
            entry.extend(list(d1))
            d2 = self.neurons[neuron_index].weight.data.detach().numpy()
            neuron_index += 1
            print("fc1 weights: ", d2.shape)
            d2 = np.ravel(d2)
            entry.extend(list(d2))
            total_params += self.get_param_count(d2)
            entry = np.ravel(entry)
            genome.append(entry)
        dc = self.cat_hidden.weight.data.detach().numpy()
        genome.append(np.ravel(dc))
        total_params += self.get_param_count(dc)
        print("cat_hidden weights: ", dc.shape)
        print(self.weights[-3][0][2].shape)
        da = self.action.weight.data.detach().numpy()
        genome.append(np.ravel(da))
        total_params += self.get_param_count(da)
        print("action weights: ", da.shape)
        print(self.weights[-2][0][2].shape)
        dv = self.value.weight.data.detach().numpy()
        genome.append(np.ravel(dv))
        total_params += self.get_param_count(dv)
        print("value weights: ", dv.shape)
        print(self.weights[-1][0][2].shape)
        print("total params: ", total_params)
        genome_shape = [len(x) for x in genome]
        print(genome_shape)
        print()

    def get_w(self):
        genome = []
        neuron_index = 0
        while neuron_index < len(self.neurons):
            entry = []
            d1 = self.neurons[neuron_index].weight.data.detach().numpy()
            neuron_index += 1
            d1 = np.ravel(d1)
            entry.extend(list(d1))
            d2 = self.neurons[neuron_index].weight.data.detach().numpy()
            neuron_index += 1
            d2 = np.ravel(d2)
            entry.extend(list(d2))
            entry = np.ravel(entry)
            genome.append(entry)
        dc = self.cat_hidden.weight.data.detach().numpy()
        genome.append(np.ravel(dc))
        da = self.action.weight.data.detach().numpy()
        genome.append(np.ravel(da))
        dv = self.value.weight.data.detach().numpy()
        genome.append(np.ravel(dv))
        return genome

    def clamp_w(self):
        for index in range(len(self.neurons)):
            self.neurons[index].weight.data = torch.clamp(self.neurons[index].weight.data, min=-1.0, max=1.0)
        self.cat_hidden.weight.data = torch.clamp(self.action.weight.data, min=-1.0, max=1.0)
        self.action.weight.data = torch.clamp(self.action.weight.data, min=-1.0, max=1.0)
        self.value.weight.data = torch.clamp(self.value.weight.data, min=-1.0, max=1.0)

    def set_w(self, w):
        neuron_index = 0
        for bi, block in self.inp_blocks.items():
            for di, dblock in block.items():
                for oi, oblock in dblock.items():
                    for item in oblock:
                        weights = item[2]
                        self.neurons[neuron_index].weight.data = weights
                        neuron_index += 1
        self.cat_hidden.weight.data = self.weights[-3][0][2]
        self.action.weight.data = self.weights[-2][0][2]
        self.value.weight.data = self.weights[-1][0][2]

    def get_action(self, state):
        probs, value = self.forward(state)
        if self.l == True:
            m = Categorical(probs)
            action = m.sample()
            prob = m.log_prob(action)
            return action, value, prob
        else:
            m = Categorical(probs)
            action = m.sample()
            #action = np.argmax(probs.detach().numpy())
            return action

class GN_model:
    def __init__(self, w, l=False):
        self.l = l
        self.w = w
        self.policy = Net(w, l)
        if self.l == True:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
            #self.optimizer = optim.SGD(self.policy.parameters(), lr=0.1, momentum=0.9)
            self.reset()

    def num_params(self):
        print(sum([param.nelement() for param in self.policy.parameters()]))

    def print_w(self):
        return self.policy.print_w()

    def get_w(self):
        return self.policy.get_w()

    def set_w(self, w):
        return self.policy.set_w(w)

    def get_action(self, state):
        if self.l == True:
            action, value, log_prob = self.policy.get_action(state)
            self.probs.append(log_prob)
            self.values.append(value)
            return action
        else:
            action = self.policy.get_action(state)
            return action

    def reset(self):
        self.rewards = []
        self.probs = []
        self.values = []

    def record_reward(self, reward):
        if reward is None:
            reward = 0
        self.rewards.append(np.float32(reward))

    # New A2C update
    def finish_episode(self):
        if len(self.rewards) < 2 or sum(self.rewards) == 0:
            self.reset()
            return

        eps = np.finfo(np.float32).eps.item()
        #gamma = 0.995
        gamma = 0.95
        R = 0.0
        policy_loss = torch.Tensor([0.0])
        value_loss = torch.Tensor([0.0])
        returns = []
        for r in self.rewards:
            R = r + gamma * R
            returns.insert(0, R.astype(np.float32))
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, value, R in zip(self.probs,
                                      self.values,
                                      returns):
            advantage = R - value.item()
            policy_loss = policy_loss - log_prob * advantage
            value_loss = value_loss + F.smooth_l1_loss(value, torch.tensor([R]))
        self.optimizer.zero_grad()
        loss = policy_loss + value_loss
        #print(returns)
        #print(policy_loss, value_loss, loss)
        loss.backward()
        #for param in self.policy.parameters():
        #    param.grad.data.clamp(-1, 1)
        self.optimizer.step()
        #self.policy.clamp_w()
        self.reset()

    # Old non-A2C update
    def update_policy(self):
        discounted_rewards = []
        GAMMA = 0.9
        if len(self.rewards) < 2:
            self.reset()
            return
        for t in range(len(self.rewards)):
            Gt = 0 
            pw = 0
            for r in self.rewards[t:]:
                Gt = Gt + GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_gradient = []
        for log_prob, Gt in zip(self.log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()
        self.reset()

class Agent:
    def __init__(self, x, y, z, learnable, color, energy,
                 action_size, genome, net_desc):
        self.colors = ["pink", "blue", "green", "orange", "purple", "red", "teal", "violet", "yellow"]
        self.start_energy = energy
        self.action_size = action_size
        self.learnable = learnable
        self.sample = False
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.inertial_damping = 0.0
        self.speed = 0.5
        self.color = color
        self.color_str = self.colors[color]
        self.genome = genome
        self.net_desc = net_desc
        self.weights = self.make_weights()
        self.model = GN_model(self.weights, self.learnable)
        self.model.reset()
        self.new_state = None
        self.state = None
        self.entity = None
        self.radius_entity = None
        self.trail_length = 0
        self.trail_alphas = [100, 60, 30]
        self.trail_entities = []
        for _ in range(self.trail_length):
            self.trail_entities.append(None)
        self.previous_stats = []
        self.previous_weights = []
        self.reset()

    def make_weights(self):
        net_desc = self.net_desc
        state = self.genome
        weights = []
        state_index = 0
        b = 0
        for index, layer_desc in enumerate(net_desc):
            entry = []
            if len(layer_desc) > 2:
                p, s1, s2, o = layer_desc
                prev_states = p
                for depth in range(prev_states):
                    for width in range(p):
                        item = state[state_index]
                        sn = s1
                        if depth > 0:
                            sn = o * 2
                        w = torch.Tensor(np.reshape(item[0:sn*s2], (s2, sn)))
                        entry.append([b, depth, width, sn, s2, w])
                        w = torch.Tensor(np.reshape(item[sn*s2:], (o, s2)))
                        entry.append([b, depth, width, s2, o, w])
                        state_index += 1
                    p -= 1
                b += 1
            else:
                item = state[state_index]
                s1, o = layer_desc
                w = torch.Tensor(np.reshape(item, (o, s1)))
                entry.append([s1, o, w])
                state_index += 1
            weights.append(entry)
        return weights

    def set_genome(self, genome):
        self.genome = genome
        self.set_w()
        self.previous_weights.append(genome)

    def set_w(self):
        self.weights = self.make_weights()
        self.model.set_w(self.weights)

    def reset(self):
        self.energy = self.start_energy
        self.frequency = 1
        self.xvel = 0
        self.yvel = 0
        self.prev_action = 0
        self.orient = 0 # 0-7 in 45 degree increments
        self.distance_travelled = 0
        self.happiness = 0
        self.fitness = 0
        self.age = 0
        self.temperature = 20
        self.previous_states = deque()
        self.previous_actions = []
        self.previous_x = []
        self.previous_y = []
        self.previous_positions = deque()
        for _ in range(self.trail_length):
            self.previous_positions.append([self.xpos, self.ypos, self.orient, self.energy])

    def get_action(self):
        if self.sample == True:
            return random.choice(range(self.action_size))
        else:
            return self.model.get_action(self.state)

class Food:
    def __init__(self, x, y, z):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.entity = None

class Protector:
    def __init__(self, x, y, z):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.xvel = 0
        self.yvel = 0
        self.speed = 0.1
        self.orient = 0 # 0-7 in 45 degree increments
        self.inertial_damping = 0.80
        self.entity = None
        self.protection_entity = None
        self.pulse_entity = None

class Predator:
    def __init__(self, x, y, z):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.xvel = 0
        self.yvel = 0
        self.speed = 0.2
        self.orient = 0 # 0-7 in 45 degree increments
        self.inertial_damping = 0.70
        self.entity = None

class Shooter:
    def __init__(self, x, y, z):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.xvel = 0
        self.yvel = 0
        self.speed = 0.2
        self.orient = 0 # 0-7 in 45 degree increments
        self.inertial_damping = 0.70
        self.shoot_cooldown = 0
        self.entity = None

class Bullet:
    def __init__(self, x, y, z, xv, yv):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.xvel = xv
        self.yvel = yv
        self.lifespan = 20
        self.entity = None

    def reuse(self, x, y, z, xv, yv):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.xvel = xv
        self.yvel = yv
        self.lifespan = 20

class Zone:
    def __init__(self, x, y, z, ztype, radius, strength):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.ztype = ztype
        self.radius = radius
        self.strength = strength
        self.entity = None
        self.radius_entity = None
        self.zone_colors = {'attractor': [51, 0, 102],
                            'repulsor': [0, 51, 102],
                            'damping': [102, 0, 51],
                            'acceleration': [102, 0, 0]}

class game_space:
    def __init__(self,
                 block_hidden_factor=1,
                 out_cat_hidden_factor=4,
                 num_prev_states=2,
                 num_recent_actions=1000,
                 learners=0.50,
                 evaluate_learner_every=10,
                 mutation_rate=0.0013, # auto-set if this is zero
                 evolve_by_block=True,
                 integer_weights=True,
                 weight_range=1,
                 area_size=50,
                 area_toroid=True,
                 num_agents=10,
                 agent_start_energy=100,
                 agent_energy_drain=1,
                 agent_view_distance=15,
                 num_protectors=3,
                 protector_safe_distance=7,
                 num_predators=6,
                 predator_view_distance=8,
                 predator_kill_distance=2,
                 num_shooters=0,
                 shooter_visible_range=20,
                 shoot_cooldown=8,
                 bullet_life=100,
                 bullet_speed=0.35,
                 bullet_radius=0.25,
                 num_food=0,
                 use_zones=False,
                 visuals=False,
                 inference=False,
                 pulse_zones=False,
                 num_previous_agents=100,
                 genome_store_size=50,
                 fitness_index=2, # 1: fitness, 2: age
                 save_every=5000,
                 record_every=200,
                 savedir="alien_ecology_save"):
        self.steps = 1
        self.spawns = 0
        self.resets = 0
        self.rebirths = 0
        self.continuations = 0
        self.deaths = 0
        self.eaten = 0
        self.shot = 0
        self.last_discovery = 0
        self.num_recent_actions = num_recent_actions
        self.num_previous_agents = num_previous_agents
        self.fitness_index = fitness_index
        self.genome_store_size = genome_store_size
        self.learners = learners
        self.evaluate_learner_every = evaluate_learner_every
        self.mutation_rate = mutation_rate
        self.evolve_by_block = evolve_by_block
        self.integer_weights = integer_weights
        self.weight_range = weight_range
        self.visuals = visuals
        self.inference = inference
        if self.inference == True:
            self.learners = 0.0
        self.pulse_zones = pulse_zones
        self.savedir = savedir
        self.stats = {}
        self.load_stats()
        self.record_every = record_every
        self.save_every = save_every
        self.num_prev_states = num_prev_states
        self.block_hidden_factor = block_hidden_factor
        self.out_cat_hidden_factor = out_cat_hidden_factor
        self.area_size = area_size
        self.area_toroid = area_toroid
        self.num_agents = num_agents
        self.agent_start_energy = agent_start_energy
        self.agent_energy_drain = agent_energy_drain
        self.num_protectors = num_protectors
        self.protector_safe_distance = protector_safe_distance
        self.num_predators = num_predators
        self.predator_view_distance = predator_view_distance
        self.predator_kill_distance = predator_kill_distance
        self.num_shooters = num_shooters
        self.shooter_visible_range = shooter_visible_range
        self.shoot_cooldown = shoot_cooldown
        self.bullet_life = bullet_life
        self.bullet_speed = bullet_speed
        self.bullet_radius = bullet_radius
        self.num_food = num_food
        self.use_zones = use_zones
        self.agent_view_distance = agent_view_distance
        self.visible_area = math.pi*(self.agent_view_distance**2)
        self.agent_types = ["evolving",
                            "learning"]
        self.agent_actions = {}
        self.recent_actions = {}
        for t in self.agent_types:
            self.agent_actions[t] = Counter()
            self.recent_actions[t] = deque()
        self.directions = ['up',
                           'upright',
                           'right',
                           'downright',
                           'down',
                           'downleft',
                           'left',
                           'upleft']
        self.obs_directions = ['up',
                               'upright',
                               'right',
                               'downright',
                               'down',
                               'downleft',
                               'left',
                               'upleft',
                               ]
        self.actions = [
                        #"rotate_right",
                        #"rotate_left",
                        #"flip",
                        #"propel",
                        "propel_up",
                        "propel_right",
                        "propel_down",
                        "propel_left",
                        #"null",
                        #"move_random",
                        ]
        self.ttypes = ["agents",
                       "bullets",
                       "shooters",
                       "food",
                       "predators",
                       "protectors"]
        bullet_obs = ["bullets_"+x for x in self.obs_directions]
        shooter_obs = ["shooters_"+x for x in self.obs_directions]
        food_obs = ["food_"+x for x in self.obs_directions]
        protector_obs = ["protectors_"+x for x in self.obs_directions]
        protector_obs.append("protector_in_range")
        predator_obs = ["predators_"+x for x in self.obs_directions]
        agent_obs = ["agent_"+x for x in self.obs_directions]
        boundary_obs = ["boundary_up",
                        "boundary_right",
                        "boundary_down",
                        "boundary_left"]
        location_obs = ["agent_xpos",
                        "agent_ypos",
                        "agent_xvel",
                        "agent_yvel"]
        other_obs = ["visible_agents",
                     "visible_food",
                     "visible_protectors",
                     "visible_predators",
                     "own_energy"]
        self.observations = []
        #self.observations.append(location_obs)
        if self.area_toroid == False:
            self.observations.append(boundary_obs)
        if self.num_food > 0:
            self.observations.append(food_obs)
        if self.num_protectors > 0:
            self.observations.append(protector_obs)
        if self.num_predators > 0:
            self.observations.append(predator_obs)
        if self.num_shooters > 0:
            self.observations.append(shooter_obs)
            self.observations.append(bullet_obs)
        self.observation_size = sum([len(x) for x in self.observations])
        self.action_size = len(self.actions)
        self.net_desc = []
        for index, item in enumerate(self.observations):
            obs = len(self.observations[index])
            hidden = int(obs * self.block_hidden_factor)
            entry = [self.num_prev_states, obs, hidden, self.action_size]
            self.net_desc.append(entry)
        self.genome_size = []
        self.make_genome_size()
        if self.mutation_rate == 0:
            params = sum([x for x in self.genome_size])
            self.mutation_rate = 1/(params*2)
        self.genome_store = []
        self.things = {}
        self.zone_types = ['attractor', 'repulsor', 'damping', 'acceleration']
        lpos = self.area_size*0.2
        rpos = self.area_size*0.8
        mpos = self.area_size*0.5
        small = self.area_size*0.04
        medium = self.area_size*0.1
        big = self.area_size*0.15
        bigger = self.area_size*0.2
        self.zone_config = [['attractor', lpos, lpos, big, 0.2],
                            ['attractor', rpos, rpos, big, 0.2],
                            ['repulsor', lpos, rpos, big, 0.2],
                            ['repulsor', rpos, lpos, big, 0.2],
                            ['repulsor', mpos, lpos, small, 2.0],
                            ['repulsor', mpos, rpos, small, 2.0],
                            ['repulsor', lpos, mpos, small, 2.0],
                            ['repulsor', rpos, mpos, small, 2.0],
                            #['acceleration', mpos, mpos, bigger, 0.3],
                            ]
        self.spawn_zones()
        self.spawn_food()
        self.previous_agents = deque()
        self.create_shooters()
        self.create_predators()
        self.create_protectors()
        self.create_genomes()
        self.things['agents'] = []
        num_learners = int(self.num_agents * self.learners)
        num_evolvable = int(self.num_agents - num_learners)
        if num_learners > 0:
            self.create_new_learner_agents(num_learners)
        if num_evolvable > 0:
            self.create_new_evolvable_agents(num_evolvable)

    def make_genome_size(self):
        genome_size = []
        for item in self.net_desc:
            p, inps, hidden, outs = item
            prev_states = p
            for depth in range(prev_states):
                for width in range(p):
                    size = 0
                    if depth == 0:
                        size += inps * hidden
                    else:
                        size += 2 * outs * hidden
                    size += hidden * outs
                    genome_size.append(size)
                p -= 1
        out_cat = len(self.observations) * self.action_size
        out_hidden = self.action_size * self.out_cat_hidden_factor
        cat_hidden = out_cat * out_hidden
        genome_size.append(cat_hidden)
        action_head = out_hidden * self.action_size
        genome_size.append(action_head)
        self.net_desc.append([out_cat, out_hidden])
        self.net_desc.append([out_hidden, self.action_size])
        value_head = out_hidden * 1
        genome_size.append(value_head)
        self.net_desc.append([out_hidden, 1])
        self.genome_size = genome_size

    def step(self):
        self.run_bullet_actions()
        self.apply_shooter_physics()
        self.run_shooter_actions()
        self.apply_predator_physics()
        self.run_predator_actions()
        self.apply_protector_physics()
        self.run_protector_actions()
        self.apply_zone_effects()
        self.set_agent_states()
        #self.make_new_states()
        self.apply_agent_physics()
        self.run_agent_actions()
        self.update_agent_status()
        if self.inference == False:
            if self.save_every > 0:
                if self.steps % self.save_every == 0:
                    self.save_genomes()
            if self.save_every > 0 and self.steps % 5000 == 0:
                self.save_stats()
        if self.steps % 100 == 0:
            self.print_stats()
        self.steps += 1

#################
# Helper routines
#################
    def entropy(self, actions):
        n_actions = len(actions)
        if n_actions <= 1:
            return 0
        value,counts = np.unique(actions, return_counts=True)
        probs = counts / n_actions
        n_classes = np.count_nonzero(probs)
        if n_classes <= 1:
            return 0
        ent = 0.
        for i in probs:
            ent -= i * math.log(i, math.e)
        return ent

    def distance(self, x1, y1, x2, y2):
        return math.sqrt(((x1-x2)**2)+((y1-y2)**2))

    def angle(self, x1, y1, x2, y2):
        angle = ((math.atan2(y2-y1, x2-x1))/math.pi+1)*180
        return angle

    def propel(self, xv, yv, orient, speed):
        xvel = xv
        yvel = yv
        # up
        if orient == 0:
            yvel += 1*speed
        # up right
        elif orient == 1:
            yvel += 0.5*speed
            xvel += 0.5*speed
        # right
        elif orient == 2:
            xvel += 1*speed
        # down right
        elif orient == 3:
            yvel -= 0.5*speed
            xvel += 0.5*speed
        # down
        elif orient == 4:
            yvel -= 1*speed
        # down left
        elif orient == 5:
            yvel -= 0.5*speed
            xvel -= 0.5*speed
        # left
        elif orient == 6:
            xvel -= 1*speed
        # up left
        elif orient == 7:
            yvel += 0.5*speed
            xvel -= 0.5*speed
        xvel = min(xvel, (speed*5))
        yvel = min(yvel, (speed*5))
        return xvel, yvel

    def update_position(self, xpos, ypos):
        if self.area_toroid == True:
            return self.update_position_toroid(xpos, ypos)
        else:
            return self.update_position_bounded(xpos, ypos)

    def update_position_toroid(self, xpos, ypos):
        nx = xpos
        ny = ypos
        if nx > self.area_size:
            nx -= self.area_size
        if nx < 0:
            nx += self.area_size
        if ny > self.area_size:
            ny -= self.area_size
        if ny < 0:
            ny += self.area_size
        return nx, ny

    def update_position_bounded(self, xpos, ypos):
        nx = xpos
        ny = ypos
        if nx > self.area_size:
            nx = self.area_size
        if nx < 0:
            nx = 0
        if ny > self.area_size:
            ny = self.area_size
        if ny < 0:
            ny = 0
        return nx, ny

    def filter_by_distance(self, x1, y1, x2, y2, radius):
        if abs(x1 - x2) <= radius:
            return True
        if abs(x2 - x1) <= radius:
            return True
        if abs(y1 - y2) <= radius:
            return True
        if abs(y2 - y1) <= radius:
            return True
        return False

    def get_safe_spawn_location(self):
        xpos = 0
        ypos = 0
        bestx = xpos
        besty = ypos
        for _ in range(20):
            xpos = random.random()*self.area_size
            ypos = random.random()*self.area_size
            predators = self.get_predators_in_radius(xpos, ypos, self.agent_view_distance)
            lowest = None
            if len(predators) < 1:
                return xpos, ypos
            if lowest == None:
                lowest = predators
                bestx = xpos
                besty = ypos
            elif predators < lowest:
                lowest = predators
                bestx = xpos
                besty = ypos
        return bestx, besty

    def detect_in_direction(self, direction, angle):
        # All 8 directions have their own angle range
        direction_angle1 = {'up': [[248, 270], [270, 292]],
                            'down': [[68, 90], [90, 112]],
                            'left': [[338, 0], [0, 22]],
                            'right': [[158, 180], [180, 202]],
                            'upright': [[203, 225], [225, 247]],
                            'upleft': [[293, 315], [315, 337]],
                            'downleft': [[23, 45], [45, 67]],
                            'downright': [[113, 135], [135, 157]],
                            }

        # Overlap between angle ranges
        direction_angle2 = {'up': [[226, 270], [270, 314]],
                            'down': [[46, 90], [90, 134]],
                            'left': [[316, 360], [0, 44]],
                            'right': [[136, 180], [180, 224]],
                            'upright':[[181, 225], [225, 269]],
                            'upleft':[[270, 315], [315, 359]],
                            'downleft':[[1, 45], [45, 89]],
                            'downright':[[91, 135], [135, 179]],
                           }
        direction_angles = direction_angle2
        if len(self.obs_directions) > 4:
            direction_angles = direction_angle1

        ranges = direction_angles[direction]
        for r in ranges:
            l, u = r
            if angle >= l and angle <= u:
                return True
        return False

    def get_things_in_direction_pos(self, x1, y1, distance, ttype, direction):
        ret = []
        for i in range(len(self.things[ttype])):
            x2 = self.things[ttype][i].xpos
            y2 = self.things[ttype][i].ypos
            if self.filter_by_distance(x1, y1, x2, y2, distance) is True:
                angle = self.angle(x1, y1, x2, y2)
                if self.detect_in_direction(direction, angle) is True:
                    d = self.distance(x1, y1, x2, y2)
                    if d <= distance:
                        ret.append(i)
        return ret

    def get_things_in_direction(self, ttype, aindex, direction):
        x1 = self.things['agents'][aindex].xpos
        y1 = self.things['agents'][aindex].ypos
        distance = self.agent_view_distance
        t1 = self.get_things_in_direction_pos(x1, y1, distance, ttype, direction)
        # If the agent is facing and close to boundary, and toroid mode is on
        # get things from the other side
        if self.area_toroid == True:
            xn = None
            yn = None
            if (y1 + distance) > self.area_size:
                if direction in ['up', 'upleft', 'upright']:
                    yn = y1 - self.area_size
            if (y1 - distance) < 0:
                if direction in ['down', 'downleft', 'downright']:
                    yn = y1 + self.area_size
            if (x1 + distance) > self.area_size:
                if direction in ['right', 'upright', 'downright']:
                    xn = x1 - self.area_size
            if (x1 - distance) < 0:
                if direction == ['left', 'upleft', 'downleft']:
                    xn = x1 + self.area_size
            if xn is not None:
                if yn is None:
                    yn = y1
            if yn is not None:
                if xn is None:
                    xn = x1
            if xn is not None and yn is not None:
                t2 = self.get_things_in_direction_pos(xn, yn, distance, ttype, direction)
                t1.extend(t2)
        return t1

    def get_things_in_radius(self, ttype, xpos, ypos, radius):
        ret = []
        for i in range(len(self.things[ttype])):
            ax = self.things[ttype][i].xpos
            ay = self.things[ttype][i].ypos
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                if self.distance(xpos, ypos, ax, ay) <= radius:
                    ret.append(i)
        return ret

    def get_visible_things(self, ttype, aindex):
        direction = self.directions[orient]
        items = self.get_things_in_direction(ttype, aindex, direction)
        return len(items)

    def get_adjacent_thing_indices(self, ttype, aindex):
        xpos = self.things['agents'][aindex].xpos
        ypos = self.things['agents'][aindex].ypos
        return self.get_things_in_radius(ttype, xpos, ypos, self.agent_view_distance)

    def get_adjacent_thing_count(self, ttype, aindex):
        xpos = self.things['agents'][aindex].xpos
        ypos = self.things['agents'][aindex].ypos
        ret = self.get_things_in_radius(ttype, xpos, ypos, self.agent_view_distance)
        return len(ret)

    def get_nearest_thing(self, ttype, aindex):
        xpos = self.things['agents'][aindex].xpos
        ypos = self.things['agents'][aindex].ypos
        distances = []
        for i in range(len(self.things[ttype])):
            ax = self.things[ttype][i].xpos
            ay = self.things[ttype][i].ypos
            radius = self.agent_view_distance
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                distances.append(self.distance(xpos, ypos, ax, ay))
        if len(distances) > 1:
            shortest_index = np.argsort(distances, axis=0)[1]
            shortest_value = distances[shortest_index]
            return shortest_index, shortest_value
        else:
            return None, None

    def apply_physics(self, ttype):
        for index in range(len(self.things[ttype])):
            self.apply_inertial_damping(ttype, index)

    def apply_inertial_damping(self, ttype, index):
        self.things[ttype][index].xvel = self.things[ttype][index].xvel * self.things[ttype][index].inertial_damping
        self.things[ttype][index].yvel = self.things[ttype][index].yvel * self.things[ttype][index].inertial_damping

    def rotate_right(self, ttype, index):
        self.things[ttype][index].orient += 1
        if self.things[ttype][index].orient > 7:
            self.things[ttype][index].orient = 0
        return 0

    def rotate_left(self, ttype, index):
        self.things[ttype][index].orient -= 1
        if self.things[ttype][index].orient < 0:
            self.things[ttype][index].orient = 7
        return 0

    def flip_thing(self, ttype, index):
        self.things[ttype][index].orient += 4
        if self.things[ttype][index].orient > 7:
            self.things[ttype][index].orient -= 7
        return 0

    def propel_thing(self, ttype, index):
        orient = self.things[ttype][index].orient
        speed = self.things[ttype][index].speed
        xvel = self.things[ttype][index].xvel
        yvel = self.things[ttype][index].yvel
        xv, yv = self.propel(xvel, yvel, orient, speed)
        self.things[ttype][index].xvel = xv
        self.things[ttype][index].yvel = yv
        return 0

    def propel_thing_in_direction(self, ttype, index, direction):
        speed = self.things[ttype][index].speed
        xvel = self.things[ttype][index].xvel
        yvel = self.things[ttype][index].yvel
        xv, yv = self.propel(xvel, yvel, direction, speed)
        self.things[ttype][index].xvel = xv
        self.things[ttype][index].yvel = yv
        return 0

    def propel_thing_up(self, ttype, index):
        return self.propel_thing_in_direction(ttype, index, 0)

    def propel_thing_right(self, ttype, index):
        return self.propel_thing_in_direction(ttype, index, 2)

    def propel_thing_down(self, ttype, index):
        return self.propel_thing_in_direction(ttype, index, 4)

    def propel_thing_left(self, ttype, index):
        return self.propel_thing_in_direction(ttype, index, 6)

    def update_thing_position(self, ttype, index):
        xpos = self.things[ttype][index].xpos
        xpos += self.things[ttype][index].xvel
        ypos = self.things[ttype][index].ypos
        ypos += self.things[ttype][index].yvel
        xpos, ypos = self.update_position(xpos, ypos)
        self.things[ttype][index].xpos = xpos
        self.things[ttype][index].ypos = ypos


########################
# Agent runtime routines
########################
    def set_agent_entity(self, index):
        xabs = self.things['agents'][index].xpos
        yabs = self.things['agents'][index].ypos
        zabs = self.things['agents'][index].zpos
        s = 0.5 + ((self.things['agents'][index].energy/self.agent_start_energy)*0.5)
        texture = "textures/" + self.things['agents'][index].color_str
        self.things['agents'][index].entity = Entity(model='sphere',
                                           color=color.white,
                                           scale=(s,s,s),
                                           position = (xabs, yabs, zabs),
                                           texture=texture)
        s = self.agent_view_distance * 2
        self.things['agents'][index].radius_entity = Entity(model='sphere',
                                                          color=color.rgba(128,128,128,10),
                                                          scale=(s,s,s),
                                                          position = (xabs, yabs, zabs))
        for n in range(self.things['agents'][index].trail_length):
            item = self.things['agents'][index].previous_positions[n]
            x, y, o, ss = item
            ss = 0.5 + ((self.things['agents'][index].energy/self.agent_start_energy)*0.5)
            self.things['agents'][index].trail_entities[n] = Entity( model='sphere',
                                                           color=color.rgba(255,255,255,self.things['agents'][index].trail_alphas[n]),
                                                           scale=(ss,ss,ss),
                                                           position = (x, y, zabs),
                                                           texture=texture)

    def spawn_new_agent(self, genome, learner):
        xpos, ypos = self.get_safe_spawn_location()
        zpos = -1
        color = 0
        if learner == False:
            color = 1
        a = Agent(xpos,
                  ypos,
                  zpos,
                  learner,
                  color,
                  self.agent_start_energy,
                  self.action_size,
                  genome,
                  self.net_desc)
        self.things['agents'].append(a)
        index = len(self.things['agents'])-1
        self.set_initial_agent_state(index)
        if self.visuals == True:
            self.set_agent_entity(index)

    def spawn_learning_agent(self, genome):
        self.spawn_new_agent(genome, True)

    def spawn_evolving_agent(self, genome):
        self.spawn_new_agent(genome, False)

    def create_new_evolvable_agents(self, num):
        genomes = random.sample(self.genome_pool, num)
        for genome in genomes:
            self.spawn_evolving_agent(genome)

    def create_new_learner_agents(self, num):
        genomes = random.sample(self.genome_pool, num)
        for genome in genomes:
            self.spawn_learning_agent(genome)

    def record_recent_actions(self, action, atype):
        t = self.agent_types[atype]
        if len(self.recent_actions[t]) > self.num_recent_actions:
            self.recent_actions[t].popleft()
        self.recent_actions[t].append(action)
        ra = Counter()
        for a in self.recent_actions[t]:
            ra[self.actions[a]] += 1
        self.agent_actions[t] = ra

    def run_agent_actions(self):
        for n in range(len(self.things['agents'])):
            xpos = self.things['agents'][n].xpos
            ypos = self.things['agents'][n].ypos
            action = int(self.things['agents'][n].get_action())
            self.things['agents'][n].previous_actions.append(action)
            self.things['agents'][n].previous_x.append(int(xpos))
            self.things['agents'][n].previous_y.append(int(ypos))
            atype = 0
            if self.things['agents'][n].learnable == True:
                atype = 1
            self.record_recent_actions(action, atype)
            self.apply_agent_action(n, action)
            self.update_agent_position(n)

    def get_agents_in_radius(self, xpos, ypos, radius):
        return self.get_things_in_radius('agents', xpos, ypos, radius)

    def get_boundary_up(self, aindex):
        if (self.things['agents'][aindex].ypos + self.agent_view_distance) >= self.area_size:
            return 1
        return 0

    def get_boundary_down(self, aindex):
        if (self.things['agents'][aindex].ypos - self.agent_view_distance) <= 0:
            return 1
        return 0

    def get_boundary_left(self, aindex):
        if (self.things['agents'][aindex].xpos - self.agent_view_distance) <= 0:
            return 1
        return 0

    def get_boundary_right(self, aindex):
        if (self.things['agents'][aindex].xpos + self.agent_view_distance) >= self.area_size:
            return 1
        return 0

    def evaluate_learner(self, index):
        if len(self.things['agents'][index].previous_stats) >= self.evaluate_learner_every:
            mpf = np.mean([x[self.fitness_index] for x in self.things['agents'][index].previous_stats])
            pfm = 0
            method = 0
            if self.inference == False:
                #prev_w = self.things['agents'][index].previous_weights
                #with open(self.savedir + "/agent_" + str(index) + "_weights.pkl", "wb") as f:
                    #f.write(pickle.dumps(prev_w))
                self.things['agents'][index].previous_weights = []
            if len(self.genome_store) > 1:
                pfm = np.mean([x[self.fitness_index] for x in self.genome_store])
            self.things['agents'][index].previous_stats = []
            if pfm > 0:
                if mpf < pfm * 0.5:
                    g = None
                    num_g = len(self.genome_store)
                    if num_g > 0:
                        g = random.choice(self.get_best_genomes_from_store(num_g, None))
                    if g is None:
                        g = self.make_random_genome()
                    self.things['agents'][index].set_genome(g)
                    self.rebirths += 1
                else:
                    self.continuations += 1
            else:
                self.continuations += 1

    def store_agent_entry(self, index):
            l = int(self.things['agents'][index].learnable)
            ae = self.entropy(self.things['agents'][index].previous_actions)
            ex = self.entropy(self.things['agents'][index].previous_x)
            ey = self.entropy(self.things['agents'][index].previous_y)
            a = self.things['agents'][index].age
            h = self.things['agents'][index].happiness + ae*50
            d = self.things['agents'][index].distance_travelled
            f = a + h
            g = self.things['agents'][index].model.get_w()
            entry = [g, f, a, h, d, l]
            self.store_genome(entry)
            self.add_previous_agent(entry)
            return entry

    def reset_agents(self, reset):
        for index in reset:
            entry = self.store_agent_entry(index)
            ae = self.entropy(self.things['agents'][index].previous_actions)
            ex = self.entropy(self.things['agents'][index].previous_x)
            ey = self.entropy(self.things['agents'][index].previous_y)
            f = entry[self.fitness_index]
            if len(self.things['agents'][index].model.rewards) > 0:
                gsfm = 0.0
                if len(self.genome_store) > (self.genome_store_size*0.75):
                    gsfm = np.mean([x[self.fitness_index] for x in self.genome_store]) * 0.75
                #measure = max(gsfm, self.agent_start_energy)
                measure = self.agent_start_energy
                reward = ((f-measure)**2)/(measure**2)
                reward += (f-measure)/measure
                reward += ae - 1.3
                reward += self.things['agents'][index].model.rewards[-1]
                print(f, "%.4f"%reward)
                reward = np.float32(reward)
                self.things['agents'][index].model.rewards[-1] = reward
            self.things['agents'][index].previous_stats.append(entry)
            self.deaths += 1
            self.resets += 1
            if self.inference == False:
                self.things['agents'][index].model.finish_episode()
                new_w = self.things['agents'][index].model.get_w()
                self.things['agents'][index].previous_weights.append(new_w)
            self.things['agents'][index].reset()
            xpos, ypos = self.get_safe_spawn_location()
            self.things['agents'][index].xpos = xpos
            self.things['agents'][index].ypos = ypos
            self.set_initial_agent_state(index)
            self.evaluate_learner(index)

    def kill_agents(self, dead):
        for index in dead:
            entry = self.store_agent_entry(index)
            f = entry[self.fitness_index]
            ae = self.entropy(self.things['agents'][index].previous_actions)
            #print(f)
            self.deaths += 1
            self.spawns += 1
            self.things['agents'][index].reset()
            xpos, ypos = self.get_safe_spawn_location()
            self.things['agents'][index].xpos = xpos
            self.things['agents'][index].ypos = ypos
            genome = entry[0]
            if self.inference == False:
                genome = self.make_new_genome(0)
            else:
                num_g = len(self.genome_store)
                if num_g > 0:
                    genome = random.choice(self.get_best_genomes_from_store(num_g, None))
            self.things['agents'][index].set_genome(genome)
            self.set_initial_agent_state(index)

    def update_agent_status(self):
        dead = set()
        reset = set()
        for index in range(len(self.things['agents'])):
            self.things['agents'][index].age += 1
            xpos = self.things['agents'][index].xpos
            ypos = self.things['agents'][index].ypos
            if self.num_shooters > 0:
                nearby_shooters = self.get_shooters_in_radius(xpos, ypos, 1)
                if len(nearby_shooters) > 0:
                    fi = random.choice(nearby_shooters)
                    self.respawn_shooter(fi)
                    self.things['agents'][index].energy += self.agent_start_energy
                    self.things['agents'][index].happiness += self.agent_start_energy
                    # Reward learning agents for eating shooter
                    if len(self.things['agents'][index].model.rewards) > 0:
                        self.things['agents'][index].model.rewards[-1] = np.float32(1.0)
            if self.num_food > 0:
                nearby_food = self.get_food_in_radius(xpos, ypos, 1)
                if len(nearby_food) > 0:
                    fi = random.choice(nearby_food)
                    self.respawn_food(fi)
                    self.things['agents'][index].energy += 50
                    self.things['agents'][index].happiness += 50
                    # Reward learning agents for eating food
                    if len(self.things['agents'][index].model.rewards) > 0:
                        self.things['agents'][index].model.rewards[-1] = np.float32(1.0)
            energy_drain = self.agent_energy_drain
            protectors = self.get_protectors_in_radius(xpos, ypos, self.protector_safe_distance)
            if len(protectors) > 0:
                energy_drain = -1
            self.things['agents'][index].energy -= energy_drain
            if self.things['agents'][index].energy <= 0:
                if self.things['agents'][index].learnable == False:
                    dead.add(index)
                else:
                    reset.add(index)
        if len(reset) > 0:
            self.reset_agents(reset)
        if len(dead) > 0:
            self.kill_agents(dead)

    def update_agent_position(self, index):
        x1 = self.things['agents'][index].xpos
        y1 = self.things['agents'][index].ypos
        o = self.things['agents'][index].orient
        e = self.things['agents'][index].energy
        x2 = x1 + self.things['agents'][index].xvel
        y2 = y1 + self.things['agents'][index].yvel
        d = self.distance(x1, y1, x2, y2)
        self.things['agents'][index].distance_travelled += d
        nx, ny = self.update_position(x2, y2)
        self.things['agents'][index].xpos = nx
        self.things['agents'][index].ypos = ny
        if len(self.things['agents'][index].previous_positions) > 0:
            self.things['agents'][index].previous_positions.popleft()
        self.things['agents'][index].previous_positions.append([x2, y2, o, e])

    def apply_agent_physics(self):
        self.apply_physics('agents')

# Agent actions
    def apply_agent_action(self, index, action):
        self.things['agents'][index].prev_action = action
        agent_function = "action_" + self.actions[action]
        class_method = getattr(self, agent_function)
        reward = class_method(index)
        if self.things['agents'][index].learnable == True:
            self.things['agents'][index].model.record_reward(reward)

    def action_null(self, index):
        return 0

    def action_rotate_right(self, index):
        return self.rotate_right('agents', index)

    def action_rotate_left(self, index):
        return self.rotate_left('agents', index)

    def action_flip(self, index):
        return self.flip_thing('agents', index)

    def action_propel(self, index):
        return propel_thing(self, ttype, index)

    def propel_agent_in_direction(self, index, direction):
        return self.propel_thing_in_direction('agents', index, direction)

    def action_propel_up(self, index):
        return self.propel_thing_up('agents', index)

    def action_propel_right(self, index):
        return self.propel_thing_right('agents', index)

    def action_propel_down(self, index):
        return self.propel_thing_down('agents', index)

    def action_propel_left(self, index):
        return self.propel_thing_left('agents', index)

    def action_move_random(self, index):
        choices = [0, 1, 2, 3]
        action = random.choice(choices)
        return self.apply_agent_action(index, action)

# Agent observations
    def make_new_state(self, index):
        state = []
        xpos = self.things['agents'][index].xpos
        ypos = self.things['agents'][index].ypos
        nearby = self.get_agents_in_radius(xpos, ypos, self.agent_view_distance)
        if len(nearby) > 0:
            for ai in nearby:
                ax = self.things['agents'][ai].xpos
                ay = self.things['agents'][ai].ypos
                angle = math.atan2(ay-ypos, ax-xpos)
                distance = self.distance(xpos, ypos, ax, ay)
                state.append([1, distance, angle])
        nearby = self.get_predators_in_radius(xpos, ypos, self.agent_view_distance)
        if len(nearby) > 0:
            for ai in nearby:
                ax = self.things['predators'][ai].xpos
                ay = self.things['predators'][ai].ypos
                angle = math.atan2(ay-ypos, ax-xpos)
                distance = self.distance(xpos, ypos, ax, ay)
                state.append([2, distance, angle])
        nearby = self.get_protectors_in_radius(xpos, ypos, self.agent_view_distance)
        if len(nearby) > 0:
            for ai in nearby:
                ax = self.things['protectors'][ai].xpos
                ay = self.things['protectors'][ai].ypos
                angle = math.atan2(ay-ypos, ax-xpos)
                distance = self.distance(xpos, ypos, ax, ay)
                state.append([3, distance, angle])
        return state

    def make_new_states(self):
        for index in range(len(self.things['agents'])):
            new_state = self.make_new_state(index)
            self.things['agents'][index].new_state = new_state

    def set_agent_states(self):
        for n in range(len(self.things['agents'])):
            self.set_agent_state(n)

    def set_initial_agent_state(self, index):
        prev_states = []
        for _ in range(self.num_prev_states):
            entry = []
            for item in self.observations:
                for n in range(len(item)):
                    entry.append(0)
            prev_states.append(entry)
        self.things['agents'][index].previous_states = deque(prev_states)

    def set_agent_state(self, index):
        current_observations = self.get_agent_observations(index)
        self.things['agents'][index].previous_states.append(current_observations)
        if len(self.things['agents'][index].previous_states) > self.num_prev_states:
            self.things['agents'][index].previous_states.popleft()
        all_states = self.things['agents'][index].previous_states
        # Assemble state:
        # For each input block, concatenate slices from each item in all_states
        state = []
        iind = 0
        for block in self.observations:
            num_items = len(block)
            entry = []
            for s in all_states:
                entry.extend(s[iind : iind+num_items])
            iind += num_items
            state.extend(entry)
        state = np.ravel(state)
        state = torch.FloatTensor(state)
        state = state.unsqueeze(0)
        self.things['agents'][index].state = state

    def get_agent_observations(self, index):
        function_names = []
        for block in self.observations:
            for n in block:
                function_names.append("get_" + n)
        observations = []
        for fn in function_names:
            m = re.match("get_([a-z]+)_([a-z]+)$", fn)
            val = None
            if m is not None:
                m1 = m.group(1)
                m2 = m.group(2)
                if m1 in self.ttypes and m2 in self.obs_directions:
                    val = len(self.get_things_in_direction(m1, index, m2))
            if val is None:
                class_method = getattr(self, fn)
                val = class_method(index)
            observations.append(val)
        return observations

    def get_own_energy(self, index):
        return self.things['agents'][index].energy/self.agent_start_energy

    def get_agent_xpos(self, index):
        return self.things['agents'][index].xpos/self.area_size

    def get_agent_ypos(self, index):
        return self.things['agents'][index].ypos/self.area_size

    def get_agent_xvel(self, index):
        return self.things['agents'][index].xvel

    def get_agent_yvel(self, index):
        return self.things['agents'][index].yvel
###################
# Protector actions
###################
    def set_protector_entity(self, index):
        xabs = self.things['protectors'][index].xpos
        yabs = self.things['protectors'][index].ypos
        zabs = self.things['protectors'][index].zpos
        s = self.protector_safe_distance*2
        texture = "textures/white"
        self.things['protectors'][index].entity = Entity(model='sphere',
                                              color=color.white,
                                              scale=(2,2,2),
                                              position = (xabs, yabs, zabs),
                                              texture=texture)
        self.things['protectors'][index].protection_entity = Entity(model='sphere',
                                                          color=color.rgba(0,102,0,50),
                                                          scale=(s,s,s),
                                                          position = (xabs, yabs, zabs))
        self.things['protectors'][index].pulse_entity = Entity(model='sphere',
                                                     color=color.rgba(0,102,0,30),
                                                     scale=(s,s,s),
                                                     position = (xabs, yabs, zabs))

    def create_protectors(self):
        self.things['protectors'] = []
        for n in range(self.num_protectors):
            self.spawn_random_protector()

    def spawn_random_protector(self):
        xpos, ypos = self.get_safe_spawn_location()
        zpos = -1
        a = Protector(xpos,
                     ypos,
                     zpos)
        self.things['protectors'].append(a)
        index = len(self.things['protectors'])-1
        if self.visuals == True:
            self.set_protector_entity(index)

    def get_protector_in_range(self, aindex):
        xpos = self.things['agents'][aindex].xpos
        ypos = self.things['agents'][aindex].ypos
        i = self.get_things_in_radius('protectors', xpos, ypos, self.protector_safe_distance)
        n = len(i)
        if n > 1:
            n = 1
        return n

    def get_protectors_in_radius(self, xpos, ypos, radius):
        return self.get_things_in_radius('protectors', xpos, ypos, radius)

    def protector_move_random(self, index):
        action = random.choice([0,1,1,1,2,3])
        if action == 0:
            pass
        elif action == 1:
            self.propel_thing('protectors', index)
        elif action == 2:
            self.rotate_right('protectors', index)
        else:
            self.rotate_left('protectors', index)

    def apply_protector_physics(self):
        return self.apply_physics('protectors')

    def update_protector_position(self, index):
        return self.update_thing_position('protectors', index)

    def run_protector_actions(self):
        for index in range(len(self.things['protectors'])):
            self.protector_move_random(index)
            self.update_protector_position(index)

##################
# Predator actions
##################
    def set_predator_entity(self, index):
        xabs = self.things['predators'][index].xpos
        yabs = self.things['predators'][index].ypos
        zabs = self.things['predators'][index].zpos
        s = self.predator_kill_distance
        texture = "textures/red"
        self.things['predators'][index].entity = Entity(model='sphere',
                                              color=color.white,
                                              scale=(s,s,s),
                                              position = (xabs, yabs, zabs),
                                              texture=texture)

    def spawn_random_predator(self):
        xpos = random.random()*self.area_size
        ypos = random.random()*self.area_size
        zpos = -1
        a = Predator(xpos,
                     ypos,
                     zpos)
        self.things['predators'].append(a)
        index = len(self.things['predators'])-1
        if self.visuals == True:
            self.set_predator_entity(index)

    def create_predators(self):
        self.things['predators'] = []
        for n in range(self.num_predators):
            self.spawn_random_predator()

    def get_predators_in_radius(self, xpos, ypos, radius):
        return self.get_things_in_radius('predators', xpos, ypos, radius)

    def predator_move_random(self, index):
        action = random.choice([0,1,2])
        if action == 0:
            self.propel_thing('predators', index)
        elif action == 1:
            self.rotate_right('predators', index)
        else:
            self.rotate_left('predators', index)

    def apply_predator_physics(self):
        return self.apply_physics('predators')

    def update_predator_position(self, index):
        return self.update_thing_position('predators', index)

    def run_predator_actions(self):
        for index in range(len(self.things['predators'])):
            xpos = self.things['predators'][index].xpos
            ypos = self.things['predators'][index].ypos
            radius = self.predator_kill_distance
            # If agents near enough to predator, kill them
            victims = self.get_agents_in_radius(xpos, ypos, radius)
            if len(victims) > 0:
                dead = []
                reset = []
                self.eaten += len(victims)
                for v in victims:
                    if self.things['agents'][v].learnable == True:
                        reset.append(v)
                    else:
                        dead.append(v)
                if len(reset) > 0:
                    self.reset_agents(reset)
                if len(dead) > 0:
                    self.kill_agents(dead)
            # If protectors are near to predators, stop predator's movement
            # and prevent them from taking actions
            # Note predators can still eat agents in this state
            p = self.get_protectors_in_radius(xpos, ypos, self.protector_safe_distance)
            if len(p) > 0:
                self.apply_inertial_damping('predators', index)
            else:
                orient = self.things['predators'][index].orient
                direction = self.directions[orient]
                distance = self.predator_view_distance
                targets = self.get_things_in_direction_pos(xpos, ypos, distance, 'agents', direction)
                if len(targets) > 0:
                    self.propel_thing('predators', index)
                else:
                    self.predator_move_random(index)
            self.update_predator_position(index)

##################
# Shooter actions
##################
    def set_shooter_entity(self, index):
        xabs = self.things['shooters'][index].xpos
        yabs = self.things['shooters'][index].ypos
        zabs = self.things['shooters'][index].zpos
        s = 1
        texture = "textures/yellow"
        self.things['shooters'][index].entity = Entity(model='sphere',
                                              color=color.white,
                                              scale=(s,s,s),
                                              position = (xabs, yabs, zabs),
                                              texture=texture)

    def spawn_random_shooter(self):
        xpos = random.random()*self.area_size
        ypos = random.random()*self.area_size
        zpos = -1
        a = Shooter(xpos,
                    ypos,
                    zpos)
        self.things['shooters'].append(a)
        index = len(self.things['shooters'])-1
        if self.visuals == True:
            self.set_shooter_entity(index)

    def respawn_shooter(self, index):
        xpos = random.random()*self.area_size
        ypos = random.random()*self.area_size
        self.things['shooters'][index].xpos = xpos
        self.things['shooters'][index].ypos = ypos
        self.things['shooters'][index].xvel = 0
        self.things['shooters'][index].yvel = 0
        self.things['shooters'][index].orient = 0
        self.things['shooters'][index].shoot_cooldown = 0

    def create_shooters(self):
        self.things['shooters'] = []
        self.things['bullets'] = []
        for n in range(self.num_shooters):
            self.spawn_random_shooter()

    def get_shooters_in_radius(self, xpos, ypos, radius):
        return self.get_things_in_radius('shooters', xpos, ypos, radius)

    def shooter_move_random(self, index):
        action = random.choice([0,1,2])
        if action == 0:
            self.propel_thing('shooters', index)
        elif action == 1:
            self.rotate_right('shooters', index)
        else:
            self.rotate_left('shooters', index)

    def apply_shooter_physics(self):
        return self.apply_physics('shooters')

    def update_shooter_position(self, index):
        return self.update_thing_position('shooters', index)

    def run_shooter_actions(self):
        for index in range(len(self.things['shooters'])):
            xpos = self.things['shooters'][index].xpos
            ypos = self.things['shooters'][index].ypos
            cooldown = self.things['shooters'][index].shoot_cooldown
            if cooldown > 0:
                self.things['shooters'][index].shoot_cooldown -= 1
            radius = self.shooter_visible_range
            shortest_index = None
            shortest_value = None
            distances = []
            for i in range(len(self.things['agents'])):
                ax = self.things['agents'][i].xpos
                ay = self.things['agents'][i].ypos
                if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                    distances.append(self.distance(xpos, ypos, ax, ay))
            if len(distances) > 0:
                shortest_index = 0
                shortest_value = distances[0]
                if len(distances) > 1:
                    shortest_index = np.argsort(distances, axis=0)[1]
                    shortest_value = distances[shortest_index]
                if cooldown < 1:
                    angle = math.atan2(ay-ypos, ax-xpos)
                    xc = math.cos(angle) * self.bullet_speed
                    yc = math.sin(angle) * self.bullet_speed
                    self.spawn_bullet(xpos, ypos, xc, yc)
                    self.things['shooters'][index].shoot_cooldown = self.shoot_cooldown
            self.shooter_move_random(index)
            self.update_shooter_position(index)

##################
# Bullet actions
##################
    def set_bullet_entity(self, index):
        xabs = self.things['bullets'][index].xpos
        yabs = self.things['bullets'][index].ypos
        zabs = self.things['bullets'][index].zpos
        s = self.bullet_radius
        c = [255, 255, 255, 255]
        self.things['bullets'][index].entity = Entity(model='sphere',
                                         color=color.rgba(*c),
                                         scale=(s,s,s),
                                         position = (xabs, yabs, zabs))

    def spawn_bullet(self, xpos, ypos, xv, yv):
        zpos = -1
        # If there are existing unused bullets, reuse one
        for index in range(len(self.things['bullets'])):
            ls = self.things['bullets'][index].lifespan
            if ls == 0:
                self.things['bullets'][index].reuse(xpos, ypos, zpos, xv, yv)
                self.things['bullets'][index].lifespan = self.bullet_life
                if self.visuals == True:
                    self.things['bullets'][index].entity.enable()
                return
        # Else spawn a new one and append it to the existing list
        b = Bullet(xpos, ypos, zpos, xv, yv)
        b.lifespan = self.bullet_life
        self.things['bullets'].append(b)
        index = len(self.things['bullets'])-1
        if self.visuals == True:
            self.set_bullet_entity(index)

    def run_bullet_actions(self):
        if 'bullets' not in self.things:
            return
        for index in range(len(self.things['bullets'])):
            xpos = self.things['bullets'][index].xpos
            ypos = self.things['bullets'][index].ypos
            # Bullets are stopped by protector field
            if 'protectors' in self.things:
                protectors = self.get_protectors_in_radius(xpos,
                                                           ypos,
                                                           self.protector_safe_distance)
                if len(protectors) > 0:
                    self.things['bullets'][index].lifespan = 0
                    if self.visuals == True:
                        self.things['bullets'][index].entity.disable()
            ls = self.things['bullets'][index].lifespan
            hit = False
            if ls > 0:
                ls = ls - 1
                self.update_bullet_position(index)
                # Check for agents hit
                victims = self.get_agents_in_radius(xpos, ypos, self.bullet_radius)
                if len(victims) > 0:
                    hit = True
                    dead = []
                    reset = []
                    self.shot += len(victims)
                    for v in victims:
                        if self.things['agents'][v].learnable == True:
                            reset.append(v)
                        else:
                            dead.append(v)
                    if len(reset) > 0:
                        self.reset_agents(reset)
                    if len(dead) > 0:
                        self.kill_agents(dead)
                if hit == True:
                    ls = 0
                # Handle bounded version - set ls to 0 of bullet hits edge
                if self.area_toroid == False:
                    if xpos == 0 or xpos == self.area_size or ypos == 0 or ypos == self.area_size:
                        ls = 0
                if ls < 0:
                    ls = 0
                self.things['bullets'][index].lifespan = ls
                if ls == 0:
                    if self.visuals == True:
                        self.things['bullets'][index].entity.disable()

    def update_bullet_position(self, index):
        return self.update_thing_position('bullets', index)

    def get_active_bullets(self):
        if 'bullets' not in self.things:
            return 0
        active = [i for i, x in enumerate(self.things['bullets']) if x.lifespan > 0]
        return active

    def get_bullets_in_radius(self, xpos, ypos, radius):
        active_b = self.get_active_bullets()
        all_b = self.get_things_in_radius('bullets', xpos, ypos, radius)
        actual = list(set(active_b).intersection(set(all_b)))
        return actual




#################
# Zone routines
#################
    def set_zone_entity(self, index):
        xabs = self.things['zones'][index].xpos
        yabs = self.things['zones'][index].ypos
        zabs = self.things['zones'][index].zpos
        ztype = self.things['zones'][index].ztype
        s = self.things['zones'][index].radius * 2
        c = self.things['zones'][index].zone_colors[ztype]
        alpha = self.things['zones'][index].strength*150
        c.append(alpha)
        self.things['zones'][index].entity = Entity(model='sphere',
                                          color=color.rgba(*c),
                                          scale=(s,s,s),
                                          position = (xabs, yabs, zabs))
        self.things['zones'][index].radius_entity = Entity(model='sphere',
                                                 color=color.rgba(*c),
                                                 scale=(s,s,s),
                                                 position = (xabs, yabs, zabs))

    def spawn_zones(self):
        if self.use_zones == False:
            return
        self.things['zones'] = []
        for vals in self.zone_config:
            ztype, xpos, ypos, radius, strength = vals
            self.spawn_zone(xpos, ypos, ztype, radius, strength)

    def spawn_zone(self, xpos, ypos, ztype, radius, strength):
        zpos = -1
        zone = Zone(xpos, ypos, zpos, ztype, radius, strength)
        self.things['zones'].append(zone)
        index = len(self.things['zones'])-1
        if self.visuals == True:
            self.set_zone_entity(index)

    def get_zones_in_radius(self, xpos, ypos, radius):
        return self.get_things_in_radius('zones', xpos, ypos, radius)

    def apply_zone_effects(self):
        if self.use_zones == False:
            return
        for index in range(len(self.things['zones'])):
            xpos = self.things['zones'][index].xpos
            ypos = self.things['zones'][index].ypos
            radius = self.things['zones'][index].radius
            strength = self.things['zones'][index].strength
            ztype = self.things['zones'][index].ztype
            affected = self.get_agents_in_radius(xpos, ypos, radius)
            if len(affected) > 0:
                for ai in affected:
                    ax = self.things['agents'][ai].xpos
                    ay = self.things['agents'][ai].ypos
                    angle = math.atan2(ay-ypos, ax-xpos)
                    xc = math.cos(angle)
                    yc = math.sin(angle)
                    axv = self.things['agents'][index].xvel
                    ayv = self.things['agents'][index].yvel
                    naxv = axv
                    nayv = ayv
                    if ztype == "acceleration":
                        naxv = axv * (1/(1 - strength))
                        nayv = ayv * (1/(1 - strength))
                    elif ztype == "damping":
                        naxv = axv * (1 - strength)
                        nayv = axv * (1 - strength)
                    elif ztype == "repulsor":
                        naxv = axv + (xc * strength)
                        nayv = ayv + (yc * strength)
                    elif ztype == "attractor":
                        naxv = axv - (xc * strength)
                        nayv = ayv - (yc * strength)
                    self.things['agents'][ai].xvel = naxv
                    self.things['agents'][ai].yvel = nayv

#################
# Food routines
#################
    def set_food_entity(self, index):
        xabs = self.things['food'][index].xpos
        yabs = self.things['food'][index].ypos
        zabs = self.things['food'][index].zpos
        s = 1
        c = [255, 153, 51, 120]
        self.things['food'][index].entity = Entity(model='sphere',
                                         color=color.rgba(*c),
                                         scale=(s,s,s),
                                         position = (xabs, yabs, zabs))

    def spawn_food(self):
        if self.num_food < 1:
            return
        self.things['food'] = []
        for index in range(self.num_food):
            xpos = random.random()*self.area_size
            ypos = random.random()*self.area_size
            zpos = -1
            food = Food(xpos, ypos, zpos)
            self.things['food'].append(food)
            if self.visuals == True:
                self.set_food_entity(index)

    def respawn_food(self, index):
        xpos = int(random.random()*self.area_size)
        ypos = int(random.random()*self.area_size)
        self.things['food'][index].xpos = xpos
        self.things['food'][index].ypos = ypos
        zpos = -1
        if self.visuals == True:
            self.things['food'][index].entity.position=(xpos, ypos, zpos)

    def get_food_in_radius(self, xpos, ypos, radius):
        return self.get_things_in_radius('food', xpos, ypos, radius)

########################################
# Routines related to genetic algorithms
########################################
    def create_genomes(self):
        self.genome_pool = []
        if os.path.exists(savedir + "/genome_store.pkl"):
            print("Loading genomes.")
            self.load_genomes()
        if len(self.genome_store) >= self.num_agents:
            self.genome_pool = self.get_best_genomes_from_store(self.num_agents, None)
        else:
            if len(self.genome_store) > 0:
                self.genome_pool = [x[0] for x in self.genome_store]
        if len(self.genome_pool) < self.num_agents:
            m = self.num_agents - len(self.genome_pool)
            print("Creating " + str(m) + " random genomes.")
            g = self.make_random_genomes(m)
            self.genome_pool.extend(g)

    def save_genomes(self):
        if len(self.genome_store) < 1:
            return
        with open(self.savedir + "/genome_store.pkl", "wb") as f:
            f.write(pickle.dumps(self.genome_store))

    def load_genomes(self):
        with open(self.savedir + "/genome_store.pkl", "rb") as f:
            self.genome_store = pickle.load(f)

    def make_single_weight(self):
        val = 0.0
        if self.integer_weights == False:
            val = random.uniform(-1*self.weight_range, self.weight_range)
        else:
            val = random.randint(-1*self.weight_range, self.weight_range)
            if val == 0:
                val = np.random.uniform(-0.1, 0.1)
        return float(val)

    def make_weights(self, num):
        if self.integer_weights == False:
            return np.random.uniform(-1*self.weight_range, self.weight_range,num)
        else:
            weights = []
            for _ in range(num):
                weights.append(self.make_single_weight())
            return weights

    def make_random_genome(self):
        genome = []
        for size in self.genome_size:
            weights = self.make_weights(size)
            genome.append(weights)
        genome = np.array(genome)
        return genome

    def make_random_genomes(self, num):
        genome_pool = []
        for _ in range(num):
            genome = self.make_random_genome()
            genome_pool.append(genome)
        return genome_pool

    def reproduce_genome(self, genome1, genome2):
        if self.evolve_by_block == True:
            return self.reproduce_genome_block(genome1, genome2)
        else:
            return self.reproduce_genome_full(genome1, genome2)

    def reproduce_genome_full(self, genome1, genome2):
        sizes = [len(x) for x in genome1]
        cg1 = []
        for item in genome1:
            cg1.extend(item)
        cg2 = []
        for item in genome1:
            cg2.extend(item)
        split = random.randint(1, len(cg1))
        entry = []
        if random.random() < 0.5:
            entry.extend(cg1[:split])
            entry.extend(cg2[split:])
        else:
            entry.extend(cg2[:split])
            entry.extend(cg1[split:])
        new_genome = []
        i = 0
        for s in sizes:
            b = entry[i:i+s]
            new_genome.append(b)
            i += s
        return new_genome

    def reproduce_genome_block(self, genome1, genome2):
        new_genome = []
        for index in range(len(genome1)):
            g1 = genome1[index]
            g2 = genome2[index]
            entry = []
            split = random.randint(1, len(g1))
            if random.random() < 0.5:
                entry = np.concatenate((g1[:split], g2[split:]))
            else:
                entry = np.concatenate((g2[:split], g1[split:]))
            new_genome.append(entry)
        return new_genome

    def mutate_genome(self, genome):
        if self.evolve_by_block == True:
            return self.mutate_genome_block(genome)
        else:
            return self.mutate_genome_full(genome)

    def mutate_genome_block(self, genome):
        new_genome = []
        for index in range(len(genome)):
            gen = genome[index]
            gen_len = len(gen)
            mutation_chance = self.mutation_rate * gen_len
            if random.random() < mutation_chance:
                index = random.choice(range(gen_len))
                val = self.make_single_weight()
                gen[index] = val
            new_genome.append(gen)
        return new_genome

    def mutate_genome_full(self, genome):
        sizes = [len(x) for x in genome]
        cg = []
        for item in genome:
            cg.extend(item)
        gen_len = len(cg)
        mutation_chance = self.mutation_rate * gen_len
        if random.random() < mutation_chance:
            index = random.choice(range(gen_len))
            val = self.make_single_weight()
            cg[index] = val
        new_genome = []
        i = 0
        for s in sizes:
            b = cg[i:i+s]
            new_genome.append(b)
            i += s
        return new_genome

    def make_new_offspring(self, genomes):
        parents = random.sample(genomes, 2)
        offspring = self.reproduce_genome(parents[0], parents[1])
        mutated = self.mutate_genome(offspring)
        return mutated

    def add_previous_agent(self, entry):
        if len(self.previous_agents) >= self.num_previous_agents:
            self.previous_agents.popleft()
        self.previous_agents.append(entry)

    def get_best_agents_from_store(self, num, atype):
        fitness = []
        if atype is not None:
            fitness = [x[self.fitness_index] for x in self.genome_store if x[5]==atype]
        else:
            fitness = [x[self.fitness_index] for x in self.genome_store]
        new_indices = []
        if len(fitness) > 2:
            #indices = np.argpartition(fitness,-num)[-num:]
            indices = range(len(fitness))
            min_fitness = min(fitness)
            min_fitness_sq = min_fitness ** 2
            for i in indices:
                num = min(10, int((fitness[i] ** 2)/min_fitness_sq))
                for n in range(num):
                    new_indices.append(i)
        if len(new_indices) > num:
            new_indices = random.sample(new_indices, num)
        return new_indices

    def get_best_genomes_from_store(self, num, atype):
        indices = self.get_best_agents_from_store(num, atype)
        if len(indices) >= num:
            return [self.genome_store[i][0] for i in indices]
        else:
            return self.make_random_genomes(num)

    def make_genome_from_store(self, atype):
        if len(self.genome_store) < 1:
            return self.make_random_genome()
        num_g = len(self.genome_store)
        genomes = self.get_best_genomes_from_store(num_g, atype)
        if len(genomes) > 1:
            return self.make_new_offspring(genomes)
        else:
            return self.make_random_genome()

    def make_gen_str(self, genome):
        full_gen = []
        for block in range(len(genome)):
            gen = genome[block]
            full_gen.extend(gen)
        gen_str = str(list(full_gen))
        return gen_str

    def is_genome_unique(self, genome):
        genome_strings = set()
        for index, entry in enumerate(self.genome_store):
            gen_str = self.full_gen_printable(entry[0])
            genome_strings.add(gen_str)
        gs = self.full_gen_printable(genome)
        if gs in genome_strings:
            return False
        else:
            return True

    def pop_min_genome_store(self):
        fitnesses = [x[self.fitness_index] for x in self.genome_store]
        min_item = np.argmin(fitnesses)
        self.genome_store.pop(min_item)

    def store_genome(self, entry):
        min_fitness = 0
        min_item = 0
        fitness = entry[self.fitness_index]
        if len(self.genome_store) > 0:
            fitnesses = [x[self.fitness_index] for x in self.genome_store]
            min_item = np.argmin(fitnesses)
            min_fitness = fitnesses[min_item]
            if len(self.genome_store) > self.genome_store_size:
                self.genome_store.pop(min_item)
        if fitness > self.agent_start_energy and fitness > min_fitness:
            self.genome_store.append(entry)
            self.last_discovery = self.spawns

    def make_new_genome(self, atype):
        return self.make_genome_from_store(atype)

    def full_gen_printable(self, gen):
        ret = ""
        for block in gen:
            ret += self.gen_printable(block)
        return ret

    def gen_printable(self, gen):
        msg = ""
        for item in gen:
            ch = "Z"
            if int(item) == -1:
                ch = "M"
            elif int(item) == 1:
                ch = "P"
            msg += ch
        return msg

    def get_genetic_diversity(self):
        if len(self.genome_store) < 1:
            return [[], 0]
        unique = []
        for item in self.genome_store[0][0]:
            if len(item) > 0:
                unique.append(set())
        for index in range(len(self.genome_store)):
            gen = self.genome_store[index][0]
            for count, g in enumerate(gen):
                gpr = self.gen_printable(g)
                unique[count].add(gpr)
        diversity = [len(x) for x in unique]
        mean_diversity = np.mean(diversity)
        return diversity, mean_diversity


######################
# Printable statistics
######################
    def record_stats(self, stat_name, stat_value):
        if self.steps % self.record_every == 0:
            if stat_name not in self.stats:
                self.stats[stat_name] = []
            self.stats[stat_name].append(stat_value)

    def save_stats(self):
        fp = os.path.join(self.savedir, "stats.json")
        with open(fp, "w") as f:
            f.write(json.dumps(self.stats))

    def load_stats(self):
        fp = os.path.join(self.savedir, "stats.json")
        if os.path.exists(fp):
            with open(fp, "r") as f:
                self.stats = json.loads(f.read())

    def make_labels(self, var, affix, saff):
        labels = ["fitness", "age", "happiness", "distance"]
        to_print = ["age"]
        if len(var) < 1:
            return ""
        lmsg = ""
        conds = ["evolving", "learning", "all     "]
        for cond, lab in enumerate(conds):
            if cond < 2:
                nvar = [x for x in var if x[5]==cond]
            else:
                nvar = var
            if len(nvar) > 0:
                lmsg += affix + lab + " agent (" + "%03d"%len(nvar) + ") stats: "
                #lmsg += "\n"
                for i, l in enumerate(labels):
                    temp = [x[i+1] for x in nvar]
                    if len(temp) > 0:
                        me = np.mean(temp)
                        self.record_stats(saff+"_"+lab+"_"+l+"_mean", me)
                        mx = max(temp)
                        self.record_stats(saff+"_"+lab+"_"+l+"_max", mx)
                        mi = min(temp)
                        self.record_stats(saff+"_"+lab+"_"+l+"_min", mi)
                        if l in to_print:
                            lmsg += "mean " + l + ": " + "%.2f"%me
                            lmsg += "  max " + l + ": " + "%.2f"%mx
                            lmsg += "  min " + l + ": " + "%.2f"%mi
                            lmsg += "\n"
                #lmsg += "\n"
        return lmsg

    def print_action_dist(self, atype):
        lmsg = atype + " agent action distribution:"
        lmsg += "\n"
        ta = sum([self.agent_actions[atype][x] for x in self.actions])
        if ta == 0:
            return ""
        bars = [int(200*(self.agent_actions[atype][x]/ta)) for x in self.actions]
        for i, x in enumerate(self.actions):
            space = (15-len(x))*" "
            bar = bars[i]*"#"
            lmsg += str(x) + ":" + space + bar + "\n"
            self.record_stats(atype+"_action_"+x, bars[i])
        lmsg += "\n"
        return lmsg

    def print_spawn_stats(self):
        msg = ""
        msg += "Spawns: " + str(self.spawns)
        msg += "  resets: " + str(self.resets)
        msg += "  rebirths: " + str(self.rebirths)
        msg += "  continuations: " + str(self.continuations)
        msg += "  deaths: " + str(self.deaths)
        msg += "  eaten: " + str(self.eaten)
        msg += "  shot: " + str(self.shot)
        msg += "\n"
        return msg

    def print_run_stats(self):
        genome_size = [x for x in self.genome_size]
        params = sum([x for x in self.genome_size])
        msg = ""
        msg += "Starting agents: " + str(self.num_agents)
        msg += "  learners: " + "%.2f"%(self.learners*self.num_agents)
        msg += "  area size: " + str(self.area_size)
        msg += "  predators: " + str(self.num_predators)
        msg += "\n"
        msg += "Action size: " + str(self.action_size)
        msg += "  state_size: " + str(self.observation_size)
        msg += "  genome size: " + str(genome_size)
        msg += "  params: " + str(params)
        msg += "\n"
        msg += "net_desc: " + str(self.net_desc)
        msg += "\n\n"
        return msg

    def print_new_state_stats(self):
        new_state_lens = [len(self.things['agents'][x].new_state) for x in range(len(self.things['agents']))]
        max_nsl = max(new_state_lens) * 3
        mean_nsl = np.mean(new_state_lens) * 3
        msg = ""
        msg += "Mean new state length: " + "%.2f"%mean_nsl
        msg += "  Max new state length: " + str(max_nsl)
        msg += "\n"
        return msg

    def print_observations(self):
        obs = "--"
        if len(self.things['agents'][0].previous_states) > 0:
            obs = self.things['agents'][0].previous_states[0]
        act = 0
        if len(self.things['agents'][0].previous_actions) > 0:
            act = self.things['agents'][0].previous_actions[-1]
        msg = ""
        msg += str(obs)
        msg += "\n"
        msg += str(act)
        msg += "\n"
        msg += "\n"
        return msg

    def get_stats(self):
        l_agents = sum([x.learnable for x in self.things['agents']])
        self.record_stats("learning agents", l_agents)
        e_agents = len(self.things['agents']) - l_agents
        self.record_stats("evolving agents", l_agents)

        msg = ""
        msg += self.print_run_stats()
        msg += "Step: " + str(self.steps)
        if self.inference == True:
            msg += " --INFERENCE--"
        msg += "\n"
        msg += "Agents: " + str(len(self.things['agents']))
        msg += "  learning: " + str(l_agents)
        msg += "  evolving: " + str(e_agents)
        msg += "  mutation rate: " + "%.5f"%self.mutation_rate
        msg += "\n\n"
        #msg += self.print_new_state_stats()
        #msg += self.print_observations()
        msg += self.print_spawn_stats()
        msg += "\n"
        msg += self.make_labels(self.previous_agents, "Previous ", "prev")
        msg += "\n"
        msg += self.make_labels(self.genome_store, "Gen_stor ", "gs")
        msg += "\n"
        gsd, mean_gsd = self.get_genetic_diversity()
        self.record_stats("genetic diversity", mean_gsd)
        msg += "Genetic diversity: " + str(gsd)
        msg += "  mean: " + "%.2f"%mean_gsd
        msg += "\n"
        for atype in self.agent_types:
            #msg += self.print_action_dist(atype)
            self.print_action_dist(atype)
        return msg

    def print_stats(self):
        os.system('clear')
        print(gs.get_stats())

# update callback for ursina
def update():
    gs.step()

    window.color = color.rgb(0,0,0)

    # Update agent positions
    for index, agent in enumerate(gs.things['agents']):
        if gs.things['agents'][index].entity is not None:
            xabs = gs.things['agents'][index].xpos
            yabs = gs.things['agents'][index].ypos
            zabs = gs.things['agents'][index].zpos
            s = 0.5 + ((gs.things['agents'][index].energy/200)*0.5)
            gs.things['agents'][index].entity.scale = (s,s,s)
            gs.things['agents'][index].entity.position = (xabs, yabs, zabs)
            gs.things['agents'][index].radius_entity.position = (xabs, yabs, zabs)
            orient = gs.things['agents'][index].orient
            gs.things['agents'][index].entity.rotation = (45*orient, 90, 0)
            for n in range(gs.things['agents'][index].trail_length):
                item = gs.things['agents'][index].previous_positions[n]
                x, y, o, ss = item
                ss = 0.5 + ((gs.things['agents'][index].energy/1000)*0.5)
                gs.things['agents'][index].trail_entities[n].position = (x, y, zabs)
                gs.things['agents'][index].trail_entities[n].rotation = (45*o, 90, 0)
                gs.things['agents'][index].trail_entities[n].scale = (ss, ss, ss)

    # Update predator positions
    for index, predator in enumerate(gs.things['predators']):
        if gs.things['predators'][index].entity is not None:
            xabs = gs.things['predators'][index].xpos
            yabs = gs.things['predators'][index].ypos
            zabs = gs.things['predators'][index].zpos
            gs.things['predators'][index].entity.position = (xabs, yabs, zabs)
            orient = gs.things['predators'][index].orient
            gs.things['predators'][index].entity.rotation = (45*orient, 90, 0)

    # Update shooter positions
    for index, shooter in enumerate(gs.things['shooters']):
        if gs.things['shooters'][index].entity is not None:
            xabs = gs.things['shooters'][index].xpos
            yabs = gs.things['shooters'][index].ypos
            zabs = gs.things['shooters'][index].zpos
            gs.things['shooters'][index].entity.position = (xabs, yabs, zabs)
            orient = gs.things['shooters'][index].orient
            gs.things['shooters'][index].entity.rotation = (45*orient, 90, 0)

    # Update bullet positions
    for index, bullet in enumerate(gs.things['bullets']):
        if gs.things['bullets'][index].entity is not None:
            xabs = gs.things['bullets'][index].xpos
            yabs = gs.things['bullets'][index].ypos
            zabs = gs.things['bullets'][index].zpos
            alpha = ((gs.things['bullets'][index].lifespan/(gs.bullet_life+1)) * 100) + 155
            gs.things['bullets'][index].entity.position = (xabs, yabs, zabs)
            gs.things['bullets'][index].entity.color = color.rgba(255,255,255,alpha)

    # Update protector positions
    for index, protector in enumerate(gs.things['protectors']):
        if gs.things['protectors'][index].entity is not None:
            xabs = gs.things['protectors'][index].xpos
            yabs = gs.things['protectors'][index].ypos
            zabs = gs.things['protectors'][index].zpos
            gs.things['protectors'][index].entity.position = (xabs, yabs, zabs)
            gs.things['protectors'][index].protection_entity.position = (xabs, yabs, zabs)
            gs.things['protectors'][index].pulse_entity.position = (xabs, yabs, zabs)
            if gs.pulse_zones == True:
                pes = abs(np.sin(gs.steps/10))*(gs.protector_safe_distance*2)
                gs.things['protectors'][index].pulse_entity.scale = (pes, pes, pes)
            orient = gs.things['protectors'][index].orient
            gs.things['protectors'][index].entity.rotation = (45*orient, 90, 0)

    # Animate pulse zones
    if gs.pulse_zones == True:
        for index, zone in enumerate(gs.things['zones']):
            radius = gs.things['zones'][index].radius
            pes = abs(np.sin(gs.steps/10))*(radius*2)
            gs.things['zones'][index].radius_entity.scale = (pes, pes, pes)


# Train the game
random.seed(123456)

print_visuals = False
inference = False

if len(sys.argv)>1:
    if "-v" in sys.argv[1:]:
        print_visuals = True
    elif "-i" in sys.argv[1:]:
        inference = True
        print_visuals = True

savedir = "alien_ecology_save"
if not os.path.exists(savedir):
    os.makedirs(savedir)

if print_visuals == True:
    app = Ursina()

    window.borderless = False
    window.fullscreen = False
    window.exit_button.visible = False
    window.fps_counter.enabled = True

    gs = game_space(visuals=True, inference=inference)
    camera_pos1 = -1 * int(gs.area_size/2)
    camera_pos2 = int(gs.area_size*2)

    camera.position -= (camera_pos1, camera_pos1, camera_pos2)
    app.run()

else:
    gs = game_space(visuals=False)
    while True:
        gs.step()

# Update report
# 1. reward is sparse
# 2. episodes take longer as training proceeds
# Training that happens after the initial boost seems to fall into local optima
# how to get initial training further towards actual real policies?
# How to model diversity/novelty in this environment?
# improve the entropy calculation of locations visited
#
# Predators cause energy drain in a radius instead of eating agents?
