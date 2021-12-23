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
        self.weights = weights
        self.l = l
        self.block = []
        for item in weights:
            if len(item) > 1:
                self.block.append([item[0][0], item[0][1], item[1][1]])
        self.num_actions = self.weights[-2][0][1]
        self.num_blocks = len(self.block)
        self.inps = [x[0] for x in self.block]
        self.out_cat = sum([x[-1] for x in self.block])
        self.blocks = {}
        for index in range(self.num_blocks):
            weights1 = self.weights[index][0][2]
            weights2 = self.weights[index][1][2]
            self.blocks[index] = nn.ModuleList()
            fc = nn.Linear(self.block[index][0], self.block[index][1])
            fc.weight.data = weights1
            self.blocks[index].append(fc)
            fc = nn.Linear(self.block[index][1], self.block[index][2])
            fc.weight.data = weights2
            self.blocks[index].append(fc)
        self.action = nn.Linear(self.out_cat, self.num_actions)
        self.action.weight.data = self.weights[-2][0][2]
        self.value = nn.Linear(self.out_cat, 1)
        self.value.weight.data = self.weights[-1][0][2]

    def forward(self, x):
        block_out = torch.empty((self.num_blocks, self.num_actions))
        current_index = 0
        for index in range(len(self.blocks)):
            i = x[0, current_index:current_index+self.inps[index]]
            a = F.relu(self.blocks[index][0](i))
            a = F.relu(self.blocks[index][1](a))
            block_out[index] = a
            current_index = current_index+self.inps[index]
        rc = torch.ravel(torch.tensor(block_out))
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
        for index in range(self.num_blocks):
            entry = []
            print("Block: " + str(index))
            d1 = self.blocks[index][0].weight.data.detach().numpy()
            print("fc0 weights: ", d1.shape)
            total_params += self.get_param_count(d1)
            b1 = self.blocks[index][0].bias.data.detach().numpy()
            print("fc0 biases: ", b1.shape)
            d2 = self.blocks[index][1].weight.detach().numpy()
            print("fc1 weights: ", d2.shape)
            total_params += self.get_param_count(d2)
            b2 = self.blocks[index][1].bias.data.detach().numpy()
            print("fc1 biases: ", b2.shape)
        da = self.action.weight.data.detach().numpy()
        total_params += self.get_param_count(da)
        print("action weights: ", da.shape)
        ba = self.action.bias.data.detach().numpy()
        print("action biases: ", ba.shape)
        dv = self.value.weight.data.detach().numpy()
        total_params += self.get_param_count(dv)
        print("value weights: ", dv.shape)
        bv = self.value.bias.data.detach().numpy()
        print("value biases: ", bv.shape)
        print("total params: ", total_params)
        genome_shape = [len(x) for x in genome]
        print(genome_shape)
        print()

    def get_w(self):
        genome = []
        for index in range(self.num_blocks):
            entry = []
            d1 = self.blocks[index][0].weight.data.detach().numpy()
            d1 = np.ravel(d1)
            entry.extend(list(d1))
            d2 = self.blocks[index][1].weight.detach().numpy()
            d2 = np.ravel(d2)
            entry.extend(list(d2))
            entry = np.ravel(entry)
            genome.append(list(entry))
        da = self.action.weight.data.detach().numpy()
        genome.append(list(np.ravel(da)))
        dv = self.value.weight.data.detach().numpy()
        genome.append(list(np.ravel(dv)))
        return genome

    def set_w(self, w):
        for index in range(self.num_blocks):
            weights1 = self.weights[index][0][2]
            weights2 = self.weights[index][1][2]
            self.blocks[index][0].weight.data = weights1
            self.blocks[index][1].weight.data = weights2
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
            self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
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
        gamma = 0.995
        R = 0.0
        policy_loss = torch.Tensor([0.0])
        value_loss = torch.Tensor([0.0])
        returns = []
        for r in self.rewards:
            R = r + gamma * R
            returns.insert(0, R.astype(np.float32))
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)
        for log_prob, value, R in zip(self.probs,
                                      self.values,
                                      returns):
            advantage = R - value.item()
            policy_loss = policy_loss - log_prob * advantage
            value_loss = value_loss + F.smooth_l1_loss(value, torch.tensor([R]))
        self.optimizer.zero_grad()
        loss = policy_loss + value_loss
        loss.backward()
        self.optimizer.step()
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
        self.trail_length = 3
        self.trail_alphas = [100, 60, 30]
        self.trail_entities = []
        for _ in range(self.trail_length):
            self.trail_entities.append(None)
        self.previous_stats = []
        self.reset()

    def make_weights(self):
        weights = []
        for index, item in enumerate(self.genome):
            entry = []
            layer_desc = self.net_desc[index]
            if len(layer_desc) > 2:
                s1, s2, o = layer_desc
                w = torch.Tensor(np.reshape(item[0:s1*s2], (s2, s1)))
                entry.append([s1, s2, w])
                w = torch.Tensor(np.reshape(item[s1*s2:], (o, s2)))
                entry.append([s2, o, w])
            else:
                s1, o = layer_desc
                w = torch.Tensor(np.reshape(item, (o, s1)))
                entry.append([s1, o, w])
            weights.append(entry)
        return weights

    def set_genome(self, genome):
        self.genome = genome
        self.set_w()

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
        self.inertial_damping = 0.75
        self.speed = 0.35
        self.previous_states = deque()
        self.previous_actions = []
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
        self.visible = 5
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
        self.visible = 5
        self.orient = 0 # 0-7 in 45 degree increments
        self.inertial_damping = 0.70
        self.entity = None

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
                 num_prev_states=1,
                 num_recent_actions=1000,
                 learners=0.50,
                 evaluate_learner_every=30,
                 mutation_rate=0.0001,
                 integer_weights=True,
                 weight_range=1,
                 area_size=70,
                 num_agents=20,
                 agent_start_energy=200,
                 agent_energy_drain=1,
                 agent_view_distance=5,
                 num_protectors=3,
                 protector_safe_distance=5,
                 num_predators=6,
                 predator_view_distance=6,
                 predator_kill_distance=2,
                 num_food=20,
                 use_zones=False,
                 visuals=False,
                 pulse_zones=False,
                 num_previous_agents=100,
                 genome_store_size=300,
                 fitness_index=2, # 1: fitness, 2: age
                 respawn_genome_store=0.9,
                 rebirth_genome_store=0.9,
                 top_n=0.2,
                 save_every=100,
                 record_every=50,
                 savedir="alien_ecology_save",
                 statsdir="alien_ecology_stats"):
        self.steps = 1
        self.spawns = 0
        self.resets = 0
        self.rebirths = 0
        self.continuations = 0
        self.births = 0
        self.deaths = 0
        self.eaten = 0
        self.num_recent_actions = num_recent_actions
        self.num_previous_agents = num_previous_agents
        self.fitness_index = fitness_index
        self.respawn_genome_store = respawn_genome_store
        self.rebirth_genome_store = rebirth_genome_store
        self.genome_store_size = genome_store_size
        self.top_n = top_n
        self.learners = learners
        self.evaluate_learner_every = evaluate_learner_every
        self.integer_weights = integer_weights
        self.weight_range = weight_range
        self.visuals = visuals
        self.pulse_zones = pulse_zones
        self.savedir = savedir
        self.statsdir = statsdir
        self.stats = {}
        self.load_stats()
        self.record_every = record_every
        self.save_every = save_every
        self.num_prev_states = num_prev_states
        self.area_size = area_size
        self.num_agents = num_agents
        self.agent_start_energy = agent_start_energy
        self.agent_energy_drain = agent_energy_drain
        self.num_protectors = num_protectors
        self.protector_safe_distance = protector_safe_distance
        self.num_predators = num_predators
        self.predator_view_distance = predator_view_distance
        self.predator_kill_distance = predator_kill_distance
        self.num_food = num_food
        self.use_zones = use_zones
        self.agent_view_distance = agent_view_distance
        self.visible_area = math.pi*(self.agent_view_distance**2)
        self.mutation_rate = mutation_rate
        self.agent_types = ["evolving",
                            "learning"]
        self.agent_actions = {}
        self.recent_actions = {}
        for t in self.agent_types:
            self.agent_actions[t] = Counter()
            self.recent_actions[t] = deque()
        self.actions = [
                        #"rotate_right",
                        #"rotate_left",
                        #"flip",
                        #"propel",
                        #"null",
                        "propel_up",
                        "propel_right",
                        "propel_down",
                        "propel_left",
                        ]
        self.observations = [
                             #"agents_up",
                             #"agents_right",
                             #"agents_down",
                             #"agents_left",
                             #"visible_agents",
                             ["food_up",
                             "food_right",
                             "food_down",
                             "food_left"],
                             #"visible_food",
                             ["protectors_up",
                             "protectors_right",
                             "protectors_down",
                             "protectors_left",
                             "protector_in_range"],
                             #"visible_protectors",
                             ["predators_up",
                             "predators_right",
                             "predators_down",
                             "predators_left"],
                             #"visible_predators",
                             #"own_energy",
                             ]
        self.observation_size = sum([len(x) for x in self.observations])
        self.net_desc = []
        for index, item in enumerate(self.observations):
            obs = len(self.observations[index])
            hidden = obs*2
            entry = [obs, hidden]
            self.net_desc.append(entry)
        self.action_size = len(self.actions)
        self.genome_size = []
        self.make_genome_size()
        self.genome_store = []
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
                            ['acceleration', mpos, mpos, bigger, 0.3]]
        self.spawn_zones()
        self.spawn_food()
        self.previous_agents = deque()
        self.create_predators()
        self.create_protectors()
        self.create_genomes()
        self.agents = []
        num_learners = int(self.num_agents * self.learners)
        num_evolvable = int(self.num_agents - num_learners)
        if num_learners > 0:
            self.create_new_learner_agents(num_learners)
        if num_evolvable > 0:
            self.create_new_evolvable_agents(num_evolvable)

    def make_genome_size(self):
        input_len = sum([x[0] for x in self.net_desc])
        if input_len != self.observation_size:
            msg = "Observation size: " + str(self.observation_size)
            msg += " does not match net_desc: " + str(self.net_desc)
            print(msg)
            sys.exit(0)
        for index in range(len(self.net_desc)):
            self.net_desc[index].append(self.action_size)
        state_size = sum([x[0] for x in self.net_desc])
        out_cat = sum([x[-1] for x in self.net_desc])
        for item in self.net_desc:
            gs = 0
            for i in range(len(item)-1):
                gs += item[i] * item[i+1]
            self.genome_size.append(gs)
        action_head = out_cat*self.action_size
        self.genome_size.append(action_head)
        self.net_desc.append([out_cat, self.action_size])
        value_head = out_cat*1
        self.genome_size.append(value_head)
        self.net_desc.append([out_cat, 1])

    def step(self):
        self.apply_protector_physics()
        self.run_protector_actions()
        self.apply_predator_physics()
        self.run_predator_actions()
        self.apply_zone_effects()
        self.set_agent_states()
        #self.make_new_states()
        self.apply_agent_physics()
        self.run_agent_actions()
        self.update_agent_status()
        if self.save_every > 0:
            if self.steps % self.save_every == 0:
                self.save_genomes()
        if self.save_every > 0 and self.steps % 500 == 0:
            self.save_stats()
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
        return self.update_position_toroid(xpos, ypos)


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

    def viewpoint(self, xpos, ypos, orient, distance):
        xv = xpos
        yv = ypos
        # up
        if orient == 0:
            yv = ypos + distance
        # up right
        elif orient == 1:
            yv = ypos + 0.5*distance
            xv = xpos + 0.5*distance
        # right
        elif orient == 2:
            xv = xpos + distance
        # down right
        elif orient == 3:
            yv = ypos - 0.5*distance
            xv = xpos + 0.5*distance
        # down
        elif orient == 4:
            yv = ypos - distance
        # down left
        elif orient == 5:
            yv = ypos - 0.5*distance
            xv = xpos - 0.5*distance
        # left
        elif orient == 6:
            xv = xpos - distance
        # up left
        elif orient == 7:
            yv = ypos + 0.5*distance
            xv = xpos - 0.5*distance
        if xv > self.area_size:
            xv -= self.area_size
        if xv < 0:
            xv += self.area_size
        if yv > self.area_size:
            yv -= self.area_size
        if yv < 0:
            yv += self.area_size
        return xv, yv

    def get_viewpoint_in_direction(self, index, direction):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        distance = self.agent_view_distance
        xv, yv = self.viewpoint(xpos, ypos, direction, distance)
        return xv, yv

    def get_viewpoint_up(self, index):
        return self.get_viewpoint_in_direction(index, 0)

    def get_viewpoint_right(self, index):
        return self.get_viewpoint_in_direction(index, 2)

    def get_viewpoint_down(self, index):
        return self.get_viewpoint_in_direction(index, 4)

    def get_viewpoint_left(self, index):
        return self.get_viewpoint_in_direction(index, 6)

    def get_viewpoint(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        orient = self.agents[index].orient
        distance = self.agent_view_distance
        xv, yv = self.viewpoint(xpos, ypos, orient, distance)
        return xv, yv

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

########################
# Agent runtime routines
########################
    def set_agent_entity(self, index):
        xabs = self.agents[index].xpos
        yabs = self.agents[index].ypos
        zabs = self.agents[index].zpos
        s = 0.5 + ((self.agents[index].energy/self.agent_start_energy)*0.5)
        texture = "textures/" + self.agents[index].color_str
        self.agents[index].entity = Entity(model='sphere',
                                           color=color.white,
                                           scale=(s,s,s),
                                           position = (xabs, yabs, zabs),
                                           texture=texture)
        for n in range(self.agents[index].trail_length):
            item = self.agents[index].previous_positions[n]
            x, y, o, ss = item
            ss = 0.5 + ((self.agents[index].energy/self.agent_start_energy)*0.5)
            self.agents[index].trail_entities[n] = Entity( model='sphere',
                                                           color=color.rgba(255,255,255,self.agents[index].trail_alphas[n]),
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
        self.agents.append(a)
        index = len(self.agents)-1
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
        for n in range(len(self.agents)):
            action = int(self.agents[n].get_action())
            self.agents[n].previous_actions.append(action)
            atype = 0
            if self.agents[n].learnable == True:
                atype = 1
            self.record_recent_actions(action, atype)
            self.apply_agent_action(n, action)
            self.update_agent_position(n)

    def get_agents_in_radius(self, xpos, ypos, radius):
        ret = []
        for i in range(len(self.agents)):
            ax = self.agents[i].xpos
            ay = self.agents[i].ypos
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                if self.distance(xpos, ypos, ax, ay) <= radius:
                    ret.append(i)
        return ret

    def get_visible_agents(self, index):
        xv, yv = self.get_viewpoint(index)
        agents = self.get_agents_in_radius(xv, yv, self.agent_view_distance*2)
        return(len(agents))

    def get_agents_up(self, index):
        xv, yv = self.get_viewpoint_up(index)
        agents = self.get_agents_in_radius(xv, yv, self.agent_view_distance)
        return(len(agents))

    def get_agents_right(self, index):
        xv, yv = self.get_viewpoint_right(index)
        agents = self.get_agents_in_radius(xv, yv, self.agent_view_distance)
        return(len(agents))

    def get_agents_down(self, index):
        xv, yv = self.get_viewpoint_down(index)
        agents = self.get_agents_in_radius(xv, yv, self.agent_view_distance)
        return(len(agents))

    def get_agents_left(self, index):
        xv, yv = self.get_viewpoint_left(index)
        agents = self.get_agents_in_radius(xv, yv, self.agent_view_distance)
        return(len(agents))

    def get_adjacent_agent_indices(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        return self.get_agents_in_radius(xpos, ypos, self.agent_view_distance)

    def get_adjacent_agent_count(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        agents = self.get_agents_in_radius(xpos, ypos, self.agent_view_distance)
        agent_count = len(agents)
        return agent_count

    def get_nearest_agent(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        distances = []
        for i in range(len(self.agents)):
            ax = self.agents[i].xpos
            ay = self.agents[i].ypos
            radius = self.agent_view_distance
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                distances.append(self.distance(xpos, ypos, ax, ay))
        if len(distances) > 1:
            shortest_index = np.argsort(distances, axis=0)[1]
            shortest_value = distances[shortest_index]
            return shortest_index, shortest_value
        else:
            return None, None

    def evaluate_learner(self, index):
        if len(self.agents[index].previous_stats) >= self.evaluate_learner_every:
            mpf = np.mean([x[self.fitness_index] for x in self.agents[index].previous_stats])
            pfm = 0
            method = 0
            if random.random() < self.rebirth_genome_store:
                method = 1
            if method == 1:
                if len(self.genome_store) > 1:
                    pfm = np.mean([x[self.fitness_index] for x in self.genome_store])
            else:
                if len(self.previous_agents) > 1:
                    pfm = np.mean([x[self.fitness_index] for x in self.previous_agents])
            self.agents[index].previous_stats = []
            if pfm > 0:
                if mpf < pfm * 0.75:
                    g = None
                    if method == 1:
                        num_g = min(len(self.genome_store), int(self.genome_store_size * self.top_n))
                        g = random.choice(self.get_best_genomes_from_store(num_g, None))
                    else:
                        num_g = min(len(self.previous_agents), int(self.num_previous_agents * self.top_n))
                        g = random.choice(self.get_best_previous_genomes(num_g, None))
                    if g is not None and len(g) == self.genome_size:
                        self.agents[index].set_genome(g)
                        self.rebirths += 1
                else:
                    self.continuations += 1

    def reset_agents(self, reset):
        for index in reset:
            l = int(self.agents[index].learnable)
            ae = self.entropy(self.agents[index].previous_actions) * 0.1
            a = self.agents[index].age
            h = self.agents[index].happiness
            d = self.agents[index].distance_travelled
            f = a + h + d
            g = self.agents[index].model.get_w()
            entry = [g, f, a, h, d, l]
            a1 = 0
            a2 = 0
            store = self.previous_agents
            if random.random() < self.rebirth_genome_store:
                store = self.genome_store
            if len(store) > 0:
                a1 = np.mean([x[self.fitness_index] for x in store])
            self.store_genome(entry)
            self.add_previous_agent(entry)
            if len(store) > 0:
                a2 = np.mean([x[self.fitness_index] for x in store])
            reward = max(-1, (a2 - a1) * 10)
            reward = 0
            if reward == 0:
                reward = ((a-self.agent_start_energy)/self.agent_start_energy) + ae
            reward = np.float32(reward)
            if len(self.agents[index].model.rewards) > 0:
                self.agents[index].model.rewards[-1] += reward
            self.agents[index].previous_stats.append(entry)
            self.deaths += 1
            self.resets += 1
            self.agents[index].model.finish_episode()
            self.agents[index].reset()
            xpos, ypos = self.get_safe_spawn_location()
            self.agents[index].xpos = xpos
            self.agents[index].ypos = ypos
            self.set_initial_agent_state(index)
            self.evaluate_learner(index)

    def kill_agents(self, dead):
        for index in dead:
            l = int(self.agents[index].learnable)
            ae = self.entropy(self.agents[index].previous_actions) * 0.1
            a = self.agents[index].age
            h = self.agents[index].happiness
            d = self.agents[index].distance_travelled
            f = a + h + d
            g = self.agents[index].model.get_w()
            entry = [g, f, a, h, d, l]
            self.deaths += 1
            self.store_genome(entry)
            self.add_previous_agent(entry)
            self.agents[index].reset()
            xpos, ypos = self.get_safe_spawn_location()
            self.agents[index].xpos = xpos
            self.agents[index].ypos = ypos
            genome = self.make_new_genome(0)
            self.agents[index].set_genome(g)
            self.set_initial_agent_state(index)
            self.spawns += 1

    def update_agent_status(self):
        dead = set()
        reset = set()
        for index in range(len(self.agents)):
            self.agents[index].age += 1
            xpos = self.agents[index].xpos
            ypos = self.agents[index].ypos
            if self.num_food > 0:
                nearby_food = self.get_food_in_radius(xpos, ypos, 1)
                if len(nearby_food) > 0:
                    fi = random.choice(nearby_food)
                    self.respawn_food(fi)
                    self.agents[index].energy += 50
                    self.agents[index].happiness += 10
                    # Reward learning agents for eating food
                    if len(self.agents[index].model.rewards) > 0:
                        self.agents[index].model.rewards[-1] = np.float32(1.0)
            energy_drain = self.agent_energy_drain
            protectors = self.get_protectors_in_radius(xpos, ypos, self.protector_safe_distance)
            if len(protectors) > 0:
                energy_drain = -1
            self.agents[index].energy -= energy_drain
            if self.agents[index].energy <= 0:
                if self.agents[index].learnable == False:
                    dead.add(index)
                else:
                    reset.add(index)
        if len(reset) > 0:
            self.reset_agents(reset)
        if len(dead) > 0:
            self.kill_agents(dead)

    def update_agent_position(self, index):
        x1 = self.agents[index].xpos
        y1 = self.agents[index].ypos
        o = self.agents[index].orient
        e = self.agents[index].energy
        x2 = x1 + self.agents[index].xvel
        y2 = y1 + self.agents[index].yvel
        d = self.distance(x1, y1, x2, y2)
        self.agents[index].distance_travelled += d
        nx, ny = self.update_position(x2, y2)
        self.agents[index].xpos = nx
        self.agents[index].ypos = ny
        self.agents[index].previous_positions.popleft()
        self.agents[index].previous_positions.append([x2, y2, o, e])

    def apply_agent_physics(self):
        for index in range(len(self.agents)):
            self.apply_agent_inertial_damping(index)

    def apply_agent_inertial_damping(self, index):
        self.agents[index].xvel = self.agents[index].xvel * self.agents[index].inertial_damping
        self.agents[index].yvel = self.agents[index].yvel * self.agents[index].inertial_damping

# Agent actions
    def apply_agent_action(self, index, action):
        self.agents[index].prev_action = action
        agent_function = "action_" + self.actions[action]
        class_method = getattr(self, agent_function)
        if self.agents[index].xvel == 0 and self.agents[index].yvel == 0:
            self.agents[index].happiness -= 1
        reward = class_method(index)
        if self.agents[index].learnable == True:
            self.agents[index].model.record_reward(reward)

    def action_null(self, index):
        return 0

    def action_rotate_right(self, index):
        self.agents[index].orient += 1
        if self.agents[index].orient > 7:
            self.agents[index].orient = 0
        return 0

    def action_rotate_left(self, index):
        self.agents[index].orient -= 1
        if self.agents[index].orient < 0:
            self.agents[index].orient = 7
        return 0

    def action_flip(self, index):
        self.agents[index].orient += 4
        if self.agents[index].orient > 7:
            self.agents[index].orient -= 7
        return 0

    def action_propel(self, index):
        orient = self.agents[index].orient
        speed = self.agents[index].speed
        xvel = self.agents[index].xvel
        yvel = self.agents[index].yvel
        xv, yv = self.propel(xvel, yvel, orient, speed)
        self.agents[index].xvel = xv
        self.agents[index].yvel = yv
        return 0

    def propel_agent_in_direction(self, index, direction):
        speed = self.agents[index].speed
        xvel = self.agents[index].xvel
        yvel = self.agents[index].yvel
        xv, yv = self.propel(xvel, yvel, direction, speed)
        self.agents[index].xvel = xv
        self.agents[index].yvel = yv
        return 0

    def action_propel_up(self, index):
        return self.propel_agent_in_direction(index, 0)

    def action_propel_right(self, index):
        return self.propel_agent_in_direction(index, 2)

    def action_propel_down(self, index):
        return self.propel_agent_in_direction(index, 4)

    def action_propel_left(self, index):
        return self.propel_agent_in_direction(index, 6)

# Agent observations
    def make_new_state(self, index):
        state = []
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        nearby = self.get_agents_in_radius(xpos, ypos, self.agent_view_distance)
        if len(nearby) > 0:
            for ai in nearby:
                ax = self.agents[ai].xpos
                ay = self.agents[ai].ypos
                angle = math.atan2(ay-ypos, ax-xpos)
                distance = self.distance(xpos, ypos, ax, ay)
                state.append([1, distance, angle])
        nearby = self.get_predators_in_radius(xpos, ypos, self.agent_view_distance)
        if len(nearby) > 0:
            for ai in nearby:
                ax = self.predators[ai].xpos
                ay = self.predators[ai].ypos
                angle = math.atan2(ay-ypos, ax-xpos)
                distance = self.distance(xpos, ypos, ax, ay)
                state.append([2, distance, angle])
        nearby = self.get_protectors_in_radius(xpos, ypos, self.agent_view_distance)
        if len(nearby) > 0:
            for ai in nearby:
                ax = self.protectors[ai].xpos
                ay = self.protectors[ai].ypos
                angle = math.atan2(ay-ypos, ax-xpos)
                distance = self.distance(xpos, ypos, ax, ay)
                state.append([3, distance, angle])
        return state

    def make_new_states(self):
        for index in range(len(self.agents)):
            new_state = self.make_new_state(index)
            self.agents[index].new_state = new_state

    def set_agent_states(self):
        for n in range(len(self.agents)):
            self.set_agent_state(n)

    def set_initial_agent_state(self, index):
        prev_states = []
        for _ in range(self.num_prev_states):
            entry = np.zeros(len(self.observations))
            prev_states.append(entry)
        self.agents[index].previous_states = deque(prev_states)

    def set_agent_state(self, index):
        self.agents[index].previous_states.popleft()
        current_observations = self.get_agent_observations(index)
        self.agents[index].previous_states.append(current_observations)
        all_states = self.agents[index].previous_states
        state = np.ravel(all_states)
        state = torch.FloatTensor(state)
        state = state.unsqueeze(0)
        self.agents[index].state = state

    def get_agent_observations(self, index):
        function_names = []
        for block in self.observations:
            for n in block:
                function_names.append("get_" + n)
        observations = []
        for fn in function_names:
            class_method = getattr(self, fn)
            val = class_method(index)
            observations.append(val)
        return observations

    def get_own_energy(self, index):
        return self.agents[index].energy/self.agent_start_energy

###################
# Protector actions
###################
    def set_protector_entity(self, index):
        xabs = self.protectors[index].xpos
        yabs = self.protectors[index].ypos
        zabs = self.protectors[index].zpos
        s = self.protector_safe_distance*2
        texture = "textures/white"
        self.protectors[index].entity = Entity(model='sphere',
                                              color=color.white,
                                              scale=(2,2,2),
                                              position = (xabs, yabs, zabs),
                                              texture=texture)
        self.protectors[index].protection_entity = Entity(model='sphere',
                                                          color=color.rgba(0,102,0,50),
                                                          scale=(s,s,s),
                                                          position = (xabs, yabs, zabs))
        self.protectors[index].pulse_entity = Entity(model='sphere',
                                                     color=color.rgba(0,102,0,30),
                                                     scale=(s,s,s),
                                                     position = (xabs, yabs, zabs))

    def create_protectors(self):
        self.protectors = []
        for n in range(self.num_protectors):
            self.spawn_random_protector()

    def spawn_random_protector(self):
        xpos, ypos = self.get_safe_spawn_location()
        zpos = -1
        a = Protector(xpos,
                     ypos,
                     zpos)
        self.protectors.append(a)
        index = len(self.protectors)-1
        if self.visuals == True:
            self.set_protector_entity(index)

    def get_protector_in_range(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        p = self.get_protectors_in_radius(xpos, ypos, self.protector_safe_distance)
        if len(p) > 0:
            return 1
        return 0

    def get_protectors_in_radius(self, xpos, ypos, radius):
        ret = []
        for i in range(len(self.protectors)):
            ax = self.protectors[i].xpos
            ay = self.protectors[i].ypos
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                if self.distance(xpos, ypos, ax, ay) <= radius:
                    ret.append(i)
        return ret

    def get_surrounding_protectors(self, index):
        xv = self.agents[index].xpos
        yv = self.agents[index].ypos
        protectors = self.get_protectors_in_radius(xv, yv, self.agent_view_distance)
        protector_count = len(protectors)
        return protector_count

    def get_visible_protectors(self, index):
        xv, yv = self.get_viewpoint(index)
        protectors = self.get_protectors_in_radius(xv, yv, self.agent_view_distance*2)
        protector_count = len(protectors)
        return protector_count

    def get_protectors_up(self, index):
        xv, yv = self.get_viewpoint_up(index)
        protectors = self.get_protectors_in_radius(xv, yv, self.agent_view_distance)
        protector_count = len(protectors)
        return protector_count

    def get_protectors_right(self, index):
        xv, yv = self.get_viewpoint_right(index)
        protectors = self.get_protectors_in_radius(xv, yv, self.agent_view_distance)
        protector_count = len(protectors)
        return protector_count

    def get_protectors_down(self, index):
        xv, yv = self.get_viewpoint_down(index)
        protectors = self.get_protectors_in_radius(xv, yv, self.agent_view_distance)
        protector_count = len(protectors)
        return protector_count

    def get_protectors_left(self, index):
        xv, yv = self.get_viewpoint_left(index)
        protectors = self.get_protectors_in_radius(xv, yv, self.agent_view_distance)
        protector_count = len(protectors)
        return protector_count

    def protector_move_random(self, index):
        action = random.choice([0,1,1,1,2,3])
        if action == 0:
            pass
        elif action == 1:
            self.protector_propel(index)
        elif action == 2:
            self.protector_rotate_right(index)
        else:
            self.protector_rotate_left(index)

    def protector_propel(self, index):
        orient = self.protectors[index].orient
        speed = self.protectors[index].speed
        xvel = self.protectors[index].xvel
        yvel = self.protectors[index].yvel
        xv, yv = self.propel(xvel, yvel, orient, speed)
        self.protectors[index].xvel = xv
        self.protectors[index].yvel = yv

    def protector_rotate_right(self, index):
        self.protectors[index].orient += 1
        if self.protectors[index].orient > 7:
            self.protectors[index].orient = 0

    def protector_rotate_left(self, index):
        self.protectors[index].orient -= 1
        if self.protectors[index].orient < 0:
            self.protectors[index].orient = 7

    def apply_protector_physics(self):
        for index in range(len(self.protectors)):
            self.apply_protector_inertial_damping(index)

    def apply_protector_inertial_damping(self, index):
        self.protectors[index].xvel = self.protectors[index].xvel * self.protectors[index].inertial_damping
        self.protectors[index].yvel = self.protectors[index].yvel * self.protectors[index].inertial_damping

    def update_protector_position(self, index):
        xpos = self.protectors[index].xpos 
        xpos += self.protectors[index].xvel
        ypos = self.protectors[index].ypos
        ypos += self.protectors[index].yvel
        xpos, ypos = self.update_position(xpos, ypos)
        self.protectors[index].xpos = xpos
        self.protectors[index].ypos = ypos

    def run_protector_actions(self):
        for index in range(len(self.protectors)):
            self.protector_move_random(index)
            self.update_protector_position(index)

##################
# Predator actions
##################
    def set_predator_entity(self, index):
        xabs = self.predators[index].xpos
        yabs = self.predators[index].ypos
        zabs = self.predators[index].zpos
        s = self.predator_kill_distance
        texture = "textures/red"
        self.predators[index].entity = Entity(model='sphere',
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
        self.predators.append(a)
        index = len(self.predators)-1
        if self.visuals == True:
            self.set_predator_entity(index)

    def create_predators(self):
        self.predators = []
        for n in range(self.num_predators):
            self.spawn_random_predator()

    def get_predators_in_radius(self, xpos, ypos, radius):
        ret = []
        for i in range(len(self.predators)):
            ax = self.predators[i].xpos
            ay = self.predators[i].ypos
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                if self.distance(xpos, ypos, ax, ay) <= radius:
                    ret.append(i)
        return ret

    def get_surrounding_predators(self, index):
        xv = self.agents[index].xpos
        yv = self.agents[index].ypos
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance)
        predator_count = len(predators)
        return predator_count

    def get_visible_predators(self, index):
        xv, yv = self.get_viewpoint(index)
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance*2)
        predator_count = len(predators)
        return predator_count

    def get_predators_up(self, index):
        xv, yv = self.get_viewpoint_up(index)
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance)
        predator_count = len(predators)
        return predator_count

    def get_predators_right(self, index):
        xv, yv = self.get_viewpoint_right(index)
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance)
        predator_count = len(predators)
        return predator_count

    def get_predators_down(self, index):
        xv, yv = self.get_viewpoint_down(index)
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance)
        predator_count = len(predators)
        return predator_count

    def get_predators_left(self, index):
        xv, yv = self.get_viewpoint_left(index)
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance)
        predator_count = len(predators)
        return predator_count

    def predator_move_random(self, index):
        action = random.choice([0,1,2])
        if action == 0:
            self.predator_propel(index)
        elif action == 1:
            self.predator_rotate_right(index)
        else:
            self.predator_rotate_left(index)

    def predator_propel(self, index):
        orient = self.predators[index].orient
        speed = self.predators[index].speed
        xvel = self.predators[index].xvel
        yvel = self.predators[index].yvel
        xv, yv = self.propel(xvel, yvel, orient, speed)
        self.predators[index].xvel = xv
        self.predators[index].yvel = yv

    def predator_rotate_right(self, index):
        self.predators[index].orient += 1
        if self.predators[index].orient > 7:
            self.predators[index].orient = 0

    def predator_rotate_left(self, index):
        self.predators[index].orient -= 1
        if self.predators[index].orient < 0:
            self.predators[index].orient = 7

    def apply_predator_physics(self):
        for index in range(len(self.predators)):
            self.apply_predator_inertial_damping(index)

    def apply_predator_inertial_damping(self, index):
        self.predators[index].xvel = self.predators[index].xvel * self.predators[index].inertial_damping
        self.predators[index].yvel = self.predators[index].yvel * self.predators[index].inertial_damping

    def update_predator_position(self, index):
        xpos = self.predators[index].xpos 
        xpos += self.predators[index].xvel
        ypos = self.predators[index].ypos
        ypos += self.predators[index].yvel
        xpos, ypos = self.update_position(xpos, ypos)
        self.predators[index].xpos = xpos
        self.predators[index].ypos = ypos

    def run_predator_actions(self):
        for index in range(len(self.predators)):
            xpos = self.predators[index].xpos
            ypos = self.predators[index].ypos
            radius = self.predator_kill_distance
            # If agents near enough to predator, kill them
            victims = self.get_agents_in_radius(xpos, ypos, radius)
            if len(victims) > 0:
                dead = []
                reset = []
                self.eaten += len(victims)
                try:
                    for v in victims:
                        if self.agents[v].learnable == True:
                            reset.append(v)
                        else:
                            dead.append(v)
                except:
                    print(victims)
                    sys.stdout.write('\a\a\a\a\a')
                    sys.stdout.flush()
                    sys.exit(0)
                if len(reset) > 0:
                    self.reset_agents(reset)
                if len(dead) > 0:
                    self.kill_agents(dead)
            # If protectors are near to predators, stop predator's movement
            # and prevent them from taking actions
            # Note predators can still eat agents in this state
            p = self.get_protectors_in_radius(xpos, ypos, self.protector_safe_distance)
            if len(p) > 0:
                self.apply_predator_inertial_damping(index)
            else:
                orient = self.predators[index].orient
                distance = self.predator_view_distance
                xv, yv = self.viewpoint(xpos, ypos, orient, distance)
                # If agents in viewfield, move forward
                targets = self.get_agents_in_radius(xv, yv, distance)
                if len(targets) > 0:
                    self.predator_propel(index)
                else:
                    self.predator_move_random(index)
            self.update_predator_position(index)

#################
# Zone routines
#################
    def spawn_zones(self):
        if self.use_zones == False:
            return
        self.zones = []
        for vals in self.zone_config:
            ztype, xpos, ypos, radius, strength = vals
            self.spawn_zone(xpos, ypos, ztype, radius, strength)

    def spawn_zone(self, xpos, ypos, ztype, radius, strength):
        zpos = -1
        zone = Zone(xpos, ypos, zpos, ztype, radius, strength)
        self.zones.append(zone)
        index = len(self.zones)-1
        if self.visuals == True:
            self.set_zone_entity(index)

    def get_zones_in_radius(self, xpos, ypos, radius):
        ret = []
        for i in range(len(self.zones)):
            ax = self.zones[i].xpos
            ay = self.zones[i].ypos
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                if self.distance(xpos, ypos, ax, ay) <= radius:
                    ret.append(i)
        return ret

    def apply_zone_effects(self):
        if self.use_zones == False:
            return
        for index in range(len(self.zones)):
            xpos = self.zones[index].xpos
            ypos = self.zones[index].ypos
            radius = self.zones[index].radius
            strength = self.zones[index].strength
            ztype = self.zones[index].ztype
            affected = self.get_agents_in_radius(xpos, ypos, radius)
            if len(affected) > 0:
                for ai in affected:
                    ax = self.agents[ai].xpos
                    ay = self.agents[ai].ypos
                    angle = math.atan2(ay-ypos, ax-xpos)
                    xc = math.cos(angle)
                    yc = math.sin(angle)
                    axv = self.agents[index].xvel
                    ayv = self.agents[index].yvel
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
                    self.agents[ai].xvel = naxv
                    self.agents[ai].yvel = nayv

    def set_zone_entity(self, index):
        xabs = self.zones[index].xpos
        yabs = self.zones[index].ypos
        zabs = self.zones[index].zpos
        ztype = self.zones[index].ztype
        s = self.zones[index].radius * 2
        c = self.zones[index].zone_colors[ztype]
        alpha = self.zones[index].strength*150
        c.append(alpha)
        self.zones[index].entity = Entity(model='sphere',
                                          color=color.rgba(*c),
                                          scale=(s,s,s),
                                          position = (xabs, yabs, zabs))
        self.zones[index].radius_entity = Entity(model='sphere',
                                                 color=color.rgba(*c),
                                                 scale=(s,s,s),
                                                 position = (xabs, yabs, zabs))

#################
# Food routines
#################
    def spawn_food(self):
        if self.num_food < 1:
            return
        self.food = []
        for index in range(self.num_food):
            xpos = random.random()*self.area_size
            ypos = random.random()*self.area_size
            zpos = -1
            food = Food(xpos, ypos, zpos)
            self.food.append(food)
            if self.visuals == True:
                self.set_food_entity(index)

    def respawn_food(self, index):
        xpos = random.random()*self.area_size
        ypos = random.random()*self.area_size
        self.food[index].xpos = xpos
        self.food[index].ypos = ypos
        zpos = -1
        if self.visuals == True:
            self.food[index].entity.position=(xpos, ypos, zpos)

    def get_food_in_radius(self, xpos, ypos, radius):
        ret = []
        for i in range(len(self.food)):
            ax = self.food[i].xpos
            ay = self.food[i].ypos
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                if self.distance(xpos, ypos, ax, ay) <= radius:
                    ret.append(i)
        return ret

    def get_surrounding_food(self, index):
        xv = self.agents[index].xpos
        yv = self.agents[index].ypos
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance)
        food_count = len(food)
        return food_count

    def get_visible_food(self, index):
        xv, yv = self.get_viewpoint(index)
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance*2)
        food_count = len(food)
        return food_count

    def get_food_up(self, index):
        xv, yv = self.get_viewpoint_up(index)
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance)
        food_count = len(food)
        return food_count

    def get_food_right(self, index):
        xv, yv = self.get_viewpoint_right(index)
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance)
        food_count = len(food)
        return food_count

    def get_food_down(self, index):
        xv, yv = self.get_viewpoint_down(index)
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance)
        food_count = len(food)
        return food_count

    def get_food_left(self, index):
        xv, yv = self.get_viewpoint_left(index)
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance)
        food_count = len(food)
        return food_count

    def set_food_entity(self, index):
        xabs = self.food[index].xpos
        yabs = self.food[index].ypos
        zabs = self.food[index].zpos
        s = 1
        c = [255, 153, 51, 120]
        self.food[index].entity = Entity(model='sphere',
                                         color=color.rgba(*c),
                                         scale=(s,s,s),
                                         position = (xabs, yabs, zabs))

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

    def make_weights(self, num):
        if self.integer_weights == False:
            return np.random.uniform(-1*self.weight_range, self.weight_range,num)
        else:
            return np.random.randint(-1*self.weight_range, self.weight_range+1, num)

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
        new_genome = []
        for index in range(len(genome)):
            gen = genome[index]
            gen_len = len(gen)
            mutation_chance = (1/self.mutation_rate) * gen_len
            if random.random() < mutation_chance:
                num_mutations = min(1, int(mutation_chance))
                index = random.sample(range(gen_len), num_mutations)
                val = 0
                if self.integer_weights == False:
                    val = random.uniform(-1*self.weight_range, self.weight_range)
                else:
                    val = random.randint(-1*self.weight_range, self.weight_range+1)
                gen[index] = val
            new_genome.append(gen)
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

    def get_best_previous_agents(self, num, atype):
        fitness = []
        if atype is not None:
            fitness = [x[self.fitness_index] for x in self.previous_agents if x[5]==atype]
        else:
            fitness = [x[self.fitness_index] for x in self.previous_agents]
        indices = []
        if len(fitness) > num:
            indices = np.argpartition(fitness,-num)[-num:]
        return indices

    def get_best_previous_genomes(self, num, atype):
        indices = self.get_best_previous_agents(num, atype)
        if len(indices) >= num:
            return [self.previous_agents[i][0] for i in indices]
        else:
            return self.make_random_genomes(num)

    def make_genome_from_previous(self, atype):
        num_g = int(self.num_previous_agents * self.top_n)
        if len(self.previous_agents) < num_g:
            return self.make_random_genome()
        genomes = self.get_best_previous_genomes(num_g, atype)
        if len(genomes) > 0:
            return self.make_new_offspring(genomes)
        else:
            return self.make_random_genome()

    def get_best_agents_from_store(self, num, atype):
        fitness = []
        if atype is not None:
            fitness = [x[self.fitness_index] for x in self.genome_store if x[5]==atype]
        else:
            fitness = [x[self.fitness_index] for x in self.genome_store]
        indices = []
        if len(fitness) > num:
            indices = np.argpartition(fitness,-num)[-num:]
        return indices

    def get_best_genomes_from_store(self, num, atype):
        indices = self.get_best_agents_from_store(num, atype)
        if len(indices) >= num:
            return [self.genome_store[i][0] for i in indices]
        else:
            return self.make_random_genomes(num)

    def make_genome_from_store(self, atype):
        num_g = int(self.genome_store_size * self.top_n)
        if len(self.genome_store) < num_g:
            return self.make_random_genome()
        genomes = self.get_best_genomes_from_store(num_g, atype)
        if len(genomes) > 0:
            return self.make_new_offspring(genomes)
        else:
            return self.make_random_genome()

    def store_genome(self, entry):
        min_fitness = 0
        min_item = 0
        fitness = entry[self.fitness_index]
        if len(self.genome_store) > 0:
            fitnesses = [x[self.fitness_index] for x in self.genome_store]
            min_item = np.argmin(fitnesses)
            min_fitness = fitnesses[min_item]
        if fitness > self.agent_start_energy and fitness > min_fitness:
            if len(self.genome_store) >= self.genome_store_size:
                self.genome_store.pop(min_item)
            self.genome_store.append(entry)

    def make_new_genome(self, atype):
        if random.random() < self.respawn_genome_store:
            return self.make_genome_from_store(atype)
        else:
            return self.make_genome_from_previous(atype)

    def get_genetic_diversity(self):
        if len(self.genome_store) < self.genome_store_size:
            return [[], 0]
        unique = []
        for item in self.genome_store[0]:
            unique.append(set())
        for index in range(len(self.genome_store)):
            gen = self.genome_store[index][0]
            for count, g in enumerate(gen):
                msg = ""
                for g1 in g:
                    c = "Z"
                    if int(g1) == -1:
                        c = "M"
                    elif int(g1) == 1:
                        c = "P"
                    msg += c
                unique[count].add(msg)
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
        fp = os.path.join(self.statsdir, "stats.json")
        with open(fp, "w") as f:
            f.write(json.dumps(self.stats))

    def load_stats(self):
        fp = os.path.join(self.statsdir, "stats.json")
        if os.path.exists(fp):
            with open(fp, "r") as f:
                self.stats = json.loads(f.read())

    def make_labels(self, var, affix, saff):
        labels = ["fitness", "age", "happiness", "distance"]
        if len(var) < 1:
            return ""
        lmsg = ""
        conds = ["evolving", "learning"]
        for cond, lab in enumerate(conds):
            nvar = [x for x in var if x[5]==cond]
            if len(nvar) > 0:
                lmsg += affix + lab + " agent (" + str(len(nvar)) + ") stats:"
                lmsg += "\n"
                for i, l in enumerate(labels):
                    temp = [x[i+1] for x in nvar]
                    if len(temp) > 0:
                        me = np.mean(temp)
                        self.record_stats(saff+"_"+lab+"_"+l+"_mean", me)
                        mx = max(temp)
                        self.record_stats(saff+"_"+lab+"_"+l+"_max", mx)
                        mi = min(temp)
                        self.record_stats(saff+"_"+lab+"_"+l+"_min", mi)
                        lmsg += "mean " + l + ": " + "%.2f"%me
                        lmsg += "  max " + l + ": " + "%.2f"%mx
                        lmsg += "  min " + l + ": " + "%.2f"%mi
                        lmsg += "\n"
                lmsg += "\n"
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
        msg += "  births: " + str(self.births)
        msg += "  rebirths: " + str(self.rebirths)
        msg += "  continuations: " + str(self.continuations)
        msg += "  deaths: " + str(self.deaths)
        msg += "  eaten: " + str(self.eaten)
        msg += "\n"
        return msg

    def print_run_stats(self):
        genome_size = [x for x in self.genome_size]
        msg = ""
        msg += "Starting agents: " + str(self.num_agents)
        msg += "  learners: " + "%.2f"%(self.learners*self.num_agents)
        msg += "  area size: " + str(self.area_size)
        msg += "  predators: " + str(self.num_predators)
        msg += "\n"
        msg += "Action size: " + str(self.action_size)
        msg += "  state_size: " + str(self.observation_size)
        msg += "  genome size: " + str(genome_size)
        msg += "\n"
        msg += "net_desc: " + str(self.net_desc)
        msg += "\n\n"
        return msg

    def print_new_state_stats(self):
        new_state_lens = [len(self.agents[x].new_state) for x in range(len(self.agents))]
        max_nsl = max(new_state_lens) * 3
        mean_nsl = np.mean(new_state_lens) * 3
        msg = ""
        msg += "Mean new state length: " + "%.2f"%mean_nsl
        msg += "  Max new state length: " + str(max_nsl)
        msg += "\n"
        return msg

    def get_stats(self):
        l_agents = sum([x.learnable for x in self.agents])
        self.record_stats("learning agents", l_agents)
        e_agents = len(self.agents) - l_agents
        self.record_stats("evolving agents", l_agents)
        agent_energy = int(sum([x.energy for x in self.agents]))
        self.record_stats("agent energy", agent_energy)
        gsitems = len(self.genome_store)
        mpf = np.mean([x[1] for x in self.previous_agents])
        bpal = 0
        bpae = 0
        pss = int(self.num_previous_agents * self.top_n)
        if len(self.previous_agents) > pss:
            bpi = self.get_best_previous_agents(pss, None)
            bpal = sum([self.previous_agents[i][5] for i in bpi])
            bpae = pss - bpal
            bplp = (bpal/pss)*100
            self.record_stats("learned_in_top_prev", bplp)
            bpep = (bpae/pss)*100
            self.record_stats("evolved_in_top_prev", bpep)
        gsal = 0
        gsae = 0
        gss = int(self.genome_store_size * self.top_n)
        if len(self.genome_store) > gss:
            gsi = self.get_best_agents_from_store(gss, None)
            gsal = sum([self.genome_store[i][5] for i in gsi])
            gsae = gss - gsal
            gplp = (gsal/gss)*100
            self.record_stats("learned_in_top_gs", gplp)
            gpep = (gsae/gss)*100
            self.record_stats("evolved_in_top_gs", gpep)

        msg = ""
        msg += self.print_run_stats()
        msg += "Step: " + str(self.steps)
        msg += "\n"
        msg += "Agents: " + str(len(self.agents))
        msg += "  learning: " + str(l_agents)
        msg += "  evolving: " + str(e_agents)
        msg += "\n\n"
        #msg += self.print_new_state_stats()
        msg += self.print_spawn_stats()
        msg += self.make_labels(self.previous_agents, "Previous ", "prev")
        msg += "Top " + str(pss) + " previous agents: learning: " + str(bpal)
        msg += "  evolving: " + str(bpae)
        msg += "\n"
        msg += "Top " + str(gss) + " agents in genome store: learning: " + str(gsal)
        msg += "  evolving: " + str(gsae)
        msg += "\n\n"
        gsd, mean_gsd = self.get_genetic_diversity()
        self.record_stats("genetic diversity", mean_gsd)
        msg += "Genetic diversity: " + str(gsd)
        msg += "  mean: " + "%.2f"%mean_gsd
        msg += "\n"
        msg += "Items in genome store: " + str(gsitems)
        msg += "\n"
        msg += self.make_labels(self.genome_store, "Genome store ", "gs")
        for atype in self.agent_types:
            #msg += self.print_action_dist(atype)
            self.print_action_dist(atype)
        return msg

    def print_stats(self):
        os.system('clear')
        print(gs.get_stats())

# Stuff to record:
# mean stats (age, fitness, etc.) of previous agents
# balance of learning/evolving in genome store and previous agents

# update callback for ursina
def update():
    gs.step()

    window.color = color.rgb(0,0,0)

    # Update agent positions
    for index, agent in enumerate(gs.agents):
        if gs.agents[index].entity is not None:
            xabs = gs.agents[index].xpos
            yabs = gs.agents[index].ypos
            zabs = gs.agents[index].zpos
            s = 0.5 + ((gs.agents[index].energy/gs.agent_start_energy)*0.5)
            gs.agents[index].entity.scale = (s,s,s)
            gs.agents[index].entity.position = (xabs, yabs, zabs)
            orient = gs.agents[index].orient
            gs.agents[index].entity.rotation = (45*orient, 90, 0)
            for n in range(gs.agents[index].trail_length):
                item = gs.agents[index].previous_positions[n]
                x, y, o, ss = item
                ss = 0.5 + ((gs.agents[index].energy/gs.agent_start_energy)*0.5)
                gs.agents[index].trail_entities[n].position = (x, y, zabs)
                gs.agents[index].trail_entities[n].rotation = (45*o, 90, 0)
                gs.agents[index].trail_entities[n].scale = (ss, ss, ss)

    # Update predator positions
    for index, predator in enumerate(gs.predators):
        if gs.predators[index].entity is not None:
            xabs = gs.predators[index].xpos
            yabs = gs.predators[index].ypos
            zabs = gs.predators[index].zpos
            gs.predators[index].entity.position = (xabs, yabs, zabs)
            orient = gs.predators[index].orient
            gs.predators[index].entity.rotation = (45*orient, 90, 0)

    # Update protector positions
    for index, protector in enumerate(gs.protectors):
        if gs.protectors[index].entity is not None:
            xabs = gs.protectors[index].xpos
            yabs = gs.protectors[index].ypos
            zabs = gs.protectors[index].zpos
            gs.protectors[index].entity.position = (xabs, yabs, zabs)
            gs.protectors[index].protection_entity.position = (xabs, yabs, zabs)
            gs.protectors[index].pulse_entity.position = (xabs, yabs, zabs)
            if gs.pulse_zones == True:
                pes = abs(np.sin(gs.steps/10))*(gs.protector_safe_distance*2)
                gs.protectors[index].pulse_entity.scale = (pes, pes, pes)
            orient = gs.protectors[index].orient
            gs.protectors[index].entity.rotation = (45*orient, 90, 0)

    if gs.pulse_zones == True:
        for index, zone in enumerate(gs.zones):
            radius = gs.zones[index].radius
            pes = abs(np.sin(gs.steps/10))*(radius*2)
            gs.zones[index].radius_entity.scale = (pes, pes, pes)


# Train the game
random.seed(1337)

print_visuals = False

if len(sys.argv)>1:
    if "v" or "-v" in sys.argv[1:]:
        print_visuals = True

savedir = "alien_ecology_save"
if not os.path.exists(savedir):
    os.makedirs(savedir)

statsdir = "alien_ecology_stats"
if not os.path.exists(statsdir):
    os.makedirs(statsdir)

if print_visuals == True:
    app = Ursina()

    window.borderless = False
    window.fullscreen = False
    window.exit_button.visible = False
    window.fps_counter.enabled = False

    gs = game_space(visuals=True)
    camera_pos1 = -1 * int(gs.area_size/2)
    camera_pos2 = int(gs.area_size*2)

    camera.position -= (camera_pos1, camera_pos1, camera_pos2)
    app.run()

else:
    gs = game_space(visuals=False)
    while True:
        gs.step()

# To do:
# Normalize inputs
# Start with a smaller number of inputs
# Reward on high entropy action prob dist
# Entropy calculated on successful actions
# Scale entropy for fitness
# Devise intrinsic reward/fitness scheme
# Split tasks?
#
# move params into a config dict
