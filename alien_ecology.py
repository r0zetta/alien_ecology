from ursina import *
import random, sys, time, os, pickle, math, json
import numpy as np
from collections import Counter, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, w, l):
        super(Net, self).__init__()
        self.w = w
        self.l = l
        self.fc_layers = nn.ModuleList()
        for item in self.w:
            s1, s2, d = item
            fc = nn.Linear(s1, s2, bias=False)
            fc.weight_data = d
            self.fc_layers.append(fc)

    def forward(self, x):
        for i in range(len(self.fc_layers)):
            if i+1 < len(self.fc_layers):
                x = F.relu(self.fc_layers[i](x))
            else:
                x = F.softmax(F.relu(self.fc_layers[i](x)), dim=-1)
        return x

    def get_w(self):
        w = []
        for fc in self.fc_layers:
            d = fc.weight_data.detach().numpy()
            d = list(np.ravel(d))
            w.extend(d)
        return w

    def set_w(self, w):
        self.w = w
        for i, item in enumerate(self.w):
            s1, s2, d = item
            self.fc_layers[i].weight_data = d

    def get_action(self, state):
        if self.l == True:
            num_actions = self.w[-1][1]
            probs = self.forward(state)
            action = np.random.choice(num_actions, p=np.squeeze(probs.detach().numpy()))
            log_prob = torch.log(probs.squeeze(0)[action])
            return action, log_prob
        else:
            with torch.no_grad():
                num_actions = self.w[-1][1]
                probs = self.forward(state)
                action = np.random.choice(num_actions, p=np.squeeze(probs.detach().numpy()))
                return action

class GN_model:
    def __init__(self, w, l=False):
        self.l = l
        self.w = w
        self.policy = Net(w, l)
        if self.l == True:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
            self.reset()

    def num_params(self):
        print(sum([param.nelement() for param in self.policy.parameters()]))

    def print_w(self):
        for item in self.w:
            print(item[0], item[1], np.array(item[2]).shape)

    def get_w(self):
        return self.policy.get_w()

    def set_w(self, w):
        return self.policy.set_w(w)

    def get_action(self, state):
        if self.l == True:
            action, log_prob = self.policy.get_action(state)
            self.log_probs.append(log_prob)
            return action
        else:
            action = self.policy.get_action(state)
            return action

    def reset(self):
        self.rewards = []
        self.log_probs = []

    def record_reward(self, reward):
        if reward is None:
            reward = 0
        self.rewards.append(reward)

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

    def save_model(self, dirname, index):
        filename = os.path.join(dirname, "policy_model_" + "%02d"%index + ".pt")
        torch.save({ "policy_state_dict": self.policy.state_dict(),
                     "policy_hidden": self.policy_hidden,
                   }, filename)

    def load_model(self, dirname, index):
        filename = os.path.join(dirname, "policy_model_" + "%02d"%index + ".pt")
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy_hidden = checkpoint["policy_hidden"]
            return True
        return False

class Predator:
    def __init__(self, x, y, z):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.xvel = 0
        self.yvel = 0
        self.speed = 0.15
        self.visible = 5
        self.orient = 0 # 0-7 in 45 degree increments
        self.inertial_damping = 0.60

class Agent:
    def __init__(self, x, y, z, learnable, color, energy,
                 state_size, action_size, hidden_size, genome):
        self.colors = ["pink", "blue", "green", "orange", "purple", "red", "teal", "violet", "yellow"]
        self.start_energy = energy
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learnable = learnable
        self.sample = False
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.color = color
        self.color_str = self.colors[color]
        self.genome = genome
        self.weights = self.make_weights()
        self.model = GN_model(self.weights, self.learnable)
        self.state = None
        self.entity = None
        self.previous_stats = []
        self.reset()

    def make_weights(self):
        weights = []
        m1 = 0
        m2 = self.state_size * self.hidden_size[0]
        w = torch.Tensor(np.reshape(self.genome[m1:m2], (self.hidden_size[0], self.state_size)))
        weights.append([self.state_size, self.hidden_size[0], w])
        if len(self.hidden_size) > 1:
            for i in range(len(self.hidden_size)):
                if i+1 < len(self.hidden_size):
                    m1 = m2
                    m2 = m1 + (self.hidden_size[i] * self.hidden_size[i+1])
                    w = torch.Tensor(np.reshape(self.genome[m1:m2],
                                     (self.hidden_size[i], self.hidden_size[i+1])))
                    weights.append([self.hidden_size[i], self.hidden_size[i+1], w])
        w = torch.Tensor(np.reshape(self.genome[m2:], (self.action_size, self.hidden_size[-1])))
        weights.append([self.hidden_size[-1], self.action_size, w])
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
        self.food_inventory = 0
        self.distance_travelled = 0
        self.happiness = 0
        self.fitness = 0
        self.age = 0
        self.temperature = 20
        self.inertial_damping = 0.80
        self.speed = 0.4
        self.previous_states = deque()

    def get_action(self):
        if self.sample == True:
            return random.choice(range(self.action_size))
        else:
            return self.model.get_action(self.state)

class Food:
    def __init__(self, x, y, z, energy):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.energy = random.randint(2, energy+2)
        self.entity = None

class Berry:
    def __init__(self, x, y, z):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.age = 0
        self.entity = None

class Pheromone:
    def __init__(self, x, y, z):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.strength = 100
        self.entity = None

# No evolution:
# learners=1.00
# evaluate_learner_every=1000000
# min_reproduction_age=200
# min_reproduction_energy=200

# All evolution:
# learners=0.00
# min_reproduction_age=50
# min_reproduction_energy=80

# Hybrid evolution:
# learners=0.50
# evaluate_learner_every=30
# min_reproduction_age=50
# min_reproduction_energy=100

class game_space:
    def __init__(self,
                 hidden_size=[32],
                 num_prev_states=1,
                 num_recent_actions=1000,
                 num_previous_agents=500,
                 genome_store_size=500,
                 top_n=0.2,
                 learners=0.25,
                 evaluate_learner_every=50,
                 use_genome_type=0,
                 mutation_rate=0.001,
                 area_size=50,
                 year_period=10*300,
                 day_period=10,
                 weather_harshness=0,
                 num_agents=30,
                 agent_start_energy=250,
                 agent_max_inventory=10,
                 num_predators=3,
                 predator_view_distance=5,
                 predator_kill_distance=2,
                 food_sources=20,
                 food_spawns=15,
                 food_dist=7,
                 food_repro_energy=15,
                 food_start_energy=10,
                 food_energy_growth=2,
                 food_max_percent=0.05,
                 food_plant_success=0.5,
                 berry_max_age=30,
                 pheromone_decay=0.90,
                 min_reproduction_age=50,
                 min_reproduction_energy=100,
                 reproduction_cost=5,
                 visuals=True,
                 reward_age_only=True,
                 respawn_genome_store=0.1,
                 rebirth_genome_store=0.9,
                 save_every=5000,
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
        self.food_picked = 0
        self.food_eaten = 0
        self.food_dropped = 0
        self.num_recent_actions = num_recent_actions
        self.num_previous_agents = num_previous_agents
        self.fitness_index = 2 # 1: fitness, 2: age
        self.respawn_genome_store = respawn_genome_store
        self.rebirth_genome_store = rebirth_genome_store
        self.genome_store_size = genome_store_size
        self.top_n = top_n
        self.learners = learners
        self.reward_age_only = reward_age_only
        self.evaluate_learner_every = evaluate_learner_every
        self.use_genome_type = use_genome_type
        self.visuals = visuals
        self.savedir = savedir
        self.statsdir = statsdir
        self.stats = {}
        self.load_stats()
        self.record_every = record_every
        self.save_every = save_every
        self.num_prev_states = num_prev_states
        self.environment_temperature = 20
        self.area_size = area_size
        self.year_period = year_period
        self.day_period = day_period
        self.weather_harshness = weather_harshness
        self.num_agents = num_agents
        self.agent_max_inventory = agent_max_inventory
        self.agent_start_energy = agent_start_energy
        self.num_predators = num_predators
        self.predator_view_distance = predator_view_distance
        self.predator_kill_distance = predator_kill_distance
        self.food_sources = food_sources
        self.food_spawns = food_spawns
        self.food_dist = food_dist
        self.food_start_energy = food_start_energy
        self.food_repro_energy = food_repro_energy
        self.food_energy_growth = food_energy_growth
        self.food_max_percent = food_max_percent
        self.food_plant_success = food_plant_success
        self.berry_max_age = berry_max_age
        self.pheromone_decay = pheromone_decay
        self.pheromones = []
        self.berries = []
        self.min_reproduction_age = min_reproduction_age
        self.min_reproduction_energy = min_reproduction_energy
        self.reproduction_cost = reproduction_cost
        self.agent_view_distance = 5
        self.visible_area = math.pi*(self.agent_view_distance**2)
        self.mutation_rate = mutation_rate
        self.agent_types = ["evolving",
                            "learning"]
        self.agent_actions = {}
        self.recent_actions = {}
        for t in self.agent_types:
            self.agent_actions[t] = Counter()
            self.recent_actions[t] = deque()
        self.actions = ["rotate_right",
                        "rotate_left",
                        "flip",
                        "propel",
                        "pick_food",
                        "drop_food",
                        "eat_food",
                        "mate",
                        #"freq_up",
                        #"freq_down",
                        #"move_random",
                        "emit_pheromone"]
        self.observations = ["visible_food",
                             "adjacent_food",
                             "food_in_range",
                             "forward_pheromones",
                             "adjacent_pheromones",
                             "visible_agents",
                             "adjacent_agents",
                             "adjacent_agent_count",
                             "mate_in_range",
                             "visible_predators",
                             "surrounding_predators",
                             "previous_action",
                             "own_age",
                             "own_energy",
                             "own_temperature",
                             "distance_moved",
                             "own_happiness",
                             "own_xposition",
                             "own_yposition",
                             "own_orientation",
                             "own_xvelocity",
                             "own_yvelocity",
                             "food_inventory",
                             "environment_temperature",
                             "visibility",
                             "age_oscillator",
                             "reproduction_oscillator",
                             "step_oscillator_day",
                             "step_oscillator_day_offset",
                             "step_oscillator_year",
                             "step_oscillator_year_offset",
                             "random_input"]
        self.hidden_size = hidden_size
        self.action_size = len(self.actions)
        self.state_size = len(self.observations)*self.num_prev_states
        self.genome_size = 0
        self.genome_size += self.state_size*self.hidden_size[0]
        if len(self.hidden_size) > 1:
            for i in range(len(self.hidden_size)):
                if i+1 < len(self.hidden_size):
                    self.genome_size += self.hidden_size[i]*self.hidden_size[i+1]
        self.genome_size += self.action_size*self.hidden_size[-1]
        self.genome_store = []
        self.previous_agents = deque()
        self.create_starting_food()
        self.create_genomes()
        self.agents = []
        num_learners = int(self.num_agents * self.learners)
        num_evolvable = int(self.num_agents - num_learners)
        if num_learners > 0:
            self.create_new_learner_agents(num_learners)
        if num_evolvable > 0:
            self.create_new_evolvable_agents(num_evolvable)
        self.create_predators()

    def step(self):
        self.set_environment_visibility()
        self.set_environment_temperature()
        self.apply_predator_physics()
        self.run_predator_actions()
        self.set_agent_states()
        self.apply_agent_physics()
        self.run_agent_actions()
        self.update_agent_status()
        self.update_pheromone_status()
        self.reproduce_food()
        self.update_berries()
        if self.save_every > 0:
            if self.steps % self.save_every == 0:
                self.save_genomes()
        if self.steps % 500 == 0:
            self.save_stats()
        self.print_stats()
        self.steps += 1

#########################
# Initialization routines
#########################
    def set_predator_entity(self, index):
        xabs = self.predators[index].xpos
        yabs = self.predators[index].ypos
        zabs = self.predators[index].zpos
        s = 2
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

    def create_genomes(self):
        self.genome_pool = []
        if os.path.exists(savedir + "/genome_store.pkl"):
            print("Loading genomes.")
            self.load_genomes()
            sg = []
            for item in self.genome_store:
                g = item[0]
                sg.append(g)
            if len(sg) > self.num_agents:
                self.genome_pool = random.sample(sg, self.num_agents)
            else:
                self.genome_pool = list(sg)
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

    def set_agent_entity(self, index):
        xabs = self.agents[index].xpos
        yabs = self.agents[index].ypos
        zabs = self.agents[index].zpos
        s = 1
        texture = "textures/" + self.agents[index].color_str
        self.agents[index].entity = Entity(model='sphere',
                                           color=color.white,
                                           scale=(s,s,s),
                                           position = (xabs, yabs, zabs),
                                           texture=texture)

    def spawn_learning_agent(self, genome):
        self.spawn_new_agent(genome, True)

    def spawn_evolving_agent(self, genome):
        self.spawn_new_agent(genome, False)

    def spawn_new_agent(self, genome, learner):
        self.spawns += 1
        xpos = random.random()*self.area_size
        ypos = random.random()*self.area_size
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
                  self.state_size,
                  self.action_size,
                  self.hidden_size,
                  genome)
        self.agents.append(a)
        index = len(self.agents)-1
        self.set_initial_agent_state(index)
        if self.visuals == True:
            self.set_agent_entity(index)

    def create_new_evolvable_agents(self, num):
        genomes = random.sample(self.genome_pool, num)
        for genome in genomes:
            self.spawn_evolving_agent(genome)

    def create_new_learner_agents(self, num):
        genomes = random.sample(self.genome_pool, num)
        for genome in genomes:
            self.spawn_learning_agent(genome)

    def set_food_entity(self, index):
        xabs = self.food[index].xpos
        yabs = self.food[index].ypos
        zabs = self.food[index].zpos
        s = 1
        texture = "textures/cherry"
        self.food[index].entity = Entity(model='sphere',
                                         color=color.white,
                                         scale=(s,s,s),
                                         position = (xabs, yabs, zabs),
                                         texture=texture)

    def create_starting_food(self):
        self.food = []
        for n in range(self.food_sources):
            xpos = random.uniform(0, self.area_size)
            ypos = random.uniform(0, self.area_size)
            for _ in range(self.food_spawns):
                z = -1
                x = xpos + random.uniform(xpos-self.food_dist, xpos+self.food_dist)
                if x < 0:
                    x += self.area_size
                if x > self.area_size:
                    x -= self.area_size
                y = ypos + random.uniform(ypos-self.food_dist, ypos+self.food_dist)
                if y < 0:
                    y += self.area_size
                if y > self.area_size:
                    y -= self.area_size
                f = Food(x, y, z, self.food_start_energy)
                self.food.append(f)
        if self.visuals == True:
            for index, f in enumerate(self.food):
                self.set_food_entity(index)

#################
# Helper routines
#################
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
        return xvel, yvel

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

########################
# Environmental effects
########################

    def set_environment_visibility(self):
        osc = np.sin(self.steps/self.day_period)
        self.agent_view_distance = 3.5 + (1.5*osc)
        self.visible_area = math.pi*(self.agent_view_distance**2)

    def set_environment_temperature(self):
        osc = np.sin(self.steps/self.year_period)
        # Climate change modifies osc multiplier and/or initial value
        osc2 = np.sin(2*self.steps/(self.year_period*10))
        osc3 = np.sin(3*(self.steps+self.year_period*5)/(self.year_period*10))
        v1 = 14 + self.weather_harshness + osc3
        v2 = 12 + self.weather_harshness + osc2
        # Max temp: 38
        # Min temp: 4
        self.environment_temperature = v1 + self.agent_view_distance + random.random()*3 + ((v2)*osc)
        self.record_stats("env_temp", self.environment_temperature)

    def set_agent_temperature(self, index):
        temp = self.environment_temperature - 5 + (self.count_adjacent_agents(index))
        self.agents[index].temperature = temp

########################
# Agent runtime routines
########################
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
            atype = 0
            if self.agents[n].learnable == True:
                atype = 1
            self.record_recent_actions(action, atype)
            self.apply_agent_action(n, action)
            self.update_agent_position(n)

    def birth_new_agent(self, index1, index2):
        self.births += 1
        self.spawns += 1
        xpos = self.agents[index1].xpos
        ypos = self.agents[index2].ypos
        zpos = -1
        g1 = self.agents[index1].model.get_w()
        g2 = self.agents[index2].model.get_w()
        s = self.reproduce_genome(g1, g2, 1)
        c = random.choice(s)
        genome = self.mutate_genome(c, 1)[0]
        c1 = self.agents[index1].color
        c2 = self.agents[index2].color
        color = np.max([c1, c2]) + 1
        if color >= len(self.agents[index1].colors):
            color = 0
        a = Agent(xpos,
                  ypos,
                  zpos,
                  False,
                  color,
                  self.agent_start_energy,
                  self.state_size,
                  self.action_size,
                  self.hidden_size,
                  genome)
        self.agents.append(a)
        ai = len(self.agents)-1
        self.set_initial_agent_state(ai)
        if self.visuals == True:
            self.set_agent_entity(ai)

    def get_agents_in_radius(self, xpos, ypos, radius):
        ret = []
        for i in range(len(self.agents)):
            ax = self.agents[i].xpos
            ay = self.agents[i].ypos
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                if self.distance(xpos, ypos, ax, ay) <= radius:
                    ret.append(i)
        return ret

    def get_viewpoint(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        orient = self.agents[index].orient
        distance = self.agent_view_distance
        xv, yv = self.viewpoint(xpos, ypos, orient, distance)
        return xv, yv

    def get_visible_agents(self, index):
        xv, yv = self.get_viewpoint(index)
        agents = self.get_agents_in_radius(xv, yv, self.agent_view_distance)
        return(len(agents))

    def get_adjacent_agent_indices(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        return self.get_agents_in_radius(xpos, ypos, self.agent_view_distance)

    def get_adjacent_agents(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        agents = self.get_agents_in_radius(xpos, ypos, self.agent_view_distance)
        return(len(agents))

    def get_adjacent_agent_count(self, index):
        return self.count_adjacent_agents(index)

    def count_adjacent_agents(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        agents = self.get_agents_in_radius(xpos, ypos, self.agent_view_distance)
        agent_count = len(agents)
        return agent_count

    def get_mate_in_range(self, index):
        ret = 0
        ai, val = self.get_nearest_agent(index)
        if ai is not None:
            if val <= 1:
                if self.agents[ai].energy >= self.min_reproduction_energy:
                    if self.agents[ai].age >= self.min_reproduction_age:
                        ret = 1
        return ret

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
            a = self.agents[index].age
            reward = a/self.agent_start_energy
            if self.reward_age_only == True:
                if len(self.agents[index].model.rewards) > 0:
                    self.agents[index].model.rewards[-1] = reward
            h = self.agents[index].happiness
            d = self.agents[index].distance_travelled
            f = a + h + d
            g = self.agents[index].model.get_w()
            entry = [g, f, a, h, d, l]
            self.store_genome(entry)
            self.add_previous_agent(entry)
            self.agents[index].previous_stats.append(entry)
            self.evaluate_learner(index)
            self.deaths += 1
            self.resets += 1
            affected = self.get_adjacent_agent_indices(index)
            for i in affected:
                self.agents[i].happiness -= 5
            self.agents[index].model.update_policy()
            self.agents[index].reset()
            self.agents[index].xpos = random.random()*self.area_size
            self.agents[index].ypos = random.random()*self.area_size
            self.set_initial_agent_state(index)

    def kill_agents(self, dead):
        for index in dead:
            l = int(self.agents[index].learnable)
            a = self.agents[index].age
            h = self.agents[index].happiness
            d = self.agents[index].distance_travelled
            f = a + h + d
            g = self.agents[index].model.get_w()
            entry = [g, f, a, h, d, l]
            self.store_genome(entry)
            self.add_previous_agent(entry)
            self.deaths += 1
            affected = self.get_adjacent_agent_indices(index)
            for i in affected:
                self.agents[i].happiness -= 5
            if self.visuals == True:
                self.agents[index].entity.disable()
                del(self.agents[index].entity)
        new_agents = [i for j, i in enumerate(self.agents) if j not in dead]
        self.agents = list(new_agents)
        if len(self.agents) < self.num_agents:
            deficit = self.num_agents - len(self.agents)
            for _ in range(deficit):
                genome = self.make_new_genome(0)
                self.spawn_evolving_agent(genome)

    def update_agent_status(self):
        dead = set()
        reset = set()
        for index in range(len(self.agents)):
            self.agents[index].age += 1
            self.set_agent_temperature(index)
            energy_drain = 1
            temperature = self.agents[index].temperature
            if temperature > 30:
                energy_drain += (temperature - 30) * 0.1
                self.agents[index].happiness -= 1
            if temperature < 5:
                energy_drain += (5 - temperature) * 0.1
                self.agents[index].happiness -= 1
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
        x2 = x1 + self.agents[index].xvel
        y2 = y1 + self.agents[index].yvel
        d = self.distance(x1, y1, x2, y2)
        self.agents[index].distance_travelled += d
        self.agents[index].xpos = x2
        self.agents[index].ypos = y2
        if self.agents[index].xpos > self.area_size:
            self.agents[index].xpos -= self.area_size
        if self.agents[index].xpos < 0:
            self.agents[index].xpos += self.area_size
        if self.agents[index].ypos > self.area_size:
            self.agents[index].ypos -= self.area_size
        if self.agents[index].ypos < 0:
            self.agents[index].ypos += self.area_size

    def apply_agent_physics(self):
        for index in range(len(self.agents)):
            self.apply_agent_inertial_damping(index)

    def apply_agent_inertial_damping(self, index):
        self.agents[index].xvel = self.agents[index].xvel * self.agents[index].inertial_damping
        self.agents[index].yvel = self.agents[index].yvel * self.agents[index].inertial_damping

##################
# Predator actions
##################
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
        if predator_count > 0:
            self.agents[index].happiness -= 1
        return predator_count

    def get_visible_predators(self, index):
        xv, yv = self.get_viewpoint(index)
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance)
        predator_count = len(predators)
        if predator_count > 0:
            self.agents[index].happiness -= 1
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
        self.predators[index].xpos += self.predators[index].xvel
        if self.predators[index].xpos > self.area_size:
            self.predators[index].xpos -= self.area_size
        if self.predators[index].xpos < 0:
            self.predators[index].xpos += self.area_size
        self.predators[index].ypos += self.predators[index].yvel
        if self.predators[index].ypos > self.area_size:
            self.predators[index].ypos -= self.area_size
        if self.predators[index].ypos < 0:
            self.predators[index].ypos += self.area_size

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

###############
# Agent actions
###############
    def apply_agent_action(self, index, action):
        self.agents[index].prev_action = action
        agent_function = "action_" + self.actions[action]
        class_method = getattr(self, agent_function)
        reward = class_method(index)
        if self.agents[index].learnable == True:
            if self.reward_age_only == True:
                self.agents[index].model.record_reward(0)
            else:
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

    def action_pick_food(self, index):
        carrying = self.agents[index].food_inventory
        if carrying >= self.agent_max_inventory:
            self.agents[index].happiness -= 5
            return 0
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        bi, bval = self.get_nearest_berry(xpos, ypos)
        if bi is not None:
            if bval <= 1:
                picked = True
                self.agents[index].happiness += 20
                self.food_picked += 1
                self.agents[index].food_inventory += 1
                self.remove_berry(bi)
                return 1
        fi, fval = self.get_nearest_food(xpos, ypos)
        if fi is not None:
            if fval <= 1:
                picked = True
                self.agents[index].happiness += 20
                self.food_picked += 1
                self.agents[index].food_inventory += 1
                self.food[fi].energy -= 5
                if self.food[fi].energy < 0:
                    self.remove_food(fi)
                return 1
        self.agents[index].happiness -= 1
        return 0

    def action_eat_food(self, index):
        reward = 0
        if self.agents[index].food_inventory > 0:
            if self.agents[index].energy >= self.agent_start_energy:
                self.agents[index].happiness -= 5
                reward = -1
            else:
                self.food_eaten += 1
                self.agents[index].energy += 20
                self.agents[index].happiness += 100
                reward = 1
            self.agents[index].food_inventory -= 1
        else:
            self.agents[index].happiness -= 1
        return reward

    def action_drop_food(self, index):
        reward = 0
        if self.agents[index].food_inventory > 0:
            xpos = self.agents[index].xpos
            ypos = self.agents[index].ypos
            self.add_berry(xpos, ypos)
            self.food_dropped += 1
            self.agents[index].food_inventory -= 1
            reward = 1
        else:
            self.agents[index].happiness -= 1
        return reward

    def action_mate(self, index):
        if self.agents[index].energy >= self.min_reproduction_energy:
            if self.agents[index].age >= self.min_reproduction_age:
                ai, val = self.get_nearest_agent(index)
                if ai is not None:
                    if val <= 1:
                        if self.agents[ai].energy >= self.min_reproduction_energy:
                            if self.agents[ai].age >= self.min_reproduction_age:
                                self.birth_new_agent(index, ai)
                                self.agents[index].happiness += 100
                                self.agents[ai].happiness += 100
                                self.agents[index].energy -= self.reproduction_cost
                                self.agents[ai].energy -= self.reproduction_cost
                                return 1
        self.agents[index].happiness -= 1
        return 0

    def action_freq_up(self, index):
        self.agents[index].frequency = self.agents[index].frequency * 0.9

    def action_freq_down(self, index):
        self.agents[index].frequency = self.agents[index].frequency * 1.1

    def action_emit_pheromone(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        self.create_new_pheromone(xpos, ypos)

    def action_move_random(self, index):
        actions = ["action_rotate_right", "action_rotate_left", "action_propel"]
        choice = random.choice(actions)
        class_method = getattr(self, choice)
        return class_method(index)

####################
# Agent observations
####################
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
        for n in self.observations:
            function_names.append("get_" + n)
        observations = []
        for fn in function_names:
            class_method = getattr(self, fn)
            val = class_method(index)
            observations.append(val)
        return observations

    def get_previous_action(self, index):
        return self.agents[index].prev_action

    def get_own_age(self, index):
        return self.agents[index].age

    def get_own_energy(self, index):
        return self.agents[index].energy

    def get_own_temperature(self, index):
        return self.agents[index].temperature

    def get_food_inventory(self, index):
        return self.agents[index].food_inventory

    def get_distance_moved(self, index):
        return self.agents[index].distance_travelled

    def get_own_happiness(self, index):
        return self.agents[index].happiness

    def get_own_xposition(self, index):
        return self.agents[index].xpos

    def get_own_yposition(self, index):
        return self.agents[index].ypos

    def get_own_orientation(self, index):
        return (self.agents[index].orient - 4) / 4

    def get_own_xvelocity(self, index):
        return self.agents[index].xvel

    def get_own_yvelocity(self, index):
        return self.agents[index].yvel

    def get_environment_temperature(self, index):
        return self.environment_temperature

    def get_visibility(self, index):
        return self.agent_view_distance

    def get_age_oscillator(self, index):
        return np.sin(self.agents[index].age/self.agents[index].frequency)

    def get_reproduction_oscillator(self, index):
        return np.sin((self.agents[index].age+(self.min_reproduction_age/2))/self.min_reproduction_age)

    def get_step_oscillator_day(self, index):
        return np.sin(self.steps/self.day_period)

    def get_step_oscillator_day_offset(self, index):
        return np.sin((self.steps+(self.day_period/2))/self.day_period)

    def get_step_oscillator_year(self, index):
        return np.sin(self.steps/self.year_period)

    def get_step_oscillator_year_offset(self, index):
        return np.sin((self.steps+(self.year_period/2))/self.year_period)

    def get_random_input(self, index):
        return random.uniform(-1, 1)

############################
# Pheromone runtime routines
############################
    def update_pheromone_alpha(self, index):
        alpha = self.pheromones[index].strength/2
        self.pheromones[index].entity.color=color.rgba(102,51,0,alpha)

    def update_pheromone_status(self):
        expired = []
        for index in range(len(self.pheromones)):
            cs = self.pheromones[index].strength
            ns = cs * self.pheromone_decay
            self.pheromones[index].strength = ns
            if self.visuals == True:
                self.update_pheromone_alpha(index)
            if ns < 1:
                expired.append(index)
        if len(expired) > 0:
            for index in expired:
                if self.visuals == True:
                    self.pheromones[index].entity.disable()
                    del(self.pheromones[index].entity)
            new_p = [i for j, i in enumerate(self.pheromones) if j not in expired]
            self.pheromones = list(new_p)

    def set_pheromone_entity(self, index):
        xabs = self.pheromones[index].xpos
        yabs = self.pheromones[index].ypos
        zabs = self.pheromones[index].zpos
        s = 2
        alpha = self.pheromones[index].strength/2
        self.pheromones[index].entity = Entity(model='sphere',
                                         color=color.rgba(102,51,0,alpha),
                                         scale=(s,s,s),
                                         position = (xabs, yabs, zabs))

    def create_new_pheromone(self, xpos, ypos):
        zpos = -1
        p = Pheromone(xpos, ypos, zpos)
        self.pheromones.append(p)
        if self.visuals == True:
            index = len(self.pheromones)-1
            self.set_pheromone_entity(index)

    def get_pheromones_in_radius(self, xpos, ypos, radius):
        radius = radius * 2
        ret = []
        for i in range(len(self.pheromones)):
            ax = self.pheromones[i].xpos
            ay = self.pheromones[i].ypos
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                if self.distance(xpos, ypos, ax, ay) <= radius:
                    ret.append(i)
        return ret

    def get_forward_pheromones(self, index):
        xv, yv = self.get_viewpoint(index)
        pheromones = self.get_pheromones_in_radius(xv, yv, self.agent_view_distance)
        return(len(pheromones))

    def get_adjacent_pheromones(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        pheromones = self.get_pheromones_in_radius(xpos, ypos, self.agent_view_distance)
        return(len(pheromones))


#######################
# Food runtime routines
#######################
    def get_food_in_radius(self, xpos, ypos, radius):
        ret = []
        for i in range(len(self.food)):
            ax = self.food[i].xpos
            ay = self.food[i].ypos
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                if self.distance(xpos, ypos, ax, ay) <= radius:
                    ret.append(i)
        return ret

    def get_visible_food(self, index):
        xv, yv = self.get_viewpoint(index)
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance)
        return len(food)

    def get_adjacent_food(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        food = self.get_food_in_radius(xpos, ypos, self.agent_view_distance)
        return len(food)

    def get_food_in_range(self, index):
        ret = 0
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        food_index, food_distance = self.get_nearest_food(xpos, ypos)
        if food_index is not None:
            if food_distance < 1:
                ret = 1
        return ret

    def get_nearest_food(self, xpos, ypos):
        distances = []
        for f in self.food:
            ax = f.xpos
            ay = f.ypos
            radius = self.agent_view_distance
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                distances.append(self.distance(xpos, ypos, ax, ay))
        if len(distances) > 0:
            shortest_index = np.argmin(distances)
            shortest_value = distances[shortest_index]
            return shortest_index, shortest_value
        else:
            return None, None

    def remove_food(self, index):
        if self.visuals == True:
            self.food[index].entity.disable()
            del(self.food[index].entity)
        self.food.pop(index)

    def reproduce_food(self):
        growth = random.random()*self.food_energy_growth
        if self.environment_temperature < 5:
            growth = -1.2
        if self.environment_temperature > 35:
            growth = -1.2
        dead = []
        for index, f in enumerate(self.food):
            self.food[index].energy += growth
            if self.food[index].energy >= self.food_repro_energy:
                if len(self.food) < self.food_max_percent*(self.area_size**2):
                    self.spawn_new_food_in_radius(self.food[index].xpos, self.food[index].ypos)
                    self.food[index].energy = self.food_start_energy
                else:
                    self.food[index].energy = self.food_repro_energy
            if self.food[index].energy <= 0:
                dead.append(index)
        if len(dead) > 0:
            if self.visuals == True:
                self.food[index].entity.disable()
                del(self.food[index].entity)
        new_food = [i for j, i in enumerate(self.food) if j not in dead]
        self.food = new_food
        if len(self.food) < 1:
            self.create_starting_food()

    def is_food_on_this_plot(self, xpos, ypos):
        for i in range(len(self.food)):
            fx = int(self.food[i].xpos)
            fy = int(self.food[i].xpos)
            if xpos == int(fx) and ypos == int(fy):
                return True
        return False

    def drop_food(self, xpos, ypos):
        z = -1
        f = Berry(xpos, ypos, z)
        self.berries.append(f)
        if self.visuals == True:
            fi = len(self.berries)-1
            self.set_berry_entity(fi)
        return True

    def spawn_new_food_in_radius(self, xpos, ypos):
        z = -1
        x = xpos + random.uniform(xpos-self.food_dist, xpos+self.food_dist)
        if x < 0:
            x += self.area_size
        if x > self.area_size:
            x -= self.area_size
        y = ypos + random.uniform(ypos-self.food_dist, ypos+self.food_dist)
        if y < 0:
            y += self.area_size
        if y > self.area_size:
            y -= self.area_size
        f = Food(x, y, z, self.food_start_energy)
        if self.is_food_on_this_plot(x, y) == False:
            self.food.append(f)
            if self.visuals == True:
                fi = len(self.food)-1
                self.set_food_entity(fi)

    def spawn_new_food_at_location(self, xpos, ypos):
        z = -1
        f = Food(xpos, ypos, z, self.food_start_energy)
        if self.is_food_on_this_plot(xpos, ypos) == False:
            self.food.append(f)
            if self.visuals == True:
                fi = len(self.food)-1
                self.set_food_entity(fi)

########################
# Berry runtime routines
########################
    def set_berry_entity(self, index):
        xabs = self.food[index].xpos
        yabs = self.food[index].ypos
        zabs = self.food[index].zpos
        s = 1
        texture = "textures/plum"
        self.berries[index].entity = Entity(model='sphere',
                                            color=color.white,
                                            scale=(s,s,s),
                                            position = (xabs, yabs, zabs),
                                            texture=texture)

    def get_berries_in_radius(self, xpos, ypos, radius):
        ret = []
        for i in range(len(self.berries)):
            ax = self.berries[i].xpos
            ay = self.berries[i].ypos
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                if self.distance(xpos, ypos, ax, ay) <= radius:
                    ret.append(i)
        return ret

    def get_visible_berries(self, index):
        xv, yv = self.get_viewpoint(index)
        berries = self.get_berries_in_radius(xv, yv, self.agent_view_distance)
        return len(berries)

    def get_adjacent_berries(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        berries = self.get_berries_in_radius(xpos, ypos, self.agent_view_distance)
        return len(berries)

    def get_berries_in_range(self, index):
        ret = 0
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        berry_index, berry_distance = self.get_nearest_berry(xpos, ypos)
        if berry_index is not None:
            if berry_distance < 1:
                ret = 1
        return ret

    def get_nearest_berry(self, xpos, ypos):
        distances = []
        for f in self.berries:
            ax = f.xpos
            ay = f.ypos
            radius = self.agent_view_distance
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                distances.append(self.distance(xpos, ypos, ax, ay))
        if len(distances) > 0:
            shortest_index = np.argmin(distances)
            shortest_value = distances[shortest_index]
            return shortest_index, shortest_value
        else:
            return None, None

    def remove_berry(self, index):
        if self.visuals == True:
            self.berry[index].entity.disable()
            del(self.berry[index].entity)
        self.berries.pop(index)

    def add_berry(self, xpos, ypos):
        zpos = -1
        berry = Berry(xpos, ypos, zpos)
        self.berries.append(berry)
        if self.visuals == True:
            index = len(self.berries)-1
            self.set_berry_entity(index)

    def update_berries(self):
        to_remove = []
        for index in range(len(self.berries)):
            self.berries[index].age += 1
            if self.berries[index].age > self.berry_max_age:
                if self.environment_temperature > 7:
                    if random.random() < self.food_plant_success:
                        xpos = self.berries[index].xpos
                        ypos = self.berries[index].ypos
                        self.spawn_new_food_at_location(xpos, ypos)
                to_remove.append(index)
        if len(to_remove) > 0:
            for index in to_remove:
                if self.visuals == True:
                    self.berries[index].entity.disable()
                    del(self.berries[index].entity)
            new_berries = [i for j, i in enumerate(self.berries) if j not in to_remove]
            self.berries = new_berries


########################################
# Routines related to genetic algorithms
########################################
    def make_random_genome(self):
        #genome = np.random.uniform(-1, 1, self.genome_size)
        genome = np.random.randint(-1, 2, self.genome_size)
        return genome

    def make_random_genomes(self, num):
        genome_pool = []
        for _ in range(num):
            genome = self.make_random_genome()
            genome_pool.append(genome)
        return genome_pool

    def reproduce_genome(self, g1, g2, num_offspring):
        new_genomes = []
        for _ in range(num_offspring):
            s = random.randint(10, len(g1)-10)
            g3 = np.concatenate((g1[:s], g2[s:]))
            new_genomes.append(g3)
            g4 = np.concatenate((g2[:s], g1[s:]))
            new_genomes.append(g4)
        return new_genomes

    def mutate_genome(self, g, num_mutations):
        new_genomes = []
        for _ in range(num_mutations):
            n = int(self.mutation_rate * len(g))
            indices = random.sample(range(len(g)), n)
            gm = g
            for index in indices:
                #val = random.uniform(-1, 1)
                val = random.randint(-1, 1)
                gm[index] = val
            new_genomes.append(gm)
        return new_genomes

    def make_new_offspring(self, genomes):
        parents = random.sample(genomes, 2)
        offspring = self.reproduce_genome(parents[0], parents[1], 1)
        chosen = random.choice(offspring)
        mutated = self.mutate_genome(chosen, 1)[0]
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
        bars = [int(300*(self.agent_actions[atype][x]/ta)) for x in self.actions]
        for i, x in enumerate(self.actions):
            space = (15-len(x))*" "
            bar = bars[i]*"#"
            lmsg += str(x) + ":" + space + bar + "\n"
            self.record_stats(atype+"_action_"+x, bars[i])
        lmsg += "\n"
        return lmsg

    def print_current_stats(self):
        ages = [x.age for x in self.agents]
        max_age = np.max(ages)
        mean_age = int(np.mean(ages))
        hap = [x.happiness for x in self.agents]
        max_hap = np.max(hap)
        min_hap = np.min(hap)
        mean_hap = int(np.mean(hap))
        d = [x.distance_travelled for x in self.agents]
        max_d = np.max(d)
        mean_d = int(np.mean(d))
        msg = ""
        msg += "mean age: " + "%.2f"%mean_age
        msg += "  max age: " + str(max_age)
        msg += "\n"
        msg += "mean happiness: " + "%.2f"%mean_hap
        msg += "  max happiness: " + str(max_hap)
        msg += "  min happiness: " + str(min_hap)
        msg += "\n"
        msg += "mean distance moved: " + "%.2f"%mean_d
        msg += "  max distance moved: " + "%.2f"%max_d
        msg += "\n\n"
        return msg

    def print_food_stats(self):
        msg = ""
        msg += "Food picked: " + str(self.food_picked)
        msg += "  eaten: " + str(self.food_eaten) 
        msg += "  dropped: " + str(self.food_dropped)
        msg += "\n\n"
        return msg

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

    def print_temp_stats(self):
        vrange = self.agent_view_distance
        etemp = self.environment_temperature
        atemp = [x.temperature for x in self.agents]
        mean_temp = int(np.mean(atemp))
        max_temp = int(max(atemp))
        min_temp = int(min(atemp))
        msg = ""
        msg += "Visual_r: " + "%.2f"%vrange
        msg += "  Env temp: " + "%.2f"%etemp
        msg += "  mean a_temp: " + str(mean_temp)
        msg += "  max a_temp: " + str(max_temp)
        msg += "  min a_temp: " + str(min_temp)
        msg += "\n\n"
        return msg

    def get_stats(self):
        num_agents = len(self.agents)
        l_agents = sum([x.learnable for x in self.agents])
        self.record_stats("learning agents", l_agents)
        e_agents = num_agents - l_agents
        self.record_stats("evolving agents", l_agents)
        agent_energy = int(sum([x.energy for x in self.agents]))
        self.record_stats("agent energy", agent_energy)
        num_food = len(self.food)
        self.record_stats("food", num_food)
        self.record_stats("food picked", self.food_picked)
        self.record_stats("food eaten", self.food_eaten)
        self.record_stats("food dropped", self.food_dropped)
        food_energy = int(sum([x.energy for x in self.food]))
        self.record_stats("food energy", food_energy)
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
        msg += "Starting agents: " + str(self.num_agents)
        msg += "  learners: " + "%.2f"%(self.learners*self.num_agents)
        msg += "  area size: " + str(self.area_size)
        msg += "  predators: " + str(self.num_predators)
        msg += "\n"
        msg += "Year length: " + str(self.year_period*6)
        msg += "  day length: " + str(self.day_period*6)
        msg += "  min reproduction age: " + str(self.min_reproduction_age)
        msg += "  min reproduction energy: " + str(self.min_reproduction_energy)
        msg += "\n"
        msg += "Action size: " + str(self.action_size)
        msg += "  state_size: " + str(self.state_size)
        msg += "  hidden: " + str(self.hidden_size)
        msg += "  genome size: " + str(self.genome_size)
        msg += "\n\n"
        msg += "Step: " + str(self.steps)
        msg += "  Food: " + str(num_food) + "  energy: " + str(food_energy)
        msg += "  Agents: " + str(num_agents)
        msg += "  learning: " + str(l_agents)
        msg += "  evolving: " + str(e_agents)
        msg += "\n\n"
        msg += self.print_spawn_stats()
        msg += self.print_temp_stats()
        msg += self.print_food_stats()
        #msg += self.print_current_stats()
        msg += self.make_labels(self.previous_agents, "Previous ", "prev")
        msg += "Top " + str(pss) + " previous agents: learning: " + str(bpal)
        msg += "  evolving: " + str(bpae)
        msg += "\n"
        msg += "Top " + str(gss) + " agents in genome store: learning: " + str(gsal)
        msg += "  evolving: " + str(gsae)
        msg += "\n\n"
        msg += "Items in genome store: " + str(gsitems)
        msg += "\n"
        msg += self.make_labels(self.genome_store, "Genome store ", "gs")
        for atype in self.agent_types:
            msg += self.print_action_dist(atype)
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

    # Change background to reflect season and time of day
    tod = gs.agent_view_distance # 1-5
    tod = (tod+2)*0.05
    osc = np.sin(gs.steps/gs.year_period)
    season = int(2 + (osc*2))
    c1 = int(100*tod)
    c2 = int(50*tod)
    bgcs = [[0, c2, c1], [0,c1,c1], [c2, c1, 0], [c1, c1, 0], [c1, c2, 0]]
    bgc = bgcs[season]
    window.color = color.rgb(*bgc)

    # Update agent positions
    for index, agent in enumerate(gs.agents):
        if gs.agents[index].entity is not None:
            xabs = gs.agents[index].xpos
            yabs = gs.agents[index].ypos
            zabs = gs.agents[index].zpos
            gs.agents[index].entity.position = (xabs, yabs, zabs)
            orient = gs.agents[index].orient
            gs.agents[index].entity.rotation = (45*orient, 90, 0)

    # Update predator positions
    for index, predator in enumerate(gs.predators):
        if gs.predators[index].entity is not None:
            xabs = gs.predators[index].xpos
            yabs = gs.predators[index].ypos
            zabs = gs.predators[index].zpos
            gs.predators[index].entity.position = (xabs, yabs, zabs)
            orient = gs.predators[index].orient
            gs.predators[index].entity.rotation = (45*orient, 90, 0)


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
# move params into a config dict
# - measure effect of GA on training
# - if GA has a neutral of positive effect, this shows that most of the agents
# configure their parameters in a similar way
# - GA-assisted learning - set learners to 1.0
# ill-performing agents will be evolved from well-performing agents
# mating mechanism also introduces evolving agents into the mix

# Experiments: graph agent age, fitness
# No evolution
# All evolution
# Hybrid
# Different hidden values: [16], [64], [32, 32]
# Hybrid with and without mating

# - how about flaggelae instead of turn and move?
# longer flaggelae mean more inertial damping, but greater "visibility"
# - queens?
# - add hunger
# - add disease
# - add poisonous food
# - add cold and hot areas
# - add light and dark areas
# - add a mechanism to inherit size, speed, resilience to temperature, etc.
# - add type and affinity for certain types
# - add sounds
# - add excretion and effect on agents and plant life
# - start github repo and report

# Other cctions:
# emit sound [n]
# kill
# attack predator
# build
# stun
# kick
