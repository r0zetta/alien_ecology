from ursina import *
import random, sys, time, os, json, re, pickle, operator, math
import numpy as np
from collections import Counter, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from scipy.spatial import distance

class Net(nn.Module):
    def __init__(self, weights):
        super(Net, self).__init__()
        self.fc_layers = []
        for item in weights:
            s1, s2, d = item
            fc = nn.Linear(s1, s2, bias=False)
            fc.weight_data = d
            self.fc_layers.append(fc)

    def forward(self, x):
        for i, fc in enumerate(self.fc_layers):
            if i < len(self.fc_layers):
                x = F.relu(fc(x))
            else:
                x = F.softmax(F.relu(fc(x)))
        return x

    def get_action(self, state):
        with torch.no_grad():
            state = state.float()
            ret = self.forward(Variable(state))
            action = torch.argmax(ret)
            return action

class GN_model:
    def __init__(self, weights):
        self.policy = Net(weights)

    def get_action(self, state):
        action = self.policy.get_action(state)
        return action

# Actions:
# increase/decrease oscillator[n] period
# emit sound [n]
# kill
# stun
# kick

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

class Agent:
    def __init__(self, x, y, z, color, energy, state_size, action_size, hidden_size, genome):
        self.colors = ["blue", "green", "orange", "purple", "red", "teal", "violet", "yellow"]
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.previous_states = deque()
        self.sample = False
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.xvel = 0
        self.yvel = 0
        self.prev_action = 0
        self.orient = 0 # 0-7 in 45 degree increments
        self.energy = energy
        self.food_inventory = 0
        self.distance_travelled = 0
        self.happiness = 0
        self.fitness = 0
        self.age = 0
        self.temperature = 20
        self.inertial_damping = 0.90
        self.speed = 0.5
        self.color = color
        self.color_str = self.colors[color]
        #self.inertial_damping = random.uniform(0.7, 0.99)
        #self.speed = random.uniform(0.3, 0.7)
        #self.color = random.choice(self.colors)
        self.genome = genome
        weights = []
        m1 = 0
        m2 = self.state_size * self.hidden_size[0]
        w = torch.Tensor(np.reshape(genome[m1:m2], (self.hidden_size[0], self.state_size)))
        weights.append([self.state_size, self.hidden_size[0], w])
        if len(self.hidden_size) > 1:
            for i in range(len(self.hidden_size)):
                if i+1 < len(self.hidden_size):
                    m1 = m2
                    m2 = m1 + (self.hidden_size[i] * self.hidden_size[i+1])
                    w = torch.Tensor(np.reshape(genome[m1:m2],
                                     (self.hidden_size[i], self.hidden_size[i+1])))
                    weights.append([self.hidden_size[i], self.hidden_size[i+1], w])
        w = torch.Tensor(np.reshape(genome[m2:], (action_size, hidden_size[-1])))
        weights.append([self.hidden_size[-1], self.action_size, w])
        self.model = GN_model(weights)
        self.state = None
        self.episode_steps = 0
        self.last_success = None
        self.entity = None

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

class Pheromone:
    def __init__(self, x, y, z):
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.strength = 100
        self.entity = None

class game_space:
    def __init__(self,
                 hidden_size=[16],
                 num_prev_states=1,
                 mutation_rate=0.001,
                 area_size=50,
                 year_length=100,
                 day_length=10,
                 num_agents=100,
                 agent_start_energy=200,
                 agent_max_inventory=10,
                 num_predators=5,
                 predator_view_distance=5,
                 predator_kill_distance=2,
                 food_sources=40,
                 food_spawns=10,
                 food_dist=5,
                 food_repro_energy=15,
                 food_start_energy=10,
                 food_energy_growth=0.1,
                 food_plant_success=1,
                 pheromone_decay=0.90,
                 min_reproduction_age=50,
                 min_reproduction_energy=150,
                 reproduction_cost=0,
                 agent_view_distance=5,
                 visuals=True,
                 save_stuff=True,
                 savedir="alien_ecology_save"):
        self.steps = 0
        self.spawns = 0
        self.births = 0
        self.deaths = 0
        self.killed = 0
        self.food_picked = 0
        self.food_eaten = 0
        self.food_planted = 0
        self.visuals = visuals
        self.savedir = savedir
        self.save_stuff = save_stuff
        self.num_prev_states = num_prev_states
        self.environment_temperature = 20
        self.area_size = area_size
        self.year_length = year_length
        self.day_length = day_length
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
        self.food_plant_success = food_plant_success
        self.pheromone_decay = pheromone_decay
        self.pheromones = []
        self.min_reproduction_age = min_reproduction_age
        self.min_reproduction_energy = min_reproduction_energy
        self.reproduction_cost = reproduction_cost
        self.agent_view_distance = agent_view_distance
        self.visible_area = math.pi*(self.agent_view_distance**2)
        self.mutation_rate = mutation_rate
        self.actions = ["null",
                        "rotate_right",
                        "rotate_left",
                        "propel",
                        "pick_food",
                        "plant_food",
                        "eat_food",
                        "mate",
                        "emit_pheromone",
                        "move_random"]
        self.agent_actions = Counter()
        self.recent_actions = deque()
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
                             "age_oscillator_fast",
                             "age_oscillator_med",
                             "age_oscillator_slow",
                             "step_oscillator_fast",
                             "step_oscillator_med",
                             "step_oscillator_med_offset",
                             "step_oscillator_slow",
                             "step_oscillator_slow_offset",
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
        self.create_new_agents(use_pool=True)
        self.create_predators()

    def step(self):
        self.steps += 1
        self.set_environment_visibility()
        self.set_environment_temperature()
        self.set_agent_states()
        self.apply_agent_physics()
        self.run_agent_actions()
        self.apply_predator_physics()
        self.run_predator_actions()
        self.update_agent_status()
        self.update_pheromone_status()
        self.reproduce_food()
        if self.steps % 500 == 0:
            if len(self.genome_store) >= 100:
                ngs = random.sample(self.genome_store, 75)
                self.genome_store = ngs
        if self.save_stuff == True:
            if self.steps % 20 == 0:
                self.save_genomes()
        self.print_stats()


# To do:
# - add predators, both moving and stationary
# - queens?
# - add climate change, both periodic and from balance of plants and organisms
# - add hunger
# - add disease
# - add poisonous food
# - add cold and hot areas
# - add a mechanism to inherit size, speed, resilience to temperature, etc.
# - add type and affinity for certain types
# - add sounds
# - add excretion and effect on agents and plant life
# - start github repo and report

#########################
# Initialization routines
#########################
    def set_predator_entity(self, index):
        xabs = self.predators[index].xpos
        yabs = self.predators[index].ypos
        zabs = self.predators[index].zpos
        s = 2
        texture = "textures/predator"
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
                a, g = item
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

    def spawn_random_agent(self, genome):
        self.spawns += 1
        xpos = random.random()*self.area_size
        ypos = random.random()*self.area_size
        zpos = -1
        a = Agent(xpos,
                  ypos,
                  zpos,
                  0,
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

    def create_new_agents(self, use_pool=False):
        self.agents = []
        if use_pool == True:
            for genome in self.genome_pool:
                self.spawn_random_agent(genome)
        else:
            for n in range(self.num_agents):
                genome = self.make_random_genome()
                self.spawn_random_agent(genome)

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

########################
# Environmental effects
########################

    def set_environment_visibility(self):
        osc = np.sin(self.steps/self.day_length)
        self.agent_view_distance = 3 + (2*osc)
        self.visible_area = math.pi*(self.agent_view_distance**2)

    def set_environment_temperature(self):
        osc = np.sin(self.steps/self.year_length)
        self.environment_temperature = 14 + self.agent_view_distance + random.random()*3 + (16*osc)

    def set_agent_temperature(self, index):
        temp = self.environment_temperature - 5 + (self.count_adjacent_agents(index))
        self.agents[index].temperature = temp

########################
# Agent runtime routines
########################
    def record_recent_actions(self, action):
        if len(self.recent_actions) > 10000:
            self.recent_actions.popleft()
        self.recent_actions.append(action)
        ra = Counter()
        for a in self.recent_actions:
            ra[self.actions[a]] += 1
        self.agent_actions = ra

    def run_agent_actions(self):
        for n in range(len(self.agents)):
            action = int(self.agents[n].get_action())
            a = self.actions[action]
            self.record_recent_actions(action)
            self.apply_agent_action(n, action)
            self.update_agent_position(n)

    def birth_new_agent(self, index1, index2):
        self.births += 1
        self.spawns += 1
        self.agents[index1].happiness += 100
        self.agents[index2].happiness += 100
        xpos = self.agents[index1].xpos
        ypos = self.agents[index2].ypos
        zpos = -1
        g1 = self.agents[index1].genome
        g2 = self.agents[index2].genome
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

    def kill_agents(self, dead):
        for index in dead:
            a = self.agents[index].age
            h = self.agents[index].happiness
            d = self.agents[index].distance_travelled
            f = a + h + d
            g = self.agents[index].genome
            self.store_genome(g, f)
            self.add_previous_agent(g, f)
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
                genome = None
                if len(self.agents) > 10:
                    if random.random() > 0.05:
                        genome = self.make_genome_from_active()
                    else:
                        genome = self.make_random_genome()
                else:
                    genome = self.make_random_genome()
                self.spawn_random_agent(genome)

    def update_agent_status(self):
        # Other things can happen here such as
        # - environmental effects
        # - happiness
        # Decrease energy and remove if the agent is dead
        dead = set()
        for index in range(len(self.agents)):
            self.agents[index].age += 1
            self.set_agent_temperature(index)
            energy_drain = 1
            temperature = self.agents[index].temperature
            if temperature > 30:
                energy_drain += (temperature - 30) * 0.1
                self.agents[index].happiness -= 1
            if temperature < 10:
                energy_drain += (10 - temperature) * 0.1
                self.agents[index].happiness -= 1
            self.agents[index].energy -= energy_drain
            if self.agents[index].energy <= 0:
                dead.add(index)
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

    def get_visible_predators(self, index):
        xv, yv = self.get_viewpoint(index)
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
                self.killed += len(victims)
                self.kill_agents(victims)
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
        return class_method(index)

    def action_null(self, index):
        return

    def action_rotate_right(self, index):
        self.agents[index].orient += 1
        if self.agents[index].orient > 7:
            self.agents[index].orient = 0

    def action_rotate_left(self, index):
        self.agents[index].orient -= 1
        if self.agents[index].orient < 0:
            self.agents[index].orient = 7

    def action_propel(self, index):
        orient = self.agents[index].orient
        speed = self.agents[index].speed
        xvel = self.agents[index].xvel
        yvel = self.agents[index].yvel
        xv, yv = self.propel(xvel, yvel, orient, speed)
        self.agents[index].xvel = xv
        self.agents[index].yvel = yv

    def action_pick_food(self, index):
        carrying = self.agents[index].food_inventory
        if carrying >= self.agent_max_inventory:
            self.agents[index].happiness -= 5
            return
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        fi, val = self.get_nearest_food(xpos, ypos)
        if val <= 1:
            self.agents[index].happiness += 20
            self.food_picked += 1
            self.agents[index].food_inventory += 1
            self.food[fi].energy -= 2
            if self.food[fi].energy < 0:
                self.remove_food(fi)

    def action_eat_food(self, index):
        if self.agents[index].food_inventory > 0:
            self.food_eaten += 1
            if self.agents[index].energy >= self.agent_start_energy:
                self.agents[index].happiness -= 5
            else:
                self.agents[index].energy += 20
                self.agents[index].happiness += 100
            self.agents[index].food_inventory -= 1

    def action_plant_food(self, index):
        if self.agents[index].food_inventory > 0:
            self.food_planted += 1
            self.agents[index].food_inventory -= 1
            xpos = self.agents[index].xpos
            ypos = self.agents[index].ypos
            if random.random() <= self.food_plant_success:
                self.plant_food(xpos, ypos)
                self.agents[index].happiness += 20

    def action_mate(self, index):
        if self.agents[index].energy < self.min_reproduction_energy:
            return
        if self.agents[index].age < self.min_reproduction_age:
            return
        if len(self.agents) < 2:
            return
        ai, val = self.get_nearest_agent(index)
        if ai is not None:
            if val <= 1:
                if self.agents[ai].energy >= self.min_reproduction_energy:
                    if self.agents[ai].age >= self.min_reproduction_age:
                        self.birth_new_agent(index, ai)
                        self.agents[index].energy -= self.reproduction_cost
                        self.agents[ai].energy -= self.reproduction_cost

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
        return self.agents[index].age * 0.1

    def get_own_energy(self, index):
        return self.agents[index].energy * 0.1

    def get_own_temperature(self, index):
        return self.agents[index].temperature

    def get_food_inventory(self, index):
        return self.agents[index].food_inventory

    def get_distance_moved(self, index):
        return self.agents[index].distance_travelled * 0.1

    def get_own_happiness(self, index):
        return self.agents[index].happiness * 0.1

    def get_own_xposition(self, index):
        return self.agents[index].xpos / self.area_size

    def get_own_yposition(self, index):
        return self.agents[index].ypos / self.area_size

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

    def get_age_oscillator_fast(self, index):
        return np.sin(self.agents[index].age)

    def get_age_oscillator_med(self, index):
        return np.sin(self.agents[index].age/self.day_length)

    def get_age_oscillator_slow(self, index):
        return np.sin(self.agents[index].age/self.year_length)

    def get_step_oscillator_fast(self, index):
        return np.sin(self.steps)

    def get_step_oscillator_med(self, index):
        return np.sin(self.steps/self.day_length)

    def get_step_oscillator_med_offset(self, index):
        return np.sin((self.steps+(self.day_length/2))/self.day_length)

    def get_step_oscillator_slow(self, index):
        return np.sin(self.steps/self.year_length)

    def get_step_oscillator_slow_offset(self, index):
        return np.sin((self.steps+(self.year_length/2))/self.year_length)

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

    def get_nearest_pheromone(self, xpos, ypos):
        distances = []
        for f in self.pheromones:
            ax = f.xpos
            ay = f.ypos
            if self.filter_by_distance(xpos, ypos, ax, ay, radius) is True:
                distances.append(self.distance(xpos, ypos, ax, ay))
        if len(distances) > 0:
            shortest_index = np.argmin(distances)
            shortest_value = distances[shortest_index]
            return shortest_index, shortest_value
        else:
            return None, None


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
            growth = -1.2 * growth
        if self.environment_temperature > 35:
            growth = (random.random()-(0.2*(self.environment_temperature-35))) * growth
        dead = []
        for index, f in enumerate(self.food):
            self.food[index].energy += growth
            if self.food[index].energy >= self.food_repro_energy:
                if len(self.food) < 0.02*(self.area_size**2):
                    self.spawn_new_food(self.food[index].xpos, self.food[index].ypos)
                    self.food[index].energy = self.food_start_energy
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

    def plant_food(self, xpos, ypos):
        z = -1
        x = xpos
        y = ypos
        f = Food(x, y, z, self.food_start_energy)
        self.food.append(f)
        if self.visuals == True:
            fi = len(self.food)-1
            self.set_food_entity(fi)

    def spawn_new_food(self, xpos, ypos):
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
            fi = len(self.food)-1
            self.set_food_entity(fi)

########################################
# Routines related to genetic algorithms
########################################
    def make_random_genome(self):
        genome = np.random.uniform(-1, 1, self.genome_size)
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
                val = random.uniform(-1, 1)
                gm[index] = val
            new_genomes.append(gm)
        return new_genomes

    def add_previous_agent(self, genome, fitness):
        if len(self.previous_agents) >= 100:
            self.previous_agents.popleft()
        self.previous_agents.append([genome, fitness])

    def get_best_previous_agents(self, num):
        fitness = [x[1] for x in self.previous_agents]
        indices = np.argpartition(fitness,-num)[-num:]
        genomes = [self.previous_agents[i][0] for i in indices]
        return genomes

    def make_genome_from_previous(self):
        if len(self.previous_agents) < 10:
            return self.make_random_genome()
        genomes = self.get_best_previous_agents(10)
        parents = random.sample(genomes, 2)
        offspring = self.reproduce_genome(parents[0], parents[1], 1)
        chosen = random.choice(offspring)
        mutated = self.mutate_genome(chosen, 1)[0]
        return mutated

    def store_genome(self, genome, fitness):
        min_fitness = 0
        min_item = 0
        if len(self.genome_store) > 0:
            fitnesses = [x[0] for x in self.genome_store]
            min_item = np.argmin(fitnesses)
            min_fitness = fitnesses[min_item]
        if fitness > self.agent_start_energy and fitness > min_fitness:
            if len(self.genome_store) >= 100:
                self.genome_store.pop(min_item)
            self.genome_store.append([fitness, genome])

    def make_genome_from_active(self):
        return self.make_genome_from_previous()



######################
# Printable statistics
######################
    def get_stats(self):
        num_agents = len(self.agents)
        agent_energy = int(sum([x.energy for x in self.agents]))
        num_food = len(self.food)
        food_energy = int(sum([x.energy for x in self.food]))
        ages = [x.age for x in self.agents]
        max_age = np.max(ages)
        mean_age = int(np.mean(ages))
        hap = [x.happiness for x in self.agents]
        max_hap = np.max(hap)
        mean_hap = int(np.mean(hap))
        d = [x.distance_travelled for x in self.agents]
        max_d = np.max(d)
        mean_d = int(np.mean(d))
        etemp = self.environment_temperature
        mtemp = int(np.mean([x.temperature for x in self.agents]))
        vrange = self.agent_view_distance
        gsitems = len(self.genome_store)
        mpf = np.mean([x[1] for x in self.previous_agents])
        gf = [x[0] for x in self.genome_store]
        gsmea = 0
        gsmax = 0
        if len(gf) > 0:
            gsmea = np.mean(gf)
            gsmax = np.max(gf)
        msg = ""
        msg += "Starting agents: " + str(self.num_agents)
        msg += " area size: " + str(self.area_size)
        msg += " Step: " + str(self.steps) + "\n"
        msg += "Action size: " + str(self.action_size)
        msg += " state_size: " + str(self.state_size)
        msg += " hidden: " + str(self.hidden_size)
        msg += " genome size: " + str(self.genome_size) + "\n"
        msg += "\n"
        msg += "Agents: " + str(num_agents) + " energy: " + str(agent_energy) + "\n"
        msg += "mean age: " + "%.2f"%mean_age
        msg += " max age: " + str(max_age)
        msg += " mean hap: " + "%.2f"%mean_hap
        msg += " max hap: " + str(max_hap)
        msg += " mean d: " + "%.2f"%mean_d
        msg += " max d: " + "%.2f"%max_d + "\n"
        msg += "\n"
        msg += "Previous agent mean fitness: " + "%.2f"%mpf + "\n"
        msg += "\n"
        msg += "Food: " + str(num_food) + " energy: " + str(food_energy) + "\n"
        msg += "\n"
        msg += "Spawns: " + str(self.spawns) + " Births: " + str(self.births)
        msg += " Deaths: " + str(self.deaths) + " Killed: " + str(self.killed) + "\n"
        msg += "\n"
        msg += "Food picked: " + str(self.food_picked)
        msg += " eaten: " + str(self.food_eaten) 
        msg += " planted: " + str(self.food_planted) + "\n"
        msg += "\n"
        msg += "Environment temp: " + "%.2f"%etemp + " mean agent temp: " + str(mtemp)
        msg += " Visual range: " + "%.2f"%vrange + "\n"
        msg += "\n"
        msg += "Items in genome store: " + str(gsitems) 
        msg += " mean fitness: " + "%.2f"%gsmea + " max fitness: " + "%.2f"%gsmax + "\n"
        msg += "\n"
        ta = sum([self.agent_actions[x] for x in self.actions])
        bars = [int(400*(self.agent_actions[x]/ta)) for x in self.actions]
        for i, x in enumerate(self.actions):
            space = (15-len(x))*" "
            bar = bars[i]*"#"
            msg += str(x) + ":" + space + bar + "\n"
        msg += "\n"
        return msg

    def print_stats(self):
        os.system('clear')
        print(gs.get_stats())

# update callback for ursina
def update():
    gs.step()

    # Change background to reflect season and time of day
    tod = gs.agent_view_distance # 1-5
    tod = (tod+2)*0.1
    osc = np.sin(self.steps/self.year_length)
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
