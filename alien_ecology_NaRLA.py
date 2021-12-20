from NaRLA import *
import numpy as np
from ursina import *
import random, sys, time, os, pickle, math, json, gc
from collections import Counter, deque

default_args = {'num_layers': 1,
                'num_neurons': 16,
                'update_type': 'sync',
                'reward_type': 'all'}
#choices=['all','task','bio','bio_then_all'])


class Agent:
    def __init__(self, x, y, z, color, energy, state_size, action_size, g):
        self.colors = ["pink", "blue", "green", "orange", "purple", "red", "teal", "violet", "yellow"]
        self.start_energy = energy
        self.state_size = state_size
        self.action_size = action_size
        self.xpos = x
        self.ypos = y
        self.zpos = z
        self.color = color
        self.color_str = self.colors[color]
        self.model = NaRLA(default_args, self.state_size, self.action_size)
        self.g = g
        if self.g is not None:
            self.set_state_dicts(g)
        self.state = None
        self.entity = None
        self.previous_stats = []
        self.reset()

    def reset(self):
        self.ep_rewards = 0.0
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
        self.speed = 0.45
        self.previous_states = deque()

    def reset_model(self):
        sd = self.get_state_dicts()
        del(self.model)
        self.model = NaRLA(default_args, self.state_size, self.action_size)
        self.set_state_dicts(sd)

    def get_state_dicts(self):
        return self.model.capture_model_state()

    def set_state_dicts(self, sd):
        return self.model.set_model_state(sd)

    def end_episode(self):
        self.model.end_episode(self.ep_rewards)
        self.reset()

    def record_reward(self, reward):
        self.ep_rewards += reward
        self.model.distribute_task_reward(reward)

    def get_action(self):
        return self.model.forward(self.state)

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
# min_reproduction_age=200
# min_reproduction_energy=200

# All evolution:
# learners=0.00
# min_reproduction_age=50
# min_reproduction_energy=80

# Hybrid evolution:
# learners=0.50
# min_reproduction_age=50
# min_reproduction_energy=100

class game_space:
    def __init__(self,
                 num_prev_states=1,
                 num_recent_actions=1000,
                 area_size=50,
                 year_period=5*50,
                 day_period=5,
                 weather_harshness=0,
                 num_agents=10,
                 agent_start_energy=300,
                 agent_max_inventory=20,
                 num_predators=3,
                 predator_view_distance=7,
                 predator_kill_distance=2,
                 food_sources=20,
                 food_spawns=20,
                 food_dist=5,
                 food_repro_energy=15,
                 food_start_energy=10,
                 food_energy_growth=3,
                 food_max_percent=0.08,
                 food_plant_success=0.8,
                 berry_max_age=50,
                 pheromone_radius=8,
                 pheromone_decay=0.95,
                 min_reproduction_age=50,
                 min_reproduction_energy=100,
                 reproduction_cost=10,
                 visuals=True,
                 savedir="alien_ecology_save",
                 statsdir="alien_ecology_stats"):
        self.steps = 1
        self.spawns = 0
        self.rebirths = 0
        self.continuations = 0
        self.resets = 0
        self.deaths = 0
        self.eaten = 0
        self.food_picked = 0
        self.food_eaten = 0
        self.food_dropped = 0
        self.num_recent_actions = num_recent_actions
        self.visuals = visuals
        self.savedir = savedir
        self.statsdir = statsdir
        self.stats = {}
        self.load_stats()
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
        self.pheromone_radius = pheromone_radius
        self.pheromone_decay = pheromone_decay
        self.pheromones = []
        self.berries = []
        self.min_reproduction_age = min_reproduction_age
        self.min_reproduction_energy = min_reproduction_energy
        self.reproduction_cost = reproduction_cost
        self.agent_view_distance = 5
        self.visible_area = math.pi*(self.agent_view_distance**2)
        self.agent_types = ["learning"]
        self.agent_actions = {}
        self.recent_actions = {}
        for t in self.agent_types:
            self.agent_actions[t] = Counter()
            self.recent_actions[t] = deque()
        self.actions = ["pick_food",
                        "eat_food",
                        "drop_food",
                        "rotate_right",
                        "rotate_left",
                        "flip",
                        "propel",
                        #"null",
                        "propel_up",
                        "propel_right",
                        "propel_down",
                        "propel_left",
                        #"mate",
                        #"freq_up",
                        #"freq_down",
                        #"move_random",
                        "emit_pheromone"
                        ]
        self.observations = ["food_up",
                             "food_right",
                             "food_down",
                             "food_left",
                             "visible_food",
                             "food_pickable",
                             "pheromone_up",
                             "pheromone_right",
                             "pheromone_down",
                             "pheromone_left",
                             "agents_up",
                             "agents_right",
                             "agents_down",
                             "agents_left",
                             "visible_agents",
                             #"can_mate",
                             #"mate_in_range",
                             "predators_up",
                             "predators_right",
                             "predators_down",
                             "predators_left",
                             "visible_predators",
                             "previous_action",
                             "own_energy",
                             "own_temperature",
                             "own_xposition",
                             "own_yposition",
                             "own_orientation",
                             "own_xvelocity",
                             "own_yvelocity",
                             "food_inventory",
                             "environment_temperature",
                             "visibility",
                             #"reproduction_oscillator",
                             "distance_moved",
                             "own_happiness",
                             "own_age",
                             #"age_oscillator",
                             #"step_oscillator_day",
                             #"step_oscillator_day_offset",
                             #"step_oscillator_year",
                             #"step_oscillator_year_offset",
                             #"random_input"
                             ]
        self.action_size = len(self.actions)
        self.state_size = len(self.observations)*self.num_prev_states
        self.genome_store = [None for x in range(self.num_agents)]
        self.previous_agents = deque()
        self.num_previous_agents = 50
        self.fitness_index = 2
        self.evaluate_learner_every = 30
        self.top_n = 1.0
        self.save_every=500
        self.record_every=50
        self.create_starting_food()
        self.load_genomes()
        self.agents = []
        self.spawn_new_agents()
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
        if self.save_every > 0 and self.steps % 100 == 0:
            self.save_stats()
        self.print_stats()
        self.steps += 1

    def save_genomes(self):
        if len(self.genome_store) < 1:
            return
        print("Saving genome_store")
        with open(self.savedir + "/genome_store.pkl", "wb") as f:
            f.write(pickle.dumps(self.genome_store))

    def load_genomes(self):
        if os.path.exists(savedir + "/genome_store.pkl"):
            with open(self.savedir + "/genome_store.pkl", "rb") as f:
                self.genome_store = pickle.load(f)

#########################
# Initialization routines
#########################
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

    def spawn_new_agents(self):
        g = []
        for item in self.genome_store:
            if item is not None:
                g.append(item[0])
            else:
                g.append(None)
        if len(g) < self.num_agents:
            for n in range(self.num_agents - len(g)):
                g.append(None)
        if len(g) > self.num_agents:
            g = random.sample(g, self.num_agents)
        for n in range(self.num_agents):
            self.spawn_new_agent(g[n])
            sd = self.agents[n].get_state_dicts()
            entry = [sd, 0, 0, 0, 0]
            self.log_genome(n, entry)
        self.save_genomes()

    def spawn_new_agent(self, g):
        self.spawns += 1
        xpos = random.random()*self.area_size
        ypos = random.random()*self.area_size
        zpos = -1
        color = 0
        a = Agent(xpos,
                  ypos,
                  zpos,
                  color,
                  self.agent_start_energy,
                  self.state_size,
                  self.action_size,
                  g)
        self.agents.append(a)
        index = len(self.agents)-1
        self.set_initial_agent_state(index)
        if self.visuals == True:
            self.set_agent_entity(index)

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
        xvel = min(xvel, (speed*10))
        yvel = min(yvel, (speed*10))
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
        # 2 - 5
        #self.agent_view_distance = 3.5 + (1.5*osc)
        # 4 - 8
        self.agent_view_distance = 6.0 + (2.0*osc)
        self.visible_area = math.pi*(self.agent_view_distance**2)

    def set_environment_temperature(self):
        osc = np.sin(self.steps/self.year_period)
        # Climate change modifies osc multiplier and/or initial value
        #osc2 = np.sin(2*self.steps/(self.year_period*10))
        #osc3 = np.sin(3*(self.steps+self.year_period*5)/(self.year_period*10))
        #v1 = 14 + self.weather_harshness + osc3
        #v2 = 12 + self.weather_harshness + osc2
        v1 = 12
        v2 = 14
        # Max temp: 34
        # Min temp: 0
        self.environment_temperature = v1 + (self.agent_view_distance*0.5) + random.random()*3 + ((v2)*osc)
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
            g = self.agents[index].get_state_dicts()
            mpf = np.mean([x[self.fitness_index] for x in self.agents[index].previous_stats])
            pfm = 0
            if len(self.genome_store) > 1:
                pfm = np.mean([x[self.fitness_index] for x in self.genome_store])
            if pfm > 0:
                if mpf < pfm * 0.75:
                    g = random.choice([x[0] for x in self.genome_store])
                    self.rebirths += 1
                else:
                    self.continuations += 1
            self.agents.pop(index)
            self.spawn_new_agent(g)
            return True

    def reset_agents(self, reset):
        gc.collect()
        for index in reset:
            a = self.agents[index].age
            h = self.agents[index].happiness
            d = self.agents[index].distance_travelled
            g = self.agents[index].get_state_dicts()
            f = a + h + d
            entry = [g, f, a, h, d]
            self.add_previous_agent(entry)
            self.log_genome(index, entry)
            affected = self.get_adjacent_agent_indices(index)
            for i in affected:
                self.agents[i].happiness -= 5
            self.agents[index].ep_rewards += a/self.agent_start_energy
            entry[0] = None
            self.agents[index].previous_stats.append(entry)
            self.deaths += 1
            self.resets += 1
            self.agents[index].end_episode()
            if len(self.agents[index].previous_stats) % 5 == 0:
                self.agents[index].reset_model()
            self.agents[index].xpos = random.random()*self.area_size
            self.agents[index].ypos = random.random()*self.area_size
            self.set_initial_agent_state(index)
            #self.evaluate_learner(index)

    def update_agent_status(self):
        dead = set()
        reset = set()
        for index in range(len(self.agents)):
            self.agents[index].age += 1
            self.set_agent_temperature(index)
            energy_drain = 1
            temperature = self.agents[index].temperature
            if temperature > 34:
                energy_drain += (temperature - 30) * 0.05
                self.agents[index].happiness -= 1
            if temperature < 4:
                energy_drain += (5 - temperature) * 0.05
                self.agents[index].happiness -= 1
            self.agents[index].energy -= energy_drain
            if self.agents[index].energy <= 0:
                reset.add(index)
        if len(reset) > 0:
            self.reset_agents(reset)

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
            self.agents[index].happiness += 1
        return predator_count

    def get_visible_predators(self, index):
        xv, yv = self.get_viewpoint(index)
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance*2)
        predator_count = len(predators)
        if predator_count > 0:
            self.agents[index].happiness += 1
        return predator_count

    def get_predators_up(self, index):
        xv, yv = self.get_viewpoint_up(index)
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance)
        predator_count = len(predators)
        if predator_count > 0:
            self.agents[index].happiness += 1
        return predator_count

    def get_predators_right(self, index):
        xv, yv = self.get_viewpoint_right(index)
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance)
        predator_count = len(predators)
        if predator_count > 0:
            self.agents[index].happiness += 1
        return predator_count

    def get_predators_down(self, index):
        xv, yv = self.get_viewpoint_down(index)
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance)
        predator_count = len(predators)
        if predator_count > 0:
            self.agents[index].happiness += 1
        return predator_count

    def get_predators_left(self, index):
        xv, yv = self.get_viewpoint_left(index)
        predators = self.get_predators_in_radius(xv, yv, self.agent_view_distance)
        predator_count = len(predators)
        if predator_count > 0:
            self.agents[index].happiness += 1
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
                        reset.append(v)
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
        if self.agents[index].xvel == 0 and self.agents[index].yvel == 0:
            self.agents[index].happiness -= 1
        reward = class_method(index)
        self.agents[index].record_reward(reward)

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
                self.agents[index].happiness += 5
                self.food_picked += 1
                self.agents[index].food_inventory += 1
                self.remove_berry(bi)
                return 1
        fi, fval = self.get_nearest_food(xpos, ypos)
        if fi is not None:
            if fval <= 1:
                picked = True
                self.agents[index].happiness += 5
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
            if self.agents[index].energy <= self.agent_start_energy:
                self.food_eaten += 1
                self.agents[index].energy += 20
                self.agents[index].happiness += 100
                reward = 0.1
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
        return 0

    def action_freq_down(self, index):
        self.agents[index].frequency = self.agents[index].frequency * 1.1
        return 0

    def action_emit_pheromone(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        self.create_new_pheromone(xpos, ypos)
        return 0

    def action_move_random(self, index):
        if "action_rotate_right" in self.actions:
            actions = ["action_rotate_right", "action_rotate_left",
                       "action_propel", "action_flip"]
        else:
            actions = ["action_propel_up", "action_propel_right",
                       "action_propel_down", "action_propel_left"]
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
        #state = torch.FloatTensor(state)
        #state = state.unsqueeze(0)
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

    def get_can_mate(self, index):
        ret = 0
        if self.agents[index].age >= self.min_reproduction_age:
            if self.agents[index].energy >= self.min_reproduction_energy:
                ret = 1
        return ret

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
        return self.agents[index].orient

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
        s = self.pheromone_radius*2
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
        pheromones = self.get_pheromones_in_radius(xv, yv, self.pheromone_radius)
        return(len(pheromones))

    def get_adjacent_pheromones(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        pheromones = self.get_pheromones_in_radius(xpos, ypos, self.pheromone_radius)
        return(len(pheromones))

    def get_pheromone_up(self, index):
        xv, yv = self.get_viewpoint_up(index)
        pheromones = self.get_pheromones_in_radius(xv, yv, self.pheromone_radius)
        return(len(pheromones))

    def get_pheromone_right(self, index):
        xv, yv = self.get_viewpoint_right(index)
        pheromones = self.get_pheromones_in_radius(xv, yv, self.pheromone_radius)
        return(len(pheromones))

    def get_pheromone_down(self, index):
        xv, yv = self.get_viewpoint_down(index)
        pheromones = self.get_pheromones_in_radius(xv, yv, self.pheromone_radius)
        return(len(pheromones))

    def get_pheromone_left(self, index):
        xv, yv = self.get_viewpoint_left(index)
        pheromones = self.get_pheromones_in_radius(xv, yv, self.pheromone_radius)
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
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance*2)
        b = self.get_berries_in_radius(xv, yv, self.agent_view_distance*2)
        return(len(food) + len(b))

    def get_adjacent_food(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        food = self.get_food_in_radius(xpos, ypos, self.agent_view_distance)
        b = self.get_berries_in_radius(xv, yv, self.agent_view_distance)
        return(len(food) + len(b))

    def get_food_up(self, index):
        xv, yv = self.get_viewpoint_up(index)
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance)
        b = self.get_berries_in_radius(xv, yv, self.agent_view_distance)
        return(len(food) + len(b))

    def get_food_right(self, index):
        xv, yv = self.get_viewpoint_right(index)
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance)
        b = self.get_berries_in_radius(xv, yv, self.agent_view_distance)
        return(len(food) + len(b))

    def get_food_down(self, index):
        xv, yv = self.get_viewpoint_down(index)
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance)
        b = self.get_berries_in_radius(xv, yv, self.agent_view_distance)
        return(len(food) + len(b))

    def get_food_left(self, index):
        xv, yv = self.get_viewpoint_left(index)
        food = self.get_food_in_radius(xv, yv, self.agent_view_distance)
        b = self.get_berries_in_radius(xv, yv, self.agent_view_distance)
        return(len(food) + len(b))

    def get_food_pickable(self, index):
        xpos = self.agents[index].xpos
        ypos = self.agents[index].ypos
        food_index, food_distance = self.get_nearest_food(xpos, ypos)
        if food_index is not None:
            if food_distance < 1:
                return 1
        berry_index, berry_distance = self.get_nearest_berry(xpos, ypos)
        if berry_index is not None:
            if berry_distance < 1:
                return 1
        return 0

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
            growth = -0.5*random.random()
        if self.environment_temperature > 35:
            growth = -0.5*random.random()
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
                if self.food[index].entity is not None:
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
        xabs = self.berries[index].xpos
        yabs = self.berries[index].ypos
        zabs = self.berries[index].zpos
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
            self.berries[index].entity.disable()
            del(self.berries[index].entity)
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
                if self.environment_temperature > 7 and self.environment_temperature < 35:
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

    def add_previous_agent(self, entry):
        if len(self.previous_agents) >= self.num_previous_agents:
            self.previous_agents.popleft()
        self.previous_agents.append(entry)

    def get_best_previous_agents(self, num):
        fitness = [x[self.fitness_index] for x in self.previous_agents]
        indices = []
        if len(fitness) > num:
            indices = np.argpartition(fitness,-num)[-num:]
        return indices

    def get_best_agents_from_store(self, num):
        fitness = [x[self.fitness_index] for x in self.genome_store]
        indices = []
        if len(fitness) > num:
            indices = np.argpartition(fitness,-num)[-num:]
        return indices

    def log_genome(self, index, entry):
        self.genome_store[index] = entry

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
        lmsg += affix + " agent (" + str(len(var)) + ") stats:"
        lmsg += "\n"
        for i, l in enumerate(labels):
            temp = [x[i+1] for x in var]
            if len(temp) > 0:
                me = np.mean(temp)
                self.record_stats(saff+"_"+l+"_mean", me)
                mx = max(temp)
                self.record_stats(saff+"_"+l+"_max", mx)
                mi = min(temp)
                self.record_stats(saff+"_"+l+"_min", mi)
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
        #msg += "  rebirths: " + str(self.rebirths)
        #msg += "  continuations: " + str(self.continuations)
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

    def print_run_stats(self):
        msg = ""
        msg += "Starting agents: " + str(self.num_agents)
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
        msg += "\n\n"
        return msg

    def get_stats(self):
        agent_energy = int(sum([x.energy for x in self.agents]))
        self.record_stats("agent energy", agent_energy)
        self.record_stats("berries", len(self.berries))
        self.record_stats("food", len(self.food))
        self.record_stats("food picked", self.food_picked)
        self.record_stats("food eaten", self.food_eaten)
        self.record_stats("food dropped", self.food_dropped)
        minv = np.mean([x.food_inventory for x in self.agents])
        self.record_stats("mean inventory", minv)
        food_energy = int(sum([x.energy for x in self.food]))
        self.record_stats("food energy", food_energy)
        gsitems = len(self.genome_store)
        mpf = np.mean([x[1] for x in self.previous_agents])
        msg = ""
        msg += self.print_run_stats()
        msg += "Step: " + str(self.steps)
        msg += "  Food: " + str(len(self.food))
        msg += "  berries: " + str(len(self.berries))
        msg += "  inventory: " + "%.2f"%minv
        msg += "\n\n"
        msg += self.print_spawn_stats()
        msg += self.print_temp_stats()
        msg += self.print_food_stats()
        #msg += self.print_current_stats()
        msg += self.make_labels(self.previous_agents, "Previous ", "prev")
        #msg += "Top " + str(pss) + " previous agents: learning: " + str(bpal)
        #msg += "  evolving: " + str(bpae)
        #msg += "\n"
        #msg += "Top " + str(gss) + " agents in genome store: learning: " + str(gsal)
        #msg += "  evolving: " + str(gsae)
        #msg += "\n\n"
        #msg += "Items in genome store: " + str(gsitems)
        #msg += "\n"
        msg += self.make_labels(self.genome_store, "Genome store ", "gs")
        for atype in self.agent_types:
            msg += self.print_action_dist(atype)
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

