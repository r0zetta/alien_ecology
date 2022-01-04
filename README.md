# Introduction
This repository contains code that implements an environment for simulated organisms to learn and evolve behaviours. The environment itself is a two-dimensional plane that can be configured as either toriodal or bounded. The environment presents a number of problems the agents must learn to solve.

1. Agents have energy that depletes during each simulation step. They can replenish energy in a number of ways, such as by consuming things or staying close to protectors.
2. Predators can be added to the environment. Predators move using very simple hard-coded logic (if they can see agents ahead they move forward, otherwise they move in a random fashion). Agents must thus learn to evade predators.
3. Shooters can be added to the environment. Shooters move randomly and will fire bullets in the direction of agents within a detection radius. Agents can consume shooters by moving onto them, which adds to the agent's energy.
4. Protectors can be added to the environment. Protectors are friendly entities that move randomly. When predators are too close to protectors they become stunned and are unable to perform actions. Agents moving close to a protector replenish energy. The protector healing field also stops bullets fired by shooters.
5. Zones can be added that affect agents. Zone types include attractor, which pull agents in, repulsors which push agents away, acceleration which decrease inertial damping, and damping fields which add extra inertial damping to agents' movement.

Agents can perform rotation and propulsion movement actions, all of which are configurable. Agents can "sense" things in different directions around their position. Sensors are configurable to either up, down, left and right, or in eight 45-degree angles from the center of the agent.

Agent logic is represented by simple feed-forward neural networks. Inputs are observations from the environment that usually include signals denoting that other things are within sensor range. Outputs of the neural network tie into agent actions. Agents can be graded on the number of steps they lived for, or by a fitness value that includes bonuses for performing actions.

Agents are trained using a combination of evolution and reinforcement learning. The weights contained in each agent's neural network are used as a genome. Genomes from organisms that perform the best are stored and used for reproduction, or to replace an ill-performing learning agent's weights after a specified evaluation period. When an evolving organism dies it is respawned at a new random location and given a genome evolved from a genome store. If a learning agent dies, backpropagation is used to train its neural network (using A2C) and it is respawned at a new random location. Learning agents are periodically evaluated on their mean age or fitness during a specified number of previous runs. If a learning agent fails its evaluation, it is replaced with a new learning agent that receives neural network weights from the genome store.

The purpose of this simulation is:
- to observe and document interesting emergent behaviours in the simulated swarm
- to understand if evolution and reinforcement learning mechanisms can be used in tandem
- to discover methods that allow agents to quickly learn policies capable of solving multiple tasks
- to discover methods that allow agents to quickly learn policies capable of solving complex tasks

# Initial experiments
I started by running some experiments designed to train policies on simple tasks, using the smallest neural networks possible.

## Experiment 1: evade predators
This simulation contains only agents and predators. The agent's state includes readings of the number of predators above, below, to the left, and to the right of the agent, within the agent's maximum view distance (state size 4, action size 4, hidden size [8], parameters 72). Agents learn to run away from predators, and this policy can be learned quickly. Evolutionary processes find optimal policies much quicker than reinforcement learning mechanisms.

## Experiment 2: collect food
This simulation contains only agents and food. Agents start with 200 energy and lose 1 energy per step. Colliding with food grants the agent 50 energy and causes the food to respawn in a new random location. Agents receive a reading of number of food above, below, to the left, and to the right of the agent, and their own energy value (state size 4, action size 4, hidden size [8], parameters 72).

## Experiment 3: follow protectors
This simulation contains only agents and protectors. Agents do not lose energy over time, and are thus rewarded for achieving longer lifespans. Agents receive a reading of number of protectors above, below, to the left, and to the right of the agent (state size 4, action size 4, hidden size [8], parameters 72).

## Experiment 4: evade predators and follow protectors
This simulation contains both predators and protectors. Agents lose energy over time and are thus rewarded for regaining energy by staying close to protectors. Agents receive readings about nearby protectors and predators and a flag that indicates whether they are within the healing field of a protector (state size 9, action size 4, hidden size 16, parameters 224).

# Findings
When presented with a small state space and limited action space, agents quickly learned good policies for certain tasks, such as evading predators, staying close to protectors. However, certain tasks such a picking up food, and multi-problem tasks were not so easily learned. This is perhaps due to the nature of this environment - rewards are typically only received at the end of an episode, and as better policies are found, agents live longer and thus episodes take longer to run. In the case of the food picking problem, the observation space is sparse (agents only see non-zero inputs when close enough to a food item). For scenarios where agents require multiple inputs (e.g. sensors for predators, sensors for protectors, and sensors for food), the neural networks become quite large (in terms of number of parameters) and thus evolution is unlikely to find good policies quickly. A number of experiments were run in an attempt to improve training in these scenarios. Details follow.

## 1. Split inputs for different tasks into "blocks"

As mentioned, multi-problem tasks were difficult to learn due the the number of inputs. I experimented with breaking down tasks into small, easily learnable blocks. Instead of attaching all inputs to one hidden layer, inputs are split into blocks (e.g. sensor readings for predators are grouped into one block, sensor readings for food are grouped into another, and so on). Each block is attached to a small hidden layer and then an output layer of the size of the number of actions the agent can perform. Output layers from each task block are then concatenated and attached to a final output block.

For example, an agent that receives four inputs for detecting predators, four for protectors, and four for food, would have the following task block architecture:

Model input: 12 values -> split into three slices each containing four values

For each input slice, feed into a separate hidden layer of 8, and then into an output layer of 4, i.e. 3 blocks each containing 64 parameters

Concatenate the three sets of four outputs into an input layer for the next step (size 12).

Connect the concatenated 12 values to a hidden layer of 8 and then a softmax layer of 4 values.

## 2. Evolve by block instead of across the entire genome

Once tasks have been split into blocks, the weights that comprise an agent's genome can also be grouped - i.e., the weights for each task block, and weights for the action head, and the weights for the value head. Thus, in the above example, each agent genome would be represented by 5 blocks. When a reproduction step occurs, instead of performing crossover on the entire concatenated weights, it is applied on a block-by-block basis. Since each sub-block is small, evolutionary algorithms may more likely find good solutions.

## 3. Use "integer" weights instead of uniform random weights

In my initial experiments, genomes were created using uniform random values. I experimented with initializing genomes using "integer" values (i.e. -1, 0, 1) to determine whether such values may allow evolutionary mechanisms to find solutions faster. Since the same genomes are used by the reinforcement learning process, zero values were replaced with small, non-zero values random.uniform(-0.1, 0.1).

The "integer" genomes created using this process can be represented as strings by converting the int value of each weight into an alphabetic representation. Genomic diversity can be studied in this manner. During very early phases of training it was observed that the genomic sequences representing task blocks quickly converged on a small number (6-9) of unique sequences that remained unchanged for a long period (using genome store sizes of 20, 50, and 100). Eventually the diversity of these sequences would reduce. The genomic diversity of the output block increased quickly and remained high during training. Further studies of weight changes of agents using reinforcement learning showed that the weights in tasks blocks remained unchanged during backpropagation. I'm unsure as to why this happened.

## 4. Improve final reward calculation to encourage longer episodes and action diversity

Initial experiments rewarded agents based on their age as follows:

```
reward = (age - start_energy)/start_energy
```

A number of different reward schemes were examined. For instance, one reward scheme used the mean age of items in the genome store instead of starting energy in the above equation. However, it failed to improve training. The current reward scheme attempts to factor in action entropy (rewarding agents that perform a distribution of actions) and presents a much larger reward for agents achieving higher longevity:

```
reward = ((age-start_energy) ** 2)/(start_energy ** 2) + (age - start_energy)/start_energy + action_entropy - 1.5
```

## 5. Add handling of previous states

Some scenarios contain moving objects (predators, protectors, shooters, bullets). It was hypothesized that agents may train better with access to previous state information, allowing them to predict trajectories. Previous states were implemented as task blocks.

```
 each prev_state is supplied such that blocks are merged with another block
 state1 state2 state3   state4 state5 state6   state7  state8  state9
 block1 block2 block3   block7 block8 block9   block13 block14 block15
    block4  block5         block10 block11         block17  block18
        block6                 block12                  block19
        out_cat                out_cat                  out_cat
```

## 6. Anneal integer weight usage

While integer weights might help evolutionary strategies find intermediate solutions quickly, they're probably not all that great for learning agents. As such, a balance between the use of integer weights and uniform random floats might be the solution to aid both strategies. Since evolutionary strategies find a unique set of task blocks very quickly, I anneal the use of integer weights to zero at the beginning of training. It is unclear whether this helps speed the training process.

## 7. Varying of activation functions, optimizers, learning rate, etc.

During experimentation, I tried many different activation functions, neural network architectures, optimizers, learning rates, mutation rates, etc. Values outside of the "default" didn't have a marked effect on the speed at which training occurred.

## 8. Selective block reproduction

Since genome store contains genomes created from both evolving and learning agents, and task blocks are mostly fixed early in training, it might be better to not perform crossover on all blocks during evolution. An implementation that sets a lower than 100% chance that a block will perform crossover, and instead simply inherit the genome from one of its parents, exists in the code. It is unclear as to whether this helps speed up the training process.

# Concerns

Statistics from the last 100 active agents were graphed during all experiments. Although values in genome store always rose during training, the last 100 active agents never improved. For evolving agents, this is understandable - many crossovers and mutations will create neural networks that perform poorly. However, active learning agents should be expected to improve over time, and this did not happen.

Even after long periods of training (many hours or even overnight), when the simulation is run in inference mode, most of the policies captured in the genome store simply didn't perform well.


# Technical details
The whole simulation is implemented in **alien_ecology.py**. If you want to try running this yourself, you will likely need to install some python packages, including numpy, ursina, and torch (pyTorch). To run the simulation, just type:

python alien_ecology.py

at the command line. 

To see visual output, append **-v** to the above command line. Visual output will allow you to watch agents learn.

To observe inference, append **-i** to the above command line. Inference will run the simulation without learners, without training, and only sample new agents from the saved genome store. Use this option to visualize trained agents.

You can view stats from the experiment by running the accompanying plot_stats.ipynb notebook.

All other options will require editing the file itself, since I didn't bother parameterizing them. Look for class game_space and edit the inputs to the init function. Here are a few tips:

- **learners** defines the split between learning agents and evolving agents. At 1.0, the simulation is all learners. At 0.0, the simulation is all evolvers.
- **num_prev_states** how many previous states are presented as input for agents.
- **evolve_by_block** if set to zero, crossover will happen across the entire genome. If set to a value between 0 and 1, will determine the chance that crossover will happen blockwise. For instance, at a value of 0.2, there will be a one in five chance that crossover will happen on a block during reproduction.
- **area_size** defines the width and height of the simulated area. It is always a square shape.
- **area_toroid** if True, agents will appear on the opposite side if they move past the area's boundaries. If false, they will no longer move if hitting the boundary. Note that bounded simulations will add observations (boundary up, down, left, right)
- **num_agents** defines the number of agents to run in the simulation. This number will be split between learners and evolvers.
- **agent_start_energy** defines agent starting energy.
- **agent_energy_drain** defines how many energy agents lose per step.
- **agent_obs_directions** (set to 4 or 8) defines whether agents sense in 4 or 8 directions,
- **agent_view_distance** how far agents can see,
- **num_predators** defines the number of predators in the simulation. If you want to have fun, increase the speed and inertial_damping values in class Predator.
- **num_protectors** defines the number of protectors in the simulation.
- **protector_safe_distance** size of protector aura
- **num_shooterss** defines the number of shooters in the simulation.
- **shoot_cooldown** how many steps before a shooter can fire a bullet, after firing.
- **bullet_life** how many steps a bullet exists for.
- **bullet_radius** size of bullets.
- **num_food** defines the number of food present in the simulation. Setting this to zero removes food completely, so remember to adjust agent observations accordingly.
- **use_zones** defines whether zones will be used during the simulation. Zones can be defined in the **self.zone_config** variable in game_space. See the code for an example of how to do this.
- **fitness_index** if set to 1 will evaluate new genomes based on fitness, and if set to 2 will base on age
- **genome_store_size** maximum items the genome store can hold.
- **integer_weights** and **weight_range** are parameters to set the randomly generated weights for new genomes. A default setting of **weight_range=1** with **integer_weights=False** will generate float weights between -1.0 and 1.0. Setting **integer_weights=True** will generate genomes containing values of -1.0, 0.0. and 1.0. Setting **weight_range** to a higher value of n (e.g. 2) will cause weights between -n and n to be generated.

Note that you can also change the simulation by commenting out items in self.actions and/or self.observations.

A second learner-only experiment (alien_ecology_NaRLA.py) is included whose agents implement logic described in: Giving Up Control: Neurons as Reinforcement Learning Agents (https://arxiv.org/abs/2003.11642). The original code for implementing NaRLA can be found here: https://github.com/Multi-Agent-Networks/NaRLA
