## Introduction
This repository contains code that implements an environment for simulated organisms to learn and evolve behaviours. The environment itself is a two-dimensional toroidal plane - organisms that travel over an edge appear at the opposite edge of the plane in a similar fashion to how pacman moves. The environment presents a number of problems the organisms must learn to solve.

1. Organisms have energy that depletes during each simulation step. Organisms can replenish energy in a number of ways, such as by consuming food or staying close to protectors.
2. Predators exist in the environment that can eat organisms. Predators move using very simple hard-coded logic (if they can see organisms ahead, move forward, otherwise move in a random fashion). Organisms must thus learn to evade predators.
3. Protectors are friendly entities that move randomly. When predators are too close to protectors they become stunned and are unable to perform actions. Note that they will still consume agents that bump into them.
4. Environmental effects are simluated. Night and day cycles affect the distance organisms can "see". Cold and hot cycles affect the drain on organisms' energy. During cold periods, organisms can increase their temperature by being close to other organisms. During hot periods, organisms must spread out.

Organisms can perform rotation and propulsion movement actions. Organisms can "see" further in the direction they're facing, but can also "sense" things above, below, to the left, and to the right of their position. Organisms can propel in the direction they're facing, as well as in up, down, left and right directions.

Organism logic is represented by simple feed-forward neural networks. Inputs are observations from the environment that include signals denoting that other organisms are close by or in view. Outputs of the neural network tie into organism actions. Agents can be graded on the number of steps they lived for, or a fitness value that includes bonuses for movement and performing actions.

Organisms are trained either by evolution, reinforcement learning, or a hybrid of the two. The weights of the neural networks are used as each organism's genome. Genomes from organisms that perform the best are stored and used for reproduction, or to replace an ill-performing learning agent's weights after a specified evaluation period. When an evolving organism dies it is respawned at a new random location and given a genome evolved from previously collected genomes. If a learning agent dies, backpropagation is used to train its neural network (using A2C) and it is respawned at a new random location. Learning agents are periodically evaluated on the mean age or fitness during previous n runs. If a learning agent fails its evaluation, it is replaced with a new learning agent that receives neural network weights from a random previously recorded genome.

The purpose of this simulation is:
- to observe and document interesting emergent behaviours in the simulated swarm
- to understand whether evolution can be used to improve reinforcement learning mechanisms
- to discover methods that allow agents to quickly learn policies capable of solving multiple tasks
- to discover methods that allow agents to quickly learn policies capable of solving complex tasks

## Experiment 1: evade predators
This simulation contains only agents and predators. The agent's state includes readings of the number of predators above, below, to the left, and to the right of the agent, within the agent's maximum view distance (state size 4, action size 4, hidden size [8], parameters 72). Agents learn to run away from predators, and this policy can be learned quickly. Evolutionary processes find optimal policies much quicker than reinforcement learning mechanisms.

## Experiment 2: collect food
This simulation contains only agents and food. Agents start with 200 energy and lose 1 energy per step. Colliding with food grants the agent 50 energy and causes the food to respawn in a new random location. Agents receive a reading of number of food above, below, to the left, and to the right of the agent, and their own energy value (state size 4, action size 4, hidden size [8], parameters 72).

## Experiment 2.5: apply collect food policy to follow protector scenario

## Experiment 3: follow protectors
This simulation contains only agents and protectors. Agents do not lose energy over time, and are thus rewarded for achieveing longer lifespans. Agents receive a reading of number of protectors above, below, to the left, and to the right of the agent (state size 4, action size 4, hidden size [8], parameters 72).

## Experiment 4: evade predators and follow protectors
This simulation contains both predators and protectors. Agents lose energy over time and are thus rewarded for regaining energy by staying close to protectors. Agents receive readings about nearby protectors and predators and a flag that indicates whether they are within the healing field of a protector (state size 9).

## Experiment 5: attractor, repulsor, and damping zones added
This experiment combines all of the previous, and contains predators, protectors, and food (state size 13). The simulation environment also contains zones that affect the agents' movement. Attractor zones pull agents to the center, repulsor zones push agents away, damping zones slow agents down, and acceleration zones speed them up. Agents do not receive readings about these zones, and must learn policies despite their effects. Predators and protectors are unaffected by these zones.

## Findings
When presented with a small state space and limited action space, agents quickly learned good policies for certain tasks, such as evading predators, staying close to protectors and colliding with food pellets. These simple tasks can be somewhat similar - for instance, the food search and follow protector tasks both require agents to move in the direction of a perceived signal.

It would be interesting to discover whether the policy learned for collecting food would work out-of-the-box in the scenario that requires agents to locate the follow a protector.

Multi-step tasks were much more difficult to learn due the the number of inputs and the sparse rewards. Perhaps breaking down tasks into small, easily learnable blocks is the way to create fast learning policies? Instead of training an agent on the complex task, might there be a way of training multiple models on sub-tasks and then combining them to solve the more complex task? Perhaps the answer is not to attach all inputs to one hidden layer, but to attach sets of inputs related to a specific task to their own hidden layer and output layer of size number of actions, concatenate all sub-output layers, and then attach them to a final output layer containing the number of actions.


For example, say the agent needs to learn tasks involving four inputs for detecting predators, four for protectors and four for food, the layers would look like this:

Model input: 12 values -> split into 3 x 4 sets

For each set of four, feed into a separate hidden layer of 8, and then into an output layer of 4, i.e. 3 blocks each containing 64 parameters

Concatenate the three sets of four outputs -> 12 output values from the three combined logic blocks

Connect the concatenated 12 values into a softmax layer of 4 values, i.e. 48 parameters (60 including A2C value head)

Total parameters: 240 (252 with A2C)

However, the way we'd approach this from an evolutionary standpoint would be to store the parameters as not one long genome, but four - one for each of the three task sub-blocks, and one for the final layer. When reproducing, we'd combine and mutate each block separately, thus preserving learned features for each task.


For now, I hope to leave the code in a state that allows anyone who downloads the repository to run and enjoy watching organisms evolve. I'll update this repository with new findings and discoveries as I make them.


# Technical details
The whole simulation is implemented in **alien_ecology.py**. If you want to try running this yourself, you will likely need to install some python packages, including numpy, ursina, and torch (pyTorch). To run the simulation, just type:

python alien_ecology.py

at the command line. To see visual output, append **-v** to the above command line. Visual output will allow you to watch agents learn.

You can view stats from the experiment by running the accompanying plot_stats.ipynb notebook.

All other options will require editing the file itself, since I didn't bother parameterizing them. Look for class game_space and edit the inputs to the init function. Here are a few tips:

- **hidden_size** defines the shape of the hidden layers in the neural network. Note that if this list contains multiple values, multiple hidden layers will be created. A default value of something like [8] or [16] is probably good. You can, of course make much larger neural networks by defining hidden to be [64, 128, 64] or something like that, but bear in mind evolutionary strategies will fail if there are too many parameters to find.
- **num_prev_states** allows you to set how many previous observation sets are present in the model's input
- **learners** defines the split between learning agents and evolving agents. At 1.0, the simulation is all learners. At 0.0, the simulation is all evolvers.
- **area_size** defines the width and height of the simulated area. It is always a square shape.
- **num_agents** defines the number of agents to run in the simulation. This number will be split between learners and evolvers.
- **agent_start_energy** defines agent starting energy.
- **agent_energy_drain** defines how many energy agents lose per step.
- **num_predators** defines the number of predators in the simulation. If you want to have fun, increase the speed and inertial_damping values in class Predator.
- **num_protectors** defines the number of protectors in the simulation.
- **num_food** defines the number of food present in the simulation. Setting this to zero removes food completely, so remember to adjust agent observations accordingly.
- **use_zones** defines whether zones will be used during the simulation. Zones can be defined in the **self.zone_config** variable in game_space. See the code for an example of how to do this.
- **fitness_index** if set to 1 will evaluate new genomes based on fitness, and if set to 2 will base on age
- **respawn_genome_store** - when evolving agents are respawned, how likely are their new genomes to come from genome_store instead of previous_agents
- **rebirth_genome_store** - same as above but for learning agents
- **top_n** defines the portion of genomes to select from either previous_agents or genome_store when reproducing or spawning new agents
- **integer_weights** and **weight_range** are parameters to set the randomly generated weights for new genomes. A default setting of **weight_range=1** with **integer_weights=False** will generate float weights between -1.0 and 1.0. Setting **integer_weights=True** will generate genomes containing values of -1.0, 0.0. and 1.0. Setting **weight_range** to a higher value of n (e.g. 2) will cause weights between -n and n to be generated.

Note that you can also change the simulation by commenting out items in self.actions and/or self.observations.

A second learner-only experiment (alien_ecology_NaRLA.py) is included whose agents implement logic described in: Giving Up Control: Neurons as Reinforcement Learning Agents (https://arxiv.org/abs/2003.11642). The original code for implementing NaRLA can be found here: https://github.com/Multi-Agent-Networks/NaRLA
