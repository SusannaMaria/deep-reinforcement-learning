import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

env = gym.make('Acrobot-v1')
env.seed(505);

from functions.basefunctions import create_tiling_grid, create_tilings,visualize_tilings, tile_encode, visualize_encoded_samples, TiledQTable, QLearningAgent, run
from functionssolutions.basefunctions import create_tiling_grid as create_tiling_grid_solution
from functionssolutions.basefunctions import create_tilings as create_tilings_solution
from functionssolutions.basefunctions import visualize_tilings as visualize_tilings_solution
from functionssolutions.basefunctions import tile_encode as tile_encode_solution
from functionssolutions.basefunctions import visualize_encoded_samples as visualize_encoded_samples_solution
from functionssolutions.basefunctions import TiledQTable as TiledQTableSolution
from functionssolutions.basefunctions import QLearningAgent as QLearningAgentSolution
from functionssolutions.basefunctions import run as runSolution

low = [-1.0, -5.0]
high = [1.0, 5.0]
create_tiling_grid(low, high, bins=(10, 10), offsets=(-0.1, 0.5)) 
create_tiling_grid_solution(low, high, bins=(10, 10), offsets=(-0.1, 0.5)) 
print("#############################################")
tiling_specs = [((10, 10), (-0.066, -0.33)),
                ((10, 10), (0.0, 0.0)),
                ((10, 10), (0.066, 0.33))]
tilings = create_tilings(low, high, tiling_specs)
tilings_solutions = create_tilings_solution(low, high, tiling_specs)
# fig = visualize_tilings(tilings)
# fig.savefig('ours.png', dpi=100)
# fig = visualize_tilings_solution(tilings)
# fig.savefig('solution.png', dpi=100)
# img=mpimg.imread('ours.png')
# img2=mpimg.imread('solution.png')
# plt.show()

samples = [(-1.2 , -5.1 ),
           (-0.75,  3.25),
           (-0.5 ,  0.0 ),
           ( 0.25, -1.9 ),
           ( 0.15, -1.75),
           ( 0.75,  2.5 ),
           ( 0.7 , -3.7 ),
           ( 1.0 ,  5.0 )]
encoded_samples = [tile_encode(sample, tilings) for sample in samples]
print("\nSamples:", repr(samples), sep="\n")
print("\nEncoded samples:", repr(encoded_samples), sep="\n")
encoded_samples_solution = [tile_encode_solution(sample, tilings) for sample in samples]
print("\nSamples:", repr(samples), sep="\n")
print("\nEncoded samples:", repr(encoded_samples), sep="\n")

visualize_encoded_samples(samples, encoded_samples, tilings)
# plt.show()
visualize_encoded_samples(samples, encoded_samples_solution, tilings_solutions)
# plt.show()

tq = TiledQTable(low, high, tiling_specs, 2)
s1 = 3; s2 = 4; a = 0; q = 1.0
print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  # check value at sample = s1, action = a
print("[UPDATE] Q({}, {}) = {}".format(samples[s2], a, q)); tq.update(samples[s2], a, q)  # update value for sample with some common tile(s)
print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  # check value again, should be slightly updated

tq = TiledQTableSolution(low, high, tiling_specs, 2)
s1 = 3; s2 = 4; a = 0; q = 1.0
print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  # check value at sample = s1, action = a
print("[UPDATE] Q({}, {}) = {}".format(samples[s2], a, q)); tq.update(samples[s2], a, q)  # update value for sample with some common tile(s)
print("[GET]    Q({}, {}) = {}".format(samples[s1], a, tq.get(samples[s1], a)))  # check value again, should be slightly updated

n_bins = 5
bins = tuple([n_bins]*env.observation_space.shape[0])
offset_pos = (env.observation_space.high - env.observation_space.low)/(3*n_bins)

tiling_specs = [(bins, -offset_pos),
                (bins, tuple([0.0]*env.observation_space.shape[0])),
                (bins, offset_pos)]

tq = TiledQTable(env.observation_space.low, 
                 env.observation_space.high, 
                 tiling_specs, 
                 env.action_space.n)
agent = QLearningAgent(env, tq)

scores = run(agent, env)

tqsol = TiledQTableSolution(env.observation_space.low, 
                 env.observation_space.high, 
                 tiling_specs, 
                 env.action_space.n)
agentsolution = QLearningAgentSolution(env, tqsol)

#scores = run(agentsolution, env)
