# Differentiable plasticity: simple binary pattern memorization and reconstruction.
#
# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This program is meant as a simple instructional example for differentiable plasticity.
# It is fully functional but not very flexible.

# Usage: python simple.py [rngseed]
#     rngseed is an optional parameter specifying the seed of the random number generator.
# To use it on a GPU or CPU, toggle comments on the 'ttype' declaration below.


import pickle as pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random


PATTERN_SIZE = 1000
NBNEUR = PATTERN_SIZE + 1  # Bias neuron not needed for this task, but included for completeness
ADAM_LR = 3e-4  # The learning rate of the Adam optimizer
RNG_SEED = 0  # Initial random seed - can be modified by passing a number as command-line argument

# Note that these patterns are likely not optimal
PROBADEGRADE = .5  # Proportion of bits to zero out in the target pattern at test time
NBPATTERNS = 5  # The number of patterns to learn in each episode
NBPRESCYCLES = 2  # Number of times each pattern is to be presented
PRESTIME = 6  # Number of time steps for each presentation
PRESTIMETEST = 6  # Same thing but for the final test pattern
INTERPRESDELAY = 4  # Duration of zero-input interval between presentations
NBSTEPS = NBPRESCYCLES * ((PRESTIME + INTERPRESDELAY) * NBPATTERNS) + PRESTIMETEST  # Total number of steps per episode

if len(sys.argv) == 2:
    RNG_SEED = int(sys.argv[1])
    print("Setting RNG_SEED to " + str(RNG_SEED))
np.set_printoptions(precision=3)
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

# ttype = torch.FloatTensor;         # For CPU
ttype = torch.cuda.FloatTensor  # For GPU


# Generate the full list of inputs for an episode.
# The inputs are returned as a PyTorch tensor of shape NbSteps x 1 x NbNeur
def generate_batch():
    inputT = np.zeros((NBSTEPS, 1, NBNEUR))  # inputTensor, initially in numpy format...

    # Create the random patterns to be memorized in an episode
    seedp = np.ones(PATTERN_SIZE)
    seedp[:PATTERN_SIZE // 2] = -1
    patterns = [np.random.permutation(seedp) for _ in range(NBPATTERNS)]

    # Now 'patterns' contains the NBPATTERNS patterns to be memorized in this episode - in numpy format
    # Choosing the test pattern, partially zero'ed out, that the network will have to complete
    testpattern = patterns[random.choice(len(patterns))].copy()
    preservedbits = np.ones(PATTERN_SIZE)
    preservedbits[:int(PROBADEGRADE * PATTERN_SIZE)] = 0
    np.random.shuffle(preservedbits)
    degradedtestpattern = testpattern * preservedbits

    # Inserting the inputs in the input tensor at the proper places
    for nc in range(NBPRESCYCLES):
        np.random.shuffle(patterns)
        for ii in range(NBPATTERNS):
            for nn in range(PRESTIME):
                numi = nc * (NBPATTERNS * (PRESTIME + INTERPRESDELAY)) + ii * (PRESTIME + INTERPRESDELAY) + nn
                inputT[numi][0][:PATTERN_SIZE] = patterns[ii][:]

    # Inserting the degraded pattern
    for nn in range(PRESTIMETEST):
        inputT[-PRESTIMETEST + nn][0][:PATTERN_SIZE] = degradedtestpattern[:]

    for nn in range(NBSTEPS):
        inputT[nn][0][-1] = 1.0  # Bias neuron.
        inputT[nn] *= 20.0  # Strengthen inputs

    # Convert from numpy to Tensor
    inputT = torch.from_numpy(inputT).type(ttype).requires_grad_(False)
    target = torch.from_numpy(testpattern).type(ttype).requires_grad_(False)

    return inputT, target


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Note: Vectors are row vectors, and matrices are transposed wrt the usual order, following pytorch conventions
        # Each *column* of w targets a single output neuron
        # The matrix of fixed (baseline) weights
        self.w = torch.from_numpy(0.01 * np.ones((NBNEUR, NBNEUR))).type(ttype).requires_grad_(True)
        # The matrix of plasticity coefficients
        self.alpha = torch.from_numpy(0.01 * np.ones((NBNEUR, NBNEUR))).type(ttype).requires_grad_(True)
        # The weight decay term / "learning rate" of plasticity - trainable, but shared across all connections
        self.eta = torch.tensor([0.01]).type(ttype).requires_grad_(True)

    def forward(self, input, yin, hebb):
        yout = F.tanh(yin.mm(self.w + torch.mul(self.alpha, hebb)) + input)
        # bmm is used to implement an outer product between yin and yout, with unsqueeze adding empty dimensions
        hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(yin.unsqueeze(2), yout.unsqueeze(1))[0]
        return yout, hebb

    @staticmethod
    def initial_state():
        return torch.zeros(1, NBNEUR).type(ttype).requires_grad_(False)

    @staticmethod
    def initial_hebb():
        return torch.zeros(NBNEUR, NBNEUR).type(ttype).requires_grad_(False)


net = Network()
optimizer = torch.optim.Adam([net.w, net.alpha, net.eta], lr=ADAM_LR)
total_loss = 0.0
all_losses = []
print_every = 10
now_time = time.time()

for num_iter in range(2000):
    # Initialize network for each episode
    y = net.initial_state()
    hebb = net.initial_hebb()
    optimizer.zero_grad()

    # Generate the inputs and target pattern for this episode
    inputs, target = generate_batch()

    # Run the episode!
    for numstep in range(NBSTEPS):
        y, hebb = net(inputs[numstep], y, hebb)

    # Compute loss for this episode (last step only)
    loss = (y[0][:PATTERN_SIZE] - target).pow(2).sum()

    # Apply backpropagation to adapt basic weights and plasticity coefficients
    loss.backward()
    optimizer.step()

    # That's it for the actual algorithm!
    # Print statistics, save files
    to = target.cpu().numpy()
    yo = y.data.cpu().numpy()[0][:PATTERN_SIZE]
    z = (np.sign(yo) != np.sign(to))
    loss_num = np.mean(z)  # Saved loss is the error rate

    total_loss += loss_num
    if (num_iter + 1) % print_every == 0:
        print(f"{num_iter + 1} ====")
        print("Target Pattern:        ", target.cpu().numpy()[-10:])
        print("Last Input Pattern:    ", np.clip(inputs.cpu().numpy()[numstep][0][-10:], -1, 1))
        print("Reconstructed Pattern: ", y.data.cpu().numpy()[0][-10:])
        previous_time = now_time
        now_time = time.time()
        print(f"Time spent on last {print_every} iters: {now_time - previous_time:0.2f}")
        total_loss /= print_every
        print(f"Mean loss over last {print_every} iters: {total_loss:0.6f}\n")

        all_losses.append(total_loss)
        with open('output_simple_' + str(RNG_SEED) + '.dat', 'wb') as fo:
            pickle.dump(net.w.data.cpu().numpy(), fo)
            pickle.dump(net.alpha.data.cpu().numpy(), fo)
            pickle.dump(y.data.cpu().numpy(), fo)  # The final y for this episode
            pickle.dump(all_losses, fo)
        with open('loss_simple_' + str(RNG_SEED) + '.txt', 'w') as fo:
            for item in all_losses:
                fo.write("%s\n" % item)
        total_loss = 0
