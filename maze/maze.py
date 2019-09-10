# Differentiable plasticity: maze exploration task.
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


# (Note: this contains an EXTREMELY EXPERIMENTAL implementation of an LSTM with plastic connections)

import argparse
import pickle
import platform
import pprint
import random
import time

import numpy as np
import torch
import torch.nn as nn
from numpy import random


np.set_printoptions(precision=4)


# 1 input for the previous reward, 1 input for numstep, 1 for whether currently on reward square, 1 "Bias" input
ADDINPUT = 4
# U, D, L, R
NBACTIONS = 4
# Receptive field size
RFSIZE = 3
TOTALNBINPUTS = RFSIZE * RFSIZE + ADDINPUT + NBACTIONS


# ttype = torch.FloatTensor    # For CPU
ttype = torch.cuda.FloatTensor  # Gor GPU


class Network(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.rule = params['rule']
        self.type = params['type']
        self.softmax = torch.nn.functional.softmax

        if params['activ'] == 'tanh':
            self.activ = torch.tanh
        elif params['activ'] == 'selu':
            self.activ = torch.selu
        else:
            raise ValueError('Must choose an activ function')

        if params['type'] == 'lstm':
            self.lstm = torch.nn.LSTM(TOTALNBINPUTS, params['hidden_size']).cuda()
        elif params['type'] == 'rnn':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hidden_size']).cuda()
            self.w = torch.nn.Parameter(
                (.01 * torch.rand(params['hidden_size'], params['hidden_size'])).cuda(),
                requires_grad=True,
            )
        elif params['type'] == 'homo':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hidden_size']).cuda()
            self.w = torch.nn.Parameter(
                (.01 * torch.rand(params['hidden_size'], params['hidden_size'])).cuda(),
                requires_grad=True,
            )
            # Homogenous plasticity: everyone has the same alpha
            self.alpha = torch.nn.Parameter(
                (.01 * torch.ones(1)).cuda(),
                requires_grad=True,
            )
            # Everyone has the same eta
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)
        elif params['type'] == 'plastic':
            self.i2h = torch.nn.Linear(TOTALNBINPUTS, params['hidden_size']).cuda()
            self.w = torch.nn.Parameter(
                (.01 * torch.rand(params['hidden_size'], params['hidden_size'])).cuda(),
                requires_grad=True,
            )
            self.alpha = torch.nn.Parameter(
                (.01 * torch.rand(params['hidden_size'], params['hidden_size'])).cuda(),
                requires_grad=True,
            )
            # Everyone has the same eta
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)
        # LSTM with plastic connections. HIGHLY EXPERIMENTAL, NOT DEBUGGED
        elif params['type'] == 'lstmplastic':
            self.h2f = torch.nn.Linear(params['hidden_size'], params['hidden_size']).cuda()
            self.h2i = torch.nn.Linear(params['hidden_size'], params['hidden_size']).cuda()
            self.h2opt = torch.nn.Linear(params['hidden_size'], params['hidden_size']).cuda()

            # Plasticity only in the recurrent connections, h to c.
            # This is replaced by the plastic connection matrices below
            # self.h2c = torch.nn.Linear(params['hidden_size'], params['hidden_size']).cuda()
            self.w = torch.nn.Parameter((.01 * torch.rand(params['hidden_size'], params['hidden_size'])).cuda(),
                                        requires_grad=True)
            self.alpha = torch.nn.Parameter(
                (.01 * torch.rand(params['hidden_size'], params['hidden_size'])).cuda(),
                requires_grad=True,
            )
            # Everyone has the same eta
            self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)

            self.x2f = torch.nn.Linear(TOTALNBINPUTS, params['hidden_size']).cuda()
            self.x2opt = torch.nn.Linear(TOTALNBINPUTS, params['hidden_size']).cuda()
            self.x2i = torch.nn.Linear(TOTALNBINPUTS, params['hidden_size']).cuda()
            self.x2c = torch.nn.Linear(TOTALNBINPUTS, params['hidden_size']).cuda()
        # An LSTM implemented "by hand", to ensure maximum simlarity with the plastic LSTM
        elif params['type'] == 'lstmmanual':
            self.h2f = torch.nn.Linear(params['hidden_size'], params['hidden_size']).cuda()
            self.h2i = torch.nn.Linear(params['hidden_size'], params['hidden_size']).cuda()
            self.h2opt = torch.nn.Linear(params['hidden_size'], params['hidden_size']).cuda()
            self.h2c = torch.nn.Linear(params['hidden_size'], params['hidden_size']).cuda()
            self.x2f = torch.nn.Linear(TOTALNBINPUTS, params['hidden_size']).cuda()
            self.x2opt = torch.nn.Linear(TOTALNBINPUTS, params['hidden_size']).cuda()
            self.x2i = torch.nn.Linear(TOTALNBINPUTS, params['hidden_size']).cuda()
            self.x2c = torch.nn.Linear(TOTALNBINPUTS, params['hidden_size']).cuda()
        else:
            raise ValueError("Which network type?")

        # From hidden to action output
        self.h2o = torch.nn.Linear(params['hidden_size'], NBACTIONS).cuda()
        # From hidden to value prediction (for A2C)
        self.h2v = torch.nn.Linear(params['hidden_size'], 1).cuda()
        self.params = params

        # Note: Vectors are row vectors, and matrices are transposed wrt the usual order, following pytorch conventions
        # Each *column* of w targets a single output neuron

    def forward(self, input, hidden, hebb):
        if self.type == 'lstm':
            # h_activ is just the h. hidden is the h and the cell state, in a tuple
            h_activ, hidden = self.lstm(input.view(1, 1, -1), hidden)
            h_activ = h_activ.view(1, -1)

        elif self.type == 'rnn':
            h_activ = self.activ(self.i2h(input) + hidden.mm(self.w))
            hidden = h_activ

        # Draft for a "manual" lstm:
        elif self.type == 'lstm_manual':
            # hidden[0] is the previous h state. hidden[1] is the previous c state
            fgt = torch.sigmoid(self.x2f(input) + self.h2f(hidden[0]))
            ipt = torch.sigmoid(self.x2i(input) + self.h2i(hidden[0]))
            opt = torch.sigmoid(self.x2opt(input) + self.h2opt(hidden[0]))
            cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, torch.tanh(self.x2c(input) + self.h2c(hidden[0])))
            h_activ = torch.mul(opt, torch.tanh(cell))
            # pdb.set_trace()
            hidden = (h_activ, cell)
            if np.isnan(np.sum(h_activ.data.cpu().numpy())) or np.isnan(np.sum(hidden[1].data.cpu().numpy())):
                raise ValueError("Nan detected !")

        elif self.type == 'lstm_plastic':
            fgt = torch.sigmoid(self.x2f(input) + self.h2f(hidden[0]))
            ipt = torch.sigmoid(self.x2i(input) + self.h2i(hidden[0]))
            opt = torch.sigmoid(self.x2opt(input) + self.h2opt(hidden[0]))
            # cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, torch.tanh(self.x2c(input) + self.h2c(hidden[0])))

            # Need to think what the inputs and outputs should be for the
            # plasticity. It might be worth introducing an additional stage
            # consisting of whatever is multiplied by ift and then added to the
            # cell state, rather than the full cell state.... But we can
            # experiment both!

            # cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, torch.tanh(self.x2c(input) + hidden[0].mm(self.w + torch.mul(self.alpha, hebb))))
            inputtocell = torch.tanh(self.x2c(input) + hidden[0].mm(self.w + torch.mul(self.alpha, hebb)))
            cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, inputtocell)

            if self.rule == 'hebb':
                hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(hidden[0].unsqueeze(2), inputtocell.unsqueeze(1))[0]
            elif self.rule == 'oja':
                # NOTE: NOT SURE ABOUT THE OJA VERSION !!
                # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
                hebb = hebb + self.eta * torch.mul(
                    (hidden[0][0].unsqueeze(1) - torch.mul(hebb, inputtocell[0].unsqueeze(0))),
                    inputtocell[0].unsqueeze(0)
                )
            h_activ = torch.mul(opt, torch.tanh(cell))
            # pdb.set_trace()
            hidden = (h_activ, cell)
            if np.isnan(np.sum(h_activ.data.cpu().numpy())) or np.isnan(np.sum(hidden[1].data.cpu().numpy())):
                raise ValueError("Nan detected !")

        elif self.type == 'plastic':
            h_activ = self.activ(self.i2h(input) + hidden.mm(self.w + torch.mul(self.alpha, hebb)))
            if self.rule == 'hebb':
                hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(hidden.unsqueeze(2), h_activ.unsqueeze(1))[0]
            elif self.rule == 'oja':
                # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
                hebb = hebb + self.eta * torch.mul(
                    (hidden[0].unsqueeze(1) - torch.mul(hebb, h_activ[0].unsqueeze(0))),
                    h_activ[0].unsqueeze(0)
                )
            else:
                raise ValueError("Must specify learning rule ('hebb' or 'oja')")
            hidden = h_activ

        elif self.type == 'homo':
            h_activ = self.activ(self.i2h(input) + hidden.mm(self.w + self.alpha * hebb))
            if self.rule == 'hebb':
                hebb = (1 - self.eta) * hebb + self.eta * torch.bmm(hidden.unsqueeze(2), h_activ.unsqueeze(1))[0]
            elif self.rule == 'oja':
                # Oja's rule. Remember that yin, yout are row vectors (dim (1,N)). Also, broadcasting!
                hebb = hebb + self.eta * torch.mul(
                    (hidden[0].unsqueeze(1) - torch.mul(hebb, h_activ[0].unsqueeze(0))),
                    h_activ[0].unsqueeze(0),
                )
            else:
                raise ValueError("Must specify learning rule ('hebb' or 'oja')")
            hidden = h_activ

        activout = self.softmax(self.h2o(h_activ), dim=-1)  # Action selection
        valueout = self.h2v(h_activ)  # Value prediction (for A2C)

        return activout, valueout, hidden, hebb

    def initialZeroHebb(self):
        return torch.zeros(self.params['hidden_size'], self.params['hidden_size'], requires_grad=False).cuda()

    def initialZeroState(self):
        if self.params['type'] == 'lstm':
            return (
                torch.zeros(1, 1, self.params['hidden_size'], requires_grad=False).cuda(),
                torch.zeros(1, 1, self.params['hidden_size'], requires_grad=False).cuda(),
            )
        elif self.params['type'] == 'lstmmanual' or self.params['type'] == 'lstmplastic':
            return (
                torch.zeros(1, self.params['hidden_size'], requires_grad=False).cuda(),
                torch.zeros(1, self.params['hidden_size'], requires_grad=False).cuda(),
            )
        elif self.params['type'] == 'rnn' or self.params['type'] == 'plastic' or self.params['type'] == 'homo':
            return torch.zeros(1, self.params['hidden_size'], requires_grad=False).cuda()
        else:
            raise ValueError("Which type?")


def train(params):
    print("Starting training...")
    print(f"Passed params:\n{pprint.pformat(params)}")
    print(pprint.pformat(dict(platform.uname()._asdict().items())))
    # Turning the parameters into a nice suffix for filenames
    param_info = "_".join(
        [
            f'{key}_{value}'
            for key, value in sorted(params.items(), key=lambda x: x[0])
            if key not in {'nb_steps', 'save_every', 'test_every'}
        ]
    )
    suffix = f'maze_{param_info}'

    # Initialize random seeds (first two redundant?)
    print("Setting random seeds")
    np.random.seed(params['rng_seed'])
    random.seed(params['rng_seed'])
    torch.manual_seed(params['rng_seed'])

    print("Initializing network")
    net = Network(params)
    print(f"Shape of all optimized parameters:\n{pprint.pformat([x.size() for x in net.parameters()])}")
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print(f"Size (numel) of all optimized elements: {allsizes}")
    print(f"Total size (numel) of all optimized elements: {sum(allsizes)}")

    print("Initializing optimizer")
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0 * params['lr'], eps=1e-4)
    # scheduler = torch.optim.lr_scheduler.step_lr(optimizer, gamma=params['gamma'], step_size=params['step_lr'])

    lab_size = params['lab_size']
    lab = np.ones((lab_size, lab_size))
    CTR = lab_size // 2

    # Simple cross maze
    # lab[CTR, 1:lab_size-1] = 0
    # lab[1:lab_size-1, CTR] = 0

    # Double-T maze
    # lab[CTR, 1:lab_size-1] = 0
    # lab[1:lab_size-1, 1] = 0
    # lab[1:lab_size-1, lab_size - 2] = 0

    # Grid maze
    lab[1:lab_size - 1, 1:lab_size - 1].fill(0)
    for row in range(1, lab_size - 1):
        for col in range(1, lab_size - 1):
            if row % 2 == 0 and col % 2 == 0:
                lab[row, col] = 1
    # Not really necessary, but nicer to not start on a wall
    # May help localization by introducing a detectable irregularity in the center?
    lab[CTR, CTR] = 0

    all_losses_objective = []
    all_losses_eval = []
    all_losses_v = []
    all_rewards = []
    loss_between_saves = 0
    now_time = time.time()

    print("Starting episodes...", flush=True)
    for episode in range(params['nb_iter']):
        PRINTTRACE = 0
        if (episode + 1) % (1 + params['print_every']) == 0:
            PRINTTRACE = 1

        # Note: it doesn't matter if the reward is on the center (reward is only computed after an action is taken).
        # All we need is not to put it on a wall or pillar (lab=1)
        rposr = 0
        rposc = 0
        if params['rp'] == 0:
            while lab[rposr, rposc]:
                rposr = np.random.randint(1, lab_size - 1)
                rposc = np.random.randint(1, lab_size - 1)
        elif params['rp'] == 1:
            # If we want to constrain the reward to fall on the periphery of the maze
            while lab[rposr, rposc] or (rposr != 1 and rposr != lab_size - 2 and rposc != 1 and rposc != lab_size - 2):
                rposr = np.random.randint(1, lab_size - 1)
                rposc = np.random.randint(1, lab_size - 1)
        # print("Reward pos:", rposr, rposc)

        # Agent always starts an episode from the center
        posc = CTR
        posr = CTR

        optimizer.zero_grad()
        loss = 0
        lossv = 0
        hidden = net.initialZeroState()
        hebb = net.initialZeroHebb()

        reward = 0.0
        rewards = []
        vs = []
        logprobs = []
        sum_reward = 0.0
        dist = 0

        for numstep in range(params['ep_len']):
            inputs_np = np.zeros((1, TOTALNBINPUTS), dtype='float32')
            inputs_np[0, 0:RFSIZE * RFSIZE] = lab[
                                            posr - RFSIZE // 2:posr + RFSIZE // 2 + 1,
                                            posc - RFSIZE // 2:posc + RFSIZE // 2 + 1,
                                            ].flatten()

            inputs = torch.from_numpy(inputs_np).cuda().requires_grad_(False)
            # Previous chosen action
            # inputs[0][num_action_chosen] = 1
            inputs[0][-1] = 1  # Bias neuron
            inputs[0][-2] = numstep
            inputs[0][-3] = reward

            # Running the network
            # y  should output probabilities; v is the value prediction
            y, v, hidden, hebb = net(inputs, hidden, hebb)

            distrib = torch.distributions.Categorical(y)
            # sample() returns a Pytorch tensor of size 1; this is needed for the backprop below
            action_chosen = distrib.sample()
            num_action_chosen = action_chosen.data[0]  # Turn to scalar

            # Target position, based on the selected action
            tgtposc = posc
            tgtposr = posr
            if num_action_chosen == 0:  # Up
                tgtposr -= 1
            elif num_action_chosen == 1:  # Down
                tgtposr += 1
            elif num_action_chosen == 2:  # Left
                tgtposc -= 1
            elif num_action_chosen == 3:  # Right
                tgtposc += 1
            else:
                raise ValueError(f"{num_action_chosen} is an invalid action, choose from {{0, 1, 2, 3}}.")

            reward = 0.0
            if lab[tgtposr][tgtposc] == 1:
                reward = -.1
            else:
                dist += 1
                posc = tgtposc
                posr = tgtposr

            # Did we hit the reward location? Increase reward and teleport!
            # Note that it doesn't matter if we teleport onto the reward, since reward is only provided after a move
            if rposr == posr and rposc == posc:
                reward += 10
                if params['rand_start'] == 1:
                    posr = np.random.randint(1, lab_size - 1)
                    posc = np.random.randint(1, lab_size - 1)
                    while lab[posr, posc] == 1:
                        posr = np.random.randint(1, lab_size - 1)
                        posc = np.random.randint(1, lab_size - 1)
                else:
                    posr = CTR
                    posc = CTR

            # Store the obtained reward, value prediction, and log-probabilities, for this time step
            rewards.append(reward)
            sum_reward += reward
            vs.append(v)
            logprobs.append(distrib.log_prob(action_chosen))

            # A2C has an entropy reward on the output probabilities, to encourage exploration.
            # Our version of PyTorch does not have an entropy() function for Distribution,
            # so we use a penalty on the sum of squares instead, which has the same basic property
            # (discourages concentration). It really does help!
            loss += params['bentropy'] * y.pow(2).sum()

            # if PRINTTRACE:
            #    print(
            #        "Probabilities:", y.data.cpu().numpy(),
            #        "Picked action:", num_action_chosen,
            #        ", got reward", reward,
            #    )

        # Do the A2C! (essentially copied from V. Mnih, https://arxiv.org/abs/1602.01783, Algorithm S3)
        R = 0
        gammaR = params['gr']
        for numstepb in reversed(range(params['ep_len'])):
            R = gammaR * R + rewards[numstepb]
            lossv += (vs[numstepb][0] - R).pow(2)
            loss -= logprobs[numstepb].data[0] * (R - vs[numstepb].data[0][0])

        all_rewards.append(sum_reward)

        if PRINTTRACE:
            print(f"\tlossv:                         {lossv.data.cpu().numpy()[0]:0.4f}")
            print(f"\tTotal reward for this episode: {sum_reward:0.1f}")
            print(f"\tTravelled Distance:            {dist}")
            print(f"\tMean 100 Ep. Reward:           {np.mean(all_rewards[-100:]):0.4f}")

        # Do we want to squash rewards for stabilization? 
        if params['squash'] == 1:
            if sum_reward < 0:
                sum_reward = -np.sqrt(-sum_reward)
            else:
                sum_reward = np.sqrt(sum_reward)
        elif params['squash'] == 0:
            pass
        else:
            raise ValueError("Incorrect value for squash parameter")

        # Mixing the reward loss and the value-prediction loss
        loss += params['blossv'] * lossv.data[0]
        loss /= params['ep_len']
        loss.backward()

        # scheduler.step()
        optimizer.step()
        # torch.cuda.empty_cache()

        loss_num = loss.item()
        loss_between_saves += loss_num
        if (episode + 1) % 10 == 0:
            all_losses_objective.append(loss_num)
            all_losses_eval.append(sum_reward)
            all_losses_v.append(float(lossv))

        # Algorithm done. Now print statistics and save files.
        if (episode + 1) % params['print_every'] == 0:
            print(f"Episode {episode + 1} {'=' * 30}")
            print(f"\tMean loss:                     {loss_between_saves / params['print_every']:0.4f}")
            loss_between_saves = 0
            previous_time = now_time
            now_time = time.time()
            print(f"\tTime spent on last {params['print_every']} iters:  {now_time - previous_time:0.4f}")
            if params['type'] in {'plastic', 'lstm_plastic'}:
                print(f"\tEta:                           {net.eta.data.cpu().numpy()[0]:0.4f}")
                print(f"\talpha[0,1]:                    {net.alpha.data.cpu().numpy()[0, 1]:0.4f}")
                print(f"\tw[0,1]:                        {net.w.data.cpu().numpy()[0, 1]:0.4f}")
            elif params['type'] == 'rnn':
                print(f"\tw[0,1]:                        {net.w.data.cpu().numpy()[0, 1]:0.4f}")

        if (episode + 1) % params['save_every'] == 0:
            print(f"\tLoss (100 ep rolling mean):    {np.mean(all_losses_objective[-100:]):0.4f}")
            print("\tSaving local files...")
            with open(f'params_{suffix}.dat', 'wb') as f:
                pickle.dump(params, f)
            with open(f'lossv_{suffix}.txt', 'w') as f:
                for item in all_losses_v:
                    print(item, file=f)
            with open(f'loss_{suffix}.txt', 'w') as f:
                for item in all_losses_eval:
                    print(item, file=f)

            torch.save(net.state_dict(), f'torchmodel_{suffix}.dat')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        help="Random seed value.",
        default=0,
    )
    parser.add_argument(
        "--bentropy",
        type=float,
        help="Coefficient for the A2C 'entropy' reward (really Simpson index concentration measure).",
        default=0.1,
    )
    parser.add_argument(
        "--blossv",
        type=float,
        help="Coefficient for the A2C value prediction loss.",
        default=.03,
    )
    parser.add_argument(
        "--lab_size",
        type=int,
        help="Size of the labyrinth, must be odd.",
        default=9,
    )
    parser.add_argument(
        "--rand_start",
        type=int,
        help="When hitting reward, should we teleport to random location (1) or center (0)?",
        default=1,
        choices=[0, 1],
    )
    parser.add_argument(
        "--rp",
        type=int,
        help="Whether the reward should be on the periphery (1) or any maze tile (0).",
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--squash",
        type=int,
        help="Squash reward through signed sqrt.",
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--activ",
        help="Activation function.",
        default='tanh',
        choices=['tanh', 'selu']
    )
    parser.add_argument(
        "--rule",
        help="Plasticity update rule.",
        default='oja',
        choices=['hebb', 'oja']
    )
    parser.add_argument(
        "--type",
        help="Network type.",
        default='rnn',
        choices=['rnn', 'homo', 'plastic', 'lstm', 'lstm_manual', 'lstm_plastic']
    )
    parser.add_argument(
        "--gr",
        type=float,
        help="gammaR: Discounting factor for rewards.",
        default=.9,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate, Adam.",
        default=1e-4,
    )
    parser.add_argument(
        "--ep_len",
        type=int,
        help="Length of episodes.",
        default=250,
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        help="Number of neurons in the recurrent layer.",
        default=200,
    )
    parser.add_argument(
        "--step_lr",
        type=int,
        help="Duration of each step in the learning rate annealing schedule.",
        default=100000000,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="Learning rate annealing factor.",
        default=0.3,
    )
    parser.add_argument(
        "--nb_iter",
        type=int,
        help="Number of learning cycles.",
        default=1000000,
    )
    parser.add_argument(
        "--save_every",
        type=int,
        help="Number of cycles between successive save points.",
        default=200,
    )
    parser.add_argument(
        "--print_every",
        type=int,
        help="Number of cycles between successive printing of information.",
        default=100,
    )

    args = parser.parse_args()
    train(vars(args))
