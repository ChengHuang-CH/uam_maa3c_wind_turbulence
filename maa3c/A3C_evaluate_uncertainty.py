"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
The most simple implementation for continuous action.

"""
import itertools

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
import math
import os, sys
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence
import time
import matplotlib.pyplot as plt
import random
import itertools

sys.path.append('..')  # add relative path of env
from env import agent_env_uncertainty as e
from matplotlib.collections import PolyCollection

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"

os.environ["OMP_NUM_THREADS"] = "1"


def get_norm(f_list, lenth, agent_index, sms):
    sample1 = norm.pdf(np.arange(lenth), loc=sms[agent_index][0], scale=sms[agent_index][1])
    sample2 = norm.pdf(np.arange(lenth), loc=sms[agent_index][2], scale=sms[agent_index][3])
    alpha = 0.5
    beta = 1 - alpha
    sample = alpha * sample1 + beta * sample2
    # normiliaze to [0.5, 1]
    sample_norm = (1 - 0.05) * (sample - np.min(sample)) / (np.max(sample) - np.min(sample)) + 0.05  # speed
    return sample_norm


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


def plot_curve(f0, f1, agent_list, sms0, sms1, delay):
    """
    https://matplotlib.org/stable/gallery/mplot3d/polys3d.html
    """
    verts = []
    verts1 = []
    # ax = plt.figure().add_subplot(projection='3d')
    color = plt.get_cmap('viridis_r')(np.linspace(0, 1, len(agent_list)))
    color1 = plt.get_cmap('twilight')(np.linspace(0, 1, len(agent_list)))
    plt.figure(figsize=(17, 4))

    nx = 1
    for i, agent_index in enumerate(agent_list):
        lenth = np.where(f0 == agent_index)[0].shape[0]
        norm0 = get_norm(f0, lenth, agent_index, sms0)
        norm1 = get_norm(f1, lenth, agent_index, sms1)

        x = np.linspace(0., lenth, lenth)
        # plt.plot(x, np.ones(lenth) * i, norm0, c='blue')
        # plt.plot(x, np.ones(lenth) * i, norm1, c='orange')
        plt.subplot(1, 4, nx)
        plt.plot(norm0, color=color[1], label=f'initial')
        plt.plot(norm1, color=color1[2], label=f'adjustment by MAA3C')

        plt.xlabel('Block index')
        plt.ylabel('Normalized speed (m/s)')

        # ax2 = plt.twinx()
        # ax2.bar(0, delay[agent_index])
        plt.tight_layout()

        nx += 1

        verts += [polygon_under_graph(x, norm0)]
        verts1 += [polygon_under_graph(x, norm1)]

    plt.legend(prop={'size': 10})
    # plt.savefig(f'speed_curve_adjustment_{test}.pdf', format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)

    facecolors = plt.get_cmap('viridis_r')(np.linspace(0, 1, len(verts)))
    facecolors1 = plt.get_cmap('viridis_r')(np.linspace(0, 1, len(verts1)))

    poly = PolyCollection(verts, facecolors=facecolors, alpha=.1)
    poly1 = PolyCollection(verts1, facecolors=facecolors1, alpha=.1)

    # ax.add_collection3d(poly, zs=range(len(agent_list)), zdir='y')
    # ax.add_collection3d(poly1, zs=range(len(agent_list)), zdir='y')
    # ax.set(xlim=(0, 150), ylim=(0, len(agent_list)), zlim=(0, 1), xlabel='x', ylabel=r'$\lambda$', zlabel='probability')

    plt.show()


def plot_curve_with_random(f0, f1, agent_list, sms0, sms1, delay, sms1_random, f1_random):
    """
    https://matplotlib.org/stable/gallery/mplot3d/polys3d.html
    """
    verts = []
    verts1 = []
    # ax = plt.figure().add_subplot(projection='3d')
    color = plt.get_cmap('viridis_r')(np.linspace(0, 1, len(agent_list)))
    color1 = plt.get_cmap('twilight')(np.linspace(0, 1, len(agent_list)))
    color2 = plt.get_cmap('viridis_r')(np.linspace(0, 1, len(agent_list)))
    plt.figure(figsize=(17, 4))

    nx = 1
    for i, agent_index in enumerate(agent_list):
        lenth = np.where(f0 == agent_index)[0].shape[0]
        norm0 = get_norm(f0, lenth, agent_index, sms0)
        norm1 = get_norm(f1, lenth, agent_index, sms1)
        norm2 = get_norm(f1_random, lenth, agent_index, sms1_random)

        x = np.linspace(0., lenth, lenth)
        # plt.plot(x, np.ones(lenth) * i, norm0, c='blue')
        # plt.plot(x, np.ones(lenth) * i, norm1, c='orange')
        plt.subplot(1, 4, nx)
        plt.plot(norm0, color=color[1], label=f'initial')
        plt.plot(norm1, color=color1[2], label=f'adjustment by MAA3C')
        plt.plot(norm2, color=color2[2], label=f'adjustment by RANDOM')

        plt.xlabel('Block index')
        plt.ylabel('Normalized speed (m/s)')

        # ax2 = plt.twinx()
        # ax2.bar(0, delay[agent_index])
        plt.tight_layout()

        nx += 1

        verts += [polygon_under_graph(x, norm0)]
        verts1 += [polygon_under_graph(x, norm1)]

        plt.legend(prop={'size': 10})
        plt.legend([f'INITIAL: {np.round(sms0[agent_index], 1)}',
                    f'MAA3C:   {np.round(sms1[agent_index], 1)}',
                    f'RANDOM:  {np.round(sms1_random[agent_index], 1)}'])
    plt.savefig(f'speed_curve_adjustment_{test}.pdf', format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)

    plt.show()


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()

        num_layers = 2
        hidden_size = a_dim
        # input: 300x150x5 output: 300x150x1 # each batch is a FLIGHT state
        self.rnn1 = torch.nn.RNN(input_size=s_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # input: 300x1x150 output: 300x1x4 [dmu1, dsigma1, dmu2, dsigma2]
        self.rnn2 = torch.nn.RNN(input_size=RNN_L2_input_size, hidden_size=4, num_layers=num_layers,
                                 batch_first=True)
        # input: 300x1x150 output: 300x1x1 [dt]
        self.rnn3 = torch.nn.RNN(input_size=RNN_L2_input_size, hidden_size=hidden_size, num_layers=num_layers,
                                 batch_first=True)
        # input: 300x1x150 output: 300x1x1 [mask]
        self.rnn4 = torch.nn.RNN(input_size=RNN_L2_input_size, hidden_size=hidden_size, num_layers=num_layers,
                                 batch_first=True)
        # input: 300x1x150 output: 300x1x1 [value]
        self.rnn5 = torch.nn.RNN(input_size=RNN_L2_input_size, hidden_size=hidden_size, num_layers=num_layers,
                                 batch_first=True)
        # input: 300x1x150 output: 300x1x1 [cancellation]
        self.rnn6 = torch.nn.RNN(input_size=RNN_L2_input_size, hidden_size=hidden_size, num_layers=num_layers,
                                 batch_first=True)
        self.distribution = torch.normal

    def forward(self, x):
        x, h_state = self.rnn1(x, None)
        seq_unpacked, _ = pad_packed_sequence(x, batch_first=True)  # unpack

        x2, _ = self.rnn2(seq_unpacked.permute(0, 2, 1), None)  # seq_unpacked.permute(0, 2, 1): 300x6x1 ---> 300x1x5
        x2 = x2.squeeze(1)  # 300x1x4--> 300x4
        out = torch.tanh(x2)

        xt, _ = self.rnn3(seq_unpacked.permute(0, 2, 1), None)  # seq_unpacked.permute(0, 2, 1): 300x6x1 ---> 300x1x1
        xt = xt.squeeze(-1)  # 300x1x1 --> 300x1
        dt = torch.relu(xt)  # dt>=0

        x4, _ = self.rnn4(seq_unpacked.permute(0, 2, 1), h_state)  # seq_unpacked.permute(0, 2, 1): 300x6x1 ---> 300x1x6
        x4 = x4.squeeze(-1)  # 300x1x1 --> 300x1
        # mask: [0,1] or [1,0] only ground delay or speed change in each step
        sign = torch.sign(x4)
        mask = torch.relu(sign)  # 0 for ground delay; 1 for speed change

        x5, _ = self.rnn5(seq_unpacked.permute(0, 2, 1), h_state)  # seq_unpacked.permute(0, 2, 1): 300x6x1 ---> 300x1x6
        x5 = x5.squeeze(-1)  # 300x1x1 --> 300x1
        values = torch.tanh(x5)

        x6, _ = self.rnn6(seq_unpacked.permute(0, 2, 1), h_state)  # seq_unpacked.permute(0, 2, 1): 300x6x1 ---> 300x1x6
        x6 = x6.squeeze(-1)  # 300x1x1 --> 300x1
        cancellation = torch.sigmoid(x6)
        cancellation = torch.where(cancellation >= torch.mean(cancellation), 1, 0)

        mu1, sigma1, mu2, sigma2, dt = out[:, 0].unsqueeze(-1) * mask, out[:, 1].unsqueeze(-1) * mask, \
                                       out[:, 2].unsqueeze(-1) * mask, out[:, 3].unsqueeze(-1) * mask, \
                                       dt * (torch.ones_like(mask) - mask)

        return mu1, sigma1, mu2, sigma2, dt, values, cancellation

    def choose_action(self, s):
        s = self.pad_pack_rnn(s)
        mu1, sigma1, mu2, sigma2, dt, _, cancellation = self.forward(s)
        a = torch.cat((mu1, sigma1, mu2, sigma2, dt, cancellation), 1)  # 300 x 5

        return a

    def loss_func(self, s, a, v_t):
        s = self.pad_pack_rnn(s)
        self.train()
        _, _, _, _, _, values, _ = self.forward(s)
        # print(v_t.shape, values.shape)
        min_size = min(v_t.shape[0], values.shape[0])
        td = v_t.mean() - values.mean()  # advantage
        c_loss = td.pow(2)

        log_prob = torch.log(a + 1e-9)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_prob  # exploration
        exp_v = log_prob.mean() * td.detach() + 0.005 * entropy.mean()
        a_loss = -exp_v
        total_loss = a_loss + c_loss
        return total_loss

    @staticmethod
    def pad_pack_rnn(observations):
        """

        :param observations: agent_num x snapshot num x 5
        :return: seq_unpad: rnn input format
        """
        sequence = []
        for n in range(observations.shape[0]):
            # idx = np.argwhere(np.all(observations[n] == 0, axis=1))
            # # delete zero rows
            # input = np.delete(observations[n], idx, axis=0)
            input = observations[n]
            tensor_input = torch.tensor(input, dtype=torch.float)
            sequence.append(tensor_input)

        pack_seq = pack_sequence(sequence, enforce_sorted=False)
        _, lens_unpacked = pad_packed_sequence(pack_seq, batch_first=True)

        pad_seq = pad_sequence(sequence, batch_first=True)
        # using lens_unpacked result
        seq_unpad = pack_padded_sequence(pad_seq, lengths=lens_unpacked, enforce_sorted=False, batch_first=True)

        return seq_unpad


class Worker():
    def __init__(self, gnet):
        super(Worker, self).__init__()
        self.g_ep_r, self.g_ep_r1, self.g_ep_r2 = 0, 0, 0
        self.gnet = gnet
        self.lnet = gnet  # local network
        self.env = env
        self.accumulative_time = np.zeros(self.env.flights_num)

    def evaluate(self, test, nn_random):
        total_step = 1
        gama = 0.99

        episodic_accumulative_time = np.zeros([MAX_EP, self.env.flights_num])
        episodic_ratio = np.zeros([MAX_EP])
        episodic_r = np.zeros([MAX_EP])
        episodic_cancellation = np.zeros([MAX_EP, self.env.flights_num])

        for episode in (range(MAX_EP)):
            # ================================ test nn start ================================================
            s = self.env.reset()
            done = False
            overdemand_num0 = len(self.env.hotspots.copy())
            ratio = 0.
            t = 0.
            ep_r = 0.

            # ====== initial states ========
            agent_list0 = self.env.agent_list.copy()
            sms0 = self.env.speed_mu_sigma.copy()
            f0 = self.env.f_index.copy()
            schedule0 = self.env.schedule_data.copy()

            for t in range(MAX_EP_STEP):
                print('-' * 20 + f'step {t}' + '-' * 20)
                n0 = self.env.agent_num
                #  network
                if nn_random == 'nn':
                    print('test network parameter')
                    try:
                        a = self.lnet.choose_action(s)
                        s_, r, ri = self.env.step(a.detach().numpy())
                        print(f'agent num: {n0} --> {self.env.agent_num};')
                        print(self.env.speed_mu_sigma[agent_list0[0]])
                    except Exception:
                        print('Err: stopped')
                        break
                if nn_random == 'random':
                    print('test random')
                    try:
                        dsigma_mu_t = np.random.uniform(low=0, high=10.0, size=(len(self.env.selective_agent_list), 6))
                        mask = np.random.randint(2, size=1)[0]
                        dsigma_mu_t[:, 0:3] *= mask
                        dsigma_mu_t[:, 4] *= (1 - mask)
                        dsigma_mu_t[:, 5] = np.where(dsigma_mu_t[:, 5] >= np.mean(dsigma_mu_t[:, 5]), 1, 0)
                        # mu1, sigma1, mu2, sigma2, dt, values
                        s_, r, ri = self.env.step(dsigma_mu_t)
                        print(f'agent num: {n0} --> {self.env.agent_num};')
                        print(self.env.speed_mu_sigma[agent_list0[0]])
                    except Exception:
                        print('Err: stopped')
                        break

                if t == MAX_EP_STEP - 1 or len(self.env.hotspots) <= 0 or self.env.agent_num == 0:
                    done = True

                ep_r += r * gama ** t

                if done:  # update global and assign to local net
                    overdemand_num1 = len(self.env.hotspots.copy())
                    ratio = (overdemand_num0 - overdemand_num1) / overdemand_num0
                    print(f'resolved conflicts: {ratio * 100}%')
                    break
                s = s_
                total_step += 1

            # ====== final states ========
            sms1 = self.env.speed_mu_sigma.copy()
            f1 = self.env.f_index.copy()
            schedule1 = self.env.schedule_data.copy()

            # num_only_delay = 0
            # num_only_speed = 0
            # num_delay_speed = 0
            # num_not_change = 0
            # num_cancellation = 0
            #
            # for agent in agent_list0:
            #     sms0_f = sms0[agent]
            #     index0_f = np.where(f0 == agent)
            #     schedule0_f = schedule0[index0_f][0]
            #
            #     sms1_f = sms1[agent]
            #     index_f1 = np.where(f1 == agent)
            #     if index_f1[0].any():  # not empty
            #         schedule1_f = schedule1[index_f1][0]
            #
            #         if schedule1_f - schedule0_f > 0 and (sms0_f == sms1_f).all():
            #             num_only_delay += 1
            #         if schedule1_f - schedule0_f == 0 and not (sms0_f == sms1_f).all():
            #             num_only_speed += 1
            #         if schedule1_f - schedule0_f > 0 and not (sms0_f == sms1_f).all():
            #             num_delay_speed += 1
            #         if schedule1_f - schedule0_f == 0 and (sms0_f == sms1_f).all():
            #             num_not_change += 1
            #     else:
            #         num_cancellation += 1
            # print(f'num_only_delay: {num_only_delay};   num_only_speed:{num_only_speed}, '
            #       f'num_delay_speed: {num_delay_speed}, num_not_change:{num_not_change},'
            #       f'num_cancellation: {num_cancellation}')

            # # ================================ test random start ================================================
            # # uncomment to compare the speed-time graph refinement with nn
            # s = self.env.reset()
            # done = False
            # overdemand_num0 = len(self.env.hotspots.copy())
            # ratio = 0.
            # t = 0.
            # ep_r = 0.
            #
            # # ====== initial states ========
            # agent_list0 = self.env.agent_list.copy()
            # sms0 = self.env.speed_mu_sigma.copy()
            # f0 = self.env.f_index.copy()
            # schedule0 = self.env.schedule_data.copy()
            #
            # for t in range(MAX_EP_STEP):
            #     print('-' * 20 + f'step {t}' + '-' * 20)
            #     n0 = self.env.agent_num
            #     #  network
            #     print('test random')
            #     dsigma_mu_t = np.random.uniform(low=0, high=1.0, size=(len(self.env.selective_agent_list), 6))
            #     mask = np.random.randint(2, size=1)[0]
            #     dsigma_mu_t[:, 0:3] *= mask
            #     dsigma_mu_t[:, 4] *= (1 - mask)
            #     dsigma_mu_t[:, 5] = np.where(dsigma_mu_t[:, 5] >= np.mean(dsigma_mu_t[:, 5]), 1, 0)
            #     # mu1, sigma1, mu2, sigma2, dt, values
            #     s_, r, ri = self.env.step(dsigma_mu_t)
            #     print(f'agent num: {n0} --> {self.env.agent_num};')
            #     print(self.env.speed_mu_sigma[agent_list0[0]])
            #
            #
            #     if t == MAX_EP_STEP - 1 or len(self.env.hotspots) <= 0 or self.env.agent_num == 0:
            #         done = True
            #
            #     ep_r += r * gama ** t
            #
            #     if done:  # update global and assign to local net
            #         overdemand_num1 = len(self.env.hotspots.copy())
            #         ratio = (overdemand_num0 - overdemand_num1) / overdemand_num0
            #         print(f'resolved conflicts: {ratio * 100}%')
            #         break
            #     s = s_
            #     total_step += 1
            #
            # # ====== final states ========
            # sms1_random = self.env.speed_mu_sigma.copy()
            # f1_random = self.env.f_index.copy()
            # schedule1_random = self.env.schedule_data.copy()
            # # ==================================== test random end ===============================================

            #  ground delay for all flights
            delay = np.zeros(self.env.flights_num)
            cancellation = np.zeros(self.env.flights_num)
            for f in range(self.env.flights_num):
                #  before
                index_f0 = np.where(f0 == f)
                schedule_f0 = schedule0[index_f0][0]  # first value of the trajectory

                # after
                index_f1 = np.where(f1 == f)
                if index_f1[0].any():
                    schedule_f1 = schedule1[index_f1][0]
                    delay[f] = schedule_f1 - schedule_f0
                else:
                    cancellation[f] = 1

            #  speed change analysis
            # plot_curve(f0, f1, agent_list0[-5:-1], sms0, sms1, delay)
            # select_agent = random.sample(agent_list0, 4)
            # plot_curve_with_random(f0, f1, select_agent, sms0, sms1, delay, sms1_random, f1_random)

            # generate test schedule
            # plot episodic return box plot
            # plot delay histogram

            episodic_accumulative_time[episode] = delay
            episodic_ratio[episode] = ratio
            episodic_r[episode] = ep_r
            episodic_cancellation[episode] = cancellation

        print(f'episodic resolved conflicts: {np.mean(episodic_ratio) * 100} %')
        # plt.figure()
        # plt.bar(range(self.env.flights_num), np.mean(episodic_accumulative_time, axis=0))
        #
        # plt.figure()
        # plt.boxplot(episodic_r)
        # plt.show()
        np.savez(f'../../../Training_files/Files/{test}_{self.env.flights_num}_{nn_random}(dryden1)_new.npz',
                 name1=episodic_accumulative_time,
                 name2=episodic_ratio,
                 name3=episodic_r,
                 name4=episodic_cancellation)


if __name__ == "__main__":
    MAX_EP = 3
    MAX_EP_STEP = 100

    # ======== revise the file name in agent_env accordingly, and the flight number on the top============
    #  test3:300 test5:200 test8:400
    test_cases = ['test50', 'test100', 'test5']
    nnrandom = ['nn', 'random']

    for combination in list(itertools.product(test_cases, nnrandom)):
        test = combination[0]
        nn_random = combination[1]

        if test == 'test50':
            flight_num = 50
        elif test == 'test100':
            flight_num = 100
        else:  # test == 'test8'
            flight_num = 200

        env = e.ATFMVariantAgentEnv(wind_uncertainty=True, phase='strategic', map_name='mueavi',
                                    flights_num=flight_num, test=test)
        N_S = env.observations_space.shape[2]  # 5
        N_A = 1

        RNN_L2_input_size = env.observations_space.shape[1]

        # model for dryden wind: strategic
        # models = ['300_11900_1211359_state_dict.pt']  # trained on 300
        models = ['100_4000_2211351_state_dict.pt']  # trained on 100
        # model for cfd wind: pretactical
        # models = ['300_6900_252131_state_dict.pt']  # trained on 300
        # models = ['100_3000_221150_state_dict.pt']  # trained on 100


        eval_t = []

        for model in models:
            gnet = Net(N_S, N_A)  # global network

            gnet.load_state_dict(torch.load(f'../../../Training_files/Files/{model}'))
            gnet.eval()

            time_t0 = time.localtime(time.time())
            # single process evaluate
            w = Worker(gnet)
            t0 = time.time()
            # ====================
            w.evaluate(test, nn_random)
            eval_t.append(time.time() - t0)

        print(f'test time: {eval_t}')


