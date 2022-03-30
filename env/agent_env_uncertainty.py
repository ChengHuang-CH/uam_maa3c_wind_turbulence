import numpy as np
import random
import sys, os

if __name__ == '__main__':
    from FlightGenerator import FlightsGenerator
else:
    from .FlightGenerator import FlightsGenerator  # for the call outside the folder

import torch
import time
from copy import deepcopy


class ATFMVariantAgentEnv:

    def __init__(self, wind_uncertainty=False, phase='strategic', map_name='mueavi',
                 flights_num=300, train=False, test='test5'):

        self.episode = 0
        """
        Dynamic Agent Number: only those flights in over-demand sectors are considered as AGENTS
        """
        self.phase = phase
        self.map_name = map_name
        self.wind_uncertainty = wind_uncertainty
        self.train = train

        self.test = test

        self.flights_num = flights_num
        self.f = FlightsGenerator(self.flights_num,
                                  phase=self.phase,
                                  map_name=self.map_name,
                                  wind_uncertainty=self.wind_uncertainty)
        print(f'Training in {self.phase} stage.')

        self.traversed_blocks_num = 700
        # ====== !revised here =======
        # if self.wind_uncertainty:
        #     self.o_dim = 4
        # else:
        self.o_dim = 3
        self.observations_space = np.zeros([self.flights_num, self.traversed_blocks_num, self.o_dim])

        self.agent_index = np.zeros(self.flights_num)  # [1,0,0,1, ..., 0];dim=1x300;flights in over-demand sectors = 1
        self.agent_num = 0

        self.hotspots = []
        self.hotspots_t = []
        self.hotspots_mtx = np.array([])
        self.agent_list = []
        self.selective_agent_list = []

        # for saving detailed info for conflict detection
        self.schedule_data = np.array([])
        self.speed_data = np.array([])
        self.f_index = np.array([])
        self.s_index = np.array([])
        self.capacity = np.array([])

        # for saving info in large aircraft (used for update profile because of the node number)
        self.large_f_index = []  # save f index of large aircraft
        self.large_s_name = []  # save trajectory nodes in large graph
        self.large_speed_data = []  # save original speed information in large graph
        self.large_schedule_data = []  # save original time information in large graph

        # for saving both info
        self.speed_mu_sigma = np.array([])
        self.aircraft_names = []
        self.payload_status = []
        self.wind_resistance = []

        self.over_wind_num = 0  # record number of all exceeding wind sectors of all flights
        self.initial_wind_num = 0

        self.initial_hotspot_num = 0

        # store accumulative information for each flight
        self.accumulative_info = np.zeros((self.flights_num, 3))
        self.stepwise_info = np.zeros((self.flights_num, 4))

        self.steps = 0

        self.done = False

    def agents_observation(self):
        """

        :param:
        :return: observations_space dim=[flights_num, 10, 5]
        """

        # clear data before each calculation
        self.agent_num = len(self.agent_list)
        if self.agent_num >= 5000:
            self.selective_agent_list = random.sample(self.agent_list, 1000)
        else:
            self.selective_agent_list = self.agent_list

        training_agent_num = len(self.selective_agent_list)

        if training_agent_num != 0:

            self.observations_space = np.zeros([training_agent_num, self.traversed_blocks_num, self.o_dim])
            '''
            observation of agent
            '''
            n = 0
            # self.hotspots, self.agent_list
            # 1. for f, flight in enumerate(self.agent_list): 2. previous error: range(self.agent_num)
            for idx, f in enumerate(self.selective_agent_list):
                row = 0
                index_in_f = np.where(self.f_index == f)
                traversed_blocks = self.s_index[index_in_f]
                traversed_blocks_t = (self.schedule_data[index_in_f] // self.f.snapshot_time).astype(int)
                traversed_blocks_capa = self.capacity[index_in_f]

                for i in range(traversed_blocks.size):
                    """
                    observation space: agent x 200 x 5
                    each row :[t, s, overdemand in s ]  # remove demand and capacity

                    """
                    self.observations_space[idx][row][0] = traversed_blocks_t[i]  # t
                    self.observations_space[idx][row][1] = traversed_blocks[i]  # s
                    # check if it is hotspot for s in t
                    if [traversed_blocks[i], traversed_blocks_t[i]] in self.hotspots_mtx.tolist():
                        # print(f'hotspots {n};f{f}; {[traversed_blocks[i], traversed_blocks_t[i]]}')
                        self.observations_space[idx][row][2] = 1  # conflict
                    else:
                        self.observations_space[idx][row][2] = 0

                    # ====== !revised here =======
                    # if self.wind_uncertainty:
                    #     self.observations_space[idx][row][3] = traversed_blocks_capa[i]  # capacity

                    if traversed_blocks_capa[i] == 1:
                        self.over_wind_num += 1

                    row = row + 1
        else:
            self.observations_space = np.zeros([1, self.traversed_blocks_num, self.o_dim])

    def step(self, actions, last_step=False):
        """

        :param actions: dmu1, dsigma1,dmu2. dsigma2, dt
        :return:
        """
        s_prev = len(self.hotspots)  # previous hotspots number
        over_wind_num_prev = self.over_wind_num
        cancelled_flight_num_prev = self.flights_num - np.unique(self.f_index).shape[0]
        self.stepwise_info = np.zeros((self.flights_num, 5))  # reset for each step

        # if last_step:
        #     print('Activate cancellation')

        for i, agent in enumerate(self.selective_agent_list):
            dmu1, dsigma1, dmu2, dsigma2, dt, cancellation = actions[i]
            # print(f'cancellation : {cancellation}')

            # if last_step:
            #     cancellation = cancellation
            # else:
            #     cancellation = 0

            # get original sparse schedule of this agent
            current_aircraft = self.aircraft_names[agent]

            # update profile based on actions for different size of aircraft
            if current_aircraft in self.f.small_aircraft:
                agent_ = np.where(self.f_index == agent)  # get flight index in the sparse schedule; tuple
                speed_agent = self.speed_data[agent_]  # get speed profile of this flight
                schedule_agent = self.schedule_data[agent_]  # get schedule of this flight

                if cancellation != 1:
                    # generate new speed profile and then reassign time for this agent
                    updated_speed_arr, updated_time_slots, updated_speed_mu_sigma = \
                        self.f.update_profile(dmu1, dsigma1, dmu2, dsigma2, dt, agent_, schedule_agent,
                                              self.speed_mu_sigma[agent], speed_agent, current_aircraft)

                else:
                    # cancel this flight
                    updated_speed_arr, \
                    updated_time_slots, \
                    updated_speed_mu_sigma = np.zeros_like(speed_agent), \
                                             np.zeros_like(schedule_agent), \
                                             np.zeros_like(self.speed_mu_sigma[agent])
                    # delete this flight
                    self.f_index = np.delete(self.f_index, agent_)
                    self.s_index = np.delete(self.s_index, agent_)
                    self.speed_data = np.delete(self.speed_data, agent_)
                    self.schedule_data = np.delete(self.schedule_data, agent_)


            else:
                agent_l = np.where(
                    self.large_f_index == agent)  # get flight index in the large aircraft schedule; tuple
                speed_agent = self.large_speed_data[agent_l]  # get speed profile of this large flight
                schedule_agent = self.large_schedule_data[agent_l]  # get schedule of this large flight
                node_names = self.large_s_name[agent_l]

                agent_ = np.where(self.f_index == agent)  # index of this flight in detailed graphs

                updated_time_slots = []
                updated_speed_arr = []

                if cancellation != 1:
                    # generate new speed profile and then reassign time for this agent
                    updated_speed_arr_l, updated_time_slots_l, updated_speed_mu_sigma = \
                        self.f.update_profile(dmu1, dsigma1, dmu2, dsigma2, dt, agent_l, schedule_agent,
                                              self.speed_mu_sigma[agent], speed_agent, current_aircraft)

                    # convert from low resolution to detailed graph

                    for idx, node_name in enumerate(node_names):
                        subnodes = self.f.G_res1.nodes[node_name]['subnodes']

                        speed_list = [updated_speed_arr_l[idx]] * len(subnodes)
                        schedule_list = [updated_time_slots_l[idx]] * len(subnodes)

                        updated_speed_arr += speed_list
                        updated_time_slots += schedule_list

                else:
                    # cancel this flight
                    updated_speed_arr_l, \
                    updated_time_slots_l, \
                    updated_speed_mu_sigma = np.zeros_like(speed_agent), \
                                             np.zeros_like(schedule_agent), \
                                             np.zeros_like(self.speed_mu_sigma[agent])
                    # delete this flight
                    # delete this flight
                    self.large_f_index = np.delete(self.large_f_index, agent_l)
                    self.large_speed_data = np.delete(self.large_speed_data, agent_l)
                    self.large_schedule_data = np.delete(self.large_schedule_data, agent_l)
                    self.large_s_name = np.delete(self.large_s_name, agent_l)
                    self.f_index = np.delete(self.f_index, agent_)
                    self.s_index = np.delete(self.s_index, agent_)
                    self.speed_data = np.delete(self.speed_data, agent_)
                    self.schedule_data = np.delete(self.schedule_data, agent_)

            # get step-wise change for this flight
            if cancellation != 1:
                increased_enroute_t = (updated_time_slots[-1] - updated_time_slots[0]) - \
                                      (schedule_agent[-1] - schedule_agent[0])
                increased_arriving_t = updated_time_slots[-1] - schedule_agent[-1]
                gradient = np.gradient(updated_speed_arr)  # acceleration
                accel_smooth = (np.abs(np.max(gradient)) - 0.1) * 10
                cancellation_punish = 0

                overt_t = (updated_time_slots[-1] - updated_time_slots[0]) - \
                          self.f.ap_database[self.aircraft_names[agent]]["envelop"][
                              self.payload_status[agent]] * 60  # constraints: max hovering time
                # ==== [update the schedule for the agent] ========
                self.speed_data[agent_] = updated_speed_arr
                self.schedule_data[agent_] = updated_time_slots
                self.speed_mu_sigma[agent] = updated_speed_mu_sigma

            else:
                increased_enroute_t = 0
                increased_arriving_t = 0
                gradient = 0
                accel_smooth = 0
                overt_t = 0
                cancellation_punish = 1

                # self.speed_mu_sigma = np.delete(self.speed_mu_sigma, agent)
                # self.payload_status = np.delete(self.payload_status, agent)
                # self.aircraft_names = np.delete(self.aircraft_names, agent)
                # self.wind_resistance = np.delete(self.wind_resistance, agent)

            # record for step-wise
            self.stepwise_info[agent][0] = increased_enroute_t
            self.stepwise_info[agent][1] = increased_arriving_t
            self.stepwise_info[agent][2] = accel_smooth
            self.stepwise_info[agent][3] = overt_t if overt_t > 0 else 0
            self.stepwise_info[agent][4] = cancellation_punish

            # record accumulative delay time
            self.accumulative_info[agent][0] += increased_enroute_t
            self.accumulative_info[agent][1] += increased_arriving_t
            self.accumulative_info[agent][2] = accel_smooth

        cancelled_flight_num_after = self.flights_num - np.unique(self.f_index).shape[0]

        if self.selective_agent_list:
            stepwise_info = np.sum(self.stepwise_info, axis=0) / self.flights_num
        else:
            stepwise_info = np.zeros(5)

        # reset capacity every step; then count again in the entry_count
        self.capacity = np.zeros_like(self.capacity)
        self.over_wind_num = 0

        # counting environment information
        self.hotspots, self.hotspots_t, self.hotspots_mtx, self.agent_list, self.capacity = \
            self.f.entry_count(self.schedule_data,
                               self.speed_data,
                               self.f_index,
                               self.s_index,
                               self.f.airblocks_num,
                               self.f.snapshot_time,
                               self.capacity,
                               self.wind_resistance)
        self.agents_observation()

        # get average en-route delay of all flights in cooperative case
        r0_n = - stepwise_info[0] * 10
        # get average arriving delay of all flights
        r1_n = - stepwise_info[1] * 0.5
        # get average smoothness of speed profile
        r2_n = - stepwise_info[2] * 10
        # hotpsot number change
        punish_by_step = 1 + self.steps * 0.01  # expect to reduce the step
        r3_n = - (len(self.hotspots) - s_prev) / self.initial_hotspot_num * 200
        # max hovering time
        r4_n = - stepwise_info[3] * self.flights_num / 3  # / self.flights_num * 20

        if self.wind_uncertainty:
            # max wind resistance
            # r5_n = - (self.over_wind_num - over_wind_num_prev) / self.initial_wind_num * 200
            # cancellation punishment
            r6_n = - stepwise_info[4] * self.flights_num * 10  # original 10
            reward_n = r0_n + r1_n + r2_n + r3_n + r4_n + r6_n
            print(f'step: {self.steps}; reward: {np.round([reward_n, r0_n, r1_n, r2_n, r3_n, r4_n, r6_n], 2)}')
            print(f'wind sectors: {over_wind_num_prev} --> {self.over_wind_num}')
            print(f'cancelled flight num: {cancelled_flight_num_prev} --> {cancelled_flight_num_after}')
        else:
            reward_n = r0_n + r1_n + r2_n + r3_n + r4_n
            print(f'step: {self.steps}; reward: {np.round([reward_n, r0_n, r1_n, r2_n, r3_n, r4_n], 2)}')

        print(f'conflicts number: {s_prev} --> {len(self.hotspots)}')

        assert np.isnan(reward_n) == False

        self.steps += 1

        return self.observations_space, reward_n, [r0_n, r1_n, r2_n, r3_n]

    def reset(self):
        """

        :return:
        """

        print(f'-----  episode {self.episode}, reset simulation ------')

        # -------------- reset flight schedule using a fixed schedule --------------
        # load using absolute path for the usage of inside and outside folder
        if self.train:
            sparse_flight_schedule = np.load(os.path.join(os.path.dirname(__file__),
                                                          f'sparse_flight_schedule(wind_True)_{self.flights_num}.npz'))
        else:
            if self.test == 'test3':
                sparse_flight_schedule = np.load(os.path.join(os.path.dirname(__file__),
                                                              f'sparse_flight_schedule(wind_False)_test3.npz'))
            elif self.test == 'test5':
                sparse_flight_schedule = np.load(os.path.join(os.path.dirname(__file__),
                                                              f'sparse_flight_schedule(wind_False)_test5.npz'))
            elif self.test == 'test8':
                sparse_flight_schedule = np.load(os.path.join(os.path.dirname(__file__),
                                                              f'sparse_flight_schedule(wind_False)_test8.npz'))
            elif self.test == 'test100':
                sparse_flight_schedule = np.load(os.path.join(os.path.dirname(__file__),
                                                              f'sparse_flight_schedule(wind_False)_test100.npz'))
            else:  # self.test == 'test50':
                sparse_flight_schedule = np.load(os.path.join(os.path.dirname(__file__),
                                                              f'sparse_flight_schedule(wind_False)_test50.npz'))
        self.schedule_data = sparse_flight_schedule['schedule_data']
        self.speed_data = sparse_flight_schedule['speed_data']
        self.f_index = sparse_flight_schedule['f_index']
        self.s_index = sparse_flight_schedule['s_index']
        self.speed_mu_sigma = sparse_flight_schedule['speed_mu_sigma']
        self.aircraft_names = sparse_flight_schedule['aircraft_names']
        self.payload_status = sparse_flight_schedule['payload_status']
        self.large_f_index = sparse_flight_schedule['large_f_index']
        self.large_s_name = sparse_flight_schedule['large_s_name']
        self.large_speed_data = sparse_flight_schedule['large_speed_data']
        self.large_schedule_data = sparse_flight_schedule['large_schedule_data']
        self.wind_resistance = sparse_flight_schedule['wind_resistance']
        self.capacity = sparse_flight_schedule['capacity']

        # -------------------------------------------------------------------------
        self.hotspots, self.hotspots_t, self.hotspots_mtx, self.agent_list, self.capacity = self.f.entry_count(
            self.schedule_data,
            self.speed_data,
            self.f_index,
            self.s_index,
            self.f.airblocks_num,
            self.f.snapshot_time,
            self.capacity,
            self.wind_resistance)
        self.initial_hotspot_num = deepcopy(len(self.hotspots))
        print(f'initial conflict num: {self.initial_hotspot_num}')

        self.accumulative_info = np.zeros((self.flights_num, 3))

        self.agents_observation()  # get self.observations_space
        self.initial_wind_num = self.over_wind_num

        # update episode and reset step to 0 for new episode
        self.episode += 1
        self.steps = 0

        return self.observations_space  # dim=[flights_num, 72, 16]


if __name__ == "__main__":
    # !!!------------------------just for example test, RL has not been included yet----------------------
    env = ATFMVariantAgentEnv()

    # while not env.done:
    # ----    new episode   ------

    observations = env.reset()
    # print(time.time() - t0)
    # # --- steps in each episode ---
    for steps in range(30):
        t0 = time.time()
        hotspots_num = len(env.hotspots.copy())
        print(f'hotspots_num:{hotspots_num}, agents flights:{len(env.selective_agent_list)}')

        env.step(actions=np.random.uniform(low=10, high=40, size=(len(env.selective_agent_list), 5)))
        print(time.time() - t0)
