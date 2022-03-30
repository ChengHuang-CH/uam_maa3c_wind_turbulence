import random
import numpy as np
import csv
import networkx as nx
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
import time
import operator
import json
import re
from mpl_toolkits.mplot3d import Axes3D


# ------------------------------------------------------------------------------------
# The air corridor is constructed. And the basic component is airblock, of which the capacity is one.
# ------------------------------------------------------------------------------------

class FlightsGenerator:
    def __init__(self, flights_num, wind_uncertainty, phase='strategic', map_name='mueavi'):
        self.phase = phase
        self.map_name = map_name
        self.wind_uncertainty = wind_uncertainty

        self.flights_num = flights_num
        self.snapshot_time = 5  # period of each snapshot (s)
        self.hour0 = 6  # operation start hour (h) >= 0
        self.hour1 = 18  # operation end hour (h) <= 24
        self.operation_start_t = self.hour0 * 60 * 60  # start from 1s
        self.operation_end_t = self.hour1 * 60 * 60  # end to 24h
        self.snapshot_num = int((self.operation_end_t - self.operation_start_t) / self.snapshot_time)
        self.capacity = 1  # each airblock can only accept one flight

        # vertices and edges of corridor from blender output.
        self.vertices = []
        self.lines = []

        if self.map_name == 'mueavi':
            self.airblocks_size = 4  # m
            with open('../../uam/corridor_files/mueavi_vertices.csv', 'r') as verticesFile:
                vertex_reader = csv.reader(verticesFile, delimiter=',')
                for line in vertex_reader:
                    vi = [float(i) for i in line]  # each vertex
                    self.vertices.append(vi)

            with open('../../uam/corridor_files/mueavi_edges.csv', 'r') as edgesFile:
                edge_reader = csv.reader(edgesFile, delimiter=',')
                for line_e in edge_reader:
                    ei = [int(j) for j in line_e]  # each vertex
                    self.lines.append(ei)

            # vertiports index in basic graph
            self.vertiports = ['l_0_1801', 'l_0_1821', 'l_0_1841', 'l_0_1861', 'l_0_1881']
            # same vertiports index in the low resolution graph
            self.vertiports_res = ['l_0_190', 'l_0_191', 'l_0_192', 'l_0_193', 'l_0_194']

        # aircraft performance
        with open('../../uam/corridor_files/drone_performance.json', 'r') as f:
            self.ap_database = json.load(f)

        self.large_aircraft = []
        self.small_aircraft = []
        for k, v in self.ap_database.items():
            if type(v['size']) != str:
                if v['size'] >= 2.0:
                    self.large_aircraft.append(k)
                else:
                    self.small_aircraft.append(k)

        # initialize the graph for airspace structure
        """
        G == G1 (involve all ground nodes) ---[merge]---> G_res1
        """
        self.remove_nodes = []
        # initial graph for 5m x 5m cubes to connect the nearest node
        self.G = nx.Graph()
        # same nodes graph with initial graph, but the edge connect the nearby 8 nodes
        # (used for the next resolution graph)
        self.G1 = nx.Graph()
        # the next resolution graph
        self.G_res1 = nx.Graph()
        self.initial_graph()  # obtain initial graph
        self.airblocks_num = len(
            list(self.G.nodes)) + 10  # plus 10 to make the index aligned after removing the 5 nodes
        print(f'Total {self.airblocks_num} nodes/airblocks in the graph')

        self.flight_schedule = np.zeros([self.flights_num, self.airblocks_num])
        self.speed_profile = np.zeros([self.flights_num, self.airblocks_num])

        self.ec = np.zeros([self.snapshot_num, self.airblocks_num])
        self.schedule_matrix_sparse = None
        self.speed_matrix_sparse = None

        # for saving detailed info for conflict detection
        self.f_index = []
        self.s_index = []
        self.speed_data = []
        self.schedule_data = []
        self.capacity = []

        # for saving info only about large aircraft
        self.large_f_index = []  # save f index of large aircraft
        self.large_s_name = []  # save trajectory nodes in large graph
        self.large_speed_data = []  # save original speed information in large graph
        self.large_schedule_data = []  # save original time information in large graph

        # for both
        self.aircraft_names = []
        self.payload_status = []
        self.speed_mu_sigma = []
        self.wind_resistance = []

        # wind profile
        if self.phase == 'strategic' and self.wind_uncertainty:
            self.wind_profile = []
            with open('../../uam/corridor_files/dryden_wind.csv', 'r') as windFile:
                header = windFile.readline()
                wind_reader = csv.reader(windFile, delimiter=',')
                for line_w in wind_reader:
                    wi = [float(j) for j in line_w]
                    self.wind_profile.append(wi)
                self.wind_profile = np.asarray(self.wind_profile)

        if self.phase == 'pretactical' and self.wind_uncertainty:
            wind_profiles = []
            external_files = ['p1.txt', 'p2.txt', 'p3.txt', 'p4.txt', 'p5.txt']
            # external_files = ['test_p1.txt', 'test_p2.txt', 'test_p3.txt', 'test_p4.txt', 'test_p5.txt']

            x = np.linspace(0, int((self.operation_end_t - self.operation_start_t) / 5),
                            int((self.operation_end_t - self.operation_start_t) / 5))
            xp = np.linspace(0, int((self.operation_end_t - self.operation_start_t) / 5), 60)
            for txt in external_files:
                wind_profile = []
                with open('../../uam/corridor_files/' + txt, 'r') as windFile:
                    header = windFile.readline()
                    content = windFile.readlines()
                    for line in content:
                        line_split = [i for i in re.split(',|\n', line) if i]
                        wind_profile.append(float(line_split[-1]))
                    # expand 60 to 8640
                    fp = wind_profile
                    y = np.interp(x, xp, fp)
                wind_profiles.append(y)

            self.wind_profiles = np.asarray(wind_profiles)  # [5, 8640]
            self.block_belong2p = self.get_block_belong()  # dim=block_num
            print()

    def get_block_belong(self):
        node2p = np.zeros(1900)
        for i, node in enumerate(list(self.G.nodes)):
            if node in self.remove_nodes:
                pass
            else:
                dis = []
                for vertiport in self.vertiports:
                    dis.append(np.linalg.norm(self.G.nodes[node]['pos'] - self.G.nodes[vertiport]['pos']))
                nearest_p = dis.index(min(dis))
                node2p[i] = nearest_p

        return node2p

    def initial_graph(self):

        cube_i = 0
        center = []
        for i in range(0, len(self.vertices), 8):
            eight_vertices = np.asarray(self.vertices[i: i + 8])  # 8 vertices of this cube
            eight_vertices_sort = eight_vertices[np.argsort(eight_vertices[:, 2])]

            # (1) get the center of each block
            center_x = np.round(np.sum(eight_vertices_sort[0:4, 0]) / 4, 2)
            center_y = np.round(np.sum(eight_vertices_sort[0:4, 1]) / 4, 2)
            center_z = np.round(np.sum(eight_vertices_sort[0, 2] + eight_vertices_sort[4, 2]) / 2, 2)
            # (2) save the node and name it by layer altitude
            # l_(layer)_(cube_index)
            flight_layer = int(center_z // self.airblocks_size)
            self.G.add_node(f'l_{flight_layer}_{cube_i}',
                            pos=np.array([center_x, center_y, center_z]),
                            FL=flight_layer,
                            index=cube_i)
            self.G1.add_node(f'l_{flight_layer}_{cube_i}',
                             pos=np.array([center_x, center_y, center_z]),
                             FL=flight_layer,
                             index=cube_i)
            center.append([center_x, center_y, center_z])

            cube_i += 1

        # search all pairs of points in a kd-tree within a distance
        node_names_G = list(self.G.nodes)
        pairs_sort = self.tree_search(center, self.airblocks_size + 0.5)
        # (3) add edge by nearest distance
        for i in range(pairs_sort.shape[0]):
            self.G.add_edge(node_names_G[pairs_sort[i][0]], node_names_G[pairs_sort[i][1]])

        # ============================= another resolution layers =============================
        node_names_G1 = list(self.G1.nodes)
        pairs_sort1 = self.tree_search(center, self.airblocks_size * np.sqrt(3) + 0.2)
        for i in range(pairs_sort1.shape[0]):
            self.G1.add_edge(node_names_G1[pairs_sort1[i][0]], node_names_G1[pairs_sort1[i][1]])

        """
                [top view of level 0 at vertiport 0]

                -----------------------------------|
                        <--- main corridor         |
                ___________________________________|
                         |      |      |
                         | 1800 | 1802 |
                         |______|______|
                         |      |      |
                         | 1801 | 1803 |
                         |______|______|

                other vertiports start from 1800  1820  1840 1860 1880;
                1801 is selcted as veriports
                """
        # remove the ground box near by vertiports after adding edge (delete nodes will delete edges).
        self.remove_nodes = ['l_0_1800', 'l_0_1802',
                             'l_0_1820', 'l_0_1822',
                             'l_0_1840', 'l_0_1842',
                             'l_0_1860', 'l_0_1862',
                             'l_0_1880', 'l_0_1882']
        self.G.remove_nodes_from(self.remove_nodes)
        self.G1.remove_nodes_from(self.remove_nodes)

        # index 0-1799 (corridor); index 1800 and after (vertiports: level 0; vetiports airspace: level 1,2,3,4)

        # (a). for corridor filter the nodes in level 0 and index>1800; only keep Fl1-4 corridor blocks
        index = 0
        # condition1 = G1.nodes[node]['FL'] >= 1 and G1.nodes[node]['index'] < 1800
        condition1 = ['FL', '>=', 1, 'index', '<', 1800]
        index = self.generate_new_graph(condition1, index, assert_num=8)

        """
        side view of  vertiport and airspace above it
        __________
        |  _____  |
        |  |_|_|  |  level 4 \
        |  |_|_|  |  level 3 ->  merge 8 boxes to one vertiport airspace level
        |_________|
        |  |_|_|  |  level 2 \
        |  |_|_|  |  level 1 ->  merge 8 boxes to one vertiport airspace level
        |_________| 
        |  |_|_|  |  level 0 ->  merge ground 2 boxes to ground level
        |_________|
        """

        # (b). for vertiports airspace
        # filter the nodes in level 0 and index<1800
        # condition2 = G1.nodes[node]['FL'] >= 1 and G1.nodes[node]['index'] >= 1800
        condition2 = ['FL', '>=', 1, 'index', '>=', 1800]
        index = self.generate_new_graph(condition2, index, assert_num=8)

        # (c). After above deletion, only nodes in level 0 are left; merge the ground 4 blocks for vertiports (G1 index>1800)
        # for ground vertiports
        # condition3 = G1.nodes[node]['FL'] >= 0 and G1.nodes[node]['index'] >= 1800
        condition3 = ['FL', '>=', 0, 'index', '>=', 1800]
        _ = self.generate_new_graph(condition3, index, assert_num=2)

        # ==== new vertiports ans airspace ====
        # 'l_0_190', 'l_2_180', 'l_4_181'
        # 'l_0_191', 'l_2_182', 'l_4_183'
        # 'l_0_192', 'l_2_184', 'l_4_185'
        # 'l_0_193', 'l_2_186', 'l_4_187'
        # 'l_0_194', 'l_2_186', 'l_4_187'
        # write edges for new graph
        pos = nx.get_node_attributes(self.G_res1, 'pos')
        centers = list(pos.values())
        pairs_sort = self.tree_search(centers, 2 * self.airblocks_size + 1)
        for i in range(pairs_sort.shape[0]):
            self.G_res1.add_edge(list(self.G_res1.nodes)[pairs_sort[i][0]], list(self.G_res1.nodes)[pairs_sort[i][1]])

    # search all pairs of points in a kd-tree within a distance
    @staticmethod
    def tree_search(lis, dis):
        """

        :param lis: target list
        :param dis: distance
        :return:
        """
        tree = cKDTree(lis)
        pairs = tree.query_pairs(dis, p=2)

        pairs = np.asarray(list(pairs))
        pairs_sort = pairs[np.argsort(pairs[:, 0])]

        return pairs_sort

    def generate_new_graph(self, condition, index, assert_num):
        ops = {'>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le,
               '==': operator.eq}

        while True:
            # remove node in level 0
            #  condition1 = ['FL', '>=', '1', 'index', '<', '1420']
            # for node in self.G1.nodes():
            #     print(node)
            #     print(self.G1.nodes[node]['FL'])

            filter_nodes = [node for node in self.G1.nodes() if
                            ops[condition[1]](self.G1.nodes[node][condition[0]], condition[2]) and
                            ops[condition[4]](self.G1.nodes[node][condition[3]], condition[5])]

            if not filter_nodes:
                break
            # print(f'filter_nodes: {filter_nodes}')
            # (a). get node
            node1 = filter_nodes[0]
            # (b). get connected neighbors
            neighbors_nodes = list(self.G1.neighbors(node1))
            neighbors_nodes.append(node1)  # all 8 cubes
            center_pos = np.array([0, 0, 0])
            valid_neighbors = []
            for node in neighbors_nodes:
                # filter the nodes in level 0 and index>1420; only keep Fl1-4 corridor blocks
                if ops[condition[1]](self.G1.nodes[node][condition[0]], condition[2]) and \
                        ops[condition[4]](self.G1.nodes[node][condition[3]], condition[5]):
                    center_pos = np.add(self.G1.nodes[node]['pos'], center_pos)
                    valid_neighbors.append(node)
            # print(f'the valid neighbors of {node1} is {valid_neighbors}')

            assert len(valid_neighbors) == assert_num  # merge 8 cubes to 1 big cube

            center_pos = center_pos / len(valid_neighbors)

            # (c). write node into the new graph
            layer = int(center_pos[2] // self.airblocks_size)
            self.G_res1.add_node(f'l_{layer}_{index}',
                                 pos=center_pos,
                                 subnodes=valid_neighbors,
                                 FL=layer,
                                 index=index
                                 )

            index += 1
            # (d). delete node from G1
            self.G1.remove_nodes_from(valid_neighbors)

        return index

    def filter_graph_detailed(self, ori, des):
        """
        filter subgraph from the original detailed graph

        """

        ori_layer, ori_index = self.G.nodes[ori]['FL'], self.G.nodes[ori]['index']
        des_layer, des_index = self.G.nodes[des]['FL'], self.G.nodes[des]['index']

        if des_index > ori_index:  # 1,3 layer
            cruise_layer = random.choice([1, 2])
        else:  # 2,4 layer
            cruise_layer = random.choice([3, 4])

        # print(f'Fly from vertiport {self.vertiports.index(ori)} to {self.vertiports.index(des)}, '
        #       f'cruise at {cruise_layer} flight layer')

        #  keep (1) assigned cruise layer nodes and (2) the vertiports nodes; fast 4e-05 s
        filter_nodes = lambda x: (self.G.nodes[x]['FL'] == cruise_layer or
                                  (1800 <= self.G.nodes[x]['index'] and self.G.nodes[x]['FL'] <= cruise_layer))
        subgraph = nx.subgraph_view(self.G, filter_node=filter_nodes)

        return subgraph

    def filter_graph_resolution(self, ori, des):
        """
        filter subgraph from the lower resolution graph

        """

        ori_layer, ori_index = self.G_res1.nodes[ori]['FL'], self.G_res1.nodes[ori]['index']
        des_layer, des_index = self.G_res1.nodes[des]['FL'], self.G_res1.nodes[des]['index']

        if des_index > ori_index:
            cruise_layer = 2
        else:
            cruise_layer = 4

        #  keep (1) assigned cruise layer nodes and (2) the vertiports nodes;
        filter_nodes = lambda x: (self.G_res1.nodes[x]['FL'] == cruise_layer or
                                  (180 <= self.G_res1.nodes[x]['index']
                                   and self.G_res1.nodes[x]['FL'] <= cruise_layer))
        subgraph = nx.subgraph_view(self.G_res1, filter_node=filter_nodes)

        return subgraph

    @staticmethod
    def generate_speed_profile(variable_speed, node_num):
        """
        generate the speed map to indicate the flight speed in specific airblocks;
        this feature allows the variable speed change for eVTol vehicles.
        [the speed vector always points from the current airblock to the next airblock]
        :return:  shape[flight num, airblocks num], element is the speed requirement in this airblock
        """

        if variable_speed:
            # the design is necessary to assign monotonic speed curve; but the adjustment can ne non-monotonic
            while True:
                # repeat trying for the acceleration requirement
                percent = np.random.uniform(3, 5, (1, 4))  # generate factor in range [3,5] after some tests.
                mean1, std1 = node_num / percent[0][0], node_num / percent[0][1]
                mean2, std2 = node_num - node_num / percent[0][2], node_num / percent[0][3]
                sample1 = norm.pdf(np.arange(node_num), loc=mean1, scale=std1)
                sample2 = norm.pdf(np.arange(node_num), loc=mean2, scale=std2)
                # gaussian mixture model (GMM)
                sample = sample1 + sample2

                # normiliaze to [0.05, 1]
                sample_norm = (sample - np.min(sample)) / (np.max(sample) - np.min(sample)) * (1 - 0.05) + 0.05  # speed
                gradient = np.gradient(sample_norm)  # acceleration
                if np.abs(np.max(gradient)) < 0.1 or np.abs(
                        np.min(gradient)) < 0.1:  # acceleration < 0.1 for smoothness
                    break

            # normiliaze to [0, 1]
            speed_array = np.round(sample_norm, 2)
            mu_sigma = [mean1, std1, mean2, std2]
        else:  # constant speed
            speed = np.round(np.random.uniform(0.5, 1.0, (1, 1))[0], 2)
            speed_array = np.ones(node_num) * speed
            mu_sigma = [0, 0, 0, 0]

        return speed_array, mu_sigma

    def update_profile(self, dmu1, dsigma1, dmu2, dsigma2, dt, agent_, schedule_agent, speed_mu_sigma,
                       speed_agent, current_aircraft):
        """
        update speed profile and time schedule
        :return:
        """
        if current_aircraft in self.small_aircraft:
            block_dim = self.airblocks_size
        else:
            block_dim = self.airblocks_size * 2

        mu1, sigma1, mu2, sigma2 = speed_mu_sigma

        if mu1 == sigma1 == mu2 == sigma2 == 0:
            speed_arr = speed_agent  # if all data is 0; keep constant speed; only consider ground delay
            speed_mu_sigma1 = np.array([0, 0, 0, 0])
        else:
            sample1 = norm.pdf(np.arange(len(agent_[0])), loc=mu1 + dmu1, scale=sigma1 + dsigma1)
            sample2 = norm.pdf(np.arange(len(agent_[0])), loc=mu2 + dmu2, scale=sigma2 + dsigma2)
            # replace the original speed_mu_sigma data with new data
            speed_mu_sigma1 = np.array([mu1 + dmu1, sigma1 + dsigma1, mu2 + dmu2, sigma2 + dsigma2])

            sample = sample1 + sample2
            # normiliaze to [0.5, 1]
            sample_norm = (sample - np.min(sample)) / (np.max(sample) - np.min(sample)) * (1 - 0.05) + 0.05  # speed
            speed_arr = np.round(sample_norm, 2)

        time_slots = []

        t = schedule_agent[0]  # accumulate from the first time

        for idx in range(len(schedule_agent)):
            if speed_arr[idx] != 0:
                t = np.round(t + block_dim / speed_arr[idx], 2)
            else:
                # speed is 0/ air holding in this snapshots
                t = np.round(t + block_dim / (speed_arr[idx] + 0.05), 2)  # set speed to 0.05 if it is 0.
            assert t != np.nan
            time_slots.append(t)

        # add ground delay
        time_slots = list(map(lambda ti: ti + dt, time_slots))

        return speed_arr, time_slots, speed_mu_sigma1

    def assign_time(self, path, speed_arr, current_aircraft, payload):
        """
        Assign time for each flight path
        :param path: single path
        :return: assigned time based on speed
        """
        time_slots = []

        for _ in range(5):  # try 5 times for this vehicle
            # retry if the assigned time exceed the time range
            time_slots = []
            if self.operation_start_t:
                self.operation_start_t += 1
            t = random.randrange(self.operation_start_t, self.operation_end_t)
            t_start = t

            if current_aircraft in self.small_aircraft:
                block_dim = self.airblocks_size
            else:
                block_dim = self.airblocks_size * 2

            for idx, node_name in enumerate(path):
                if speed_arr[idx] != 0:
                    t = np.round(t + block_dim / speed_arr[idx], 2)
                else:
                    # speed is 0/ air holding in this snapshots
                    t = t + 5.0
                time_slots.append(t)
            t_end = time_slots[-1]

            if t_end < self.operation_end_t \
                    and t_end - t_start < self.ap_database[current_aircraft]["envelop"][payload] * 60:
                break

        return time_slots

    def generate_flights(self):
        """
        flight schedule is discretized to sparse structure row(f_index list) col(s_index list) and data(schedule_data,
        speed_data) instead of a [f_num, s_num] matrix to save memory and speed up the process of counting.

        :return:
        """
        print('generating flights ...')
        num = 0

        p = np.array([1, 0])  # if assigning variable speed [70% true, 30% false]
        a_size = np.array([0.5, 0.5])  # if assigning small size aircraft [70% small UAV, 30% large UAV]
        ef = np.array([0.5, 0.5])  # if empty or full payload

        while True:
            small_aircraft = np.random.choice([True, False], p=a_size.ravel())  # random select the size of aircraft

            if small_aircraft:
                vertiports = self.vertiports
            else:
                vertiports = self.vertiports_res

            origin_vertiports = vertiports[random.randrange(len(vertiports))]
            destination_vertiports = vertiports[random.randrange(len(vertiports))]

            if origin_vertiports == destination_vertiports:  # avoid generating the same vertiport for both
                continue

            if small_aircraft:
                subgraph = self.filter_graph_detailed(origin_vertiports, destination_vertiports)
            else:
                subgraph = self.filter_graph_resolution(origin_vertiports, destination_vertiports)

            # find all path from origin to destination vertiport
            paths = nx.all_shortest_paths(subgraph, source=origin_vertiports, target=destination_vertiports)
            paths = list(paths)  # convert generator to list
            # print(f'availabel paths: {len(paths)}')
            if len(paths) > 10:
                paths = random.sample(paths, 10)  # random keep 10 paths for this generation episode

            for j in range(len(paths)):
                path_j = paths[j]
                node_num = len(path_j)

                if small_aircraft:
                    current_aircraft = self.small_aircraft[random.randrange(len(self.small_aircraft))]
                else:
                    current_aircraft = self.large_aircraft[random.randrange(len(self.large_aircraft))]

                time_slots = []

                for _ in range(5):  # try 5 times to generate with current size and origin-destination paths
                    payload = np.random.choice(['hover_t_empty', 'hover_t_full'], p=ef.ravel())  # empty or full payload
                    variable_speed = np.random.choice([True, False], p=p.ravel())  # random change in each loop
                    # speed for each node in this path
                    speed_arr, mu_sigma = self.generate_speed_profile(variable_speed, node_num)
                    # assign time for each node in this path
                    time_slots = self.assign_time(path_j, speed_arr, current_aircraft, payload)

                    if time_slots:
                        break

                # after several attempts, abandon current path generation and try next path
                if not time_slots or \
                        time_slots[-1] - time_slots[0] >= self.ap_database[current_aircraft]["envelop"][payload] * 60:
                    break
                # print(current_aircraft, payload, time_slots[-1] - time_slots[0],
                #       self.ap_database[current_aircraft]["envelop"][payload] * 60)
                # double check to make sure: en-route flight time < max hovering time
                assert time_slots[-1] - time_slots[0] < self.ap_database[current_aircraft]["envelop"][payload] * 60

                # append aircraft information only if above conditions are promised.
                self.aircraft_names.append(current_aircraft)
                self.payload_status.append(payload)
                self.wind_resistance.append(self.ap_database[current_aircraft]["envelop"]["resist_max"])
                # ==================

                for idx, node_name in enumerate(path_j):

                    if speed_arr[idx] != 0 or time_slots[idx] != 0:
                        if small_aircraft:
                            # for constructing sparse matrix
                            self.f_index.append(num)
                            self.s_index.append(self.G.nodes[node_name]['index'])
                            self.speed_data.append(speed_arr[idx])
                            self.schedule_data.append(time_slots[idx])
                            self.capacity.append(0)
                        else:
                            # large aircraft information
                            self.large_f_index.append(num)
                            self.large_s_name.append(node_name)
                            self.large_speed_data.append(speed_arr[idx])
                            self.large_schedule_data.append(time_slots[idx])

                            # ==== convert information in low resolution blocks to corresponding detailed blocks ===

                            subnodes = self.G_res1.nodes[node_name]['subnodes']
                            # print(f'{num} {current_aircraft} {payload} the node {node_name} consists of : {subnodes}')

                            num_list = [num] * len(subnodes)
                            s_list = [self.G.nodes[subnode_name]['index'] for subnode_name in subnodes]

                            speed_list = [speed_arr[idx]] * len(subnodes)
                            schedule_list = [time_slots[idx]] * len(subnodes)
                            capacity_list = [0] * len(subnodes)  # 0 is normal; 1 is close

                            self.f_index += num_list  # append list to current list
                            self.s_index += s_list
                            self.speed_data += speed_list
                            self.schedule_data += schedule_list
                            self.capacity += capacity_list

                self.speed_mu_sigma.append(mu_sigma)
                num += 1

                # break inner loop
                if num >= self.flights_num:
                    break
            # break outer loop
            if num >= self.flights_num:
                break
        # sparse structure
        self.schedule_data = np.asarray(self.schedule_data)
        self.speed_data = np.asarray(self.speed_data)
        self.f_index = np.asarray(self.f_index)
        self.s_index = np.asarray(self.s_index)
        self.speed_mu_sigma = np.asarray(self.speed_mu_sigma)  # dim = traj num * 4

        return self.schedule_data, self.speed_data, \
               self.f_index, self.s_index, self.speed_mu_sigma, \
               self.aircraft_names, self.payload_status, \
               self.large_f_index, self.large_s_name, \
               self.large_speed_data, self.large_schedule_data, \
               self.wind_resistance, self.capacity

    def entry_count(self, schedule_data, speed_data, f_index, s_index, airblocks_num, snapshot_time, capacity,
                    wind_resistance):
        """

        :param schedule_data:
        :param speed_data:
        :param f_index:
        :param s_index:
        :param airblocks_num:
        :param snapshot_time:
        :param wind_uncertainty: wind profile of each sector in the 24 hours; at strategic they share one file;
        :return:
        """

        # MemoryError: Unable to allocate 189. GiB for an array with shape (17280, 1000, 1470)
        # T_F_S = np.zeros([self.snapshot_num, self.flights_num, self.airblocks_num])

        hotspots = []
        hotspots_t = []
        traversing_f = []

        for s in range(airblocks_num):
            # (1) get schedule values in this airblock
            index_list = np.where(s_index == s)
            # and filter the values in this airblock from the list
            schedule_in_s = schedule_data[index_list]
            speed_in_s = speed_data[index_list]
            f_in_s = f_index[index_list]

            # (2) check the pairwise values less than certain bound to detect possible conflict
            if list(schedule_in_s) and list(schedule_in_s)[
                0] > 0:  # if not empty [at leat one flight traversed this sector]
                # expand the dimension by adding zero row since the kdtree requires 2-d input
                schedule_in_s_ex = list(np.vstack((schedule_in_s, np.zeros(schedule_in_s.shape))).T)
                tree = cKDTree(schedule_in_s_ex)
                # check pairwise distance ; if the time difference is less than 2* 5s, the two flights has
                # very high possibility in this airblock at the same snapshot.
                pairs = tree.query_pairs(2 * snapshot_time, p=2)
                pairs = list(pairs)
                if pairs:  # conflict detected in this airblock
                    for pair in pairs:
                        snapshot_id0 = int(schedule_in_s[pair[0]] // snapshot_time)
                        snapshot_id1 = int(schedule_in_s[pair[1]] // snapshot_time)

                        if snapshot_id0 == snapshot_id1:  # hotspot
                            # get hospot number and traversing flights id
                            hotspots.append(s)
                            hotspots_t.append(snapshot_id0)
                            traversing_f.append(f_in_s[pair[0]])
                            traversing_f.append(f_in_s[pair[1]])
                            # print(f's{s}, t{snapshot_id0}: {f_in_s[pair[0]]} and f{f_in_s[pair[1]]} conflicted')

                # in addition to those flight in hotspot, the flight traversing max wind areas will also be recorded;
                # get the flight and its schedule traversing this block; compare the wind at its traversing time with
                # its max wind resistance speed.
                if self.phase == 'strategic' and self.wind_uncertainty:
                    # print(f'schedule_in_s:{schedule_in_s}')
                    for i, t in enumerate(schedule_in_s):
                        if t > 0 and self.wind_profile[int(t // snapshot_time), 4] > wind_resistance[f_in_s[i]]:
                            # print(self.wind_profile[int(t // self.snapshot_time), 4], wind_resistance[f_in_s[i]])
                            traversing_f.append(f_in_s[i])

                            # revise the capacity of this flight in global schedule
                            f_ = np.where(f_index == f_in_s[i])  # get f list of this flight from global schedule
                            s_ = s_index[f_]  # get s list of this flight from global schedule
                            idx_ = np.where(s_ == s)[0]  # find the index of this s in the schedule list
                            capacity_ = capacity[f_]
                            capacity_[idx_] = 1  # revise the capacity
                            capacity[f_] = capacity_

                            # ========== regard wind effect flight as conflict =================
                            hotspots.append(s)
                            hotspots_t.append(int(t // snapshot_time))

                if self.phase == 'pretactical' and self.wind_uncertainty:
                    p = int(self.block_belong2p[s])
                    for i, t in enumerate(schedule_in_s):
                        index = int((t - self.operation_start_t) // snapshot_time)
                        if t > self.operation_end_t:
                            continue
                        if t > 0 and self.wind_profiles[p][index] > wind_resistance[f_in_s[i]]:
                            traversing_f.append(f_in_s[i])

                            # revise the capacity of this flight in global schedule
                            f_ = np.where(f_index == f_in_s[i])  # get f list of this flight from global schedule
                            s_ = s_index[f_]  # get s list of this flight from global schedule
                            idx_ = np.where(s_ == s)[0]  # find the index of this s in the schedule list
                            capacity_ = capacity[f_]
                            capacity_[idx_] = 1  # revise the capacity
                            capacity[f_] = capacity_

                            # ========== regard wind effect flight as conflict =================
                            hotspots.append(s)
                            hotspots_t.append(int(t // snapshot_time))

        hotspots_mtx = np.vstack((np.asarray(hotspots), np.asarray(hotspots_t))).T

        # remove repeated index
        traversing_f = list(set(traversing_f))
        # print([336, 4596] in hotspots_comb.tolist())

        return hotspots, hotspots_t, hotspots_mtx, traversing_f, capacity


if __name__ == "__main__":
    flights_num = 50
    wind_uncertainty = False

    g = FlightsGenerator(flights_num, phase='strategic', wind_uncertainty=wind_uncertainty)
    """generate new schedule"""
    t0 = time.time()
    # schedule_data, speed_data, f_index, s_index, speed_mu_sigma, aircraft_names, payload_status, \
    # large_f_index, large_s_name, large_speed_data, large_schedule_data, wind_resistance, capacity = g.generate_flights()
    # # save generated sparse data
    # np.savez_compressed(f'sparse_flight_schedule(wind_{wind_uncertainty})_test{flights_num}.npz',
    #                     schedule_data=schedule_data,
    #                     speed_data=speed_data,
    #                     f_index=f_index,
    #                     s_index=s_index,
    #                     speed_mu_sigma=speed_mu_sigma,
    #                     aircraft_names=aircraft_names,
    #                     payload_status=payload_status,
    #                     large_f_index=large_f_index,
    #                     large_s_name=large_s_name,
    #                     large_speed_data=large_speed_data,
    #                     large_schedule_data=large_schedule_data,
    #                     wind_resistance=wind_resistance,
    #                     capacity=capacity)

    """load existed schedule"""

    sparse_flight_schedule = np.load(f'sparse_flight_schedule(wind_{wind_uncertainty})_test{flights_num}.npz')
    schedule_data = sparse_flight_schedule['schedule_data']
    speed_data = sparse_flight_schedule['speed_data']
    f_index = sparse_flight_schedule['f_index']
    s_index = sparse_flight_schedule['s_index']
    speed_mu_sigma = sparse_flight_schedule['speed_mu_sigma']
    aircraft_names = sparse_flight_schedule['aircraft_names']
    payload_status = sparse_flight_schedule['payload_status']
    large_f_index = sparse_flight_schedule['large_f_index']
    large_s_name = sparse_flight_schedule['large_s_name']
    large_speed_data = sparse_flight_schedule['large_speed_data']
    large_schedule_data = sparse_flight_schedule['large_schedule_data']
    wind_resistance = sparse_flight_schedule['wind_resistance']
    capacity = sparse_flight_schedule['capacity']

    hotspots, hotspots_t, hotspots_mtx, traversing_f, capacity = g.entry_count(schedule_data, speed_data, f_index,
                                                                               s_index,
                                                                               g.airblocks_num,
                                                                               g.snapshot_time,
                                                                               capacity,
                                                                               wind_resistance)
    print(f'{time.time() - t0:.2f}s consumed!')
    print(f'{len(hotspots)}/{g.snapshot_num} spatio-temporal conflicts')
    print(f'{len(traversing_f)}/{flights_num} flights involved')

    # for agent in range(flights_num):
    #     current_type = aircraft_names[agent]
    #     current_payload = payload_status[agent]
    #     if current_type in g.large_aircraft:
    #         agent_ = np.where(f_index == agent)  # get flight index in the sparse schedule; tuple
    #         speed_agent = speed_data[agent_]  # get speed profile of this flight
    #         schedule_agent = schedule_data[agent_]  # get schedule of this flight
    #         print(agent, speed_agent)
