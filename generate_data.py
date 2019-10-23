import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import colors
from copy import deepcopy
from orderedset import OrderedSet
from collections import defaultdict
import ray
import time

class IsingModel:
    def __init__(self, size, T = 1, J = 1, h = 0):
        self.size = size # size of lattice
        self.T = T # k_B * temperature (default 1)
        self.J = J # strength of interaction
        self.h = h # strength of magnetic field

        self.num_spin_updates = 0

    def initialize(self):
        self.state = np.random.choice([-1, 1], (self.size, self.size))


    def update_mh(self, steps = 10000):
        for _ in range(steps):
            r_ind, c_ind = np.random.choice(self.size, 2)

            energy = self.J * self.state[r_ind][c_ind]*(self.state[(r_ind-1)%self.size][c_ind] + \
                                                self.state[(r_ind+1)%self.size][c_ind] + \
                                                self.state[r_ind][(c_ind-1)%self.size] + \
                                                self.state[r_ind][(c_ind+1)%self.size])
            energy += -self.h * (self.state[r_ind][c_ind]) # generally absent

            prob = min(1, float(np.e**(-2*energy/self.T)))

            if np.random.random() < prob:
                self.state[r_ind][c_ind] *= -1


    def display(self):
        cmap = colors.ListedColormap(['blue', 'red'])
        bounds=[-1,0,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(self.state, interpolation='nearest', origin='lower',
                    cmap=cmap, norm=norm)
    def getNN(self, site_indices, site_ranges, num_NN):
        '''
            site_indices: [i,j], site to get NN of
            site_ranges: [Nx,Ny], boundaries of the grid
            num_NN: number of nearest neighbors, usually 1
            function which gets NN on any d dimensional cubic grid
            with a periodic boundary condition
        '''

        Nearest_Neighbors = list();
        for i in range(len(site_indices)):
            for j in range(-num_NN,num_NN+1): #of nearest neighbors to include
                if(j == 0): continue;
                NN = list(deepcopy(site_indices)); #don't want to overwite;
                NN[i] = (NN[i] + j)%(site_ranges[i]);
                Nearest_Neighbors.append(tuple(NN))
        return Nearest_Neighbors;

    def SW_BFS(self, bonded, clusters, start, beta, nearest_neighbors = 1):
        '''
        function currently cannot generalize to dimensions higher than 2...
        main idea is that we populate a lattice with clusters according to SW using a BFS from a root coord
        :param lattice: lattice
        :param bonded: 1 or 0, indicates whether a site has been assigned to a cluster
               or not
        :param clusters: dictionary containing all existing clusters, keys are an integer
                denoting natural index of root of cluster
        :param start: root node of graph (x,y)
        :param beta: temperature
        :param J: strength of lattice coupling
        :param nearest_neighbors: number or NN to probe
        :return:
        '''
        N = self.state.shape;
        visited = np.zeros(N); #indexes whether we have visited nodes during
                                     #this particular BFS search
        if(bonded[tuple(start)] != 0): #cannot construct a cluster from this site
            return bonded, clusters, visited;

        p = 1 - np.exp(-2 * beta * self.J); #bond forming probability

        queue = list();


        queue.append(start);
        index = tuple(start)
        clusters[index] = [index];
        cluster_spin = self.state[index]
        color = np.max(bonded) + 1;

        ## need to make sub2ind work in arbitrary dimensions

        #whatever the input coordinates are
        while(len(queue) > 0):
            #print(queue)
            r = tuple(queue.pop(0));
            ##print(x,y)
            if(visited[r] == 0): #if not visited
                visited[r] = 1;
                #to see clusters, always use different numbers
                bonded[r] = color;
                NN = self.getNN(r,N, nearest_neighbors);
                for nn_coords in NN:
                    rn = tuple(nn_coords);
                    if(self.state[rn] == cluster_spin and bonded[rn] == 0\
                       and visited[rn] == 0): #require spins to be aligned
                        random = np.random.rand();
                        if (random < p):  # accept bond proposal
                            queue.append(rn); #add coordinate to search
                            clusters[index].append(rn) #add point to the cluster
                            bonded[rn] = color; #indicate site is no longer available


        return bonded, clusters, visited;

    def run_cluster_epoch(self, nearest_neighbors = 1):
        """
        Implements 1 step of the Swendsen Wang algorithm
        """
        #simulation parameters
        beta = 1.0 / self.T
        Nx, Ny = self.state.shape

        #scan through every element of the lattice

        #propose a random lattice site to generate a cluster
        bonded = np.zeros((Nx,Ny));
        clusters = dict();  # keep track of bonds
        ## iterate through the entire lattice to assign bonds
        ## and clusters
        for i in range(Nx):
            for j in range(Ny):
                print("Nx, Ny", i, j)
                ## at this point, we do a BFS search to create the cluster
                bonded, clusters, visited = self.SW_BFS(bonded, clusters, [i,j], beta, nearest_neighbors=1);

        print("finish with SW_BFS!")
        for cluster_index in clusters.keys():
            [x0, y0] = np.unravel_index(cluster_index, (Nx,Ny));
            r = np.random.rand();
            if(r < 0.5):
                for coords in clusters[cluster_index]:
                    [x,y] = coords;
                    #print(Lattice[x,y], end=', '); #check clusters
                    self.state[x,y] = -1*self.state[x,y];

        return self.state;

    def update_SW(self, steps = 1):
        """
        Runs some steps of the Swendsen Wang algorithm
        """
        for _ in range(steps):
            self.run_cluster_epoch()

    def update_wolff_step(self):
        """
        Runs some steps of the Wolff algorithm
        """
        beta = 1 / self.T
        #Choose random site i.
        x, y = np.random.randint(self.size, size=2)
        s_i = self.state[x, y]
        cluster = {(x, y)} #indices of cluster (implemented as a set)
        # cluster = OrderedSet([(x, y)])
        queue = [(x, y)]
        visited = defaultdict(int)

        while len(queue) > 0: # Repeat for all elements added to the cluster
            x, y = queue.pop()
            # print("len visited", len(visited))
            # Check neighboring sites
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    x_prime = (x + dx)%(self.size)
                    y_prime = (y + dy)%(self.size)
                    if ((x_prime, y_prime) not in cluster) & (visited[(x_prime, y_prime)] == 0) & (x_prime != 0 or y_prime != 0):
                        candidate = self.state[x_prime, y_prime]
                        if candidate == s_i:
                            rand = np.random.rand()
                            if rand < 1 - np.exp(-2 * beta): # join site j to cluster with probability p = 1 − exp(−2β).
                                cluster.add((x_prime, y_prime))
                                queue.append((x_prime, y_prime))
            visited[(x, y)] = 1
        # invert all spins in the cluster
        # cluster = list(cluster)
        # print("cluster", cluster)
        # print(tuple(zip(cluster)))
        # print("cluster", cluster)
        # print("zipcluster", tuple(zip(*cluster)))
        self.state[tuple(zip(*cluster))] *= -1
        # for x, y in cluster:
        #     self.state[x, y] *= -1

        return cluster


    def update_wolff(self, steps = 1):
        """
        Runs some steps of the Swendsen Wang algorithm
        """
        for _ in range(steps):
            cluster = self.update_wolff_step()
            print(len(cluster))

            self.num_spin_updates += len(cluster)

    def save(self):
        np.save("ising_samples/ising6561x6561_temp_{}_sample{}.npy".format(2.269185, time.time()), self.state)
        self.num_spin_updates = 0

ray.init()
num_times = 1
num_workers = 4
N = 2187

@ray.remote
def generate_data(lattice_file):
    i = IsingModel(size = N, T = 2.269185)
    # i.state = np.load(lattice_file)
    i.initialize()
    i.update_wolff(1000)
    for _ in range(100000):
        i.update_wolff()
        if i.num_spin_updates >= N ** 2:
            i.save()

    return i.state

# i = generate_data()
# np.random.randint(100, size=2)

# Start Ray.
# ray.init()

# @ray.remote
# def generate_data():
#     i = IsingModel(size = 6561, T = 2.269185)
#     i.initialize()
#     i.update_SW(steps = 1)
#     return i.state

import time
tic = time.time()
data = []
for _ in range(num_times):
    result_ids = []
    lattice_files = ["ising_samples/ising6561x6561_temp_2.269185_sample1571857139.575895.npy",
                     "ising_samples/ising6561x6561_temp_2.269185_sample1571854227.0954611.npy",
                     "ising_samples/ising6561x6561_temp_2.269185_sample1571854112.154505.npy",
                     "ising_samples/ising6561x6561_temp_2.269185_sample1571854084.229632.npy"]
    for seed in range(num_workers):
        result_ids.append(generate_data.remote(lattice_files[seed]))
    data.append(ray.get(result_ids))
# data = np.array(data)
# toc = time.time()
# np.set_printoptions(threshold=10000)
# print(data)
# print("shape", data.shape)
# print("time", toc-tic)
# i[:100, :100]
# import matplotlib.pyplot as plt
# plt.imshow(i)
# plt.show()
# np.save("ising_samples/ising6561x6561_temp_{}.npy".format(2.269185), data)
ray.shutdown()
