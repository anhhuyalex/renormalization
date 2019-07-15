import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import colors
from copy import deepcopy

class IsingModel:
    def __init__(self, size, T = 1, J = 1, h = 0):
        self.size = size # size of lattice
        self.T = T # k_B * temperature (default 1)
        self.J = J # strength of interaction 
        self.h = h # strength of magnetic field
    
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
                ## at this point, we do a BFS search to create the cluster
                bonded, clusters, visited = self.SW_BFS(bonded, clusters, [i,j], beta, nearest_neighbors=1);
        
        
        for cluster_index in clusters.keys():
            [x0, y0] = np.unravel_index(cluster_index, (Nx,Ny));
            r = np.random.rand();
            if(r < 0.5):
                for coords in clusters[cluster_index]:
                    [x,y] = coords;
                    #print(Lattice[x,y], end=', '); #check clusters
                    self.state[x,y] = -1*self.state[x,y];

        return self.state;
    
    def update_SW(self, steps = 30):
        """
        Runs some steps of the Swendsen Wang algorithm
        """
        for _ in range(steps):
            self.run_cluster_epoch()
        

for temp in [1, 2.269185, 5]:
    data = []
    for _ in range(10000):
        i = IsingModel((81), 2.269185)
        i.initialize()
        for _ in range(10):
            i.update_SW(1)
        data.append(i.state)
    data = np.array(data)
    np.save("ising81x81_temp_{}.npy".format(temp), data)