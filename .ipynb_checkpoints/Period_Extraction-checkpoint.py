import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits import mplot3d
import math
from sklearn.cluster import KMeans
from ripser import Rips
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import ruptures as rpt
import gudhi as gd
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from mpi4py import MPI
from Period_Functions import *

print("Imports Complete!")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#From Data_Prep.py
n = 68
fix_seasons = pd.read_csv(r"~/TDA/Data/2004_2012/Fix_Seasons_LaGrange_68.csv")
years = [x for x in range(2004,2012)]
seasons = ["SPRING", "SUMMER", "FALL", "WINTER"]

col = "CHLcal"
time_series = np.array(fix_seasons[col])
total_window = len(time_series)
#total_window = 2000
time_step = 1.0/(4*n)
periods = []
collective_periods = np.zeros(total_window, dtype='double')

w = 12*n
dim = 8
d = int(n / 2)
n_cells = 6

for i in range(total_window):
    if i % size == rank:
        print("Rank: " + str(rank) + " " + str(i) + " / " + str(total_window))
        window = get_window_from_series(time_series, w, i)
        swe = sliding_window_embedding(window, dim, d)
        diagram = plot_persistent_homology_diagram(swe, False)
        if diagram == 0:
            periods.append(0)
            print("Not enough data to do SWE, returning 0")
            continue
        l1, l2 = calculate_norms(diagram)
        try:
            landmarks, cells = generate_voronoi_cells(swe, n_cells)
        except ValueError:
            periods.append(0)
            print("Failed to make Voronoi cells, returning 0")
            continue
        jumps = generate_vector_jumps(landmarks,cells)
        summary = Jump_Summary(jumps, 20)
        periods.append(estimate_period(summary, time_step))

#Gather periods
comm.Barrier()
if rank == 0:
    print("Period Analysis Complete!")

loc = 0
for i in range(total_window):
    r = i % size
    if r == 0:
        if rank == 0:
            collective_periods[i] = periods[loc]
            loc += 1
    else:
        if rank == 0:
            collective_periods[i] = comm.recv(source=r, tag=int(i / size))
        elif rank == r:
            comm.send(periods[loc], dest=0, tag=loc)
            loc += 1

graph_name = 'Graphs/2004_2012/LaGrange_' + col + '_w' + str(w) + '_n' + str(dim) + '_d' + str(d) + '.png'
if rank == 0:
    print("All Periods Received!")
    index = [i / (4.0 * n) for i in range(len(collective_periods))]
    plt.scatter(index,collective_periods)
    plt.xlabel('Years since 2004')
    plt.ylabel('Period (In Years)')
    plt.savefig(graph_name)
    print("Analysis Complete. ", graph_name, " successfully created.")