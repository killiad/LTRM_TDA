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

print("Imports complete")
print("Beginning Data Preperation")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

predicted_df = pd.read_csv(r"~/TDA/Data/water_full.csv")
time_df = pd.read_csv(r"~/TDA/Data/water_data_qfneg.csv")
predicted_df.insert(0,'TIME',time_df['TIME'],True)

years = [x for x in range(2004,2012)]
predicted_df = predicted_df[predicted_df['YEAR'].isin(years)]

#predicted_df = predicted_df[predicted_df['FLDNUM']=='Havana, IL']
predicted_df["MONTH"] = pd.DatetimeIndex(predicted_df["DATE"]).month
predicted_df["DATE"] = pd.to_datetime(predicted_df["DATE"])
predicted_df["SEASON"] = predicted_df["MONTH"]
seasons = {3 : 'SPRING',
           4 : 'SPRING',
           5 : 'SPRING',
           6 : 'SUMMER',
           7 : 'SUMMER',
           8 : 'SUMMER',
           9 : 'FALL',
           10 : 'FALL',
           11: 'FALL',
           12: 'WINTER',
           1: 'WINTER',
           2: 'WINTER'}
predicted_df = predicted_df.replace({"SEASON" : seasons})

stratum = {'Main channel' : 1, 'Side channel' : 2, 'Backwater area contiguous to the main channel' : 3}
predicted_df = predicted_df.replace({"STRATUM" : stratum})
predicted_df.sort_values(by=["DATE","TIME"], inplace=True)

date = []
for index, row in predicted_df.iterrows():
    time = row["TIME"]
    time = time.split(":")
    h = int(time[0])
    m = int(time[1])
    date.append(row["DATE"] + pd.DateOffset(hours=h, minutes=m))
date = np.array(date)
predicted_df["DATE"] = date

#impute = predicted_df[predicted_df['YEAR']==2001]
#impute = impute[impute['SEASON']=='FALL']
#year = [2002 for x in range(len(impute))]
#year = np.array(year)
#impute.YEAR = year
#lst = []
#for index, row in impute['DATE'].items():
#    lst.append(row.replace(year=2002))
#lst = np.array(lst)
#impute.DATE = lst
#predicted_df = predicted_df.append(impute)
#predicted_df.sort_values(by=["DATE","TIME"], inplace=True)

n = 540
years = []
#years = [x for x in range(1997,2003)]
years += [x for x in range(2004,2012)]
seasons = ["SPRING", "SUMMER", "FALL", "WINTER"]
run_test = False

if run_test:
    for year in years:
        for season in seasons:
            test = predicted_df[predicted_df['YEAR']==year]
            test = test[test['SEASON']==season]
            if(len(test) < n):
                print("Year ", year, " season ", season, " has ", len(test), " points!")

columns = {x : [] for x in predicted_df.columns}
fix_seasons = pd.DataFrame(columns)
for year in years:
    year_df = predicted_df[predicted_df["YEAR"]==year]
    for season in seasons:
        seasonal_df = year_df[year_df["SEASON"]==season]
        counter = 0
        for index, row in seasonal_df.iterrows():
            if counter >= n:
                break
            fix_seasons.loc[len(fix_seasons.index)] = row
            counter += 1

print("Data Preperation Complete")
print("Node ", rank, " will begin periodicity analysis")

#will use a window of size n*4
w = n*3
dim = 30
d = 6
time_series = np.array(fix_seasons["TEMP"])
total_window = len(time_series)
collective_periods = np.zeros(total_window, dtype='double')
time_step = 1.0/(4*n)
periods = []
for i in range(total_window):
    if(i % size != rank):
        continue

    print("Rank: ", rank, "\tStep: ", i, "/", total_window)
    window = get_window_from_series(time_series, w, i)
    swe = sliding_window_embedding(window, dim, d)
    diagram = plot_persistent_homology_diagram(swe, False)
    if diagram == 0:
        periods.append(0)
        print("Not enough data to do SWE, returning 0")
        continue
    l1, l2 = calculate_norms(diagram)
    try:
        landmarks, cells = generate_voronoi_cells(swe, dim)
    except ValueError:
        periods.append(0)
        print("Failed to make Voronoi cells, returning 0")
        continue
    jumps = generate_vector_jumps(landmarks,cells)
    summary = Jump_Summary(jumps, 20)
    periods.append(estimate_period(summary, time_step))

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

graph_name = 'Graphs/TEMP_w' + str(w) + '_s' + str(s) + '_n' + str(dim) + '_d' + str(d) + '.png'
if rank == 0:
    print("All Periods Received!")
    index = [i / (4.0 * n) for i in range(len(collective_periods))]
    plt.scatter(index,collective_periods)
    plt.xlabel('Years since 2004')
    plt.ylabel('Period (In Years)')
    plt.savefig(graph_name)
    print("Analysis Complete. ", graph_name, " successfully created.")