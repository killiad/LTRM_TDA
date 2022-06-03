#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
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

#Takes in jumps in a Voronoi cell, along with a threshold for what a dominant jump is, and calculates
#relevant statistics and the predicted period, if possible
class Jump_Summary:
    def __init__(self, jumps, epsilon):
        self.jumps = jumps
        self.epsilon = epsilon
        self.small_jump_totals = {i : [] for i in range(len(jumps))}
        self.large_jumps = {i : [] for i in range(len(jumps))}
        self.small_jump_totals_variance = []
        self.large_jumps_average = []
        self.period_average = 0
        self.period_variance = 0
        
        for landmark in range(len(jumps)):
            i = 0
            jump_lst = jumps[landmark]
            for jump in jump_lst:
                if jump < epsilon:
                    i += 1
                else:
                    self.large_jumps[landmark].append(jump)
                    self.small_jump_totals[landmark].append(i)
                    i = 0
            if len(self.large_jumps[landmark]) == 0:
                continue
            try:
                avg = sum(self.small_jump_totals[landmark]) / len(self.small_jump_totals[landmark])
                self.small_jump_totals_variance.append(sum((x-avg)**2 for x in self.small_jump_totals[landmark]) / len(self.small_jump_totals[landmark]))
            except:
                print("No small jumps for Landmark ", landmark)
            self.large_jumps_average.append(sum(self.large_jumps[landmark]) / len(self.large_jumps[landmark]))
        try:
            self.period_average = sum(self.large_jumps_average) / len(self.large_jumps_average)
            self.period_variance = sum((x-self.period_average)**2 for x in self.large_jumps_average) / len(self.large_jumps_average)
        except:
            print("Not enough data to get period estimate, returning 0")
            self.period_average = 0
            self.period_variance = 0
    
    def print_summary(self):
        print("Epsilon: ", self.epsilon)
        print("Number of Landmarks: ", len(self.jumps))
        print("")
        for i in range(len(self.jumps)):
            try:
                avg = sum(self.small_jump_totals[i]) / len(self.small_jump_totals[i])
                lavg = sum(self.large_jumps[i]) / len(self.large_jumps[i])
                print("Landmark ", i, " Small Jump Totals: ", self.small_jump_totals[i])
                print("Landmark ", i, " Small Jump Totals Average: ", avg)
                print("Landmark ", i, " Small Jump Totals Variance: ", 
                    sum((x-avg)**2 for x in self.small_jump_totals[i]) / len(self.small_jump_totals[i]))
                print("Landmark ", i, " Large Jumps: ", self.large_jumps[i])
                print("Landmark ", i, " Large Jumps Average: ", lavg)
                print("Landmark ", i, " Large Jump Variance: ", sum((x-lavg)**2 for x in self.large_jumps[i]) / len(self.large_jumps[i]))
                print("")
            except:
                print("Not enough data for Landmark ", i)
                print("")
        print("Average Period (in steps): ", self.period_average)
        print("Period (in steps) Variance: ", self.period_variance)

#Turns a time series into a specific subset of the data
def get_window_from_series(time_series, m, end):
    window = time_series[max([0,end-m]):end+1]
    return window

#Turns a time series into a collection of windows  to slide along
def sliding_window_embedding(time_series, n, d):
    size = len(time_series)
    swe = [0 for x in range(size - n*d)]
    for i in range(len(swe)):
        swe[i] = [time_series[i + k*d] for k in range(n+1)]
    return np.array(swe)

#If possible, makes a persistence diagram from the data
def plot_persistent_homology_diagram(data, plot=True):
    try:
        rips = Rips(verbose = False)
        diagrams = rips.fit_transform(data)
        if plot:
            rips.plot(diagrams)
        return diagrams
    except:
        return 0

#Calculates the l1 and l2 norms of the persistence diagram
def calculate_norms(diagram, dimension=1):
    l1 = 0
    l2 = 0
    for entry in diagram[dimension]:
        l1 = l1 + abs(entry[0]) + abs(entry[1])
        l2 = l2 + entry[0]*entry[0]+entry[1]*entry[1]
    l2 = math.sqrt(l2)
    return[l1,l2]

def dist(p1, p2):
    lst = [(p1[i] - p2[i])**2 for i in range(len(p1))]
    return math.sqrt(sum(lst))

#Selects n equally spaced apart points to be center points in the swe and assigns each point  in the SWE
#to a corresponding Voronoi cell
def generate_voronoi_cells(swe, num_cells):
    kmeans = KMeans(n_clusters=num_cells).fit(swe)
    landmarks = kmeans.cluster_centers_
    cells = {i : [] for i in range(len(landmarks))}
    time_index = 0
    for point in swe:
        distances = [dist(point,landmark) for landmark in landmarks]
        cell = np.argmin(distances)
        cells[cell].append(time_index)
        time_index += 1
    return landmarks, cells

#Gets the distance between time series indicies in a Voronoi cell
def generate_vector_jumps(landmarks,cells):
    j = {i : [] for i in range(len(landmarks))}
    for landmark in range(len(landmarks)):
        t = cells[landmark]
        for index in range(1,len(t)):
            j[landmark].append(t[index]-t[index-1])
    return j

#Converts the estimated period into the proper time unit
def estimate_period(summary, time_step):
    return summary.period_average * time_step

#DEPRICATED
def voronoi_estimations(swe, epsilon, time_step, min_cells, max_cells):
    voronoi = []
    periods = []
    for cell_num in range(min_cells,max_cells+1):
        print("Vornoi Cell Number Amount: ", cell_num)
        landmarks, cells = generate_voronoi_cells(swe, cell_num)
        jumps = generate_vector_jumps(landmarks,cells)
        summary = Jump_Summary(jumps, epsilon)
        if summary.valid_summary:
            periods.append(estimate_period(summary, time_step))
            voronoi.append(cell_num)
    return voronoi, periods