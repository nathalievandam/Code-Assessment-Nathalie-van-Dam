# -*- coding: utf-8 -*-
"""
CODE ASSESSMENT FINAL - Nathalie van Dam
"""

# Prevent printing of warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%% Import libraries

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import islice
import string

#%% Define variables

n_tags = 2  # Define number of tags (In this case 2: A & B)
distance_max = 1.5  # Define the maximum distance in [m], in this case 1.5
mae_max = 0.1  # Define the allowed mean absolute error between distance measured by tags and distance determined from beacon data in [m]
dev_max = 0.2  # Define the allowed maximum difference between distance measured by tags and distance determined from beacon data in [m]

# Choose which two tags to evaluate (in this case just A & B)
# and the time instance at which to plot markers for the tag positions (in this case starting point so at t = 0)

tag1 = 'A'
tag2 = 'B'
time_p = 0

#%% Read and clean tag data and tag position data measured by beacon from csv files

def read_clean_tagdata(n_tags):
    
    tags_list = list(islice(string.ascii_uppercase, n_tags))
        
    tag = {}
    for i in tags_list:
        
        # Create tag dictionary where you can reach the data of tag A using: tag[A], etc.
        filename = f'tag{i}.csv'
        tag[i] = pd.read_csv(filename)  
        
        # Remove units in column names
        tag[i].columns = tag[i].columns.str.replace(r'\s*\[.*\]', '', regex=True) 
            
    return tag, tags_list

def read_clean_positiondata(tags_list):
    
    position = pd.read_csv('position.csv')

    # Remove units in column names
    position.columns = position.columns.str.replace(r'\s*\[.*\]', '', regex=True) 

    # Label tagID correctly in position dataframe (A, B, etc. for each time instance instead of all A)
    unique_times = position['Time'].unique()
    for time in unique_times:
        position.loc[position['Time'] == time, 'TagID'] = tags_list
        
    return position


tag, tags_list = read_clean_tagdata(n_tags)  

position = read_clean_positiondata(tags_list)


#%% Check if tag distances correct

# Calculate distances between two tags from position data measured by beacon
    
x = {}
y = {}
for i in tags_list:
    x[i] = position[position['TagID'] == f'{i}']['x'].reset_index(drop=True)
    y[i] = position[position['TagID'] == f'{i}']['y'].reset_index(drop=True)

distance_beacon = np.sqrt((x[f'{tag1}']-x[f'{tag2}'])**2 + (y[f'{tag1}'] - y[f'{tag2}'])**2)


# Select distance between tag 1 & 2 (in this case A & B), measured by tag 1

distance_tag = tag[tag1]['Distance'][tag[tag1]['ContactID'] == tag2]
        
#%% Determine reliability of sensors by comparing distance measured by tags with distance from position data

# Calculate mean absolute error between two tags, and between tags and beacon

mae_tag1_tag2 = abs(tag[tag1]['Distance'] - tag[tag2]['Distance']).mean()
mae_tag1_Beacon = abs(tag[tag1]['Distance'].values - distance_beacon.values).mean()


# Calculate max difference between tag and beacon distance

largestdev_tag1_Beacon = abs(tag[tag1]['Distance'].values - distance_beacon.values).max()


# Print mean absolute difference

print(f'The mean absolute difference between distance measured by tag {tag1} and {tag2} = {mae_tag1_tag2} [m]')
print(f'The mean absolute difference between distance measured by tag {tag1} and beacon = {mae_tag1_Beacon} [m]')
print(f'The largest deviation between distance measured by tag {tag1} and beacon = {largestdev_tag1_Beacon} [m]')


# Print whether reliable, based on defined maximum allowed difference

if (mae_tag1_Beacon < mae_max) & (largestdev_tag1_Beacon < dev_max):
    print(f'Sensor data is reliable (absolute mean difference < {mae_max} [m] and largest deviation < {dev_max} [m])')


#%% Calculate contact time between two tags

contact_points = distance_tag[distance_tag < distance_max]
time_step = tag[tag1]['Time'][1] - tag[tag1]['Time'][0]
contact_time = len(contact_points)*time_step

print(f'The contact time was {contact_time} [s] in total')


#%% Plot

# Plot positions between taggs
plt.figure()
sns.lineplot(data = position[position['TagID'] == tag1], x = 'x', y='y', sort = False, label = f'Position tag {tag1}')
sns.lineplot(data = position[position['TagID'] == tag2],  x = 'x', y='y', sort = False, label= f'Position tag {tag2}')


# Plot positions where distance < distance_max
condition = distance_tag < distance_max
sns.lineplot(x = x[tag1][condition], y = y[tag1][condition], sort = False, label = f'Tag {tag1} when distance <1.5 m', color = sns.color_palette()[3])
sns.lineplot(x = x[tag2][condition], y = y[tag2][condition], sort = False, label = f'Tag {tag2} when distance <1.5 m', color = sns.color_palette()[2])


# Plot position at time time_p
plt.plot(position.loc[(position['TagID'] == tag1) & (position['Time'] == time_p), 'x'].values, position.loc[(position['TagID'] == tag1) & (position['Time'] == time_p), 'y'].values, '^', markersize = 10, label = f'Position {tag1} at time = {time_p} [s]', color = sns.color_palette()[0])
plt.plot(position.loc[(position['TagID'] == tag2) & (position['Time'] == time_p), 'x'].values, position.loc[(position['TagID'] == tag2) & (position['Time'] == time_p), 'y'].values, '^', markersize = 10, label = f'Position point {tag2} at time = {time_p} [s]', color = sns.color_palette()[1])


# Plot characteristics
plt.title('Movement of tags in room')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

