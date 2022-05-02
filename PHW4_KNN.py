import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import math

# Read data
dataset = pd.read_csv('data/KNN_DATA.csv')
features = ['HEIGHT(cm)', 'WEIGHT(kg)']
target = ['T_SHIRT_SIZE']

# input user data
height = int(input("Enter your Height: "))
weight = int(input("Enter your Weight: "))
k = int(input("Enter K: "))


def scalingData(data):
    # input user data
    data.loc[len(data)] = (height, weight, None)

    # normalization
    weights = np.array(data.loc[:, ['WEIGHT(kg)']]).reshape(-1)
    heights = np.array(data.loc[:, ['HEIGHT(cm)']]).reshape(-1)

    stdscaler = StandardScaler()
    scaled_weights = stdscaler.fit_transform(weights[:, np.newaxis]).reshape(-1)
    scaled_heights = stdscaler.fit_transform(heights[:, np.newaxis]).reshape(-1)

    scaled_data = pd.DataFrame({
        'HEIGHT(cm)': scaled_heights,
        'WEIGHT(kg)': scaled_weights,
        'T_SHIRT_SIZE': data['T_SHIRT_SIZE']
    })

    return scaled_data


def calDistance(point1, point2):
    distance = 0

    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2

    # Return square root of distance
    return math.sqrt(distance)


def distance_rank(data, features, k):
    # User data
    test = [data[features[0]].iloc[-1], data[features[1]].iloc[-1]]

    # Calculate the distance between the entered value and the dataset
    distance = []
    for i, row in data.iterrows():
        origin = [row[features[0]], row[features[1]]]
        distance.append(calDistance(origin, test))

    # Add Distance column
    data['Distance'] = distance
    data.loc[len(data) - 1, 'Distance'] = None

    # Add Rank column
    data['Rank'] = data['Distance'].rank(method='min')

    # Exclude values beyond range 'k'
    data.loc[data['Rank'] > k, 'Rank'] = None

    return data


def calResult(data):
    k_data = data.loc[data['Rank'] <= k, :]
    k_data = k_data.reset_index()

    # count M, L
    m_count = 0
    l_count = 0
    i = 0
    for i in range(len(k_data)):
        if k_data.loc[i, 'T_SHIRT_SIZE'] == 'M':
            m_count = m_count + 1;
        else:
            l_count = l_count + 1;
    #print('M=', m_count)
    #print('L=', l_count)

    # Predict SIZE
    if m_count > l_count:
        size = 'M'
    else:
        size = 'L'

    # Print prediction value
    print('Predicted SIZE : ', size)



# Scale
scaled_data = scalingData(dataset)
print('\n==== Scaled Data =====\n', scaled_data)

# Calculate distance & rank
ranked_data = distance_rank(scaled_data, features, k)
print('\n\n==== Ranked Data ====\n', ranked_data)

# Result
print('\n\n==== Result ====')
calResult(ranked_data)
