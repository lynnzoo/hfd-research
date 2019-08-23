import numpy as np
import pandas as pd
# import googlemaps
import os
import re
from collections import Counter, defaultdict
# from gurobipy import *
from sklearn.cluster import KMeans


def split_by_vehicle_type(merged, save=True):
    ambulances = merged.loc[(merged['vehicletype'] == 'A')]
    medics = merged.loc[(merged['vehicletype'] == 'M')]
    engines = merged.loc[(merged['vehicletype'] == 'E')]
    ladders = merged.loc[(merged['vehicletype'] == 'L') | (merged['vehicletype'] == 'T')]
    if save is True:
        ambulances.to_csv('ambulances_forclusters.csv', index=False)
        medics.to_csv('medics_forclusters.csv', index=False)
        engines.to_csv('engines_forclusters.csv', index=False)
        ladders.to_csv('ladders_forclusters.csv', index=False)
    return ambulances, medics, engines, ladders


def run_kmeans(df, name, fraction=1, K=250, save=True, output=True):
    """
    Written by Lynn Zhu
    """
    coordsdf = df[['longitude', 'latitude']]
    if fraction != 1:
        samp = coordsdf.sample(frac=fraction, replace=False, random_state=123)
        kmeans = KMeans(init='k-means++', n_clusters=K, n_init=10, random_state=123).fit(samp)
    else:
        kmeans = KMeans(init='k-means++', n_clusters=K, n_init=10, random_state=123).fit(coordsdf)

    centers = kmeans.cluster_centers_
    centers_df = pd.DataFrame(centers, columns=['longitude', 'latitude'])
    centers_df['cluster'] = centers.index

    counts = Counter(kmeans.labels_)
    counts_df = pd.DataFrame.from_dict(counts, orient='index')
    counts_df['cluster'] = counts_df.index
    counts_df.columns = ['counts', 'cluster']

    final_df = centers_df.merge(counts_df, on='cluster')

    filename = name + '.csv'
    if save is True:
        final_df.to_csv(os.path.join(os.getcwd(), "data", filename), index=False)
    if output is True:
        return final_df, kmeans


def kmeans_predict(kmeans, predict_on, colname):
    """
    Written by Lynn Zhu
    :param kmeans: sklearn KMeans model
    :param predict_on: dataframe with latitude and longitude to predict on
    :param colname: column name of prediction
    :return: predict_on dataframe with new column, the prediction
    """
    predict_on[colname] = kmeans.predict(predict_on[["longitude", "latitude"]])
    return predict_on


def read_cluster_data(filename):
    """
    Written by: Lynn Zhu
    """
    column_names = ["longitude", "latitude", "cluster", "count"]
    df = pd.read_csv(filename, header=0,  names=column_names)
    return df


def read_station_data(station_filename, baysdf):
    # STATION: station number
    # NUM_DT: number of drive through bays
    # NUM_NDT: number of non-drive through bays
    # NUM_BAYS_UPDATED: total number of bays
    # NUM_AMBULANCES: capacity if all bays are used for ambulances
    # NUM_ENGINES: capacity if all bays are used for engines
    # NUM_AMBULANCE_DT: capacity if *only drive through* bays are used for ambulances
    # NUM_AMBULANCE_NDT: capacity if *only non-drive through* bays are used for ambulances
    station_names = ["Station Number", "Address", "Latitude", "Longitude", "Bays"]
    bay_names = ["Station Number", "DT Bays", "NDT Bays", "Total Bays", "Ambulances", "Engines",
                 "Ambulances DT", "Ambulances NDT"]
    stations = pd.read_csv(station_filename, header=0, names=station_names)
    baysdf.columns = bay_names
    final = pd.concat([stations, baysdf], axis=1)
    final.columns = ["Station Number", "Address", "Latitude", "Longitude", "Bays", "Station Number.1", "DT Bays",
                   "NDT Bays", "Total Bays", "Ambulances", "Engines", "Ambulances DT", "Ambulances NDT"]
    final.drop(['Station Number.1'], axis=1)
    final.to_csv(os.path.join(os.getcwd(), "data", "station_data.csv"), index=False)
    return final


def get_time_matrix(filename, demand, stations, apikey):
    gmaps = googlemaps.Client(key=apikey)
    num_stations = stations.shape[0]
    num_demand = demand.shape[0]
    # origins = {}
    # destinations = {}
    time_matrix = np.zeros((num_stations, num_demand))  # STATION = ROWS, DEMAND POINTS = COLUMN
    iteration = 0
    for index, row in stations.iterrows():
        for index2, row2 in demand.iterrows():
            iteration += 1
            station_address = (row['Latitude'], row['Longitude'])
            demand_address = (row2['Latitude'], row2['Longitude'])
            time = gmaps.distance_matrix(station_address, demand_address, mode='driving')['rows'][0]['elements'][0][
                'duration']['value']
            time_matrix[index, index2] = time
            if iteration % 500 == 0:
                print("Iteration: ", iteration)
    filename = os.path.join(os.getcwd(), "data", filename)
    np.savetxt(filename, time_matrix, delimiter=',')
    return time_matrix


def calculate_covering_set(distance_matrix, threshold, stations, demand):
    covering_set_idx = {}
    covering_set_num = {}
    for demand_idx, description in demand.iterrows():
        covering_idx = []
        covering_num = []
        for station_idx, station_d in stations.iterrows():
            if distance_matrix[station_idx, demand_idx] <= threshold * 60:
                covering_idx.append(station_idx)
                covering_num.append(station_d['Station Number'])

        covering_set_idx[demand_idx] = covering_idx
        covering_set_num[demand_idx] = covering_num

    return covering_set_idx, covering_set_num


# def calculate_station_covering_set(distance_matrix, threshold, stations, demand):
#     # num_stations = stations.shape[0]
#     # num_demand = demand.shape[0]
#
#     station_covering_set_idx = {}
#     station_covering_set_num = {}
#
#     for station_idx, station_d in stations.iterrows():
#         demand_covered = []
#         for demand_idx, demand_d in demand.iterrows():
#             if distance_matrix[station_idx, demand_idx] <= threshold * 60:
#                 demand_covered.append(demand_idx)
#         station_covering_set_idx[station_idx] = tuple(demand_covered)
#         station_num = station_d['Station Number']
#         station_covering_set_num[station_num] = demand_covered
#     return station_covering_set_idx, station_covering_set_num

#
# def MCLP(model_name, num_vehicles, demand, stations, covering_set, vehicle_type, solution_filename):
#     num_demand = demand.shape[0]
#     num_stations = stations.shape[0]
#
#     mod = Model(model_name)
#     d = {}  # binary variables for each demand point
#     s = {}  # binary variables for each station
#
#     for i in range(num_demand):
#         d[i] = mod.addVar(vtype=GRB.BINARY, name=model_name + " Demand Point " + str(demand.iloc[i, 0]))
#
#     for j in range(num_stations):
#         s[j] = mod.addVar(lb=0, ub=stations.loc[j, vehicle_type], vtype=GRB.INTEGER,
#                           name=model_name + " Station " + str(stations.iloc[j, 0]))
#     mod.update()
#
#     for i in range(num_demand):
#         mod.addConstr(quicksum(s[j] for j in covering_set[i]) >= d[i])
#     mod.addConstr(quicksum(s[j] for j in range(num_stations)) <= num_vehicles)
#
#     mod.setObjective(quicksum(demand.loc[i, 'count'] * d[i] for i in range(num_demand)), GRB.MAXIMIZE)
#
#     mod.update()
#     mod.optimize()
#     solution_filename = os.path.join(os.getcwd(), "data", solution_filename)
#     mod.write(solution_filename)
#     #     variables = mod.getVars()
#     return mod


def MEXCLP(model_name, busy_fraction, num_vehicles, demand, stations, covering_set, vehicle_type, solution_filename):
    num_demand = demand.shape[0]
    num_stations = stations.shape[0]

    MEX = Model(model_name)
    y = {}  # binary variables for each demand point
    x = {}  # binary variables for each station

    for i in range(num_demand):
        for k in range(num_vehicles):
            l = k + 1
            y[(i, k)] = MEX.addVar(vtype=GRB.BINARY, name="Demand Point " + str(
                demand.loc[i, 'cluster']) + " Covered %d Times" % l)

    for j in range(num_stations):
        x[j] = MEX.addVar(lb=0, ub=stations.loc[j, vehicle_type], vtype=GRB.INTEGER,
                          name="Station " + str(stations.iloc[j, 0]))
    MEX.update()

    for i in range(num_demand):
        MEX.addConstr(quicksum(x[j] for j in covering_set[i]) >= quicksum(y[(i, k)] for k in range(num_vehicles)))

    MEX.addConstr(quicksum(x[j] for j in range(num_stations)) <= num_vehicles)

    MEX.setObjective(quicksum(
        demand.loc[i, 'count'] * y[(i, k)] * (1 - busy_fraction) * (busy_fraction ** (k - 1)) for i in range(num_demand)
        for k in range(num_vehicles)), GRB.MAXIMIZE)

    MEX.update()
    MEX.optimize()
    solution_filename = os.path.join(os.getcwd(), "data", solution_filename)
    MEX.write(solution_filename)
    return MEX


def MEXCLP_diminishing(model_name, reward_matrix, busy_fraction, num_vehicles, demand, stations, covering_set,
                       vehicle_type, solution_filename):
    num_demand = demand.shape[0]
    num_stations = stations.shape[0]

    MEX_d = Model(model_name)
    y = {}  # binary variables for each demand point
    x = {}  # binary variables for each station

    for i in range(num_demand):
        for k in range(num_vehicles):
            l = k + 1
            y[(i, k)] = MEX_d.addVar(vtype=GRB.BINARY, name="Demand Point " + str(
                demand.loc[i, 'cluster']) + " Covered %d Times" % l)

    for j in range(num_stations):
        x[j] = MEX_d.addVar(lb=0, ub=stations.loc[j, vehicle_type], vtype=GRB.INTEGER,
                            name="Station " + str(stations.iloc[j, 0]))
    MEX_d.update()

    for i in range(num_demand):
        MEX_d.addConstr(quicksum(x[j] for j in covering_set[i]) >= quicksum(y[(i, k)] for k in range(num_vehicles)))

    MEX_d.addConstr(quicksum(x[j] for j in range(num_stations)) <= num_vehicles)

    MEX_d.setObjective(quicksum(
        reward_matrix[i, k] * y[(i, k)] * (1 - busy_fraction) * (busy_fraction ** (k - 1)) for i in range(num_demand)
        for k in range(num_vehicles)), GRB.MAXIMIZE)

    MEX_d.update()
    MEX_d.optimize()
    solution_filename = filename = os.path.join(os.getcwd(), "data", solution_filename)
    MEX_d.write(solution_filename)
    return MEX_d


def calc_MCLP_performance(model_object, demand, stations, filename1, filename2, filename3):
    model_variables = model_object.getVars()
    model_objective = model_object.objVal
    demand_dict = {}
    station_dict = {}
    num_demand = demand.shape[0]
    num_stations = stations.shape[0]
    num_covered = 0
    pop_covered = 0
    station_with = 0
    vehicles_used = 0
    total_pop = demand['count'].sum()
    for var in model_variables:
        if 'Demand Point' in var.varName:

            numbers = [int(s) for s in var.varName.split() if s.isdigit()]
            demand_dict[numbers[0]] = var.x
            if var.x > 0:
                num_covered += 1
                pop_covered += demand.loc[numbers[0], 'count']

        elif 'Station ' in var.varName:

            number_station = [int(s) for s in var.varName.split() if s.isdigit()]
            station_dict[number_station[0]] = var.x
            if var.x > 0:
                station_with += 1
                vehicles_used += var.x

    # print(pop_covered, total_pop)
    percent_pop_covered = (pop_covered / total_pop) *100
    percent_num_covered = (num_covered / num_demand) *100
    percent_station = (station_with / num_stations) *100

    filename1 = os.path.join(os.getcwd(), "data", filename1)
    filename2 = os.path.join(os.getcwd(), "data", filename2)
    pd.DataFrame.from_dict(data=demand_dict, orient='index').to_csv(filename1, header=False)
    pd.DataFrame.from_dict(data=station_dict, orient='index').to_csv(filename2, header=False)
    print("")
    print("MCLP Performance:")
    print("% of Incidents Covered 1 Time: ", percent_pop_covered, "%")
    print("")
    print("% of Demand Points Covered 1 Time: ", percent_num_covered, "%")
    print("")
    print("Percent of Stations with a Vehicle: ", percent_station, "%")
    print("")
    print("Number of Vehicles Used: ", vehicles_used,)
    print("")
    print("Objective Function: ", model_objective)
    print("")


def calc_MEXCLP_performance(model_object, stations, covering_set, demand, filename1, filename2, filename3):
    model_variables = model_object.getVars()
    objective_function = model_object.objVal
    num_stations = stations.shape[0]
    num_demand_points = demand.shape[0]
    total_population = demand['count'].sum()

    MEX_demand = model_variables[0:-num_stations]
    MEX_stations = model_variables[-num_stations:]

    max_demand = 0
    max_station = 0
    demand_dict = defaultdict(int)
    station_dict = defaultdict(int)

    for demand_var in MEX_demand:
        nums = [int(s) for s in demand_var.varName.split() if s.isdigit()]
        # print(nums)
        if nums[1] > max_demand and demand_var.x > 0:
            max_demand = nums[1]
        if nums[1] >= demand_dict[nums[0]] and demand_var.x > 0:
            demand_dict[nums[0]] = nums[1]
    print("The maximum number of times a demand point is covered:", max_demand)
    pd.DataFrame.from_dict(data=demand_dict, orient='index').to_csv(filename1, header=False)

    covered_dict = {}
    for idx in range(0, int(max_demand)):
        cover_time = idx + 1
        num_covered = 0
        pop_covered = 0
        for key, value in demand_dict.items():
            if value >= cover_time:
                num_covered += 1
                pop_covered += demand.loc[key, 'count']
        # print(pop_covered, total_population)
        percent_pop_covered = pop_covered / total_population
        percent_num_covered = num_covered / num_demand_points
        covered_dict[cover_time] = (percent_num_covered, percent_pop_covered)

    num_vehicles = 0
    for station_var in MEX_stations:

        nums = [int(s) for s in station_var.varName.split() if s.isdigit()]
        if station_var.x > max_station:
            max_station = station_var.x
        station_dict[nums[0]] = station_var.x
        num_vehicles += station_var.x
    pd.DataFrame.from_dict(data=station_dict, orient='index').to_csv(filename2, header=False)

    station_cover_dict = {}
    for idx in range(0, int(max_station)):
        cover_time = idx + 1
        station_cover = 0
        for key, value in demand_dict.items():
            if value >= cover_time:
                station_cover += 1

        percent_station_covered = station_cover / num_stations
        station_cover_dict[cover_time] = percent_station_covered

    return ("Percentage of Demand Points and Percentage of Incidents Covered X+ Times:", covered_dict,
            "Percentage of Stations With X+ Number of Vehicles:", station_cover_dict, "Number of Vehicles Used",
            num_vehicles,
            "Objective Value:", objective_function)


def dim_returns(demand, num_vehicles):
    num_demand = demand.shape[0]
    reward_matrix = np.zeros((num_demand, num_vehicles))
    for idx in range(0, num_demand):
        max_count = demand.loc[idx, 'count']
        for num_cover in range(0, num_vehicles):
            value = max_count * (1/(num_cover + 1.0))
            reward_matrix[idx, num_cover] = value
    return reward_matrix











