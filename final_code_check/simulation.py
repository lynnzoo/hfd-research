import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
import copy
import os
np.random.seed(55)

# Module written by Ashwin and modularized/organized by Lynn

class Vehicle:
    # Types of vehicles we are including: E, L, M, A,
    def __init__(self, name, vehicle_type, assigned_station):
        self.name = name
        self.type = vehicle_type
        self.station = assigned_station
        self.is_free = True
        self.time2free = 0

    def dispatch(self, time):
        self.is_free = False
        self.time2free = time

    def update(self, subtracted):
        self.time2free -= subtracted
        if self.time2free <= 0:
            self.time2free = 0
            self.is_free = True

    def give_state(self):
        return self.name, self.type, self.station, self.is_free, self.time2free


class Station:
    def __init__(self, name, assigned_vehicles):
        self.name = name
        self.free_vehicles = assigned_vehicles
        self.dispatched_vehicles = {}

    def return_vehicle(self, vehicle_name, vehicle_object):
        self.free_vehicles[vehicle_name] = vehicle_object

    def dispatch_vehicle(self, vehicle_name, vehicle_object):
        self.dispatched_vehicles[vehicle_name] = vehicle_object
        del self.free_vehicles[vehicle_name]

    def give_state(self):
        return (self.name, "Vehicles Available for Dispatch", list(self.free_vehicles.keys()),
                "Vehicles Currently Out of Station", list(self.dispatched_vehicles.keys()))

    def diagnose(self):
        print(self.name)
        for vehicle in self.dispatched_vehicles.keys():
            print(self.dispatched_vehicles[vehicle].give_state())
        return (self.name, "Vehicles Available for Dispatch", list(self.free_vehicles.keys()),
                "Vehicles Currently Out of Station", list(self.dispatched_vehicles.keys()))


def initialize(allocation):
    station_dict = {}
    global_vehicle_dict = {}

    for index, row in allocation.iterrows():
        name = int(row['Station'])
        vehicles = {}
        num_engines = row['Engines']
        num_ladders = row['Ladders']
        num_amb = row['Ambulances']
        num_medics = row['Medics']
        for engine in range(0, num_engines):
            vehicle_name = 'E' + str(name) + "_" + str(engine)
            vehicles[vehicle_name] = Vehicle(vehicle_name, 'E', name)
            global_vehicle_dict[vehicle_name] = vehicles[vehicle_name]

        for ladder in range(0, num_ladders):
            vehicle_name = 'L' + str(name) + "_" + str(ladder)
            vehicles[vehicle_name] = Vehicle(vehicle_name, 'L', name)
            global_vehicle_dict[vehicle_name] = vehicles[vehicle_name]

        for amb in range(0, num_amb):
            vehicle_name = 'A' + str(name) + "_" + str(amb)
            vehicles[vehicle_name] = Vehicle(vehicle_name, 'A', name)
            global_vehicle_dict[vehicle_name] = vehicles[vehicle_name]

        for medic in range(0, num_medics):
            vehicle_name = 'M' + str(name) + "_" + str(medic)
            vehicles[vehicle_name] = Vehicle(vehicle_name, 'M', name)
            global_vehicle_dict[vehicle_name] = vehicles[vehicle_name]

        station_dict[name] = Station(name, vehicles)
    print("System Initialized")
    return station_dict, global_vehicle_dict


def read_hfd(hfd_allocation_file):
    hfd = pd.read_csv(hfd_allocation_file, header=0)
    hfd = hfd[:-1]
    return hfd


def read_dispatch(dispatch_file):
    protocol = pd.read_csv(dispatch_file, header=0)
    return protocol


def find_closest(vehicle_type, dp, station_dict, time_matrix, stations):
    closest_station = time_matrix[:, dp].argsort()
    iteration = 0

    for station_idx in closest_station:
        # print(station_idx)
        iteration += 1
        station_num = stations.loc[station_idx, 'Station Number']
        free_vehicles = station_dict[station_num].give_state()[2]
        for vehicle in free_vehicles:
            if vehicle_type == vehicle[0]:
                travel_time = time_matrix[station_idx, dp]
                return vehicle, travel_time, station_num

    for station in station_dict.keys():
        station_dict[station].diagnose()
        print(station_dict[station].give_state())

    print('No available vehicles', 'Vehicle Type', vehicle_type, 'Demand Point', dp, "Iteration", iteration)


def send_vehicle(vehicle_name, time, station_num, station_dict, vehicle_dict_free, vehicle_dict_dispatched):
    vehicle_object = vehicle_dict_free[vehicle_name]
    vehicle_object.dispatch(time)
    vehicle_dict_dispatched[vehicle_name] = vehicle_object
    station_dict[station_num].dispatch_vehicle(vehicle_name, vehicle_object)
    del vehicle_dict_free[vehicle_name]


def extract_week(data, startday, endday):
    indices = []
    for index, row in data.iterrows():
        timeob = datetime.strptime(row['Dispatch_Date'], '%Y-%m-%d %H:%M:%S')
        if timeob.day < startday or timeob.day > endday:
            indices.append(index)
    new = data.drop(indices)
    return new


def read_station_data(station_filename, bay_filename, station_names, bay_names):
    # Variables
    # STATION: station number
    # NUM_DT: number of drive through bays
    # NUM_NDT: number of non-drive through bays
    # NUM_BAYS_UPDATED: total number of bays
    # NUM_AMBULANCES: capacity if all bays are used for ambulances
    # NUM_ENGINES: capacity if all bays are used for engines
    # NUM_AMBULANCE_DT: capacity if *only drive through* bays are used for ambulances
    # NUM_AMBULANCE_NDT: capacity if *only non-drive through* bays are used for ambulances
    stations = pd.read_csv(station_filename, header=0, names=station_names)
    bays = pd.read_csv(bay_filename, header=0, names=bay_names)
    bays.drop('Station Number', axis=1, inplace=True)
    final = pd.concat([stations, bays], axis=1)

    return final


def get_dispatch_times(demand, stations, old, emal):
    num_stations = stations.shape[0]
    num_demand = demand.shape[0]
    old = np.transpose(old)
    new = copy.copy(old)
    new = new / 60.0
    # print(new[0,:])
    iteration = 0
    dispatch_time_dict = defaultdict(list)
    for index, row in demand.iterrows():
        subset = emal[emal['cluster_11_17'] == index]
        # print(subset)
        for index2, row2 in stations.iterrows():
            iteration += 1

            station_num = row2['Station Number']
            subset2 = subset[subset['station'] == station_num]
            subset_amb = subset2[subset2['unit'].str.contains('A')]
            subset_medic = subset2[subset2['unit'].str.contains('M')]
            subset_engine = subset2[subset2['unit'].str.contains('E')]
            subset_ladder = subset2[subset2['unit'].str.contains('L')]
            response_a = subset_amb['dispatchtime'].tolist()
            response_m = subset_medic['dispatchtime'].tolist()
            response_e = subset_engine['dispatchtime'].tolist()
            response_l = subset_ladder['dispatchtime'].tolist()
            merged = response_a + response_m + response_e + response_l
            if merged != []:
                mean_response = np.mean(merged)
                new[index, index2] = mean_response
                # print(mean_response)

            if response_a != []:
                key = (index, station_num, 'A')
                dispatch_time_dict[key] = response_a
            else:
                key = (index, station_num, 'A')
                dispatch_time_dict[key] = [old[index, index2] / 60.0]
            if response_m != []:
                key = (index, station_num, 'M')
                dispatch_time_dict[key] = response_m
            else:
                key = (index, station_num, 'M')
                dispatch_time_dict[key] = [old[index, index2] / 60.0]
            if response_e != []:
                key = (index, station_num, 'E')
                dispatch_time_dict[key] = response_e
            else:
                key = (index, station_num, 'E')
                dispatch_time_dict[key] = [old[index, index2] / 60.0]
            if response_l != []:
                key = (index, station_num, 'L')
                dispatch_time_dict[key] = response_l
            else:
                key = (index, station_num, 'L')
                dispatch_time_dict[key] = [old[index, index2] / 60.0]

    return dispatch_time_dict, new


def prep_sim_data(EMAL):
    simulationdf = EMAL.groupby((EMAL['eventnum'] != EMAL['eventnum'].shift()).cumsum().values).first()
    # simulationdf.to_csv("EMAL_dropped.csv")
    sim_train = simulationdf[~simulationdf['eventnum'].astype(str).str.startswith('18')]
    sim_test = simulationdf[simulationdf['eventnum'].astype(str).str.startswith('18')]

    year = input("What year would you like to get simulation data from (choices: 2012-2017)? (default: 2017) :  ")
    month = input("What month would you like to get simulation data from? "
                  "Please capitalize and use the full name of the month. (default: March) :  ")
    if year == "":
        year = 2017
    if month == "":
        month = "March"
    month = str(month)
    year = str(year)
    month_year = month+'-'+year
    simulation_data = sim_train[sim_train['mnth_yr'] == month_year]
    hist_simulation_data = EMAL[EMAL["mnth_yr"] == month_year]
    return simulation_data, hist_simulation_data


def derive_time_distributions(merged_file):
    time_df = merged_file[merged_file['Dispatch_Date'].notnull()]

    ems = time_df[time_df['EMS_or_Fire_Event'] == "EMS"]
    fire = time_df[time_df['EMS_or_Fire_Event'] == "Fire"]
    ems_to_distributions = defaultdict(list)
    fire_to_distributions = defaultdict(list)
    ems_complete_distributions = defaultdict(list)
    fire_complete_distributions = defaultdict(list)

    emal = ['E', 'M', 'A', 'L']

    for index, row in ems.iterrows():
        if 'A' in row['unit'][0]:
            ems_complete_distributions['A'].append(row['timetocomplete'])

            ems_to_distributions['A'].append(row['turnouttime'])
        elif 'E' in row['unit'][0]:
            ems_complete_distributions['E'].append(row['timetocomplete'])

            ems_to_distributions['E'].append(row['turnouttime'])

        elif 'M' in row['unit'][0]:

            ems_complete_distributions['M'].append(row['timetocomplete'])
            ems_to_distributions['M'].append(row['turnouttime'])
        elif 'L' in row['unit'][0]:
            ems_to_distributions['L'].append(row['turnouttime'])
            ems_complete_distributions['L'].append(row['timetocomplete'])
    for index, row in fire.iterrows():
        if 'A' in row['unit'][0]:
            fire_to_distributions['A'].append(row['turnouttime'])
            fire_complete_distributions['A'].append(row['timetocomplete'])

        elif 'E' in row['unit'][0]:
            fire_to_distributions['E'].append(row['turnouttime'])
            fire_complete_distributions['E'].append(row['timetocomplete'])

        elif 'M' in row['unit'][0]:
            fire_to_distributions['M'].append(row['turnouttime'])
            fire_complete_distributions['M'].append(row['timetocomplete'])
        elif 'L' in row['unit'][0]:
            fire_to_distributions['L'].append(row['turnouttime'])
            fire_complete_distributions['L'].append(row['timetocomplete'])

    return ems_to_distributions, ems_complete_distributions, fire_to_distributions, fire_complete_distributions


def dispatch_redux(code, dp, juris_in, dispatch_time_dict, time_matrix, stations, ems_dispatch,
                   fire_dispatch, station_dict, vehicle_dict_free, vehicle_dict_dispatched,
                   ems_to_time, fire_to_time, ems_complete_time, fire_complete_time, maximum, mode):
    elam = ['E', 'L', 'A', 'M']
    response_times = []
    vehicles_sent = []
    stations_used = []
    turnout_times = []
    if code[0:2] == 'FE':
        # print("EVENT IS EMS")
        df = ems_dispatch[ems_dispatch['TYPE'] == code]
        num_levels = df.shape[0]
        # print("Number of Levels", num_levels)
        level_dict = defaultdict(dict)
        final_level = None
        min_time_level = 1000000
        for level in range(0, num_levels):
            # print("We are on level:", level + 1)
            row = df[df['Level'] == level + 1]
            # print("PROTOCOL", row)
            holder_dict = defaultdict(list)

            for idx in range(0, maximum):
                for vtype in elam:
                    # print("Checking stuff", vtype, row[vtype].values[0], idx+1)
                    if row[vtype].values[0] == idx + 1:
                        holder_dict[idx + 1].append(vtype)

            iteration = 0

            # print("LEVEL ", level+1, "BRANCHES ", holder_dict)
            for key, value in holder_dict.items():
                min_time_vehicle = 10000000
                for vtype in value:
                    iteration += 1

                    vehicle_name, travel_time, station_num = find_closest(vtype, dp, station_dict,
                                                                           time_matrix, stations)

                    if travel_time < min_time_vehicle:
                        min_time_vehicle = travel_time

                        vt2s = vehicle_name
                        s2s = station_num

                # print("VEHICLE WITHIN LEVEL ", level + 1, "AND BRANCH ", key, "WITH MIN TIME:", vt2s, "TIME:", min_time_vehicle)

                level_dict[level + 1][vt2s] = [vt2s, travel_time, s2s]
            # print("Minimum time within a level", min_time_vehicle, "Minimum time overall", min_time_level)
            if min_time_vehicle < min_time_level:
                final_level = level + 1
                min_time_level = min_time_vehicle

        for v2s in level_dict[final_level].keys():

            station_num = level_dict[final_level][v2s][2]
            turnout_time = np.random.choice(ems_to_time[v2s[0]])
            complete_time = np.random.choice(ems_complete_time[v2s[0]])
            if mode == 'p':

                dispatch_time = np.random.choice(dispatch_time_dict[(dp, station_num, v2s[0])])
            else:
                dispatch_time = level_dict[final_level][v2s][1]

            if v2s[0] == 'A':
                dispatch_time = dispatch_time * 1.13
            if v2s[0] == 'M':
                dispatch_time = dispatch_time * 1.15
            if v2s[0] == 'L':
                dispatch_time = dispatch_time * .92
            total_time = turnout_time + dispatch_time + complete_time
            total_time = round(total_time, 3)
            send_vehicle(v2s, total_time, station_num, station_dict,
                         vehicle_dict_free, vehicle_dict_dispatched)

            response_times.append(dispatch_time)
            vehicles_sent.append(v2s)
            stations_used.append(station_num)
            turnout_times.append(turnout_time)

    elif code[0:2] == 'FF':
        df = fire_dispatch[fire_dispatch['TYPE'] == code]
        for vtype in elam:
            num_to_send = int(df[vtype].values[0])
            for num in range(0, num_to_send):
                vehicle_name, travel_time, station_num = find_closest(vtype, dp, station_dict, time_matrix, stations)
                # dispatch_penalty = np.random.gamma(k, theta)
                if mode == 'p':
                    dispatch_time = np.random.choice(dispatch_time_dict[(dp, station_num, vehicle_name[0])])
                else:
                    dispatch_time = travel_time
                # if vehicle_name[0] == 'A' or vehicle_name[0] == "M":
                # travel_time = travel_time * 1.2
                if vehicle_name[0] == 'A':
                    dispatch_time = dispatch_time * 1.13
                if vehicle_name[0] == 'M':
                    dispatch_time = dispatch_time * 1.15
                if vehicle_name[0] == 'L':
                    dispatch_time = dispatch_time * .92
                turnout_time = np.random.choice(fire_to_time[vehicle_name[0]])
                complete_time = np.random.choice(fire_complete_time[vehicle_name[0]])

                total_time = dispatch_time + turnout_time + complete_time
                total_time = round(total_time, 3)
                # print("Flag 3")
                # print("vehicle", vehicle_name)
                # print("Turnout time", turnout_time, "dispatch time", travel_time, "completion time", complete_time)
                send_vehicle(vehicle_name, total_time, station_num, station_dict, vehicle_dict_free,
                             vehicle_dict_dispatched)

                response_times.append(travel_time)
                vehicles_sent.append(vehicle_name)
                stations_used.append(station_num)
                turnout_times.append(turnout_time)

    return (vehicles_sent, response_times, stations_used, turnout_times)


def simulator(name, incident_list, allocation, ems_dispatch, fire_dispatch, dispatch_time_dict, time_matrix, stations,
              ems_to_time, fire_to_time, ems_complete_time, fire_complete_time, maximum, mode='p'):
    time_matrix = np.transpose(time_matrix)
    station_dict, vehicle_dict_free = initialize(allocation)
    vehicle_dict_dispatched = {}
    CLOCK = datetime.strptime(incident_list.loc[incident_list.index[0], 'Dispatch_Date'], '%Y-%m-%d %H:%M:%S')
    print("Starting Time", CLOCK)
    event_nums = []
    units = []
    station_come = []
    enroute = []
    onscene = []
    dispatch_times = []
    to_times = []
    vehicle_types = []
    codes = []
    longitudes = []
    latitudes = []
    incident_juris = []
    in_juris = []

    num_iter = 0
    for index, incident in incident_list.iterrows():
        juris_in = incident['incident_juris']
        if juris_in == 0:
            continue
            # print("INDEX", index)
        new_time = datetime.strptime(incident['Dispatch_Date'], '%Y-%m-%d %H:%M:%S')

        holder = new_time - CLOCK
        difference = holder.total_seconds() / 60

        code = incident['type']
        eventnum = incident['eventnum']

        latitude = incident['latitude']
        longitude = incident['longitude']
        dp = incident['cluster_11_17']
        for key, vehicle in list(vehicle_dict_dispatched.items()):

            vehicle.update(difference)

            station = vehicle.give_state()[2]

            vehicle_name = vehicle.give_state()[0]

            if vehicle.give_state()[3] == True:
                station_dict[station].return_vehicle(vehicle_name, vehicle)
                vehicle_dict_free[vehicle_name] = vehicle
                del vehicle_dict_dispatched[key]

        if CLOCK.day != new_time.day:
            print("WORLD CLOCK", new_time.strftime('%m/%d/%Y %H:%M:%S'))
            print(num_iter, " Events Successfully Executed")
            print('EVENT NUM', eventnum)

        CLOCK = new_time

        vehicles, times, stations_used, turnout_times = dispatch_redux(code, dp, juris_in, dispatch_time_dict,
                                                                       time_matrix,
                                                                       stations, ems_dispatch, fire_dispatch,
                                                                       station_dict, vehicle_dict_free,
                                                                       vehicle_dict_dispatched, ems_to_time,
                                                                       fire_to_time, ems_complete_time,
                                                                       fire_complete_time, maximum, mode)
        # print(eventnum, vehicles)
        for dispatch_idx in range(0, len(vehicles)):
            event_nums.append(eventnum)
            units.append(vehicles[dispatch_idx])
            station_come.append(stations_used[dispatch_idx])
            # print(CLOCK + timedelta(minutes = turnout_times[dispatch_idx]))
            enroute.append(CLOCK + timedelta(minutes=turnout_times[dispatch_idx]))
            onscene.append(
                CLOCK + timedelta(minutes=turnout_times[dispatch_idx]) + timedelta(minutes=times[dispatch_idx]))
            dispatch_times.append(times[dispatch_idx])
            to_times.append(turnout_times[dispatch_idx])
            vehicle_types.append(vehicles[dispatch_idx][0])
            codes.append(code)
            latitudes.append(latitude)
            longitudes.append(longitude)
            incident_juris.append(juris_in)
            if juris_in == np.int64(stations_used[dispatch_idx]):
                in_juris.append(1)
            else:
                in_juris.append(0)
            # if len(vehicles) > 1:
            # print("VEHICLE BEING SENT", vehicles[dispatch_idx])
            # print("TURNOUT TIMES", turnout_times[dispatch_idx])

        num_iter += 1

    store = {'eventnum': event_nums, 'unit': units, 'station': station_come,
             'vehicletype': vehicle_types, 'type': codes, 'enroute': enroute, 'onscene': onscene,
             'dispatchtime': dispatch_times, 'turnouttime': to_times, 'longitude': longitudes, 'latitude': latitudes,
             'incident_juris': incident_juris, 'in_juris': in_juris}

    final = pd.DataFrame.from_dict(data=store)
    filename = name+'.csv'
    final.to_csv(filename, index=False)

    return final


def counting_emal_units_sent(merged):
    """
    Written by: Erin Kreus
    Purpose: Uses MASTER Spatial Joined file of all dispatches to determine how many of each unit type
    including only Ambulances, Engines, Ladders, and Medics in order to allow for comparisons among
    dispatching protocal and what was actually sent. It writes to CSV: 'actual_vehicle_type_responses.csv'
    :param codes: path to csv of events and number of Engine, ladder, ambulance, and medic units
    """
    # get dataset of only vehicles that we care about for dispatching (Ambulances, Ladders, Engines, and Medics)
    emal_merged = merged.loc[(merged['vehicletype'] == "A") |
                             (merged['vehicletype'] == "L") |
                             (merged['vehicletype'] == "E") |
                             (merged['vehicletype'] == "M")]

    # Create a pivot table to count how many of each unit type was dispatched for each incident (event number)
    emal_counts = pd.pivot_table(emal_merged, values='station', index=['eventnum', 'type'], columns=['vehicletype'],
                                 aggfunc='count')

    # Flatten the Pivot Table (get rid of wierd formatting create by pivot_table function)
    emal_counts = pd.DataFrame(emal_counts.to_records())

    # Fill in NA values with 0
    emal_counts[['A', 'E', 'L', 'M']] = emal_counts[['A', 'E', 'L', 'M']].fillna(value=0)

    emal_counts.to_csv(os.path.join(os.getcwd(), "data", "actual_vehicle_type_responses.csv"))


def simulator_hist(name, dispatch_list, allocation, dispatch_time_dict, time_matrix, stations, mode='p'):
    time_matrix = np.transpose(time_matrix)
    station_dict, vehicle_dict_free = initialize(allocation)
    vehicle_dict_dispatched = {}
    CLOCK = datetime.strptime(dispatch_list.loc[dispatch_list.index[0], 'Dispatch_Date'], '%Y-%m-%d %H:%M:%S')
    print("Starting Time", CLOCK)
    event_nums = []
    units = []
    station_come = []
    enroute = []
    onscene = []
    dispatch_times = []
    to_times = []
    vehicle_types = []
    codes = []
    longitudes = []
    latitudes = []
    incident_juris = []
    in_juris = []

    num_iter = 0
    for index, dispatch in dispatch_list.iterrows():

        juris_in = dispatch['incident_juris']
        # if juris_in == 0:
        # continue
        # print("INDEX", index)
        # print(type(dispatch['Dispatch_Date']))
        new_time = datetime.strptime(dispatch['Dispatch_Date'], '%Y-%m-%d %H:%M:%S')

        holder = new_time - CLOCK
        difference = holder.total_seconds() / 60

        code = dispatch['type_hfd']
        eventnum = dispatch['eventnum']

        latitude = dispatch['latitude']
        longitude = dispatch['longitude']
        dp = dispatch['cluster_11_17']
        for key, vehicle in list(vehicle_dict_dispatched.items()):

            vehicle.update(difference)

            station = vehicle.give_state()[2]

            vehicle_name = vehicle.give_state()[0]

            if vehicle.give_state()[3] == True:
                station_dict[station].return_vehicle(vehicle_name, vehicle)
                vehicle_dict_free[vehicle_name] = vehicle
                del vehicle_dict_dispatched[key]

        if CLOCK.day != new_time.day:
            print("WORLD CLOCK", new_time.strftime('%m/%d/%Y %H:%M:%S'))
            print(num_iter, " Dispatches Successfully Executed")
            print('EVENT NUM', eventnum)

        CLOCK = new_time

        unit_2_dispatch = dispatch['unit'][0]
        vehicle_name, travel_time, station_num = find_closest(unit_2_dispatch, dp, station_dict, time_matrix, stations)
        turnout_time = dispatch['turnouttime']
        complete_time = dispatch['timetocomplete']
        if mode == 'p':

            dispatch_time = np.random.choice(dispatch_time_dict[(dp, station_num, vehicle_name[0])])
        else:
            dispatch_time = travel_time

        # if vehicle_name[0] == 'A':
        # dispatch_time = dispatch_time * 1.13
        # if vehicle_name[0] == 'M':
        # dispatch_time = dispatch_time * 1.15
        # if vehicle_name[0] == 'L':
        # dispatch_time = dispatch_time * .92
        total_time = turnout_time + dispatch_time + complete_time
        total_time = round(total_time, 3)
        send_vehicle(vehicle_name, total_time, station_num, station_dict,
                     vehicle_dict_free, vehicle_dict_dispatched)

        event_nums.append(eventnum)
        units.append(vehicle_name)
        station_come.append(station_num)
        enroute.append(dispatch['enroute'])
        onscene.append(CLOCK + timedelta(minutes=turnout_time) + timedelta(minutes=dispatch_time))
        dispatch_times.append(dispatch_time)
        to_times.append(turnout_time)
        vehicle_types.append(vehicle_name[0])
        codes.append(code)
        latitudes.append(latitude)
        longitudes.append(longitude)
        incident_juris.append(juris_in)
        if juris_in == np.int64(station_num):
            in_juris.append(1)
        else:
            in_juris.append(0)

        num_iter += 1

    store = {'eventnum': event_nums, 'unit': units, 'station': station_come,
             'vehicletype': vehicle_types, 'type': codes, 'enroute': enroute, 'onscene': onscene,
             'dispatchtime': dispatch_times, 'turnouttime': to_times, 'longitude': longitudes, 'latitude': latitudes,
             'incident_juris': incident_juris, 'in_juris': in_juris}

    final = pd.DataFrame.from_dict(data=store)
    filename = name+'.csv'
    final.to_csv(filename, index=False)
    return final

#
# def creating_dispatching_protocol(codes_dispatching, fire_codes, ems_codes_other, ems_codes_levels):
#     """
#     Written by: Erin Kreus and Lynn Zhu
#     Purpose: Uses dispatching protocol files to format for use in validation
#     :param codes: path to two csvs: one with fire dispatching protocol and one with EMS dispatching protocol
#     """
#     # Import four datasets:
#
#     # 1-codes dataset that includes all incident codes and their meanings (given to us by HFD)
#     codes_dispatching = pd.read_csv(all_codes)
#
#     # 2-fire appropriate responses dataset provided by HFD
#     fire_codes=pd.read_csv(fire_codes)
#
#     # 3-EMS appropriate responses dataset for select groups provided by HFD because they were missing from
#     # other flow charts
#     ems_codes_other=pd.read_csv(ems_codes_extra)
#
#     # 4-EMS appropriate responses dataset that was created by interpreting HFD dispatching charts
#     ems_codes_levels=pd.read_csv(ems_codes_final)
#
#     # Now, make two dispatching protocal datasets: one for EMS and one for Fire
#
#     # First, make EMS dispatching dataset by:
#
#     # 1-Determine the last two characters of the incident type. This is used later to merge with the EMS responses
#     # In other words, the last two digits give us the medical incident "identifier"
#     codes_dispatching['identifier'] = codes_dispatching['TYPE'].str[-2:]
#
#     # 2-Determine the first two characters of the incident type. This lets us figure out if it Fire or EMS
#     codes_dispatching['prefix'] = codes_dispatching['TYPE'].str[:2]
#
#     # 3-Make an EMS code file by filtering for "FE"
#     codes_ems_master=codes_dispatching.loc[(codes_dispatching['prefix']=="FE")]
#
#     # 4-Create four datasets for different levels of EMS Responses (differing levels refer to closest vehicle)
#     # There are two possible levels of each EMS incident response
#     ems_codes1=ems_codes_levels.loc[(ems_codes_levels['Level']==1)]
#     ems_codes2=ems_codes_levels.loc[(ems_codes_levels['Level']==2)]
#     ems_codes3=ems_codes_levels.loc[(ems_codes_levels['Level']==3)]
#     ems_codes4=ems_codes_levels.loc[(ems_codes_levels['Level']==4)]
#
#     # 5-For each level, merge the EMS response with the code on the "identifier" (unique two level identifier)
#     codes_ems_master1=pd.merge(codes_ems_master,ems_codes1,how='left',on="identifier")
#     codes_ems_master2=pd.merge(codes_ems_master,ems_codes2,how='left',on="identifier")
#     codes_ems_master3=pd.merge(codes_ems_master,ems_codes3,how='left',on="identifier")
#     codes_ems_master4=pd.merge(codes_ems_master,ems_codes4,how='left',on="identifier")
#
#
#     # 6-Drop any rows where we are missing the identifier. This excludes codes that WE DO NOT KNOW THE
#     # APPROPRIATE DISPATCH FOR or DO NOT HAVE A SECOND LEVEL. We also only want to keep the first entry
#     codes_ems_master1=codes_ems_master1.dropna(subset=['E', 'L','A','M','Level'], how="all")
#     codes_ems_master2=codes_ems_master2.dropna(subset=['E', 'L','A','M','Level'], how="all")
#     codes_ems_master3=codes_ems_master3.dropna(subset=['E', 'L','A','M','Level'], how="all")
#     codes_ems_master4=codes_ems_master4.dropna(subset=['E', 'L','A','M','Level'], how="all")
#     codes_ems_master1=codes_ems_master1.drop_duplicates(subset='TYPE', keep="first")
#     codes_ems_master2=codes_ems_master2.drop_duplicates(subset='TYPE', keep="first")
#     codes_ems_master3=codes_ems_master3.drop_duplicates(subset='TYPE', keep="first")
#     codes_ems_master4=codes_ems_master4.drop_duplicates(subset='TYPE', keep="first")
#
#     # 7-concat the two datasets for level 1-4 back together
#     ems_dispatching_guide = pd.concat([codes_ems_master1, codes_ems_master2,
#                                        codes_ems_master3, codes_ems_master4],ignore_index=True,sort=False)
#
#     # 8-Drop the one that is an exception
#     ems_dispatching_guide=ems_dispatching_guide.loc[ems_dispatching_guide['TYPE'] != "FEREC1"]
#
#     # 7-concat the two datasets by adding in the ones given to us explicitly
#     # by HFD
#     ems_dispatching_guide = pd.concat([ems_dispatching_guide,
#                                        ems_codes_other],ignore_index=True,sort=False)
#
#     # Second, make EMS dispatching dataset by:
#
#     # 1-Make an Fire code file by filtering for "FF"
#     codes_fire_master=codes_dispatching.loc[(codes_dispatching['prefix']=="FF")]
#
#     # 2-For each level, merge the Fire response with the code on the "identifier"
#     fire_dispatching_guide=pd.merge(codes_fire_master,fire_codes,how='left',on="TYPE")
#
#     # 3-Drop types that we do not have the dispatching protocal for. This excludes codes that WE DO NOT KNOW THE
#     # APPROPRIATE DISPATCH FOR or DO NOT HAVE A SECOND LEVEL. We also only want to keep the first entry
#     fire_dispatching_guide=fire_dispatching_guide.dropna(subset=['E', 'L', 'A', 'M'], how="all")
#     fire_dispatching_guide=fire_dispatching_guide.drop_duplicates(subset='TYPE', keep="first")
#
#     # 4-Create a level variable of value "1" because only one level for all fires
#     fire_dispatching_guide['Level']=1
#
#     # export to csv
#     fire_dispatching_guide.to_csv(os.path.join(os.getcwd(), "data", "fire_dispatching_protocol.csv"))
#     ems_dispatching_guide.to_csv(os.path.join(os.getcwd(), "data", "ems_dispatching_protocol.csv"))


def generate_new(hfd_allocation, ranked_file, sortby, amb, med, ladder, engine, num):
    hfd_allocation = hfd_allocation.set_index('Station')

    ranked = pd.read_csv(ranked_file, header=0, index_col='start_inc_juris')
    ranked = ranked.drop(301, axis=0)
    ranked_sorted = ranked.sort_values(by=[sortby], ascending=False)
    sort_list = ranked_sorted.index.values.tolist()
    print(sort_list)
    for idx in range(0, amb):
        amount = int(hfd_allocation.loc[str(sort_list[idx]), "Ambulances"])

        new_amount = int(amount + num)
        hfd_allocation.loc[str(sort_list[idx]), "Ambulances"] = int(new_amount)

    for idx in range(0, med):
        amount = int(hfd_allocation.loc[str(sort_list[idx]), "Medics"])

        new_amount = int(amount + num)
        hfd_allocation.loc[str(sort_list[idx]), "Medics"] = int(new_amount)
    for idx in range(0, ladder):
        amount = int(hfd_allocation.loc[str(sort_list[idx]), "Ladders"])

        new_amount = int(amount + num)
        hfd_allocation.loc[str(sort_list[idx]), "Ladders"] = int(new_amount)
    for idx in range(0, engine):
        amount = int(hfd_allocation.loc[str(sort_list[idx]), "Engines"])

        new_amount = int(amount + num)
        hfd_allocation.loc[str(sort_list[idx]), "Engines"] = int(new_amount)

    hfd_allocation = hfd_allocation.reset_index()
    return hfd_allocation

