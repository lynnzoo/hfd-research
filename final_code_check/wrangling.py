import pandas as pd
import numpy as np
import os
import warnings
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point


def data_wrangle(incident_files, dispatch_files, sep=False, save=True, output=True):
    """
    written by Lynn Zhu
    purpose: sequentially use all functions to wrangle the data from raw form
    :param incident_files: a list of all incident csv file paths
    :param dispatch_files: a list of all dispatch csv file paths
    :param sep: if True, saves individual cleaned and backup files corresponding to each file
    in dispatch_files and incident_files
    :param save: boolean, if True, saves whole cleaned and whole backup incidents files as csvs
    :param output: if True, returns two pandas data frames, a cleaned and joined incident pandas data frame
    and a cleaned and joined dispatch pandas data frame
    :return: if output = False, returns nothing
    """
    # clean and concatenate all the dispatch and incident files respectively
    # if sep = True, individual cleaned and backup files corresponding to each file in dispatch_files and incident_files
    # will be saved
    cleaned_incidents, backup_incidents = concatenate_files(incident_files, clean_func=clean_incidents, sep=sep)
    cleaned_dispatches, backup_dispatches = concatenate_files(dispatch_files, clean_func=clean_dispatches, sep=sep)
    total_incidents = cleaned_incidents.shape[0] + backup_incidents.shape[0]
    total_dispatches = cleaned_dispatches.shape[0] + backup_dispatches.shape[0]
    print("INPUT")
    print("%s total incidents" % total_incidents)
    print("%s total dispatches" % total_dispatches)

    # backup dispatches for the incidents that got put in backup
    unique_backup_incidents = backup_incidents.eventnum.unique()
    matched_backup_dispatches = cleaned_dispatches[cleaned_dispatches.eventnum.isin(unique_backup_incidents)]
    backup_dispatches = pd.concat([backup_dispatches, matched_backup_dispatches])
    cleaned_dispatches = cleaned_dispatches[~cleaned_dispatches.eventnum.isin(unique_backup_incidents)]

    # remove all incidents without dispatches
    all_incidents, all_dispatches, no_dispatches, no_incidents = overlapping_eventnums(cleaned_incidents,
                                                                                       cleaned_dispatches)

    backup_incidents = pd.concat([backup_incidents, no_dispatches], sort=False)
    if no_incidents.shape[0] != 0:
        backup_dispatches = pd.concat([backup_dispatches, no_incidents])
        warnings.warn(
            "There are dispatches that do not match to an incident. This is unexpected."
            "There were %s dispatches that did not match to an incident" % (no_incidents.shape[0]))

    # data wrangling of times and fire/EMS categorization
    all_dispatches = wrangle_dates_times(all_dispatches)
    all_dispatches = calculate_response_times(all_dispatches)
    all_incidents, all_dispatches = categorize(all_incidents, all_dispatches)

    # data wrangling of negative response/dispatch times
    all_incidents, all_dispatches, negative_incidents, negative_dispatches = remove_negative_times(all_incidents,
                                                                                                   all_dispatches)

    backup_incidents = pd.concat([backup_incidents, negative_incidents], sort=False)
    backup_dispatches = pd.concat([backup_dispatches, negative_dispatches], sort=False)

    # data wrangling of invalid vehicle types
    all_incidents, all_dispatches, invalid_incidents, invalid_dispatches = remove_invalid_vehicles(all_incidents,
                                                                                                   all_dispatches)

    backup_incidents = pd.concat([backup_incidents, invalid_incidents], sort=False)
    backup_dispatches = pd.concat([backup_dispatches, invalid_dispatches], sort=False)

    if all_incidents.shape[0] + backup_incidents.shape[0] != total_incidents:
        warnings.warn("There are missing incidents after wrangling. This can be indicative of a serious"
                      " error. There are %s incidents missing." % (total_incidents.shape[0] - (all_incidents.shape[0] +
                                                                   backup_incidents.shape[0])))

    # if save = True, save all data frames as csvs
    if save:
        # save incident files
        all_incidents.to_csv(os.path.join(os.getcwd(), "data", "all_incidents.csv"), index=False)
        backup_incidents.to_csv(os.path.join(os.getcwd(), "data", "backup_incidents.csv"), index=False)
        all_dispatches.to_csv(os.path.join(os.getcwd(), "data", "all_dispatches.csv"), index=False)
        backup_dispatches.to_csv(os.path.join(os.getcwd(), "data", "backup_dispatches.csv"), index=False)
        # no_dispatches.to_csv(os.path.join(os.getcwd(), "data", "no_dispatch_incidents.csv"), index=False)
        # save dispatch files
        # if no_incidents.shape[0] != 0:
        #     no_incidents.to_csv(os.path.join(os.getcwd(), "data", "no_incident_dispatches.csv"), index=False)
        # negative_incidents.to_csv(os.path.join(os.getcwd(), "data", "negative_times_incidents.csv"), index=False)
        # negative_dispatches.to_csv(os.path.join(os.getcwd(), "data", "negative_times_dispatches.csv"), index=False)
        # invalid_incidents.to_csv(os.path.join(os.getcwd(), "data", "invalid_vehicle_incidents.csv"), index=False)
        # invalid_dispatches.to_csv(os.path.join(os.getcwd(), "data", "invalid_vehicle_dispatches.csv"), index=False)

    # print summary
    print("")
    print("INCIDENTS WRANGLING")
    print("%s incidents in backup" % backup_incidents.shape[0])
    print("     %s incidents without dispatches" % no_dispatches.shape[0])
    print("     %s incidents associated ONLY to dispatches with negative response/dispatch times"
          % negative_incidents.shape[0])
    print("     %s incidents associated ONLY to dispatches with invalid vehicles"
          % invalid_incidents.shape[0])
    print("%s%% of incidents removed during wrangling" %
          (round(((total_incidents - all_incidents.shape[0])/total_incidents)*100, 2)))
    print('%s incidents remaining' % all_incidents.shape[0])

    print("")
    print("DISPATCHES WRANGLING")
    print("%s dispatches in backup" % backup_dispatches.shape[0])
    if no_incidents.shape[0] != 0:
        print("     %s dispatches without incidents" % no_incidents.shape[0])
    print("     %s dispatches with negative response/dispatch times"
          % negative_dispatches.shape[0])
    print("     %s dispatches with invalid vehicles"
          % invalid_dispatches.shape[0])
    print("%s%% of dispatches removed during wrangling" %
          (round(((total_dispatches - all_dispatches.shape[0])/total_dispatches)*100, 2)))
    print('%s dispatches remaining' % all_dispatches.shape[0])
    print("")

    # return both pandas data frames if desired
    if output:
        return all_incidents, all_dispatches


def clean_incidents(filename):
    # , save=False):
    """
    written by Lynn Zhu
    purpose: import and clean an incidents csv file
    :param filename: file path to the incidents file
    :param save: boolean, if True, saves the cleaned incidents file and a backup file for incidents removed during cleaning
    :return: the cleaned incidents file as a pandas data frame
    """
    # import incidents csv file
    df = pd.read_csv(filename, header=None,
                     names=['eventnum', 'type', 'address', 'oh', 'longitude', 'latitude'],
                     dtype={'eventnum': object, 'type': str, 'address': str, 'oh': object,
                            'longitude': str, 'latitude': str})
    og_rows = df.shape[0]
    df.drop(['address', 'oh'], axis=1)

    # check for invalid eventnums, and delete any rows found
    df = df[~df['eventnum'].str.contains('[A-Za-z]', na=True)]
    df = df[df['eventnum'].str.contains('^[0-9]{10}$', na=True)]
    clean_sql_rows = df.shape[0]
    if og_rows - clean_sql_rows > 4:
        warnings.warn(
            "The inputted data is not formatted as expected. "
            "There were %s incidents deleted for not having a valid event number from %s" % (og_rows - clean_sql_rows,
                                                                                             filename))
    # remove extra quotations and spaces
    df = df.replace(r"^'\s?", '', regex=True)
    df = df.replace(r"\s?'$", '', regex=True)

    # treat blank entries as NA
    df.replace('', np.nan, inplace=True)

    # drop all of the rows with NA values in latitude, longitude, or type
    nona = df.dropna(subset=["latitude", "longitude", "type"])

    # backup all rows that will be dropped
    backup = df[~df.index.isin(nona.index)]

    # add decimal to latitude and longitude coordinates
    nona = nona.reset_index(drop=True)
    nona["latitude"] = pd.to_numeric(nona["latitude"]) * 0.000001
    nona["longitude"] = pd.to_numeric(nona["longitude"]) * 0.000001

    # if save:
    #     head, tail = os.path.split(filename)
    #     name = "clean_" + tail
    #     new_filename = os.path.join(head, name)
    #     nona.to_csv(open(new_filename, "w+"), index=False, header=True)
    #
    #     if backup.shape[0] != 0:
    #         backup_name = "backup_" + tail
    #         backup_name = os.path.join(head, backup_name)
    #         backup.to_csv(open(backup_name, "w+"), index=False, header=True)

    return nona, backup


def clean_dispatches(filename):
    # , save=False):
    """
    written by Lynn Zhu
    purpose: import and clean a dispatches csv file
    :param filename: file path to the dispatches file
    :param save: boolean, if True, saves the cleaned dispatches file and a backup file for dispatches removed during cleaning
    :return: the cleaned dispatches file as a pandas data frame
    """
    # import dispatches csv file
    df = pd.read_csv(filename, header=None, names=['eventnum', 'unit', 'station', 'vehicle', 'enroute', 'onscene'],
                     dtype={'eventnum': object, 'unit': str, 'station': str, 'vehicle': str,
                            'enroute': str, 'onscene': str})
    og_rows = df.shape[0]
    df.drop(['vehicle'], axis=1)

    # check for invalid eventnums, and delete any rows found
    df = df[~df['eventnum'].str.contains('[A-Za-z]', na=True)]
    df = df[df['eventnum'].str.contains('^[0-9]{10}$', na=True)]
    clean_sql_rows = df.shape[0]
    if og_rows - clean_sql_rows > 4:
        warnings.warn(
            "The inputted data is not formatted as expected. "
            "There were %s incidents deleted for not having a valid event number from %s" % (og_rows - clean_sql_rows,
                                                                                             filename))
    # remove extra quotations and spaces
    df = df.replace(r"^'\s?", '', regex=True)
    df = df.replace(r"\s?'$", '', regex=True)

    # treat blank entries as NA
    df.replace('', np.nan, inplace=True)

    # drop all of the rows with NA values in unit, enroute, or onscene
    nona = df.dropna(subset=["unit", "enroute", "onscene"])

    # backup all rows that will be dropped
    backup = df[~df.index.isin(nona.index)]

    nona = nona.reset_index(drop=True)

    # if save is True:
    #     head, tail = os.path.split(filename)
    #     name = "clean_" + tail
    #     new_filename = os.path.join(head, name)
    #     nona.to_csv(open(new_filename, "w+"), index=False, header=True)
    #
    #     if backup.shape[0] != 0:
    #         backup_name = "backup_" + tail
    #         backup_name = os.path.join(head, backup_name)
    #         backup.to_csv(open(backup_name, "w+"), index=False, header=True)

    return nona, backup


def concatenate_files(files, clean_func, sep=False):
    """
    written by Lynn Zhu
    purpose: cleans and concatenates incidents into one pandas data frame, and one backup data frame
    :param files: list of all csv file names
    :param clean_func: function used to clean files
    :param sep: if True, saves individual cleaned and backup files corresponding to each file in incident_files as csvs
    :return: pandas data frame of all cleaned incidents
    """
    # clean each raw file using a designated clean function
    cleaned = [clean_func(file) for file in files]

    # concatenate all cleaned and backup files
    whole_cleaned = pd.concat([df_tuple[0] for df_tuple in cleaned])
    whole_backup = pd.concat([df_tuple[1] for df_tuple in cleaned])

    # cast eventnum as an int
    whole_cleaned["eventnum"] = pd.to_numeric(whole_cleaned["eventnum"])
    whole_backup["eventnum"] = pd.to_numeric(whole_backup["eventnum"])

    return whole_cleaned, whole_backup


def overlapping_eventnums(incidents, dispatches):
    """
    written by Lynn Zhu
    :param incidents: incidents pandas data frame, cleaned
    :param dispatches: dispatches pandas data frame, cleaned
    :return: new all_incidents and all_dispatches pandas data frames with rows removed which have an eventnum which
    isn't in the other data frame
    no_incidents and no_dispatches pandas data frames to hold those rows which were removed
    """
    # get unique eventnums from incidents and dispatches
    incident_eventnums = incidents.eventnum.unique()
    dispatch_eventnums = dispatches.eventnum.unique()

    # eventnums of incidents with no dispatches
    no_dispatch_eventnums = np.setdiff1d(incident_eventnums, dispatch_eventnums, assume_unique=True)

    # eventnums of dispatches with no incidents
    no_incident_eventnums = np.setdiff1d(dispatch_eventnums, incident_eventnums, assume_unique=True)

    # remove all the incidents without any dispatches from all_incidents
    new_incidents = incidents[~incidents.eventnum.isin(no_dispatch_eventnums)]
    no_dispatches = incidents[incidents.eventnum.isin(no_dispatch_eventnums)]

    # remove all the dispatches without any incident from all_dispatches
    new_dispatches = dispatches[~dispatches.eventnum.isin(no_incident_eventnums)]
    no_incidents = dispatches[dispatches.eventnum.isin(no_incident_eventnums)]

    return new_incidents, new_dispatches, no_dispatches, no_incidents


def wrangle_dates_times(dispatches):
    """
    written by Erin Kreus and Lynn Zhu
    purpose: break down and convert dates and times into usable formats
    :param dispatches: pandas data frame of all dispatches
    :return: updated dispatches pandas data frame
    """
    # convert en-route and on-scene time to date/time format
    dispatches['enroute'] = pd.to_datetime(dispatches['enroute'], format="%Y%m%d %H:%M:%S")
    dispatches['onscene'] = pd.to_datetime(dispatches['onscene'], format="%Y%m%d %H:%M:%S")

    # extract the hour of day that the dispatch left (en-route)
    dispatches['hour'] = dispatches['enroute'].dt.hour

    # extract the date that the dispatch left (en-route)
    dispatches['date'] = dispatches['enroute'].dt.date

    # extract the year that the dispatch left (en-route)
    dispatches['year'] = dispatches['enroute'].dt.year

    # extract the month year that the dispatch left (en-route)
    dispatches['mnth_yr'] = dispatches['enroute'].apply(lambda x: x.strftime('%B-%Y'))

    # determine the *dispatch* times by determining the difference between the enroute and onscene time
    dispatches['dispatchtime'] = dispatches['onscene'] - dispatches['enroute']

    # convert the dispatch time to minutes
    dispatches['dispatchtime'] = dispatches['dispatchtime']/np.timedelta64(1, 'm')

    return dispatches


def calculate_response_times(dispatches):
    """
    written by Erin Kreus and Lynn Zhu
    purpose: determine *response* times by calculating the minimum en-route time for the incident
    :param dispatches: pandas data frame of all dispatches
    :return: updated dispatches pandas data frame
    """
    # get the minimum dispatch en route time for each event number
    dispatches_first_time = dispatches.groupby(['eventnum'])['enroute'].min().reset_index(name='min_enroute')
    dispatches = pd.merge(dispatches, dispatches_first_time, on='eventnum', how='left')

    # calculate the response time
    dispatches['responsetime'] = dispatches['onscene'] - dispatches['min_enroute']

    # convert the response time to minutes
    dispatches['responsetime'] = dispatches['responsetime']/np.timedelta64(1, 'm')

    return dispatches


def categorize(incidents, dispatches):
    """
    written by Erin Kreus and Lynn Zhu
    purpose: categorize vehicle types and incidents
    :param incidents: pandas data frame of all incidents
    :param dispatches: pandas data frame of all dispatches
    :return: updated incidents and dispatches pandas data frames
    """
    # extract the vehicle type (Ambulance, Medic, Ladder, Engine, etc.) of the vehicle from the dispatches dataset
    dispatches['vehicletype'] = dispatches['unit'].str.replace('\d+', '')

    # determine whether the incident was a fire or EMS incident using the first two characters of the type
    incidents = incidents.assign(prefix=incidents['type'].str[:2])
    incidents = incidents.assign(fire_ems=pd.np.where(incidents.prefix.str.contains("FE"), "ems", "fire"))
    # incidents['prefix'] = incidents['type'].str[:2]
    # incidents['fire_ems'] = pd.np.where(incidents.prefix.str.contains("FE"), "ems", "fire")
    return incidents, dispatches


def remove_negative_times(all_incidents, all_dispatches):
    """
    written by Lynn Zhu
    purpose: filter out dispatches with negative response/dispatch times and incidents ONLY associated with a
    dispatch with a negative response/dispatch time
    :param all_incidents: concatenated all incidents pandas data frame, cleaned
    :param all_dispatches: concatenated all dispatches pandas data frame, cleaned
    :return incidents and dispatches pandas data frames without negative times, and backup incident and dispatch data
    frames
    """
    incidents_copy = all_incidents.copy(deep=True)
    dispatches_copy = all_dispatches.copy(deep=True)
    total_num = dispatches_copy.shape[0]

    # filter out dispatches with a negative response/dispatch times
    negative_times = dispatches_copy[(dispatches_copy['dispatchtime'] <= 0)]
    dispatches_copy = dispatches_copy[(dispatches_copy['dispatchtime'] > 0)]
    negative_times = pd.concat([negative_times,
                                dispatches_copy[(dispatches_copy['responsetime'] <= 0)]])
    dispatches_copy = dispatches_copy[(dispatches_copy['responsetime'] > 0)]

    if total_num != dispatches_copy.shape[0] + negative_times.shape[0]:
        warnings.warn(
            "There are missing dispatches after filtering for negative times. This can be indicative of a serious"
            " error. There are %s dispatches "
            "missing." % (total_num - (dispatches_copy.shape[0] + negative_times.shape[0])))

    # remove incidents which were ONLY associated to dispatches with a negative response/dispatch time
    incidents_copy, dispatches_copy, no_dispatches, no_incidents = overlapping_eventnums(incidents_copy,
                                                                                         dispatches_copy)
    if no_incidents.shape[0] != 0:
        warnings.warn(
            "There are dispatches that do not match to an incident. This can be indicative of a serious error."
            "There were %s dispatches that did not match to an incident." % (no_incidents.shape[0]))

    return incidents_copy, dispatches_copy, no_dispatches, negative_times


def remove_invalid_vehicles(all_incidents, all_dispatches):
    """
    written by Lynn Zhu
    purpose: remove dispatches which have no vehicle type or vehicle type of ARS, LSI, OEC, cascade (CC), rehab (RH) and
    incidents which are ONLY associated to a dispatch with an invalid vehicle type
    :param all_incidents: concatenated all incidents pandas data frame, cleaned
    :param all_dispatches: concatenated all dispatches pandas data frame, cleaned
    :return: incidents and dispatches pandas data frames without invalid vehicles, and backup incident and dispatch data
    frames
    """
    incidents_copy = all_incidents.copy(deep=True)
    dispatches_copy = all_dispatches.copy(deep=True)

    # filter out invalid vehicle types
    no_invalid_vehicles = dispatches_copy[~(dispatches_copy.vehicletype == '')]
    no_invalid_vehicles = no_invalid_vehicles[~no_invalid_vehicles.vehicletype.isin(["ARS", "LSI", "OEC", "RH", "CC"])]

    backup = dispatches_copy[~dispatches_copy.index.isin(no_invalid_vehicles.index)]

    # remove incidents which were ONLY associated to dispatches with an invalid vehicle type
    incidents_copy, no_invalid_vehicles, no_dispatches, no_incidents = overlapping_eventnums(incidents_copy,
                                                                                             no_invalid_vehicles)

    if no_incidents.shape[0] != 0:
        warnings.warn(
            "There are dispatches that do not match to an incident. This can be indicative of a serious error."
            "There were %s dispatches that did not match to an incident." % (no_incidents.shape[0]))

    return incidents_copy, no_invalid_vehicles, no_dispatches, backup


def bays_capacity(raw_bays, save=True):
    """
    written by Erin Kreus and Lynn Zhu
    purpose: add new variables to determine bay capacity at each station from raw bays capacity file
    :param save: if True, writes new csv file: 'bays_capacity.csv'
    """
    bays = pd.read_csv(raw_bays)

    # create 5 new variables
    # num_bays_updated: total number of bays
    bays['num_bays_updated'] = bays['num_dt'] + bays['num_ndt']

    # num_ambulances: capacity if ALL bays are used for ambulances
    bays['num_ambulances'] = bays['num_dt'] * 2 + bays['num_ndt']

    # num_engines: capacity if all bays are used for engines
    bays['num_engines'] = bays['num_dt'] + bays['num_ndt']

    # num_ambulance_dt: capacity if ONLY DRIVE THROUGH bays are used for ambulances
    bays['num_ambulance_dt'] = bays['num_dt'] * 2

    # num_ambulance_ndt: capacity if ONLY NON DRIVE THROUGH bays are used for ambulances
    bays['num_ambulance_ndt'] = bays['num_ndt']

    if save:
        bays.to_csv(os.path.join(os.getcwd(), "data", "bays_capacity.csv"), index=False)

    return bays


def spatial_join(territories_filename, incidents, save=False):
    geometry = [Point(xy) for xy in zip(incidents.longitude, incidents.latitude)]
    # df = incidents.drop(['longitude', 'latitude'], axis=1)
    crs = {'init': 'epsg:4326'}
    gdf = GeoDataFrame(incidents, crs=crs, geometry=geometry)
    territories = gpd.read_file(territories_filename)
    territories = territories.rename(columns={"D1": "in_territory"})

    territories = territories.to_crs({'init': 'epsg:4326'})
    territories_join = territories[["in_territory", "geometry"]]
    spatial_joined = gpd.sjoin(gdf, territories_join, op="within", how="left")
    spatial_joined = spatial_joined.drop(["geometry", "index_right"], axis=1)
    spatial_joined[['in_territory']] = spatial_joined[['in_territory']].fillna(value=0)

    # print(spatial_joined.shape)
    if save:
        gdf.to_file(os.path.join(os.getcwd(), "data", "all_incidents_geometry.shp"), index=False)
        spatial_joined.to_csv(os.path.join(os.getcwd(), "data", "all_incidents_spatial_joined.csv"), index=False)
    return spatial_joined


def join_incidents_dispatches(incidents_spatial_joined, dispatches, save=True):
    """
    Written by Lynn Zhu
    Purpose: join incidents and dispatches on eventnum
    :param incidents_spatial_joined: pandas data frame of all incidents which have been spatially joined to a territory
    :param dispatches: pandas data frame of all dispatches
    :param save: if True, save joined pandas data frame as csv
    :return: returns joined pandas data frame of incidents and dispatches
    """
    # merge the dispatch and incidents dataset
    joined = pd.merge(incidents_spatial_joined, dispatches, on='eventnum', how='inner')

    # sort by event number and time enroute
    joined = joined.sort_values(['eventnum', 'enroute'], ascending=[True, True])

    # add whether the dispatch station matched the territory of the incident
    joined["dispatch_match"] = np.where(joined['station'] == joined['in_territory'], 1.0, 0.0)

    if save:
        joined.to_csv(os.path.join(os.getcwd(), "data", "master_joined.csv"), index=False)

    return joined


def split_ems_fire(spatial_joined, save=True):
    """
    Written by Lynn Zhu
    :param spatial_joined: spatial joined pandas data frame
    :param save: if True, saves EMS and fire splits as csvs
    :return: EMS and fire splits of spatial_joined if output = True
    """
    # create an EMS and a fire dataset
    data_ems = spatial_joined.loc[spatial_joined['fire_ems'] == 'ems']
    data_fire = spatial_joined.loc[spatial_joined['fire_ems'] == 'fire']

    if save:
        data_ems.to_csv(os.path.join(os.getcwd(), "data", "data_ems.csv"), index=False)
        data_fire.to_csv(os.path.join(os.getcwd(), "data", "data_fire.csv"), index=False)

    return data_ems, data_fire


def import_joined_file(filename):
    """
    Written by: Lynn Zhu
    Purpose: read in any joined csv accurately into a pandas dataframe
    :param filename: name of csv file to be read in
    :return: a pandas dataframe of the imported csv
    """
    df = pd.read_csv(filename, dtype={'eventnum': int, 'type': object, 'longitude': float,
                                      'latitude': float, 'prefix': object, 'fire_ems': object,
                                      'in_territory': int, 'unit': object, 'station': int,
                                      'enroute': object, 'onscene': object, 'hour': int, 'date': object,
                                      'year': int, 'mnth_yr': object, 'dispatchtime': float,
                                      'min_enroute': object, 'responsetime': float, 'vehicletype': object,
                                      'dispatch_match': int})
    return df


def emal_dataset(merged, fire_dispatch, ems_dispatch, timecombined):
    """
    Written by: Erin Kreus and Lynn Zhu
    Purpose: Get EMAL Dispatches that we know dispatching protocol for Validation and Dispatching Simulator
    """
    # Time Report Data: Convert Dispatch Date and In Service Date in time report file to date time
    timecombined['Dispatch_Date'] = pd.to_datetime(timecombined['Dispatch_Date'], format="%m/%d/%y %H:%M")
    timecombined['In_Service_Date'] = pd.to_datetime(timecombined['In_Service_Date'], format="%m/%d/%y %H:%M")

    # Time Report Data: Rename columns for merging
    timecombined = timecombined.rename(index=str,
                                       columns={"Unit": "unit", "Event_Type": "type_hfd", "Event_Number": "eventnum"})

    # Dispatching Protocol: Concat the fire dispatching protocol with the EMS dispatching protocol one
    inthedataset = pd.concat([fire_dispatch, ems_dispatch], ignore_index=True)

    # Dispatching Protocol: Create a dummy variable for merging later so we can determine if we know the
    # Dispatching Protocol for a given dispatch in list of event types we have dispatching protocol for
    inthedataset['indataset'] = 1

    # Dispatching Protocol: Rename TYPE column for Merging in list of event types we have dispatching protocol for
    inthedataset = inthedataset.rename(index=str, columns={"TYPE": "type"})

    # Dispatching Protocol: Drop unnecessary columns in list of event types we have dispatching protocol for
    inthedataset = inthedataset[['type', 'indataset']]

    # Dispatching Protocol: Drop duplicates in list of event types (because there are multiple levels)
    inthedataset = inthedataset.drop_duplicates()

    # Dispatches File: Ensure dispatches file dates are date time
    merged['enroute'] = pd.to_datetime(merged['enroute'], format="%m/%d/%Y %H:%M:%S")
    merged['onscene'] = pd.to_datetime(merged['onscene'], format="%m/%d/%Y %H:%M:%S")

    # PART II: MERGE TIME REPORTS AND DISPATCHING FILE

    # Merge the dispatches file with the time reports on the eventnumber and unit
    merged_allunits = pd.merge(merged, timecombined, how='left', on=['eventnum', 'unit'])

    # Calculate the turnout time and convert to minutes
    merged_allunits['turnouttime'] = merged_allunits['enroute'] - merged_allunits['Dispatch_Date']
    merged_allunits['turnouttime'] = merged_allunits['turnouttime'] / np.timedelta64(1, 'm')

    # Calculate the time to complete and convert to minutes
    merged_allunits['timetocomplete'] = merged_allunits['In_Service_Date'] - merged_allunits['onscene']
    merged_allunits['timetocomplete'] = merged_allunits['timetocomplete'] / np.timedelta64(1, 'm')

    # Filter for EMAL (Engine, Medic, Ambulance, and Ladder) Units only
    merged_allunits = merged_allunits.loc[(merged_allunits['vehicletype'] == "A") |
                                          (merged_allunits['vehicletype'] == "L") |
                                          (merged_allunits['vehicletype'] == "E") |
                                          (merged_allunits['vehicletype'] == "M")]

    # PART III: REMOVE DISPATCHES WE DON'T KNOW DISPATCHING PROTOCOL FOR

    # Merge the all dispatches with time reports file with the list showing the units we have dispatching protocol for
    merged_times = pd.merge(merged_allunits, inthedataset, how='left', on='type')

    # Exclude dispatches we don't know dispatching protocol for
    merged_times = merged_times.loc[(merged_times['indataset'] == 1)]

    # Drop dummy variable for in the dispatching protocol column
    merged_times = merged_times.drop(['indataset'], axis=1)

    # Write to CSV
    merged_times.to_csv(os.path.join(os.getcwd(), "data", "EMAL_dispatches.csv"))
    timecombined = timecombined.dropna(subset=['Dispatch_Date'])
    return merged_times


def unit_profiles_dataset(master, correct_response_ems):
    """
    Written by: Erin Kreus
    """
    engines = master.loc[(master['vehicletype'] == 'E')]
    ladders = master.loc[(master['vehicletype'] == 'L')]
    ambulances = master.loc[(master['vehicletype'] == 'A')]
    medics = master.loc[(master['vehicletype'] == 'M')]

    # get correct response datasets
    engines_correct = correct_response_ems.loc[correct_response_ems['vehicletype'] == 'E']
    ladders_correct = correct_response_ems.loc[correct_response_ems['vehicletype'] == 'L']

    engines.to_csv(os.path.join(os.getcwd(), "unit_profiles_pipeline", "engines_data_unitprofiles.csv"))
    ladders.to_csv(os.path.join(os.getcwd(), "unit_profiles_pipeline", "ladders_data_unitprofiles.csv"))
    ambulances.to_csv(os.path.join(os.getcwd(), "unit_profiles_pipeline", "ambulances_data_unitprofiles.csv"))
    medics.to_csv(os.path.join(os.getcwd(), "unit_profiles_pipeline", "medics_data_unitprofiles.csv"))
    engines_correct.to_csv(os.path.join(os.getcwd(), "unit_profiles_pipeline", "engines_data_unitprofiles_correct.csv"))
    ladders_correct.to_csv(os.path.join(os.getcwd(), "unit_profiles_pipeline", "ladders_data_unitprofiles_correct.csv"))
