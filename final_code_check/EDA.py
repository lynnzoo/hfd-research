import pandas as pd
import numpy as np
import os
import plotly.graph_objs as go
import plotly.io as pio
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import warnings
warnings.filterwarnings("ignore")
from datetime import timedelta
from collections import defaultdict
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
from functools import partial


def find_chains(spatial_joined, subset=None, save=True):
    """
    Written by Lynn Zhu
    :param spatial_joined: pandas dataframe of spatial joined incidents and dispatches
    :param subset: a two item list, take a subset of the dataframe between [x,y] , if None, will take full
    :param save: if True, save results of find_chains where pandas dataframe is saved as csv and dictionaries are saved
    as pickle files
    :param output: if True, return results of find_chains
    :return: dictionaries and pandas dataframe (results)
    """
    aml_dispatches = spatial_joined[spatial_joined.vehicletype.isin(["A", "M", "L"])]
    aml_dispatches.sort_values(by="eventnum", ascending=True, inplace=True)
    aml_dispatches.reset_index(drop=True, inplace=True)
    aml_dispatches["enroute"] = pd.to_datetime(aml_dispatches["enroute"], format="%m/%d/%Y %H:%M:%S")
    aml_dispatches["onscene"] = pd.to_datetime(aml_dispatches["onscene"], format="%m/%d/%Y %H:%M:%S")
    df = aml_dispatches[["eventnum", "station", "enroute", "onscene", "vehicletype", "unit",
                         "fire_ems", "incident_juris", "in_juris"]]

    # select only rows in which in_juris == 0 and which are in a jurisdiction covered by the HFD
    distress_responses = df[df.in_juris == 0]
    distress_responses = distress_responses[distress_responses.incident_juris != 0]

    # order by eventnum (chronologically)
    distress_responses.sort_values(by="eventnum", ascending=True, inplace=True)
    distress_responses.reset_index(drop=True, inplace=True)

    subset1 = input("What row number should the subset start at? "
                    "If you would like to use entire data, press enter without inputting a number.")
    if subset1:
        subset2 = input("What row number should the subset end at? The max is s%" % spatial_joined.shape[0])
        subset = [subset1, subset2]

    # take a subset to speed up
    if subset:
        subset = distress_responses.iloc[subset[0]:subset[1], :]
    else:
        subset = distress_responses.copy()

    distress_left = subset.copy()
    marked_chains = subset.copy()
    marked_chains["chain_id"] = 0
    chain_id = 1
    chains = defaultdict(partial(np.ndarray, 0))
    first_vehicles = defaultdict(partial(np.ndarray, 0))
    while distress_left.shape[0] != 0:
        row = distress_left.iloc[0, :]
        idx = row.name
        marked_chains.at[idx, 'chain_id'] = chain_id
        gone_time = row.enroute + timedelta(hours=2)
        while_away = distress_responses[(distress_responses['eventnum'] > row.eventnum) &
                                        (distress_responses['incident_juris'] == row.station) &
                                        (distress_responses['vehicletype'] == row.vehicletype) &
                                        (distress_responses['enroute'] <= gone_time) &
                                        (distress_responses['station'] != row.station)]
        chain = 1
        while while_away.shape[0] != 0:
            next_row = while_away.iloc[0, :]
            next_idx = next_row.name
            marked_chains.at[next_idx, 'chain_id'] = chain_id
            if next_row.eventnum in distress_left.eventnum.unique():
                distress_left.drop(distress_left[distress_left.eventnum == next_row.eventnum].index, inplace=True)
            chain += 1
            next_gone_time = row.enroute + timedelta(hours=2)
            while_away = distress_responses[(distress_responses['eventnum'] > next_row.eventnum) &
                                            (distress_responses['incident_juris'] == next_row.station) &
                                            (distress_responses['vehicletype'] == next_row.vehicletype) &
                                            (distress_responses['enroute'] <= next_gone_time) &
                                            (distress_responses['station'] != next_row.station)]
        chains[row.incident_juris] = np.append(chains[row.incident_juris], chain)
        first_vehicles[row.incident_juris] = np.append(first_vehicles[row.incident_juris], row.unit)
        chain_id += 1
        distress_left.drop(distress_left.head(1).index, inplace=True)
    if save is True:
        pickle.dump(chains, open(os.path.join(os.getcwd(), "data", 'chains_result.p'), 'wb+'))
        pickle.dump(first_vehicles, open(os.path.join(os.getcwd(), "data", 'chains_first_vehicles.p'), 'wb+'))
        marked_chains.to_csv(os.path.join(os.getcwd(), "data", 'marked_chains.csv'))

    return chains, first_vehicles, marked_chains


def combine_chain_files(chain_result_files, first_vehicle_files):
    """
    Written by Lynn Zhu
    Combines pickle files from previously run find chains.
    :param chain_result_files:
    :param first_vehicle_files:
    :return:
    """
    chain_results = []
    first_vehicles = []
    for chain_result in chain_result_files:
        temp_dict = pickle.load(open(chain_result, 'rb+'))
        chain_results.append(temp_dict)
    for vehicle_result in first_vehicle_files:
        temp_dict = pickle.load(open(vehicle_result, 'rb+'))
        first_vehicles.append(temp_dict)
    keys = chain_results[0].keys()
    combined_chain_results = {key: np.concatenate([r[key] for r in chain_results]) for key in keys}
    combined_first_vehicles = {key: np.concatenate([fv[key] for fv in first_vehicles]) for key in keys}
    return combined_chain_results, combined_first_vehicles


def chain_analysis(chains, first_vehicles, save=True):
    """
    Written by Lynn Zhu
    """
    avg = defaultdict(float)
    for key, value in chains.items():
        avg[key] = np.average(value)

    avg_df = pd.DataFrame.from_dict(avg, orient='index', dtype=float)
    avg_df['start_inc_juris'] = avg_df.index
    avg_df.columns=['average', 'start_inc_juris']
    avg_df = avg_df[["start_inc_juris", "average"]]
    for key, value in first_vehicles.items():
        first_vehicles[key] = first_vehicles[key].astype('<U1')
    counts = defaultdict(dict)
    for key, value in first_vehicles.items():
        counts[key] = collections.Counter(value)
    if save is True:
        avg_df.to_csv(os.path.join(os.getcwd(), "data", 'average_chain_lengths.csv'), index=False)
        pickle.dump(counts, open(os.path.join(os.getcwd(), "data", 'chain_first_vehicle_counts.p'), 'wb+'))
    return avg_df, counts


def is_appropriate_response(row):
    """
    Written by: Erin Kreus and Lynn Zhu
    Purpose: determines if the vehicle dispatched was appropriate based on the incident type, for one row
    """
    if (row['app_unit_indicator'] == "A1") & (row['vehicletype'] in (["E", "L", "A"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "AI") & (row['vehicletype'] in (["A", "E", "L", "M", "SQ"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "A4") & (row['vehicletype'] in (["A"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "A6") & (row['vehicletype'] in (["A", "M", "SQ"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "A7") & (row['vehicletype'] in (["A"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "A8") & (row['vehicletype'] in (["A"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "B7") & (row['vehicletype'] in (["M", "SQ"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "B8") & (row['vehicletype'] in (["M", "SQ"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "C1") & (row['vehicletype'] in (["M", "A", "SQ"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "C2") & (row['vehicletype'] in (["M", "A", "SQ", "L", "E"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "C3") & (row['vehicletype'] in (["M", "A", "SQ", "L", "E"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "C6") & (row['vehicletype'] in (["M", "A", "SQ", "L", "E"])):
        val = 'Correct Response'
    elif (row['app_unit_indicator'] == "D1") & (row['vehicletype'] in (["M", "A", "SQ", "L", "E"])):
        val = 'Correct Response'
    else:
        val = "Incorrect Response"
    return val


def analyse_ems_response(data_ems, save=True):
    """
    Written by: Erin Kreus and Lynn Zhu
    Purpose: analyzes whether each dispatch to an EMS incident was a correct response
    :param spatial_joined:: pandas dataframe of dispatches and incidents spatial joined
    """
    copy = data_ems.copy(deep=True)
    # # 1-get ending indicator that will later be used to determine if this was a correct vehicle type
    copy['app_unit_indicator'] = copy['type'].str[-2:]
    #
    # # filter out non-EMS incidents
    # correct_response = copy.loc[copy['fire_ems'] == 'ems']

    # filter for vehicles we examining in the study
    correct_response = copy.loc[(data_ems['vehicletype'] == 'L') |
                                (copy['vehicletype'] == 'T') |
                                (copy['vehicletype'] == 'E') |
                                (copy['vehicletype'] == 'A') |
                                (copy['vehicletype'] == 'M') |
                                (copy['vehicletype'] == 'SQ')]

    # filter the response indicators for EMS incidents
    correct_response = correct_response.loc[(correct_response['app_unit_indicator'] == 'A1') |
                                            (correct_response['app_unit_indicator'] == 'AI') |
                                            (correct_response['app_unit_indicator'] == 'A4') |
                                            (correct_response['app_unit_indicator'] == 'A6') |
                                            (correct_response['app_unit_indicator'] == 'A7') |
                                            (correct_response['app_unit_indicator'] == 'A8') |
                                            (correct_response['app_unit_indicator'] == 'B7') |
                                            (correct_response['app_unit_indicator'] == 'B8') |
                                            (correct_response['app_unit_indicator'] == 'C1') |
                                            (correct_response['app_unit_indicator'] == 'C2') |
                                            (correct_response['app_unit_indicator'] == 'C3') |
                                            (correct_response['app_unit_indicator'] == 'C6') |
                                            (correct_response['app_unit_indicator'] == 'D1')]

    correct_response['response'] = correct_response.apply(is_appropriate_response, axis=1)
    if save is True:
        correct_response.to_csv(os.path.join(os.getcwd(), "data", "correct_response_ems.csv"), index=False)
    return correct_response


def calculate_helping_fraction(df, save=True):
    """
    Written by: Lynn Zhu
    Purpose: calculates the fraction of the dispatches from a station which are responding
    outside of its territory (helping another territory)
    """
    station_list = df['station'].unique()
    helping_fraction_dict = defaultdict(float)
    for station in station_list:
        subframe = df.loc[df['station'] == station]
        total = subframe.shape[0]
        numerator = subframe['dispatch_match'].sum()
        helping_fraction_dict[station] = [1-float(numerator/total)]
    helping_fractions = pd.DataFrame.from_dict(helping_fraction_dict, orient='index', dtype=float,
                                               columns=['helping_fraction'])
    helping_fractions['territory'] = helping_fractions.index
    helping_fractions = helping_fractions[['territory', 'helping_fraction']]
    if save is True:
        helping_fractions.to_csv(os.path.join(os.getcwd(), "data", "helping_fractions.csv"), index=False)

    return helping_fractions


def calculate_distress_fraction(df, save=True):
    """
    Written by: Lynn Zhu
    Purpose: calculates the fraction of the dispatches for a territory where the unit responding
    is not from the territory's station (is in distress)
    """
    territory_list = df['in_territory'].unique()
    territory_list = territory_list[~pd.isnull(territory_list)]
    distress_fraction_dict = defaultdict(float)
    for terr in territory_list:
        subframe = df.loc[df['in_territory'] == terr]
        total = subframe.shape[0]
        numerator = subframe['in_territory'].sum()
        distress_fraction_dict[terr] = [1-(numerator/total)]
    distress_fractions = pd.DataFrame.from_dict(distress_fraction_dict, orient='index', dtype=float,
                                                columns=['distress_fraction'])
    distress_fractions['territory'] = distress_fractions.index
    distress_fractions=distress_fractions[['territory', 'distress_fraction']]
    if save is True:
        distress_fractions.to_csv(os.path.join(os.getcwd(), "data", "distress_fractions.csv"), index=False)

    return distress_fractions


def median_response_by_station(master, save=True):
    """
    Written by Lynn Zhu
    Purpose: to calculate medians response times by station
    """
    medians = master.groupby(['station', 'year'])['responsetime'].median().reset_index(name='median')
    if save is True:
        medians.to_csv(os.path.join(os.getcwd(), "data", "median_by_station.csv"), index=False)


def save_plot(fig, name):
    if not os.path.exists('figures'):
        os.mkdir('figures')
    figpath = 'figures/' + name + '.png'
    pio.write_image(fig, file=figpath, format='png')

# FIGURES
# figure 1: cleaning/wrangling diagram

# figure 2:
def plot_num_incidents(merged, name):
    """
    Written by: Erin Kreus and Lynn Zhu
    Purpose: plots number of total incidents responded to by HFD over time (from 2012-2017),
    broken down by EMS and fire incidents
    :param merged: pandas dataframe of dispatches and incidents left joined on eventnumber
    :param name: name of plot image to be saved, as string
    """
    # get only the one that arrived first
    first_responders = merged.groupby((merged['eventnum'] != merged['eventnum'].shift()).cumsum().values).first()

    # use first_responders which is our dispatch data set with the first response time (this gives us the count of
    # incidents) and exclude second half of 2011 and first half of 2018 because the data for those years are incomplete
    first_responders = first_responders.loc[first_responders['year'] != 2011]
    first_responders = first_responders.loc[first_responders['year'] != 2018]

    # create an EMS and a fire dataset to make the graphs
    data_ems = first_responders.loc[first_responders['fire_ems'] == 'ems']
    data_fire = first_responders.loc[first_responders['fire_ems'] == 'fire']

    # calculate the count of incidents by month using a group by month-year
    responsetime_emscnts = data_ems.groupby(['mnth_yr']).size().reset_index(name='counts')
    responsetime_firecnts = data_fire.groupby(['mnth_yr']).size().reset_index(name='counts')

    # convert to month year to date time for graphing and sort by month year
    responsetime_emscnts['mnth_yr'] = pd.to_datetime(responsetime_emscnts['mnth_yr'], format="%B-%Y")
    responsetime_firecnts['mnth_yr'] = pd.to_datetime(responsetime_firecnts['mnth_yr'], format="%B-%Y")
    responsetime_emscnts = responsetime_emscnts.sort_values(by='mnth_yr')
    responsetime_firecnts = responsetime_firecnts.sort_values(by='mnth_yr')

    # create traces for graphing
    trace0 = go.Scatter(x=responsetime_emscnts['mnth_yr'], y=responsetime_emscnts['counts'], name='EMS Incidents',
                        line=dict(width=2))
    trace1 = go.Scatter(x=responsetime_firecnts['mnth_yr'], y=responsetime_firecnts['counts'], name='Fire Incidents',
                        line=dict(width=2))
    traces = [trace0, trace1]

    # use plotly package to create graphs
    layout = go.Layout(
        barmode='overlay',
        title='Total Number of Incidents Responded to by the HFD <br> 2012-2017',
        xaxis=dict(title='Date'),
        width=700,
        height=500,
        yaxis=dict(title='Number of Incidents', range=[0, 30000]))
    fig = dict(data=traces, layout=layout)

    save_plot(fig, name)


# figure 3:
def plot_dispatch_times(data_ems, data_fire, name):
    """
    Written by: Erin Kreus and Lynn Zhu
    Purpose: plots histogram of number of dispatches by the time of day, broken down by EMS and fire dispatches
    :param merged:: pandas dataframe of dispatches and incidents left joined on eventnumber
    :param name: name of plot image to be saved, as string
    """
    # 1-create two new datasets-an EMS and a fire dataset to make it easier to make traces
    # data_ems = joined.loc[joined['fire_ems'] == 'ems']
    # data_fire = joined.loc[joined['fire_ems'] == 'fire']

    # 2-calculate the number of incidents by hour using a group by hour.
    timeofday_e = data_ems.groupby(['hour']).size().reset_index(name='counts')
    timeofday_f = data_fire.groupby(['hour']).size().reset_index(name='counts')

    # 3-create traces for the graph to plot total dispatches by Time of Day (one trace for EMS, one trace for fire)
    trace0 = go.Bar(
                x=['12AM', '1AM', '2AM', '3AM', '4AM', '5AM', '6AM', '7AM', '8AM', '9AM', '10AM', '11AM', '12PM',
                   '1PM', '2PM', '3PM', '4PM', '5PM', '6PM', '7PM', '8PM', '9PM', '10PM', '11PM'],
                y=timeofday_e['counts'],
                name='EMS Incidents')
    trace1 = go.Bar(
                x=['12AM', '1AM', '2AM', '3AM', '4AM', '5AM', '6AM', '7AM', '8AM', '9AM', '10AM', '11AM', '12PM',
                   '1PM', '2PM', '3PM', '4PM', '5PM', '6PM', '7PM', '8PM', '9PM', '10PM', '11PM'],
                y=timeofday_f['counts'],
                name='Fire Incidents')

    # 4-create the plot using plotly package
    traces = [trace0, trace1]
    layout = go.Layout(
        barmode='stack',
        title='The HFD Total Individuals Dispatches by Time of Day <br> 2012-2018',
        xaxis=dict(title='Time of Day'),
        width=700,
        height=500,
        yaxis=dict(title='Total Dispatches'))

    fig = go.Figure(data=traces, layout=layout)
    save_plot(fig, name)


# figure 4:
def prop_correct_response(correct_response_ems, name):
    """
    Written by: Erin Kreus and Lynn Zhu
    Purpose: plot the proportion of EMS incidents where the first unit on-scene was an
    appropriate unit for the response from 2012-2018
    """
    # 1-sort by event number and time arrived on-scene
    correct_ems_first = correct_response_ems.sort_values(['eventnum', 'onscene'], ascending=[True, True])

    # 2-get only the one that arrived first
    correct_ems_first = correct_ems_first.groupby((correct_ems_first['eventnum']
                                                   != correct_ems_first['eventnum'].shift()).cumsum().values).first()

    # 3-make data set with only correct responses
    correct_response_ems_first = correct_ems_first.loc[correct_ems_first['response'] == 'Correct Response']

    # 3-get total count of incidents by station
    all_counts = correct_ems_first.groupby(['mnth_yr']).size().reset_index(name='total')

    # 4-get total count of incidents by station that were correct
    correct_counts = correct_response_ems_first.groupby(['mnth_yr']).size().reset_index(name='correct')

    # 5-merge the total count of correct and total data set
    correct_first = pd.merge(all_counts, correct_counts, on=['mnth_yr'], how='left')

    # 6-get percent of incidents where the first response was the correct response and sort the result
    correct_first['proportion'] = 100*correct_first['correct']/correct_first['total']
    correct_first['mnth_yr'] = pd.to_datetime(correct_first['mnth_yr'], format="%B-%Y")
    correct_first_analysis = correct_first.sort_values(by='mnth_yr')

    # 7-create traces for graphing
    trace0 = go.Scatter(x=correct_first_analysis['mnth_yr'], y=correct_first_analysis['proportion'],
                        line=dict(width=2))
    traces = [trace0]

    # 8-use plotly package to create graphs
    layout = go.Layout(
        barmode='overlay',
        title='Percentage of Incidents when the First Unit On-Scene '
                'was an Appropriate Unit <br> EMS Responses Only <br> 2012-2018',
        xaxis=dict(title='Date'),
        width=700,
        height=500,
        yaxis=dict(title='Percentage of Incidents', range=[50, 100]))
    fig = dict(data=traces, layout=layout)
    # plot(fig, filename='styled-line')

    save_plot(fig, name)


# figure 5:
def plot_hist_correct(correct_response_ems, name):
    """
    Written by: Erin Kreus
    Purpose: plots normalized distribution of response times for dispatches divided by
    correct and incorrect responses.
    """
    # 1-make two separate data sets for the traces for each type of response (correct and incorrect)
    responsecorrect = correct_response_ems.loc[correct_response_ems['response'] == 'Correct Response']
    responseincorrect = correct_response_ems.loc[correct_response_ems['response'] == 'Incorrect Response']

    # 2-create an upper bound value for the graph equal to the 99th percentile
    upperbound = correct_response_ems['responsetime'].quantile(.99)

    # 3-filter the data to be less than the upperbound
    responsecorrect = responsecorrect[(responsecorrect['responsetime'] < upperbound)]
    responseincorrect = responseincorrect[(responseincorrect['responsetime'] < upperbound)]

    # 4-Create traces for the graph to plot the distributions by response correctness
    trace1 = go.Histogram(x=responsecorrect['responsetime'],
                          histnorm='probability', marker=dict(), nbinsx=225, opacity=0.65, name='Correct Responses')
    trace2 = go.Histogram(x=responseincorrect['responsetime'],
                          histnorm='probability', marker=dict(), nbinsx=225, opacity=.9, name='Incorrect Responses')
    traces = [trace2, trace1]

    # 5-create the plot using plotly package
    layout = go.Layout(
        barmode='overlay',
        title='Normalized Distribution of Response Times for Correct/Incorrect Responses <br> EMS Responses Only',
        xaxis=dict(title='Response Time'),
        width=800,
        height=500,
        yaxis=dict(title='Probability'))
    fig = go.Figure(data=traces, layout=layout)
    # plot(fig, filename='overlaid histogram')

    save_plot(fig, name)


# figure 6
def fire_gone(correct_response_ems, spatial_joined, name):
    """
    Written by: Erin Kreus
    Purpose: determine how frequently for fire incidents, a fire unit
    (engine or ladder) is unavailable to respond because it is busy
    unnecessarily assisting an EMS incident
    """
    # STEP 1: CREATE TWO ADDITIONAL DATASETS
    # a-Engines and Ladders are unnecessarily sent to an EMS Incident
    # get dataset where Engines/Ladders are unnecessarily responding to an EMS incident
    dispatches_engines = correct_response_ems.loc[(correct_response_ems['response'] == 'Incorrect Response') &
                                                  ((correct_response_ems['vehicletype'] == "E") |
                                                   (correct_response_ems['vehicletype'] == "L"))]
    # remove all unnecessary columns
    dispatches_engines_bad = dispatches_engines[['station', 'onscene', 'vehicletype']]

    # create an indicator variable for all points that says the fire vehicle was busy at an EMS incident
    dispatches_engines_bad['vehicle_at_ems'] = 1

    # rename the station column as the merging column to make merging simpler
    dispatches_engines_bad = dispatches_engines_bad.rename(index=str, columns={"station": "MergingColumn"})
    dispatches_engines_bad["MergingColumn"] = dispatches_engines_bad["MergingColumn"].astype(int)

    # convert enroute time to datetime (to avoid error messages later)
    dispatches_engines_bad['onscene'] = pd.to_datetime(dispatches_engines_bad.onscene)

    # sort dataset by enroute time
    dispatches_engines_bad = dispatches_engines_bad.sort_values(by=['onscene'])

    # drop NA rows
    dispatches_engines_bad = dispatches_engines_bad.dropna()

    # b-Engines and Ladders are sent to an Fire Incident
    # get dataset of fire incidents only
    dispatches_fire_only = spatial_joined.loc[(spatial_joined['fire_ems'] == "fire")]

    # get dataset of first dispatches only (so we do not double-count incidents)
    dispatches_fire_only = dispatches_fire_only.sort_values(by=['onscene'])
    dispatches_fire_only = dispatches_fire_only.groupby(
        (dispatches_fire_only['eventnum'] != dispatches_fire_only['eventnum'].shift()).cumsum().values).first()

    # create a new column that determines if the vehicle first sent was busy
    dispatches_fire_only['juris_indicator'] = pd.np.where((dispatches_fire_only['station'] ==
                                                           dispatches_fire_only['incident_juris']), 0, 1)
    dispatches_fire_only['busyorno'] = pd.np.where((dispatches_fire_only['juris_indicator'] == 1), "busy", "notbusy")

    # filter for busy dispatches only
    dispatches_fire_first = dispatches_fire_only.loc[(dispatches_fire_only['busyorno'] == "busy")]

    # filter for Engine and Ladder dispatches only
    dispatches_fire_first = dispatches_fire_first.loc[((dispatches_fire_first['vehicletype'] == "E") |
                                                      (dispatches_fire_first['vehicletype'] == "L"))]
    # drop unnecessary columns
    dispatches_fire_first = dispatches_fire_first[['onscene', 'date', 'year', 'hour', 'incident_juris', 'vehicletype']]

    # rename Jurisdiction column as the Merging Column to make it easier to run later
    dispatches_fire_first = dispatches_fire_first.rename(index=str, columns={"incident_juris": "MergingColumn"})

    # ensure enroute time is a date time variable
    dispatches_fire_first['onscene'] = pd.to_datetime(dispatches_fire_first.onscene)

    # drop na values
    dispatches_fire_first = dispatches_fire_first.dropna()

    # sort by enroute time
    dispatches_fire_first=dispatches_fire_first.sort_values(by=['onscene'])

    # STEP 2: MERGE THE TWO DATASETS TOGETHER TO GET FIRE INCIDENTS WHERE THE FIRE VEHICLE WAS BUSY ASSISTING EMS
    # merge datasets where time the fire incident occured within an hour of the fire truck being dispatched
    # to an unnecessary EMS incident
    fire_but_busy_with_ems =pd.merge_asof(dispatches_fire_first, dispatches_engines_bad, direction='forward',
                                          on='onscene', by=['MergingColumn', 'vehicletype'],
                                          tolerance=pd.Timedelta('3200s'))
    # Determine whether the vehicle was busy with an EMS incident
    fire_but_busy_with_ems=fire_but_busy_with_ems.loc[(fire_but_busy_with_ems['vehicle_at_ems'] == 1)]

    # STEP 3: MAKE TABLE THAT DETERMINES PERCENTAGE OF THE TIME A FIRE VEHICLE IS BUSY WITH EMS
    # make table with number of times an fire vehicle is busy with EMS when there is a fire
    table500 = fire_but_busy_with_ems.groupby(['year']).size().reset_index(name='Number_Fire_Bad')
    # get count of total fire incidents
    table501 = dispatches_fire_first.groupby(['year']).size().reset_index(name='Total')
    # get table
    table502 = pd.merge(table500, table501, on="year")
    # determine percentage of the time
    table502['percentage'] = 100*table502['Number_Fire_Bad']/table502['Total']

    # STEP 4: Graph results
    # create traces for graphing
    trace0 = go.Scatter(x=table502['year'], y=table502['percentage'], name='Percentage',
                        line=dict(width=2))
    traces = [trace0]

    # use plotly package for graphing
    layout = go.Layout(
            barmode='overlay',
            title='Percentage of Time Engines/Ladders are Unavailable <br> '
                  'for Fire Incidents while Busy with EMS Incidents<br> 2011-2018',
            xaxis=dict(title='Year'),
            width=700,
            height=500,
            yaxis=dict(title='Percent of Fire Incidents (%)', range=[0, 15]))
    fig = dict(data=traces, layout=layout)
    # plot(fig, filename='styled-line')

    save_plot(fig, name)


# figure 8
def plot_incident_response_time(merged, name):
    """
    Written by: Erin Kreus and Lynn Zhu
    Purpose: plots the median and 90th percentile response time for first unit on-scene over time (2012-2018)'
    :param merged:: pandas dataframe of dispatches and incidents left joined on eventnumber
    """
    # 1-use only dispatch data set with the first response time (this gives us the count of incidents)
    first_responders = merged.groupby((merged['eventnum'] != merged['eventnum'].shift()).cumsum().values).first()

    # 2-calculate the median and 90th quantile of incidents by month using a group by month-year
    first_median = first_responders.groupby('mnth_yr')['responsetime'].median().reset_index(name='median')
    first90 = first_responders.groupby('mnth_yr')['responsetime'].quantile(.9).reset_index(name='90th')

    # 3-convert to month year to date time for graphing and sort by month year
    first_median['mnth_yr'] = pd.to_datetime(first_median['mnth_yr'], format="%B-%Y")
    first90['mnth_yr'] = pd.to_datetime(first90['mnth_yr'], format="%B-%Y")
    first_median = first_median.sort_values(by='mnth_yr')
    first90 = first90.sort_values(by='mnth_yr')

    # 4-create traces for graphing
    trace0 = go.Scatter(x=first_median['mnth_yr'], y=first_median['median'], name='Median Response Time',
                        line=dict(width=2))
    trace1 = go.Scatter(x=first90['mnth_yr'], y=first90['90th'], name='90th Percentile Response Time',
                        line=dict(width=2))
    traces = [trace0, trace1]

    # 5-use plotly package to create graphs
    layout = go.Layout(
        barmode='overlay',
        title='Median and 90th Percentile Response Time for First Unit On-Scene <br> 2012-2018',
        xaxis=dict(title='Date'),
        width=700,
        height=500,
        yaxis=dict(title='Response Time (minutes)', range=[3, 10]))
    fig = dict(data=traces, layout=layout)
    # plot(fig, filename='styled-line')
    save_plot(fig, name)


# figure 9
def plot_dispatch_dist(merged, name):
    """
    Written by: Erin Kreus and Lynn Zhu
    Purpose: plot the distributions of dispatch times broken down by vehicle type, for dispatches between 2012-2018
    :param merged:: pandas dataframe of dispatches and incidents left joined on eventnumber
    """
    # 1-make three seperate data sets for the traces for each type of vehicle studied (ladder, engine, and ambulance)
    vehicle_eng = merged.loc[merged['vehicletype'] == 'E']
    vehicle_lad = merged.loc[(merged['vehicletype'] == 'L') | (merged['vehicletype'] == 'T')]
    vehicle_amb = merged.loc[(merged['vehicletype'] == 'M') | (merged['vehicletype'] == 'A')]

    # 2-create an upperbound value for the graph equal to the 99th percentile
    upperbound = merged['dispatchtime'].quantile(.99)

    # 3-filter the data to be less than the upperbound
    data_filtered_eng = vehicle_eng[(vehicle_eng['dispatchtime'] < upperbound)]
    data_filtered_lad = vehicle_lad[(vehicle_lad['dispatchtime'] < upperbound)]
    data_filtered_amb = vehicle_amb[(vehicle_amb['dispatchtime'] < upperbound)]

    # 5-Create traces for the graph to plot the distributions by type
    trace1 = go.Histogram(x=data_filtered_eng['dispatchtime'], marker=dict(color='#2ca02c'), nbinsx=225, opacity=0.5,
                          name='Engines')
    trace2 = go.Histogram(x=data_filtered_lad['dispatchtime'], marker=dict(color='#d62728'), nbinsx=225, opacity=.74,
                          name='Ladders')
    trace3 = go.Histogram(x=data_filtered_amb['dispatchtime'], marker=dict(color='#17becf'), nbinsx=225, opacity=0.5,
                          name='Ambulances')
    traces = [trace3, trace1, trace2]

    # 4-create the plot using plotly package
    layout = go.Layout(
        barmode='overlay',
        title='Distribution of Dispatch Times among Ambulance, Ladder, and Engine Units <br> 2012-2018',
        xaxis=dict(title='Dispatch Time (minutes)'),
        width=700,
        height=500,
        yaxis=dict(title='Number of Dispatches'))
    fig = go.Figure(data=traces, layout=layout)
    # plot(fig, filename='overlaid histogram')

    save_plot(fig, name)

# figure 10: geographic distribution of median response times (maps)


# figure 11: comparing incident counts and response times in the houston area (maps)


# figure 12 station 37:
def plot_station37(spatial_joined, name):
    """
    Written by: Erin Kreus
    Purpose: plots distribution of response times for station 37 which represents
    the 90th percentile for all station median response times for 2017
    """
    # filter the dataset to only include 2018 data and station 37 data
    station37 = spatial_joined.loc[(spatial_joined['year'] == 2017) & (spatial_joined['station'] == 37.0)]

    # filter the data to be less than 30 in response times
    station37 = station37[(station37['responsetime'] < 30)]

    # create trace for the graph to plot the distributions by response times for station 37
    trace1 = go.Histogram(x=station37['responsetime'],marker=dict(),nbinsx = 75)
    traces = [trace1]

    # create the plot using plotly package
    layout = go.Layout(
        barmode='overlay',
        title='Distribution of Response Times for Station 37 <br> 2017 only',
        xaxis=dict(title='Response Time'),
        width=800,
        height=500,
        yaxis=dict(title='Number of Dispatches'))
    fig = go.Figure(data=traces, layout=layout)
    # plot(fig, filename='overlaid histogram')

    save_plot(fig, name)


# figure 12 station 72:
def plot_station72(spatial_joined, name):
    """
    Written by: Erin Kreus
    Purpose: plots distribution of response times for station 72 which represents
    the 10th percentile for all station median response times for 2017
    """
    # filter the dataset to only include 2018 data and station 72 data
    station72 = spatial_joined.loc[(spatial_joined['year'] == 2017) & (spatial_joined['station'] == 72.0)]

    # filter the data to be less than 30 in response times
    station72 = station72[(station72['responsetime'] < 30)]

    # create trace for the graph to plot the distributions by response times for station 37
    trace1 = go.Histogram(x=station72['responsetime'], marker=dict(), nbinsx=75)
    traces = [trace1]

    # create the plot using plotly package
    layout = go.Layout(
        barmode='overlay',
        title='Distribution of Response Times for Station 72 <br> 2017 only',
        xaxis=dict(title='Response Time'),
        width=800,
        height=500,
        yaxis=dict(title='Number of Dispatches'))
    fig = go.Figure(data=traces, layout=layout)
    # plot(fig, filename='overlaid histogram')

    save_plot(fig, name)


# figure 15:
def plot_hist_jurisdictions(spatial_joined, name):
    """
    Written by: Erin Kreus
    Purpose: plots normalized distribution of response times for dispatches for the HFD over time
    (from 2011-2018), broken down by correct and incorrect jurisdiction.
    :param spatial_joined: pandas dataframe of dispatches and incidents left joined on eventnumber
    """
    # create two datasets for graphing-(1) for dispatches who travelled outside their
    # jurisdiction and (2) for dispatches who did not travel outside their jurisdiction
    spatial_joined['juris_indicator'] = pd.np.where((spatial_joined['station'] ==
                                                     spatial_joined['incident_juris']), 0, 1)
    dispatches_jurisdiction_correct = spatial_joined.loc[spatial_joined['juris_indicator'] == 0]
    dispatches_jurisdiction_incorrect = spatial_joined.loc[spatial_joined['juris_indicator'] == 1]

    # filter the two datasets that have a response time <30 for a limit for the graph.
    dispatches_jurisdiction_correct = dispatches_jurisdiction_correct[
        (dispatches_jurisdiction_correct['responsetime'] < 30)]
    dispatches_jurisdiction_incorrect = dispatches_jurisdiction_incorrect[
        (dispatches_jurisdiction_incorrect['responsetime'] < 30)]

    # create traces for the graph to plot the distributions by response correctness
    trace1 = go.Histogram(x=dispatches_jurisdiction_correct['responsetime'], histnorm='probability', marker=dict(),
                          nbinsx=225, opacity=0.65, name='Within Territory')
    trace2 = go.Histogram(x=dispatches_jurisdiction_incorrect['responsetime'], histnorm='probability', marker=dict(),
                          nbinsx=225, opacity=.9, name='Outside Territory')
    traces = [trace2, trace1]

    # create the plot using plotly package
    layout = go.Layout(
        barmode='overlay',
        title='Normalized Distribution of Response Times for Within/Outside Jurisdiction <br> 2011-2018',
        xaxis=dict(title='Response Time'),
        width=800,
        height=500,
        yaxis=dict(title='Probability'))
    fig = go.Figure(data=traces, layout=layout)
    # plot(fig, filename='overlaid histogram')

    save_plot(fig, name)


# table 1: proportion of "slow" responses requiring a vehicle to leave an assigned jurisdiction
def table_over10_minutes_eda(spatial_joined, name):
    """
   Written by: Erin Kreus
   Purpose: determine for dispatches with response times >10 minutes,
   how frequently the unit came from outside of the incident jurisdiction
   """
    # filter all dispatches to exclude those <10 minutes
    spatial_joined['juris_indicator'] = pd.np.where((spatial_joined['station'] ==
                                                     spatial_joined['incident_juris']), 0, 1)
    data_testinglong = spatial_joined.loc[spatial_joined['responsetime'] > 10]

    # replace the busy indicator variable with text indicator to
    # assist with creating pivot table
    data_testinglong['busy'] = pd.np.where((data_testinglong['juris_indicator'] == 0),
                                           "correctstation", "incorrectstation")

    # create a pivot table to determine the count of dispatches with response times >10 minutes
    # by station (within jurisdiction or outside jurisdiction)
    table = pd.pivot_table(data_testinglong, values='eventnum', index=['year'], columns=['busy'], aggfunc='count')

    # flatten pivot table to convert to dataframe
    flattened = pd.DataFrame(table.to_records())

    # get proportions from counts
    flattened['proportion_wrongstation'] = 100 * flattened['incorrectstation'] / (
                flattened['correctstation'] + flattened['incorrectstation'])
    flattened['proportion_rightstation'] = 100 * flattened['correctstation'] / (
                flattened['correctstation'] + flattened['incorrectstation'])

    # drop unnecessary columns
    flattened = flattened.drop(['correctstation', 'incorrectstation'], axis=1)

    # save table in tables folder

    if not os.path.exists('tables'):
        os.mkdir('tables')
    tablename = name + '.csv'
    flattened.to_csv(os.path.join(os.getcwd(), "tables", tablename), index=False)


# table 2: number of dispatches in each archetype, 2011 - 2018
def table_case_study_overall(spatial_joined, name):
    """
   Written by: Erin Kreus
   Purpose: determine the counts associated with each dispatch type from the case
   studies
   """
    # rename the merged dataset
    casestudy_meta = spatial_joined

    # create an indicator variable of if the event occured within the jurisdiction
    casestudy_meta['juris_indicator'] = pd.np.where((casestudy_meta['station'] ==
                                                     casestudy_meta['incident_juris']), 0, 1)

    # Create a new variable for if the dispatch was busy or no
    casestudy_meta['busyness'] = pd.np.where((casestudy_meta['juris_indicator'] == 1), "busy", "notbusy")

    # create a new variable for if the dispatch was slow (>=10 minutes) or not slow (<10 minutes)
    casestudy_meta['sloworno'] = pd.np.where((casestudy_meta['responsetime'] >= 10), "slow", "notslow")

    # create a pivot table
    table = pd.pivot_table(casestudy_meta, values='eventnum', index=['sloworno'], columns=['busyness'], aggfunc='count')

    # flatten the pivot table so we can convert it to csv
    flattened = pd.DataFrame(table.to_records())

    # save table in tables folder
    if not os.path.exists('tables'):
        os.mkdir('tables')
    tablename = name + '.csv'
    flattened.to_csv(os.path.join(os.getcwd(), 'tables', tablename), index=False)


# csvs for figure 16 - 19
def gis_case_studies_data(spatial_joined, stations):
    """
    Written by: Erin Kreus
    Purpose: create output for 4 case studies associated with the analysis.
    Notably, these cannot be looped due to differing incident paths.
    """
    # import stations data
    stations = pd.read_csv(stations)

    # 1-codes dataset that includes incident descriptors
    # rename stations columns to allow for merging with dispatches file
    stations1 = stations[['STATION', 'LAT', 'LONG']]
    stations1.columns = ['station', 'latitude', 'longitude']

    # prepare data for gis data by excluding unnecessary columns
    gis_prep = spatial_joined[['eventnum', 'unit', 'incident_juris', 'latitude', 'longitude', 'station']]

    #####################CASE STUDY 1#################################

    # get data for case study example 1 by filtering by event number
    gis_type1 = gis_prep.loc[gis_prep['eventnum'] == 1803200927]

    # add descriptor for GIS graph as location of incident
    gis_type1['Descriptor'] = 'Location of Incident'

    # drop unnecessary columns
    gis_type1 = gis_type1[['station', 'latitude', 'longitude', 'Descriptor']]

    # determine the number of unique stations involved in the incident
    stations_type1_list = gis_type1.station.unique()

    # filter stations dataset for only involved stations
    stations_type1 = stations1[stations1['station'].isin(stations_type1_list)]

    # rename station descriptor column with the responding unit location
    stations_type1['Descriptor'] = 'Responding Ambulance Location'

    # stack the datasets together
    gis_type1 = pd.concat([gis_type1, stations_type1], ignore_index=True, sort=False)

    #####################CASE STUDY 2#################################

    # get data for case study example 2 by filtering by event number
    gis_type2 = gis_prep.loc[gis_prep['eventnum'] == 1808030525]

    # add descriptor for GIS graph as location of incident
    gis_type2['Descriptor'] = 'Location of Incident'

    # drop unnecessary columns
    gis_type2 = gis_type2[['station', 'latitude', 'longitude', 'Descriptor']]

    # determine the number of unique stations involved in the incident
    stations_type2_list = gis_type2.station.unique()

    # filter stations dataset for only involved stations
    stations_type2 = stations1[stations1['station'].isin(stations_type2_list)]

    # rename station descriptor column with the responding unit location
    stations_type2['Descriptor'] = 'Responding Ambulance Location'

    # stack the datasets together
    gis_type2 = pd.concat([gis_type2, stations_type2], ignore_index=True, sort=False)

    #####################CASE STUDY 3#################################

    # get data for case study example 1 by filtering by event number
    gis_type3 = gis_prep.loc[gis_prep['eventnum'] == 1606200818]

    # add descriptor for GIS graph as location of incident
    gis_type3['Descriptor'] = 'Location of Incident'

    # drop unnecessary columns
    gis_type3 = gis_type3[['station', 'latitude', 'longitude', 'Descriptor']]

    # determine the number of unique stations involved in the incident
    stations_type3_list = gis_type3.station.unique()

    # filter stations dataset for only involved stations
    stations_type3 = stations1[stations1['station'].isin(stations_type3_list)]

    # rename station descriptor column with the responding unit location
    stations_type3['Descriptor'] = pd.np.where(stations_type3['station'] == 25, "Responding Engine Location",
                                               "Responding Ambulance Location")

    # get location of correct ambulance
    gis_type3_pt2 = gis_prep.loc[gis_prep['eventnum'] == 1606200800]

    # add descriptor for GIS graph as location of incident
    gis_type3_pt2['Descriptor'] = 'Territory Ambulance Location'

    # drop unnecessary columns
    gis_type3_pt2 = gis_type3_pt2[['station', 'latitude', 'longitude', 'Descriptor']]

    # stack the datasets together
    gis_type3 = pd.concat([gis_type3, stations_type3, gis_type3_pt2], ignore_index=True, sort=False)

    #####################CASE STUDY 4#################################

    # get data for case study example 1 by filtering by event number
    gis_type4 = gis_prep.loc[gis_prep['eventnum'] == 1712210353]

    # add descriptor for GIS graph as location of incident
    gis_type4['Descriptor'] = 'Location of Incident'

    # drop unnecessary columns
    gis_type4 = gis_type4[['station', 'latitude', 'longitude', 'Descriptor']]

    # determine the number of unique stations involved in the incident
    stations_type4_list = gis_type4.station.unique()

    # filter stations dataset for only involved stations
    stations_type4 = stations1[stations1['station'].isin(stations_type4_list)]

    # rename station descriptor column with the responding unit location
    stations_type4['Descriptor'] = pd.np.where(stations_type4['station'] == 29, "Responding Engine Location",
                                               "Responding Medic Location")

    # get location of correct ambulance
    gis_type4_pt2 = gis_prep.loc[gis_prep['eventnum'] == 1712210348]

    # add descriptor for GIS graph as location of incident
    gis_type4_pt2['Descriptor'] = 'Territory Medic Location'

    # drop unnecessary columns
    gis_type4_pt2 = gis_type4_pt2[['station', 'latitude', 'longitude', 'Descriptor']]

    # stack the datasets together
    gis_type4 = pd.concat([gis_type4, stations_type4, gis_type4_pt2], ignore_index=True, sort=False)

    if not os.path.exists(os.path.join(os.getcwd(), "GIS", "data")):
        os.makedirs(os.path.join(os.getcwd(), "GIS", "data"))

    # dave data in GIS folder
    gis_type1.to_csv(os.path.join(os.getcwd(), "GIS", "data", "gis_type1.csv"))
    gis_type2.to_csv(os.path.join(os.getcwd(), "GIS", "data", "gis_type2.csv"))
    gis_type3.to_csv(os.path.join(os.getcwd(), "GIS", "data", "gis_type3.csv"))
    gis_type4.to_csv(os.path.join(os.getcwd(), "GIS", "data", "gis_type4.csv"))


# figure 20a:
def plot_case_study_frequency_correct(correct_response_ems, name):
    """
    Written by: Erin Kreus
    Purpose: determine how frequently each archetype occurs for
    correct responses
    """
    # get dataset of correct responses
    correct_response_ems_correct = correct_response_ems.loc[(correct_response_ems['response'] == "Correct Response")]

    # create variable for if the vehicle was busy
    correct_response_ems_correct['busyness'] = pd.np.where((correct_response_ems_correct['station'] ==
                                                            correct_response_ems_correct['incident_juris']),
                                                           "notbusy", "busy")
    # create a variable for if it is slow or not
    correct_response_ems_correct['sloworno'] = pd.np.where((correct_response_ems_correct['responsetime'] >= 10),
                                                           "slow", "notslow")

    # make (and flatten) a pivot table to get the counts in each type
    table = pd.pivot_table(correct_response_ems_correct, values='eventnum', index=['busyness'],
                           columns=['sloworno'], aggfunc='count')
    flattened = pd.DataFrame(table.to_records())

    # Graph results
    # create traces for graphing
    trace1 = go.Bar(x=['Outside Territory', 'Within Territory'], y=flattened['notslow'], name='Timely Response')
    trace2 = go.Bar(x=['Outside Territory', 'Within Territory'], y=flattened['slow'], name='Slow Response')
    traces = [trace1, trace2]

    # use plotly package for graphing
    layout = go.Layout(
            barmode='group',
            title='Frequency of Archetypes among Correct Dispatches  <br> EMS Dispatches Only<br> 2011-2018',
            xaxis=dict(title='Year'),
            width=700,
            height=500,
            yaxis=dict(title='Number of Dispatches',range=[0, 1500000]))
    fig = dict(data=traces, layout=layout)
    # plot(fig, filename='styled-line')

    save_plot(fig, name)


# figure 20b:
def plot_case_study_frequency_incorrect(correct_response_ems, name):
    """
    Written by: Erin Kreus
    Purpose: determine how frequently each archetype occurs for incorrect responses
    """
    # get dataset of correct responses
    correct_response_ems_incorrect = correct_response_ems.loc[(correct_response_ems['response'] ==
                                                               "Incorrect Response")]

    # create variable for if the vehicle was busy
    correct_response_ems_incorrect['busyness'] = pd.np.where((correct_response_ems_incorrect['station'] ==
                                                              correct_response_ems_incorrect['incident_juris']),
                                                             "notbusy", "busy")

    # create a variable for if it is slow or not
    correct_response_ems_incorrect['sloworno'] = pd.np.where((correct_response_ems_incorrect['responsetime'] >= 10),
                                                             "slow", "notslow")

    # make (and flatten) a pivot table to get the counts in each type
    table = pd.pivot_table(correct_response_ems_incorrect, values='eventnum', index=['busyness'],
                           columns=['sloworno'],aggfunc='count')
    flattened = pd.DataFrame(table.to_records())

    # Graph results
    # create traces for graphing
    trace1 = go.Bar(x=['Outside Territory', 'Within Territory'], y=flattened['notslow'], name='Timely Response')
    trace2 = go.Bar(x=['Outside Territory', 'Within Territory'], y=flattened['slow'], name='Slow Response')
    traces = [trace1,trace2]

    # use plotly package for graphing
    layout = go.Layout(
            barmode='group',
            title='Frequency of Archetypes among Incorrect Dispatches  <br> EMS Dispatches Only<br> 2011-2018',
            xaxis=dict(title='Year'),
            width=700,
            height=500,
            yaxis=dict(title='Number of Dispatches', range=[0, 500000]))
    fig = dict(data=traces, layout=layout)
    # plot(fig, filename='styled-line')

    save_plot(fig, name)


# figure 21 and figure 22:
def avg_chain_length_bar(avg_df, name, save=True):
    if save is True:
        font = {'family': 'monospace',
                'weight': 'bold',
                'size': 22}
        plt.rc('font', **font)
        plt.figure(figsize=(20, 10))
        # bar graph
        # distress.columns = ['start_inc_juris', 'distress_fraction']
        # helping.columns = ["start_inc_juris", 'helping_fraction']
        # final_avg_df = avg_df.merge(helping, how='left', on='start_inc_juris')
        # final_avg_df = final_avg_df.merge(distress, how='left', on='start_inc_juris')
        result = avg_df.sort_values('average')
        plt.rcParams['figure.figsize'] = 20, 10
        sns.set(style="whitegrid", font_scale=1.5)
        fig = sns.barplot(x='start_inc_juris', y="average", data=avg_df, order=result['start_inc_juris'])
        plt.ylim(1, 1.5)
        plt.title("Average Chain Length for Chains Beginning in Each Territory")
        plt.xlabel("Starting Station Territory")
        plt.ylabel("Average Chain Length")
        plt.xticks(rotation=270)
        # plt.savefig(os.path.join(os.getcwd(), "figures", figname1))
        # plt.show(fig)

        # save plot
        if not os.path.exists('figures'):
            os.mkdir('figures')
        figure = fig.get_figure()
        figure.savefig(os.path.join(os.getcwd(), "figures", name))

        # # scatter plot
        # f, ax = plt.subplots()
        # points = ax.scatter(final_avg_df.distress_fraction, final_avg_df.average, c=final_avg_df.helping_fraction)
        # f.colorbar(points)
        # plt.xlabel("Distress Fraction")
        # plt.ylabel("Average Chain")
        # plt.title("Distress Fraction of Station vs Average Length of Chain Originating in their Territory, "
        #           "Color = Helping Fraction")

        # # save plot
        # figname2 = 'figure22.png'
        # figure = points.get_figure()
        # figure.savefig(os.path.join(os.getcwd(), "figures", figname2))
        # # plt.savefig(os.path.join(os.getcwd(), "figures", figname2))

# figures not yet in report:


# figures no longer in report:
def plot_median_response_TOD(merged, name):
    """
    Written by: Erin Kreus and Lynn Zhu
    Purpose: plots the median incident response time by time of day
    :param merged:: pandas dataframe of dispatches and incidents left joined on eventnumber
    """
    # 1-use only dispatch data with the first response time
    first_responders = merged.groupby((merged['eventnum'] != merged['eventnum'].shift()).cumsum().values).first()

    # 2-calculate the average response time by hour using a group by hour
    responsetime = first_responders.groupby('hour')['dispatchtime'].median().reset_index(name='median')

    # 3-make trace for plot with the xaxis being the time of day and the axis being the median response time
    trace0 = [go.Bar(
                x=['12AM', '1AM', '2AM', '3AM', '4AM', '5AM', '6AM', '7AM', '8AM', '9AM', '10AM', '11AM', '12PM',
                   '1PM', '2PM', '3PM', '4PM', '5PM', '6PM', '7PM', '8PM', '9PM', '10PM', '11PM'],
                y=responsetime['median']
        )]

    # 4-create the plot using plotly package
    layout = go.Layout(
        title='Houston Fire Department Median Incident Dispatch Time by Time of Day <br>'
              ' (Median Time from Station Departure to Incident Arrival) <br> 2012-2018',
        xaxis=dict(title='Time of Day'),
        width=700,
        height=500,
        yaxis=dict(title='Median Dispatch Time (Minutes)'),
        showlegend=False
    )
    fig = go.Figure(data=trace0, layout=layout)
    # iplot(fig, filename='basic-bar')

    save_plot(fig, name)
