import pandas as pd
import numpy as np
import os
import plotly.graph_objs as go
import plotly.io as pio
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import collections
# import warnings
# warnings.filterwarnings("ignore")
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)

#
# def validation_wrangling(hfd_sim, MEXCLP_restricted, MEXCLP_unrestricted, add_5_amb,
#                          infinite, hfd_sim_hist, MEXCLP_restricted_hist, add_5_amb_hist, real_data,
#                          save=True, output=True):
#     """
#     Written by: Lynn Zhu
#     Purpose: Combine all simulated validation datasets and find response times
#     """
#     # Import all simulated data and create new variable indicating which model it came from
#     hfd_sim['whichmodel'] = "Current Allocation"
#     MEXCLP_restricted['whichmodel'] = "MEXCLP Capacity"
#     MEXCLP_unrestricted['whichmodel'] = "MEXCLP No Capacity"
#     add_5_amb['whichmodel'] = "Five Additional Ambulances"
#     infinite['whichmodel'] = "Infinite Capacity"
#     hfd_sim_hist['whichmodel'] = "Historical Current Allocation"
#     MEXCLP_restricted_hist['whichmodel'] = "Historical MEXCLP Capacity"
#     add_5_amb_hist['whichmodel'] = "Historical Five Additional Ambulances"
#
#     sim_list = [hfd_sim, MEXCLP_restricted, MEXCLP_unrestricted, add_5_amb, infinite,
#                 hfd_sim_hist, MEXCLP_restricted_hist, add_5_amb_hist]
#
#     # Read in actual data for comparison and label it as the actual data
#     real_data['whichmodel'] = "Actual Data"
#
#     # GET RESPONSE TIMES FOR EACH DATASET
#     # convert enroute/onscene time in simulated dataset to datetime
#
#     for simulated in sim_list:
#         simulated['enroute'] = pd.to_datetime(simulated['enroute'], format="%Y-%m-%d %H:%M:%S")
#         simulated['onscene'] = pd.to_datetime(simulated['onscene'], format="%Y-%m-%d %H:%M:%S")
#         # part 1: Get the minimum dispatch en route time for each event number
#         simulated_first_time = simulated.groupby(['eventnum'])['enroute'].min().reset_index(name='min_enroute')
#
#         # part 2: create a new column in the dispatches dataset
#         simulated = pd.merge(simulated, simulated_first_time, on='eventnum', how='left')
#         # part 3: calculate the response time
#         simulated['responsetime'] = simulated['onscene'] - simulated['min_enroute']
#
#         # part 4: convert the response time to minutes
#         simulated['responsetime'] = simulated['responsetime'] / np.timedelta64(1, 'm')
#
#     sim_list.append(real_data)
#     simulated = simulated.append([sim_list], sort=False)
#
#     # get ending indicator that will later be used to determine if this was the correct vehicle type
#     simulated['app_unit_indicator'] = simulated['type'].str[-2:]
#
#     # get if the simulated data is EMS or Fire
#     simulated['prefix'] = simulated['type'].str[:2]
#     simulated['fire_EMS'] = pd.np.where(simulated.prefix.str.contains("FE"), "EMS", "Fire")
#
#     if save:
#         simulated.to_csv(os.path.join(os.getcwd(), "data", "validation_simulated.csv"), index=False)
#     if output:
#         return simulated


def validation_wrangling(simulated, simulated2, simulated3, simulated4,
                         simulated6, simulated_historical, simulated_historical2,
                         simulated_historical3, emal, save=True, output=True):
    """
    Written by: Erin Kreu
    Purpose: Combine all simulated validation datasets and find response times
    """
    simulated['whichmodel'] = "Current Allocation"
    simulated2['whichmodel'] = "MEXCLP Capacity"
    simulated3['whichmodel'] = "MEXCLP No Capacity"
    simulated4['whichmodel'] = "Five Additional Ambulances"
    simulated6['whichmodel'] = "Infinite Capacity"
    simulated_historical['whichmodel'] = "Historical Current Allocation"
    simulated_historical2['whichmodel'] = "Historical MEXCLP Capacity"
    simulated_historical3['whichmodel'] = "Historical Five Additional Ambulances"

    # Read in actual data for comparison and label it as the actual data
    emal['whichmodel'] = "Actual Data"

    # Import all simulated data and create new variable indicating which model it came from

    # Read in actual data for comparison and label it as the actual data
    emal['whichmodel'] = "Actual Data"

    # Drop data missing a dispatch date because the simulator does this
    emal = emal.dropna(subset=['Dispatch_Date'])

    # Filter only for March 2017
    emal = emal.loc[(emal['mnth_yr'] == 'March-2017')]

    # GET RESPONSE TIMES FOR EACH DATASET
    # convert enroute/onscene time in simualted dataset to datetime
    simulated['enroute'] = pd.to_datetime(simulated['enroute'], format="%Y-%m-%d %H:%M:%S")
    simulated['onscene'] = pd.to_datetime(simulated['onscene'], format="%Y-%m-%d %H:%M:%S")
    simulated2['enroute'] = pd.to_datetime(simulated2['enroute'], format="%Y-%m-%d %H:%M:%S")
    simulated2['onscene'] = pd.to_datetime(simulated2['onscene'], format="%Y-%m-%d %H:%M:%S")
    simulated3['enroute'] = pd.to_datetime(simulated3['enroute'], format="%Y-%m-%d %H:%M:%S")
    simulated3['onscene'] = pd.to_datetime(simulated3['onscene'], format="%Y-%m-%d %H:%M:%S")
    simulated4['enroute'] = pd.to_datetime(simulated4['enroute'], format="%Y-%m-%d %H:%M:%S")
    simulated4['onscene'] = pd.to_datetime(simulated4['onscene'], format="%Y-%m-%d %H:%M:%S")
    simulated6['enroute'] = pd.to_datetime(simulated6['enroute'], format="%Y-%m-%d %H:%M:%S")
    simulated6['onscene'] = pd.to_datetime(simulated6['onscene'], format="%Y-%m-%d %H:%M:%S")
    simulated_historical['enroute'] = pd.to_datetime(simulated_historical['enroute'], format="%Y-%m-%d %H:%M:%S")
    simulated_historical['onscene'] = pd.to_datetime(simulated_historical['onscene'], format="%Y-%m-%d %H:%M:%S")
    simulated_historical2['enroute'] = pd.to_datetime(simulated_historical2['enroute'], format="%Y-%m-%d %H:%M:%S")
    simulated_historical2['onscene'] = pd.to_datetime(simulated_historical2['onscene'], format="%Y-%m-%d %H:%M:%S")
    simulated_historical3['enroute'] = pd.to_datetime(simulated_historical3['enroute'], format="%Y-%m-%d %H:%M:%S")
    simulated_historical3['onscene'] = pd.to_datetime(simulated_historical3['onscene'], format="%Y-%m-%d %H:%M:%S")

    # get response time for first dataset

    # part 1: Get the minimum dispatch en route time for each event number
    simulated_first_time = simulated.groupby(['eventnum'])['enroute'].min().reset_index(name='min_enroute')

    # part 2: create a new column in the dispatches dataset
    simulated = pd.merge(simulated, simulated_first_time, on='eventnum', how='left')

    # part 3: calculate the response time
    simulated['responsetime'] = simulated['onscene'] - simulated['min_enroute']

    # part 4: convert the response time to minutes
    simulated['responsetime'] = simulated['responsetime'] / np.timedelta64(1, 'm')

    # GET RESPONSE TIMES FOR REMAINING ALLOCATIONS
    # get response times for remaining datasets (we use the same method outlined above, but omit
    # more detailed comments)
    # simulated2
    simulated_first_time = simulated2.groupby(['eventnum'])['enroute'].min().reset_index(name='min_enroute')
    simulated2 = pd.merge(simulated2, simulated_first_time, on='eventnum', how='left')
    simulated2['responsetime'] = simulated2['onscene'] - simulated2['min_enroute']
    simulated2['responsetime'] = simulated2['responsetime'] / np.timedelta64(1, 'm')

    # simulated3
    simulated_first_time = simulated3.groupby(['eventnum'])['enroute'].min().reset_index(name='min_enroute')
    simulated3 = pd.merge(simulated3, simulated_first_time, on='eventnum', how='left')
    simulated3['responsetime'] = simulated3['onscene'] - simulated3['min_enroute']
    simulated3['responsetime'] = simulated3['responsetime'] / np.timedelta64(1, 'm')

    # simulated4
    simulated_first_time = simulated4.groupby(['eventnum'])['enroute'].min().reset_index(name='min_enroute')
    simulated4 = pd.merge(simulated4, simulated_first_time, on='eventnum', how='left')
    simulated4['responsetime'] = simulated4['onscene'] - simulated4['min_enroute']
    simulated4['responsetime'] = simulated4['responsetime'] / np.timedelta64(1, 'm')

    # simulated6
    simulated_first_time = simulated6.groupby(['eventnum'])['enroute'].min().reset_index(name='min_enroute')
    simulated6 = pd.merge(simulated6, simulated_first_time, on='eventnum', how='left')
    simulated6['responsetime'] = simulated6['onscene'] - simulated6['min_enroute']
    simulated6['responsetime'] = simulated6['responsetime'] / np.timedelta64(1, 'm')

    # simulated_historical
    simulated_first_time = simulated_historical.groupby(['eventnum'])['enroute'].min().reset_index(name='min_enroute')
    simulated_historical = pd.merge(simulated_historical, simulated_first_time, on='eventnum', how='left')
    simulated_historical['responsetime'] = simulated_historical['onscene'] - simulated_historical['min_enroute']
    simulated_historical['responsetime'] = simulated_historical['responsetime'] / np.timedelta64(1, 'm')

    # simulated_historical2
    simulated_first_time = simulated_historical2.groupby(['eventnum'])['enroute'].min().reset_index(name='min_enroute')
    simulated_historical2 = pd.merge(simulated_historical2, simulated_first_time, on='eventnum', how='left')
    simulated_historical2['responsetime'] = simulated_historical2['onscene'] - simulated_historical2['min_enroute']
    simulated_historical2['responsetime'] = simulated_historical2['responsetime'] / np.timedelta64(1, 'm')

    # simulated_historical3
    simulated_first_time = simulated_historical3.groupby(['eventnum'])['enroute'].min().reset_index(name='min_enroute')
    simulated_historical3 = pd.merge(simulated_historical3, simulated_first_time, on='eventnum', how='left')
    simulated_historical3['responsetime'] = simulated_historical3['onscene'] - simulated_historical3['min_enroute']
    simulated_historical3['responsetime'] = simulated_historical3['responsetime'] / np.timedelta64(1, 'm')

    # Get a stacked dataset by appending the datasets
    simulated = simulated.append([simulated2, simulated3, simulated4, simulated6, simulated_historical,
                                  simulated_historical2, simulated_historical3, emal], sort=False)

    # get ending indicator that will later be used to determine if this was the correct vehicle type
    simulated['app_unit_indicator'] = simulated['type'].str[-2:]

    # get if the simulated data is EMS or Fire
    simulated['prefix'] = simulated['type'].str[:2]
    simulated['fire_EMS'] = pd.np.where(simulated.prefix.str.contains("FE"), "EMS", "Fire")

    if save:
        simulated.to_csv(os.path.join(os.getcwd(), "data", "validation_simulated.csv"), index=False)
    if output:
        return simulated

def validation_90th_reponsetimes(simulated):
    """
    Written by: Erin Kreus
    Purpose: Calculate the 90th percentile response time for each model
    :param codes: path csvs:
    """
    #1-create pivot table
    times_90th= pd.pivot_table(simulated, values='responsetime', index=['vehicletype'], columns=['whichmodel'], aggfunc=lambda x: np.percentile(x, 90))

    #2-flatten pivot table
    times_90th = pd.DataFrame(times_90th.to_records())

    times_90th.to_csv(os.path.join(os.getcwd(), "data", "90th_response_times_by_model.csv"))


def validation_90th_dispatch_times(simulated):
    """
    Written by: Erin Kreus
    Purpose: Calculate the 90th percentile dispatch time for each model
    :param codes: path csvs:
    """
    #1-create pivot table
    times_90th= pd.pivot_table(simulated, values='dispatchtime', index=['vehicletype'], columns=['whichmodel'], aggfunc=lambda x: np.percentile(x, 90))

    #2-flatten pivot table
    times_90th = pd.DataFrame(times_90th.to_records())

    times_90th.to_csv(os.path.join(os.getcwd(), "data", "90th_dispatch_times_by_model.csv"))


def validation_50th_response_times(simulated):
    """
    Written by: Erin Kreus
    Purpose: Calculate the 50 percentile response time for each model
    :param codes: path csvs:
    """
    #1-create pivot table
    times_50th= pd.pivot_table(simulated, values='responsetime', index=['vehicletype'], columns=['whichmodel'], aggfunc=lambda x: np.percentile(x, 50))

    #2-flatten pivot table
    times_50th = pd.DataFrame(times_50th.to_records())

    times_50th.to_csv(os.path.join(os.getcwd(), "data", "50th_response_times_by_model.csv"))


def validation_50th_dispatch_times(simulated):
    """
    Written by: Erin Kreus
    Purpose: Calculate the 50th percentile dispatch time for each model
    :param codes: path csvs:
    """
    #1-create pivot table
    times_50th= pd.pivot_table(simulated, values='dispatchtime', index=['vehicletype'], columns=['whichmodel'],
                               aggfunc=lambda x: np.percentile(x, 50))

    #2-flatten pivot table
    times_50th = pd.DataFrame(times_50th.to_records())

    times_50th.to_csv(os.path.join(os.getcwd(), "data", "50th_dispatch_times_by_model.csv"))


def count_dispatches(simulated):
    """
    Written by: Erin Kreus
    Purpose: Count the number of dispatches for each model
    :param codes: path csvs:
    """
    # 1-create pivot table
    count_dispatches = pd.pivot_table(simulated, values='eventnum', index=['vehicletype'], columns=['whichmodel'],
                                      aggfunc='count')

    # 2-flatten pivot table
    count_dispatches = pd.DataFrame(count_dispatches.to_records())

    count_dispatches.to_csv(os.path.join(os.getcwd(), "data", "dispatchescount_bymodel.csv"))


def gis_90th(simulated):
    """
    Written by: Erin Kreus
    Purpose: Get 90th percentile Response times for ambulances by jurisdiction
    :param codes: path csvs:
    """
    # Filter only for ambulances and five additional model
    filtered=simulated.loc[(simulated['whichmodel'] == 'Current Allocation')
                           | (simulated['whichmodel'] == 'Five Additional Ambulances')]
    filtered=filtered.loc[filtered['vehicletype'] == 'A']

    # 1-create pivot table
    byjurisdiction= pd.pivot_table(filtered, values='responsetime', index=['incident_juris'],
                                   columns=['vehicletype','whichmodel'], aggfunc=lambda x: np.percentile(x, 90))

    # 2-flatten pivot table
    byjurisdiction = pd.DataFrame(byjurisdiction.to_records())

    byjurisdiction.to_csv(os.path.join(os.getcwd(), "data", "90thresponsetime_byjuris.csv"))
    return byjurisdiction


def gis_50th(simulated):
    """
    Written by: Erin Kreus
    Purpose: Get Median  Response times for ambulances by jurisdiction
    :param codes: path csvs:
    """
    # Filter only for ambulances and five additional
    filtered = simulated.loc[(simulated['whichmodel'] == 'Current Allocation') | (
                simulated['whichmodel'] == 'Five Additional Ambulances')]
    filtered = filtered.loc[filtered['vehicletype'] == 'A']

    # 1-create pivot table
    byjurisdiction50 = pd.pivot_table(filtered, values='responsetime', index=['incident_juris'],
                                      columns=['vehicletype', 'whichmodel'], aggfunc=lambda x: np.percentile(x, 50))

    # Flatten the pivot table
    byjurisdiction50 = pd.DataFrame(byjurisdiction50.to_records())
    byjurisdiction50.to_csv(os.path.join(os.getcwd(), "data", "50thresponsetime_byjuris.csv"))
    return byjurisdiction50


def f(row):
    if ((row['app_unit_indicator'] == "A1") & (row['vehicletype'] in (["E","L","A"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "AI") & (row['vehicletype'] in (["A","E","L","M","SQ"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "A4") & (row['vehicletype'] in (["A","M"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "A6") & (row['vehicletype'] in (["A","M","SQ"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "A7") & (row['vehicletype'] in (["A"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "A8") & (row['vehicletype'] in (["A"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "B7") & (row['vehicletype'] in (["M","SQ"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "B8") & (row['vehicletype'] in (["M","SQ"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "C1") & (row['vehicletype'] in (["M","A","SQ"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "C2") & (row['vehicletype'] in (["M","A","SQ","L","E"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "C3") & (row['vehicletype'] in (["M","A","SQ","L","E"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "C6") & (row['vehicletype'] in (["M","A","SQ","L","E"]))):
        val = 'Correct Response'
    elif ((row['app_unit_indicator'] == "D1") & (row['vehicletype'] in (["M","A","SQ","L","E"]))):
        val = 'Correct Response'
    else:
        val = "Incorrect Response"
    return val


def correct_response_validation(simulated):
    """
    Written by: Erin Kreus
    Purpose: Determine correct response percentage for each model
    :param codes: path csvs:
    """
    #since we are only looking at EMS incidents, we filter out non-EMS incidents
    simulated_EMS =simulated.loc[simulated['fire_EMS'] == 'EMS']

    #filter the response indicators for EMS incidents
    simulated_EMS =simulated_EMS.loc[(simulated_EMS['app_unit_indicator'] == 'A1') |
                                     (simulated_EMS['app_unit_indicator'] == 'AI') |
                                     (simulated_EMS['app_unit_indicator'] == 'A4') |
                                     (simulated_EMS['app_unit_indicator'] == 'A6') |
                                     (simulated_EMS['app_unit_indicator'] == 'A7') |
                                     (simulated_EMS['app_unit_indicator'] == 'A8') |
                                     (simulated_EMS['app_unit_indicator'] == 'B7') |
                                     (simulated_EMS['app_unit_indicator'] == 'B8') |
                                     (simulated_EMS['app_unit_indicator'] == 'C1') |
                                     (simulated_EMS['app_unit_indicator'] == 'C2') |
                                     (simulated_EMS['app_unit_indicator'] == 'C3') |
                                     (simulated_EMS['app_unit_indicator'] == 'C6') |
                                     (simulated_EMS['app_unit_indicator'] == 'D1')]

    simulated_EMS['response'] = simulated_EMS.apply(f, axis=1)

    # get dataset for each model and get the first response only
    simulated_suggested = simulated_EMS[(simulated_EMS['whichmodel'] == "MEXCLP Capacity")]
    correct_EMS_first1 = simulated_suggested.sort_values(['eventnum', 'onscene'], ascending=[True, True])
    correct_EMS_first1 = correct_EMS_first1.groupby((correct_EMS_first1['eventnum'] != correct_EMS_first1['eventnum'].shift()).cumsum().values).first()

    simulated_current = simulated_EMS[(simulated_EMS['whichmodel'] == "Current Allocation")]
    correct_EMS_first2 = simulated_current.sort_values(['eventnum', 'onscene'], ascending=[True, True])
    correct_EMS_first2 = correct_EMS_first2.groupby((correct_EMS_first2['eventnum'] != correct_EMS_first2['eventnum'].shift()).cumsum().values).first()

    simulated_actual = simulated_EMS[(simulated_EMS['whichmodel'] == "Actual Data")]
    correct_EMS_first3 = simulated_actual.sort_values(['eventnum', 'onscene'], ascending=[True, True])
    correct_EMS_first3 = correct_EMS_first3.groupby((correct_EMS_first3['eventnum'] != correct_EMS_first3['eventnum'].shift()).cumsum().values).first()

    simulated_suggested2 = simulated_EMS[(simulated_EMS['whichmodel'] == "MEXCLP No Capacity")]
    correct_EMS_first4 = simulated_suggested2.sort_values(['eventnum', 'onscene'], ascending=[True, True])
    correct_EMS_first4 = correct_EMS_first4.groupby((correct_EMS_first4['eventnum'] != correct_EMS_first4['eventnum'].shift()).cumsum().values).first()

    simulated_suggested3 = simulated_EMS[(simulated_EMS['whichmodel'] == "Five Additional Ambulances")]
    correct_EMS_first5 = simulated_suggested3.sort_values(['eventnum', 'onscene'], ascending=[True, True])
    correct_EMS_first5 = correct_EMS_first5.groupby((correct_EMS_first5['eventnum'] != correct_EMS_first5['eventnum'].shift()).cumsum().values).first()

    simulated_suggested5 = simulated_EMS[(simulated_EMS['whichmodel'] == "Infinite Capacity")]
    correct_EMS_first7 = simulated_suggested5.sort_values(['eventnum', 'onscene'], ascending=[True, True])
    correct_EMS_first7 = correct_EMS_first7.groupby((correct_EMS_first7['eventnum'] != correct_EMS_first7['eventnum'].shift()).cumsum().values).first()

    simulated_suggested6 = simulated_EMS[(simulated_EMS['whichmodel'] == "Historical Current Allocation")]
    correct_EMS_first8 = simulated_suggested6.sort_values(['eventnum', 'onscene'], ascending=[True, True])
    correct_EMS_first8 = correct_EMS_first8.groupby((correct_EMS_first8['eventnum'] != correct_EMS_first8['eventnum'].shift()).cumsum().values).first()

    simulated_suggested7 = simulated_EMS[(simulated_EMS['whichmodel'] == "Historical MEXCLP Capacity")]
    correct_EMS_first9 = simulated_suggested7.sort_values(['eventnum', 'onscene'], ascending=[True, True])
    correct_EMS_first9 = correct_EMS_first9.groupby((correct_EMS_first9['eventnum'] != correct_EMS_first9['eventnum'].shift()).cumsum().values).first()

    simulated_suggested8 = simulated_EMS[(simulated_EMS['whichmodel'] == "Historical Five Additional Ambulances")]
    correct_EMS_first10 = simulated_suggested8.sort_values(['eventnum', 'onscene'], ascending=[True, True])
    correct_EMS_first10 = correct_EMS_first10.groupby((correct_EMS_first10['eventnum'] != correct_EMS_first10['eventnum'].shift()).cumsum().values).first()

    #append the dataframes
    correctresponses = correct_EMS_first1.append([correct_EMS_first2,correct_EMS_first3,
                                                 correct_EMS_first4,correct_EMS_first5,
                                                 correct_EMS_first7,correct_EMS_first8,
                                                 correct_EMS_first9,correct_EMS_first10,
                                                 correct_EMS_first1], sort=False)

    #Get pivot table which shows the % of responses for EMS where the first unit on scene is the correct one
    pivot = pd.pivot_table(correctresponses, values='eventnum', index=['whichmodel'], columns=['response'], aggfunc='count')

    #flatten the pivot table
    pivot = pd.DataFrame(pivot.to_records())

    #get percentage correct
    pivot['Inaccuracy']=100*pivot['Incorrect Response']/(pivot['Correct Response']+pivot['Incorrect Response'])

    pivot.to_csv(os.path.join(os.getcwd(), "data", "inaccuracyofdispatches.csv"))


def plot_50th_changes(fiftybyjuris):
    '''
    Written by: Erin Kreus
    Purpose: Plots 50th percentile response time under current allocation and under 5 additional ambulances allocation
    '''
    # Rename the columns
    fiftybyjuris.columns = ['incident_juris', 'current', 'addition']

    # filter only for incident territories that we added ambulances to
    fiftybyjuris = fiftybyjuris.loc[(fiftybyjuris['incident_juris'] == 35) |
                                    (fiftybyjuris['incident_juris'] == 73) |
                                    (fiftybyjuris['incident_juris'] == 8) |
                                    (fiftybyjuris['incident_juris'] == 33) |
                                    (fiftybyjuris['incident_juris'] == 46)]

    # Sort by the incident jurisdiction
    fiftybyjuris = fiftybyjuris.sort_values(by=['incident_juris'])

    # Use the plotly package to plot the response times
    trace1 = go.Bar(
        x=['Territory 8', 'Territory 33', 'Territory 35', 'Territory 46', 'Territory 73'],
        y=fiftybyjuris['current'],
        name='Current Allocation',
        marker=dict(
            color='rgb(55, 83, 109)'
        )
    )
    trace2 = go.Bar(
        x=['Territory 8', 'Territory 33', 'Territory 35', 'Territory 46', 'Territory 73'],
        y=fiftybyjuris['addition'],
        name='5 Additional Allocation',
        marker=dict(
            color='rgb(10, 150, 109)'
        )
    )
    data = [trace1, trace2]
    layout = go.Layout(title='50th Percentile Response Times for Ambulances',
                       xaxis=dict(
                           tickfont=dict(
                               size=14,
                               color='rgb(107, 107, 107)'
                           )
                       ),
                       yaxis=dict(
                           title='Response Time (Minutes)',
                           titlefont=dict(
                               size=16,
                               color='rgb(107, 107, 107)'
                           ),
                           tickfont=dict(
                               size=14,
                               color='rgb(107, 107, 107)'
                           )
                       ),
                       legend=dict(
                           x=0,
                           y=1.0,
                           bgcolor='rgba(255, 255, 255, 0)',
                           bordercolor='rgba(255, 255, 255, 0)'
                       ),
                       barmode='group',
                       bargap=0.15,
                       bargroupgap=0.1
                       )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='style-bar')

    if not os.path.exists('figures'):
        os.mkdir('figures')
    pio.write_image(fig, 'figures/validation_fig1.png')


def plot_90th_changes(ninetybyjuris):
    '''
    Written by: Erin Kreus
    Purpose: Plots 90th percentile response time under current allocation and under 5 additional ambulances allocation
    '''
    # Load in the datasource (50th percentile response times) by jurisdiction
    # Rename the columns
    ninetybyjuris.columns = ['incident_juris', 'current', 'addition']

    # filter only for incident territories that we added ambulances to
    ninetybyjuris = ninetybyjuris.loc[(ninetybyjuris['incident_juris'] == 35) |
                                      (ninetybyjuris['incident_juris'] == 73) |
                                      (ninetybyjuris['incident_juris'] == 8) |
                                      (ninetybyjuris['incident_juris'] == 33) |
                                      (ninetybyjuris['incident_juris'] == 46)]

    # Sort by the incident jurisdiction
    ninetybyjuris = ninetybyjuris.sort_values(by=['incident_juris'])

    # Use the plotly package to plot the response times
    trace1 = go.Bar(
        x=['Territory 8', 'Territory 33', 'Territory 35', 'Territory 46', 'Territory 73'],
        y=ninetybyjuris['current'],
        name='Current Allocation',
        marker=dict(
            color='rgb(55, 83, 109)'
        )
    )
    trace2 = go.Bar(
        x=['Territory 8', 'Territory 33', 'Territory 35', 'Territory 46', 'Territory 73'],
        y=ninetybyjuris['addition'],
        name='5 Additional Allocation',
        marker=dict(
            color='rgb(10, 150, 109)'
        )
    )
    data = [trace1, trace2]
    layout = go.Layout(title='50th Percentile Response Times for Ambulances',
                       xaxis=dict(
                           tickfont=dict(
                               size=14,
                               color='rgb(107, 107, 107)'
                           )
                       ),
                       yaxis=dict(
                           title='Response Time (Minutes)',
                           titlefont=dict(
                               size=16,
                               color='rgb(107, 107, 107)'
                           ),
                           tickfont=dict(
                               size=14,
                               color='rgb(107, 107, 107)'
                           )
                       ),
                       legend=dict(
                           x=0,
                           y=1.0,
                           bgcolor='rgba(255, 255, 255, 0)',
                           bordercolor='rgba(255, 255, 255, 0)'
                       ),
                       barmode='group',
                       bargap=0.15,
                       bargroupgap=0.1
                       )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='style-bar')

    if not os.path.exists('figures'):
        os.mkdir('figures')
    pio.write_image(fig, 'figures/validation_fig2.png')
