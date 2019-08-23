# Written by Lynn Zhu
import wrangling
import EDA
import modeling
import validation
import simulation
import os
import time
import pandas as pd
import numpy as np
import argparse
import rpy2.robjects as robjects


def run_pipeline(incident_files, dispatch_files, time_files, raw_bays,
                 territories_shp, stations, fire_protocols, ems_protocols, HFD_allocation,
                 args, save_data=True, save_figs=False):
    """
    Written by Lynn Zhu
    """
    # wrangling
    if not args.skip_wrangling:
        tik = time.time()
        print("")
        all_incidents, all_dispatches = wrangling.data_wrangle(incident_files, dispatch_files, sep=False,
                                                                          save=save_data, output=True)
        print("Incidents and dispatches cleaned.")
        bays = wrangling.bays_capacity(raw_bays, save=save_data)
        print("Bays capacity determined.")
        # cleaning_wrangling_2.subdivide_codes(codes)
        # print("Codes subdivided.")

        spatial_joined = wrangling.spatial_join(territories_shp, all_incidents, save=save_data)
        print("Spatial join of incidents into territories complete.")

        master = wrangling.join_incidents_dispatches(spatial_joined, all_dispatches,
                                                     save=save_data)
        print("Incidents and dispatches joined on eventnum, %s rows. Check that this is the same number of rows as "
              "previously there were cleaned dispatches." % master.shape[0])

        all_ems, all_fire = wrangling.split_ems_fire(master, save=save_data)
        print("EMS and Fire split complete.")

        # print("Data joined with out times.")

        print("Data wrangling complete.")
        tok = time.time()
        minutes = (tok-tik)/60
        print("%s minutes spent on wrangling data." % minutes)

    else:
        print("")
        master = wrangling.import_joined_file(os.path.join(os.getcwd(), "data", "master_joined.csv"))
        data_ems = wrangling.import_joined_file(os.path.join(os.getcwd(), "data", "data_ems.csv"))
        data_fire = wrangling.import_joined_file(os.path.join(os.getcwd(), "data", "data_fire.csv"))
        bays = pd.read_csv(os.path.join(os.getcwd(), "data", "bays_capacity.csv"))
        all_incidents = pd.read_csv(os.path.join(os.getcwd(), "data", "all_incidents.csv"))
        print("Data loaded.")

    tik = time.time()
    # EDA
    if not args.skip_wrangling:
        print("")
        # calculate median response times
        EDA.median_response_by_station(master, save=save_data)
        print("Calculated median response times.")

        # calculate distress and helping fractions
        helping = EDA.calculate_helping_fraction(master, save=save_data)
        distress = EDA.calculate_distress_fraction(master, save=save_data)
        print("Calculated helping and distress fractions.")
    #
        correct_response_ems = EDA.analyse_ems_response(data_ems, save=save_data)
        print("Determined correct responses for EMS incidents.")

        wrangling.unit_profiles_dataset(master, correct_response_ems)
        print("Unit profile datasets created.")
    else:
        helping = pd.read_csv(os.path.join(os.getcwd(), "data", "helping_fractions.csv"))
        distress = pd.read_csv(os.path.join(os.getcwd(), "data", "distress_fractions.csv"))
        correct_response_EMS = pd.read_csv(os.path.join(os.getcwd(), "data", "correct_response_EMS.csv"))

    # EDA: chain analysis
    if args.run_find_chains:
        chains, first_vehicles, marked_chains = EDA.find_chains(master, save=False)
        print("Completed finding chains.")
    else:
        print("")
        print("Finding chains step skipped, please load chain data:")
        print("Put all pickle result files starting with 'chains_result' and 'first_vehicle' into the data folder.")
        input("Press Enter to proceed...")
        chain_result_files = []
        first_vehicle_files = []
        data = os.listdir(os.path.join(os.getcwd(), "data"))
        for file in data:
            name = file.split(os.sep)[-1]
            if name.startswith("chains_result"):
                chain_result_files.append(os.path.join(os.getcwd(), "data", file))
            elif name.startswith("first_vehicle"):
                first_vehicle_files.append(os.path.join(os.getcwd(), "data", file))
        print("%s chain_result files found" % len(chain_result_files))
        print("%s first_vehicle files found" % len(first_vehicle_files))
        chains, first_vehicles = EDA.combine_chain_files(chain_result_files, first_vehicle_files)

    average_chain_df, first_vehicles_types = EDA.chain_analysis(chains, first_vehicles, save=True)
    print("Completed chain analysis.")

    # make and save EDA figures in the order they are referenced in the final report
    if save_figs is True:
        print("")
        # figure 1 is data cleaning / wrangling diagram
        EDA.plot_num_incidents(master, 'figure2')
        EDA.plot_dispatch_times(data_ems, data_fire, 'figure3')
        EDA.prop_correct_response(correct_response_ems, 'figure4')
        EDA.plot_hist_correct(correct_response_ems, 'figure5')
        EDA.fire_gone(correct_response_ems, master, 'figure6')
        # figure 7a and 7b are per-station incident counts in Houston Area 2012, 2016 (maps)
        EDA.plot_incident_response_time(master, 'figure8')
        EDA.plot_dispatch_dist(master, 'figure9')
        # figure 10 is a geographic distribution of median response times (maps)
        # figure 11 is a comparison of incident counts and response times in the houston area (maps)
        EDA.plot_station37(master, 'figure12_station37')
        EDA.plot_station72(master, 'figure12_station72')
        # figure 13 is a geographic distribution of responses taking greater than 10 min (maps)
        # figure 14a and 14 b are station helper fractions and station distress fractions (maps)
        EDA.plot_hist_jurisdictions(master, 'figure15')
        # generate data for figures 16-19, which are all case study diagrams (maps)
        EDA.gis_case_studies_data(master, stations)
        EDA.plot_case_study_frequency_correct(correct_response_ems, 'figure20a')
        EDA.plot_case_study_frequency_incorrect(correct_response_ems, 'figure20b')
        EDA.avg_chain_length_bar(average_chain_df, "figure21", save=save_figs)
        EDA.table_over10_minutes_eda(master, 'table1')
        EDA.table_case_study_overall(master, 'table2')
        print("Running R script")
        robjects.r.source("script.R")
        print(
            "EDA figures generated and saved into figures folder, and tables generated and saved into tables folder.")
    else:
        print("Skipped EDA figure generation.")

    print("EDA complete.")
    tok = time.time()
    minutes = (tok - tik) / 60
    print("%s minutes spent on EDA." % minutes)

    # modeling

    # create 250 demand points for each vehicle type, as well as the 250 demand points for simulator
    if args.run_clustering:
        ambulances, medics, engines, ladders = modeling.split_by_vehicle_type(master,
                                                                              save=save_data)
        l11_17 = ladders[ladders.year != 2018]
        a11_17 = ambulances[ambulances.year != 2018]
        e11_17 = engines[engines.year != 2018]
        m11_17 = medics[medics.year != 2018]
        sim_df = master[master.vehicletype.isin(['A', 'M', 'L', 'E'])]
        sim_df = sim_df.drop_duplicates(subset=['eventnum'])
        sim_11_17 = sim_df[sim_df.year != 2018]
        sim_18 = sim_df[sim_df.year == 2018]
        ambulance_clusters, _ = modeling.kmeans(a11_17, name='ambulance_clusters', fraction=0.25,
                                                save=save_data,
                                                output=True, K=250)
        medic_clusters, _ = modeling.kmeans(m11_17, name='medic_clusters', fraction=0.50, save=save_data,
                                            output=True, K=250)
        engine_clusters, _ = modeling.kmeans(e11_17, name='engine_clusters', fraction=0.50, save=save_data,
                                             output=True, K=250)
        ladder_clusters, _ = modeling.kmeans(l11_17, name='ladder_clusters', fraction=1, save=save_data,
                                             output=True, K=250)
        clusters_11_17, kmeans11_17 = modeling.kmeans(sim_11_17, fraction=0.20, save=save_data,
                                                      output=True, K=250)
        clusters_18, kmeans18 = modeling.kmeans(sim_18, fraction=1, save=save_data, output=True, K=250)
        print("Completed clustering.")
        out_times_all = wrangling.out_times(time_files)
        EMAL = wrangling.EMAL_dataset(master, fire_protocols, ems_protocols,
                                      out_times_all)
        EMAL = modeling.kmeans_predict(kmeans11_17, EMAL, colname="cluster_11_17")
        EMAL = modeling.kmeans_predict(kmeans18, EMAL, colname="cluster_18")
        print("Completed cluster assignments.")
    else:
        print("")
        print("To skip clustering, please ensure your cluster files been loaded into the data folder."
              "Name the modeling clusters by the vehicle type, and the simulator clusters by year. "
              "The clusters needed are: ambulance_clusters.csv, medic_clusters.csv, engine_clusters.csv, "
              "ladder_clusters.csv, clusters_11_17, clusters_18.")
        input("Press Enter to proceed...")
        ambulance_clusters = modeling.read_cluster_data(
            os.path.join(os.getcwd(), "data", 'ambulance_clusters.csv'))
        medic_clusters = modeling.read_cluster_data(
            os.path.join(os.getcwd(), "data", 'medic_clusters.csv'))
        engine_clusters = modeling.read_cluster_data(
            os.path.join(os.getcwd(), "data", 'engine_clusters.csv'))
        ladder_clusters = modeling.read_cluster_data(
            os.path.join(os.getcwd(), "data", 'ladder_clusters.csv'))
        clusters_11_17 = modeling.read_cluster_data(os.path.join(os.getcwd(), "data", 'clusters11_17.csv'))
        clusters_18 = modeling.read_cluster_data(os.path.join(os.getcwd(), "data", 'clusters18.csv'))
        print("Loaded clusters from csv.")
        EMAL = pd.read_csv(os.path.join(os.getcwd(), "data", 'EMAL_dispatches_w_dp.csv'))
        print("Loaded cluster assignments.")

    # load station capacity data
    station_data = modeling.read_station_data(stations, bays)

    # get time matrix for each vehicle cluster using google maps API
    if args.run_google_maps_api:
        print("")
        print("Get a google maps api key (with sufficient funds) and save it in the data file as api_key.txt.")
        input("Press Enter to proceed...")
        with open(os.path.join(os.getcwd(), "data", "raw", 'api_key.txt'), 'r') as api:
            api_key = api.read()
        print("API key in use:")
        print(api_key)
        ambulance_time_matrix = modeling.get_time_matrix(ambulance_clusters, station_data, api_key,
                                                         "ambulance_time_matrix.csv")
        medic_time_matrix = modeling.get_time_matrix(medic_clusters, station_data, api_key, "medic_time_matrix.csv")
        engine_time_matrix = modeling.get_time_matrix(engine_clusters, station_data, api_key, "engine_time_matrix.csv")
        ladder_time_matrix = modeling.get_time_matrix(ladder_clusters, station_data, api_key, "ladder_time_matrix.csv")

        # ems_time_matrix = modeling.get_time_matrix(ems_clusters, station_data, api_key, "ems_time_matrix.csv")
        # fire_time_matrix = modeling.get_time_matrix(fire_clusters, station_data, api_key, "fire_time_matrix.csv")
    else:
        # ems_time_matrix = np.loadtxt(os.path.join(os.getcwd(), "data", 'ems_time_matrix.csv'), delimiter=',')
        # fire_time_matrix = np.loadtxt(os.path.join(os.getcwd(), "data", 'fire_time_matrix.csv'), delimiter=',')
        ambulance_time_matrix = np.loadtxt(os.path.join(os.getcwd(), "data",
                                                        "ambulance_time_matrix.csv"), delimiter=',')
        medic_time_matrix = np.loadtxt(os.path.join(os.getcwd(), "data", "medic_time_matrix.csv"), delimiter=',')
        engine_time_matrix = np.loadtxt(os.path.join(os.getcwd(), "data", "engine_time_matrix.csv"), delimiter=',')
        ladder_time_matrix = np.loadtxt(os.path.join(os.getcwd(), "data", "ladder_time_matrix.csv"), delimiter=',')
        simulation_time_matrix = np.loadtxt(os.path.join(os.getcwd(), "data",
                                                         "simulation_time_matrix.csv"), delimiter=',')

    # simulation
    sim_data, hist_sim_data = simulation.prep_sim_data(EMAL)
    if args.run_simulators:
        HFD_allocation = simulation.read_hfd(HFD_allocation)

        # simulate with protocol-based simulator
        dispatch_time_dict, new_time_matrix = simulation.get_dispatch_times(clusters_11_17, station_data,
                                                                            simulation_time_matrix, EMAL)
        sim_matrix_T = np.transpose(simulation_time_matrix)
        sim_matrix_T = sim_matrix_T / 60.0

        ems_to_time, ems_complete_time, fire_to_time, fire_complete_time = simulation.derive_time_distributions(
            EMAL)
        print("")
        print(
            "Ensure all of the allocations are in the data folder. These allocations are created from the modeling "
            "results. The allocations needed are the MEXCLP restricted allocation, the MEXCLP unrestricted allocation, "
            "and the infinite allocation.")
        input("Press Enter to proceed...")
        MEXCLP_restricted_alloc = pd.read_csv(os.path.join(os.getcwd(), "data", 'MEXCLP_independent_merged.csv'),
                                              header=0)
        MEXCLP_unrestricted_alloc = pd.read_csv(
            os.path.join(os.getcwd(), "data", 'MEXCLP_independent_allocation.csv'),
            header=0)
        infinite_alloc = pd.read_csv(os.path.join(os.getcwd(), "data", 'infinite_allocation.csv'), header=0)
        add_5_amb_alloc = simulation.generate_new('HFD_allocation.csv', 'chain_average_lengths.csv', 'multiplied',
                                                  5, 0, 0, 0, 1)

        hfd_sim = simulation.simulator(sim_data, HFD_allocation, ems_protocols, fire_protocols,
                                       dispatch_time_dict, sim_matrix_T, station_data, ems_to_time, fire_to_time,
                                       ems_complete_time, fire_complete_time, 2, 'p')

        MEXCLP_restricted = simulation.simulator("MEXCLP_restricted_simulation", sim_data, MEXCLP_restricted_alloc,
                                                 ems_protocols, fire_protocols, dispatch_time_dict,
                                                 sim_matrix_T, station_data, ems_to_time, fire_to_time,
                                                 ems_complete_time,
                                                 fire_complete_time, 2, 'p')

        MEXCLP_unrestricted = simulation.simulator("MEXCLP_unrestricted_simulation", sim_data,
                                                   MEXCLP_unrestricted_alloc, ems_protocols, fire_protocols,
                                                   dispatch_time_dict, sim_matrix_T, station_data, ems_to_time,
                                                   fire_to_time, ems_complete_time, fire_complete_time, 2, 'p')

        infinite = simulation.simulator("infinite_simulation", sim_data, infinite_alloc, ems_protocols,
                                        fire_protocols, dispatch_time_dict, sim_matrix_T, station_data, ems_to_time,
                                        fire_to_time, ems_complete_time, fire_complete_time, 2, 'p')

        add_5_amb = simulation.simulator("add_5_amb", sim_data, add_5_amb_alloc, ems_protocols, fire_protocols,
                                         dispatch_time_dict, sim_matrix_T, station_data, ems_to_time, fire_to_time,
                                         ems_complete_time, fire_complete_time, 2, 'p')

        # simulate with historical-based simulator
        hfd_sim_hist = simulation.simulator_hist("hfd_sim_hist", hist_sim_data, HFD_allocation, dispatch_time_dict,
                                                 sim_matrix_T, station_data, 'p')
        MEXCLP_restricted_hist = simulation.simulator_hist("MEXCLP_restricted_hist", hist_sim_data,
                                                           MEXCLP_restricted_alloc, dispatch_time_dict,
                                                           sim_matrix_T,
                                                           station_data, 'p')
        add_5_amb_hist = simulation.simulator_hist("add_5_amb_hist", hist_sim_data, add_5_amb_alloc,
                                                   dispatch_time_dict,
                                                   sim_matrix_T, station_data, 'p')
        print("Completed simulation.")

    print("Data science pipeline complete! Congrats :)")


def main():
    """
    Written by Lynn Zhu
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_wrangling", action='store_true', dest='skip_wrangling',
                        help="Flag to skip wrangling when all intermediate files"
                             "have already been generated and saved into data folder")
    parser.add_argument("--skip_save_data", action='store_true', dest='skip_save_data',
                        help="Flag to skip saving intermediate data tables")
    parser.add_argument("--run_find_chains", action='store_true', dest='run_find_chains',
                        help="Flag to run chain analysis step")
    parser.add_argument("--skip_figures", action='store_true', dest='skip_figures',
                        help="Flag to skip generating and saving EDA figures")
    parser.add_argument("--run_clustering", action='store_true', dest='run_clustering',
                        help="Flag to run clustering step")
    parser.add_argument("--run_google_maps_api", action='store_true', dest='run_google_maps_api',
                        help="Flag to run the Google Maps API to create a distance matrix")
    parser.add_argument("--run_simulators", action='store_true', dest='run_simulators',
                        help="Flag to run the protocol and historical simulators")
    args = parser.parse_args()
    # use os.listdir("data")
    data_files = os.listdir(os.path.join(os.getcwd(), "data", "raw"))


    # codes = os.path.join(os.getcwd(), "data", "raw", "incidentdescriptions.csv")
    stations = os.path.join(os.getcwd(), "data", "raw", "HFD_stations.csv")
    raw_bays = os.path.join(os.getcwd(), "data", "raw", "raw_bays.csv")
    territories_shp = os.path.join(os.getcwd(), "data", "raw", "territories.shp")
    fire_protocols = os.path.join(os.getcwd(), "data", "raw", "fire_dispatching_protocol.csv")
    fire_protocols = pd.read_csv(os.path.join(os.getcwd(), "data", fire_protocols))
    ems_protocols = os.path.join(os.getcwd(), "data", "raw", "ems_dispatching_protocol.csv")
    ems_protocols = pd.read_csv(os.path.join(os.getcwd(), "data", ems_protocols))
    current_allocation = os.path.join(os.getcwd(), "data", "raw", "HFD_allocation.csv")

    # separate incident, dispatch, time, and hospital files
    incident_files = []
    dispatch_files = []
    time_files = []
    hospital_files = []

    for file in data_files:
        name = file.split(os.sep)[-1]
        if name.startswith("incidents_"):
            incident_files.append(os.path.join(os.getcwd(), "data", "raw", file))
        elif name.startswith("dispatches_"):
            dispatch_files.append(os.path.join(os.getcwd(), "data", "raw", file))
        elif name.startswith("timereports_"):
            time_files.append(os.path.join(os.getcwd(), "data", "raw", file))
        elif name.startswith("hospital_"):
            hospital_files.append(os.path.join(os.getcwd(), "data", "raw", file))

    # call run_pipeline with lists
    run_pipeline(incident_files, dispatch_files, time_files, raw_bays, territories_shp, stations,
                 fire_protocols, ems_protocols, current_allocation, args, save_data=(not args.skip_save_data),
                 save_figs=(not args.skip_figures))


if __name__ == "__main__":
    main()
