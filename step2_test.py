from os import path
import sys, json, time
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import pickle
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
import geopy.distance
from scipy.special import softmax, log_softmax
# Get Directory
# BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
BASE_DIR = path.dirname("C:/Users/alex/Desktop/Thesis/repo/thesis-amz/step2_test/")
print(BASE_DIR)

## model_path=path.join(BASE_DIR, 'data/model_build_outputs/all_zones_ordered.pickle')
## with open(model_path, 'rb') as out_file:
##     all_zones_ordered = pickle.load( out_file)
## print( all_zones_ordered
## )
# Read input data
prediction_routes_path = path.join(BASE_DIR, 'data/route_data/route_data_544.json')
#prediction_routes_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_route_data.json')
actual_sequences_path = path.join(BASE_DIR, 'data/actual_sequences/actual_sequences.json')
print(prediction_routes_path)
route_data_df = pd.read_json(prediction_routes_path,
                             orient='index')  # shape: (6, 544)
actual_sequences_df = pd.read_json(actual_sequences_path,
                                   orient='index')  # shape: (6, 6112)
route_actual_sequence_data_df = (pd.merge(route_data_df, actual_sequences_df,
                                          left_index=True, right_index=True, how="inner")) # shape: (7, 544)

print(route_actual_sequence_data_df)

route_time_path = path.join(BASE_DIR, 'data/travel_times/travel_times_544.json')
#route_time_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_travel_times.json')
with open(route_time_path, newline='') as in_file:
    model_apply_time_data= json.load(in_file)

model_apply_route_data = route_data_df

def sequence_of_zones(row):
    sequence = []
    actual = sorted(list(row["actual"]), key=lambda l: row["actual"][l])
    actual.append(actual[0])
    # finding the sequence of zones

    for i, stop in enumerate(actual):
        if len(sequence) == 0 or i == len(actual) - 1:
            sequence.append("0")
        elif row["stops"][stop]["zone_id"] is None:
            continue
        elif row["stops"][stop]["zone_id"] != sequence[-1]:
            sequence.append(row["stops"][stop]["zone_id"])
        # drop the stop if it the stop has no zone id

    return sequence

def random_sequence_of_zones(row):
    sequence = []
    actual = sorted(list(row["actual"]), key=lambda l: row["actual"][l])
    actual.append(actual[0])
    # finding the sequence of zones

    for i, stop in enumerate(actual):
        if len(sequence) == 0 or i == len(actual) - 1:
            sequence.append("0")
        if row["stops"][stop]["zone_id"] is None:
            continue
        elif row["stops"][stop]["zone_id"] != sequence[-1]:
            sequence.append(row["stops"][stop]["zone_id"])
        # drop the stop if it the stop has no zone id
    sequence.pop(len(sequence) - 1)
    sequence.pop(0)
    random.shuffle(sequence)
    return ['0'] + sequence + ['0']


# apply transformations

def group_lst(lst):
    grouped_lst = [{lst[0]: 1}]
    for el in lst[1:]:
        if el in grouped_lst[-1]:
            grouped_lst[-1][el] += 1
        else:
            grouped_lst = grouped_lst + [{el: 1}]
    return grouped_lst


def keep_highest_values(lst):
    result = []
    seen = {}
    for d in lst:
        for key, value in d.items():
            if key not in seen:
                seen[key] = value
                result.append(d)
            elif value > seen[key]:
                result.remove({key: seen[key]})
                seen[key] = value
                result.append(d)
    return result


def get_reduced_sequence(lst):
    result = []
    zones_nstops_lst = keep_highest_values(group_lst(lst))
    for dict in zones_nstops_lst:
        for key, value in dict.items():
            result += [key]
    return result


def reduced_sequence_of_zones(row):
    sequence = []
    actual = sorted(list(row["actual"]), key=lambda l: row["actual"][l])
    actual.append(actual[0])
    # finding the sequence of zones

    for i, stop in enumerate(actual):
        if len(sequence) == 0 or i == len(actual) - 1:
            sequence.append("0")
        elif row["stops"][stop]["zone_id"] is None:
            continue
        else:
            sequence.append(row["stops"][stop]["zone_id"])
        # drop the stop if it the stop has no zone id
    sequence = get_reduced_sequence(sequence)

    return sequence + ["0"]

route_actual_sequence_data_df["sequences_of_zones"] = route_actual_sequence_data_df.apply(
    lambda row: sequence_of_zones(row),
    axis=1)
route_actual_sequence_data_df["reduced_sequences_of_zones"] = route_actual_sequence_data_df.apply(
    lambda row: reduced_sequence_of_zones(row),
    axis = 1)
route_actual_sequence_data_df["random_sequences_of_zones"] = route_actual_sequence_data_df.apply(
    lambda row: random_sequence_of_zones(row),
    axis = 1)

print(route_actual_sequence_data_df["sequences_of_zones"][0])
print(route_actual_sequence_data_df["reduced_sequences_of_zones"][0])
print(route_actual_sequence_data_df["random_sequences_of_zones"][0])

#print(route_actual_sequence_data_df)
def compute_euclidean_distance_matrix(locations):
    """Creates callback to return distance between points."""
    distances = np.zeros((len(locations),len(locations)  ))
    for from_counter, from_node in enumerate(locations):
        # distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                # distances[from_counter][to_counter] = (int(
                #     100*math.hypot((from_node[0] - to_node[0]), #multiplied by 100 because of int
                #                (from_node[1] - to_node[1]))))
                distances[from_counter][to_counter] = 100*geopy.distance.distance(from_node,to_node).km
    return distances

def create_data_model(prob_mat):
    """Stores the data for the problem."""
    # Note that distances SHOULD BE integers; multiply by 100 for probabilities
    data = {}
    data['distance_matrix'] = prob_mat
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def get_routes(solution, routing, manager):
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes

def create_time_matrix_test(stopslist, jsondata):
    times = np.zeros((len(stopslist), len(stopslist)))
    for i, source in enumerate(stopslist):
        for j, dest in enumerate(stopslist):
            times[i][j] = jsondata[source][dest]
    return times

def step2(stops, time_mat, inv_zones_selected, predicted_sequence_of_zones):
    locations = []  # initialize with loc of depot
    for i in stops:
        locations.append((stops[i]['lat'], stops[i]['lng']))
    dist_mat = compute_euclidean_distance_matrix(locations)
    zone_list = []
    for stop in stops:
        zone_list.append(stops[stop]["zone_id"])
    #print("zone_list: ", len(zone_list), zone_list)
    zones_in_hist = [v for k, v in inv_zones_selected.items()]
    #print("zones_in_hist: ", len(zones_in_hist), zones_in_hist)

    NoZones_idx = [i + 1 for i, j in enumerate(zone_list[1:]) if j not in zones_in_hist]
    #print("NoZones_idx: ", len(NoZones_idx), NoZones_idx)
    for NoZoneStop in NoZones_idx:
        dist_list = dist_mat[NoZoneStop].tolist()  # list(dist_mat[NoZoneStop].values())
        nearest_idx = dist_list.index(min([i for i in dist_list if i > 0]))
        zone_list[NoZoneStop] = zone_list[nearest_idx]
    zones_selected = {v: k for k, v in inv_zones_selected.items()}
    zones_selected[None] = 0
    zone_list_idx = list(map(zones_selected.get, zone_list))
    #print("zone_list_idx: ", len(zone_list_idx), zone_list_idx)

    # Register callback with the solver.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    zone_order_predicted = predicted_sequence_of_zones
    #print("zone_order_predicted: ", len(zone_order_predicted), zone_order_predicted)
    #print("inv_zones_selected: ", inv_zones_selected)
    #print("zones_selected: ", zones_selected)
    zones_selected = {v: k for k, v in inv_zones_selected.items()}
    #print("zones_selected: ", zones_selected)
    zone_order = [list(map(zones_selected.get, zone_order_predicted))]
    #print("zone_order: ", len(zone_order[0]), zone_order )

    combi_mat = np.zeros((len(stops), len(stops)))
    #print("stops: ", len(stops), stops)
    for i in range(len(stops)):
        for j in range(len(stops)):
            # compute for zone penalty w_i = [2 0.35 2.13 4.09 2.32 4.02 6.10]
            if zone_list_idx[i] == zone_list_idx[j]:
                zone_penalty = 0.35
            elif zone_order[0].index(zone_list_idx[j]) - 1 == zone_order[0].index(zone_list_idx[i]):
                zone_penalty = 2.13
            elif zone_order[0].index(zone_list_idx[j]) - 2 == zone_order[0].index(zone_list_idx[i]):
                zone_penalty = 4.09
            elif zone_order[0].index(zone_list_idx[j]) + 1 == zone_order[0].index(zone_list_idx[i]):
                zone_penalty = 2.32
            elif zone_order[0].index(zone_list_idx[j]) + 2 == zone_order[0].index(zone_list_idx[i]):
                zone_penalty = 4.02
            else:
                zone_penalty = 6.10

            combi_mat[i][j] = 2*time_mat[i][j] + zone_penalty
    data = create_data_model(10000*combi_mat)
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Register callback with the solver.
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 120

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        stop_order = get_routes(solution, routing, manager)[0][:-1]
        return stop_order

def step2_tt(stops, time_mat, inv_zones_selected, predicted_sequence_of_zones):
    locations = []  # initialize with loc of depot
    for i in stops:
        locations.append((stops[i]['lat'], stops[i]['lng']))
    dist_mat = compute_euclidean_distance_matrix(locations)
    zone_list = []
    for stop in stops:
        zone_list.append(stops[stop]["zone_id"])
    #print("zone_list: ", len(zone_list), zone_list)
    zones_in_hist = [v for k, v in inv_zones_selected.items()]
    #print("zones_in_hist: ", len(zones_in_hist), zones_in_hist)

    NoZones_idx = [i + 1 for i, j in enumerate(zone_list[1:]) if j not in zones_in_hist]
    #print("NoZones_idx: ", len(NoZones_idx), NoZones_idx)
    for NoZoneStop in NoZones_idx:
        dist_list = dist_mat[NoZoneStop].tolist()  # list(dist_mat[NoZoneStop].values())
        nearest_idx = dist_list.index(min([i for i in dist_list if i > 0]))
        zone_list[NoZoneStop] = zone_list[nearest_idx]
    zones_selected = {v: k for k, v in inv_zones_selected.items()}
    zones_selected[None] = 0
    zone_list_idx = list(map(zones_selected.get, zone_list))
    #print("zone_list_idx: ", len(zone_list_idx), zone_list_idx)

    # Register callback with the solver.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    zone_order_predicted = predicted_sequence_of_zones
    #print("zone_order_predicted: ", len(zone_order_predicted), zone_order_predicted)
    #print("inv_zones_selected: ", inv_zones_selected)
    #print("zones_selected: ", zones_selected)
    zones_selected = {v: k for k, v in inv_zones_selected.items()}
    #print("zones_selected: ", zones_selected)
    zone_order = [list(map(zones_selected.get, zone_order_predicted))]
    #print("zone_order: ", len(zone_order[0]), zone_order )

    combi_mat = np.zeros((len(stops), len(stops)))
    #print("stops: ", len(stops), stops)
    #for i in range(len(stops)):
    #    for j in range(len(stops)):
    #        # compute for zone penalty w_i = [2 0.35 2.13 4.09 2.32 4.02 6.10]
    #        if zone_list_idx[i] == zone_list_idx[j]:
    #            zone_penalty = 0.35
    #        elif zone_order[0].index(zone_list_idx[j]) - 1 == zone_order[0].index(zone_list_idx[i]):
    #            zone_penalty = 2.13
    #        elif zone_order[0].index(zone_list_idx[j]) - 2 == zone_order[0].index(zone_list_idx[i]):
    #            zone_penalty = 4.09
    #        elif zone_order[0].index(zone_list_idx[j]) + 1 == zone_order[0].index(zone_list_idx[i]):
    #            zone_penalty = 2.32
    #        elif zone_order[0].index(zone_list_idx[j]) + 2 == zone_order[0].index(zone_list_idx[i]):
    #            zone_penalty = 4.02
    #        else:
    #            zone_penalty = 6.10
#
    #        combi_mat[i][j] = 2*time_mat[i][j] + zone_penalty
    data = create_data_model(10000*2*time_mat)
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Register callback with the solver.
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 120

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        stop_order = get_routes(solution, routing, manager)[0][:-1]
        return stop_order

op_dict = {}
i = 0
for index, model_apply_route_data_T0 in model_apply_route_data.iterrows():
    i += 1
    print("Iteration: ", i , " Working on index: ", index)
    routeID = model_apply_route_data_T0.name
    route_timedata = model_apply_time_data[index]
    station_id0= model_apply_route_data_T0['station_code']
    ## zones_ordered = all_zones_ordered[station_id0]

    route_T0 = pd.DataFrame(model_apply_route_data_T0['stops']).T.sort_values(by='type',
                                        ascending=False)
    stops =  route_T0.to_dict('index')
    #print("stops: ", len(stops), stops)


    try:
        ## route_T0['zone_matrix_id'] = route_T0.zone_id.map(zones_ordered).fillna(0).astype(int)
        station_lat, station_lng = route_T0.lat[0],route_T0.lng[0]
        ## zoneidx_list = route_T0.zone_matrix_id.unique().tolist()

        time_mat = create_time_matrix_test([*stops], route_timedata)
        time_mat = time_mat / time_mat.sum(axis=1, keepdims=True)  # normalize
        #print("time_mat: ", len(time_mat), time_mat)

        ## inv_zones_ordered = {v: k for k, v in zones_ordered.items()}
        ## inv_zones_selected = {k: inv_zones_ordered[ zoneidx_list[k] ] for k in  range(len(zoneidx_list)) }

        predicted_sequence_of_zones = route_actual_sequence_data_df.loc[routeID]['reduced_sequences_of_zones']

        inv_zones_selected = {k: zone for k,zone in enumerate(set(predicted_sequence_of_zones), start=1)}
        inv_zones_selected[0] = '0'
        #print(inv_zones_selected)

        op_seq = step2( stops, time_mat, inv_zones_selected, predicted_sequence_of_zones)
        not_in_opseq = list(set([i for i in range(len(route_T0))]) - set(op_seq))
        # print("not_in_opseq", not_in_opseq)
        op_seq = op_seq + not_in_opseq
        op_list = route_T0.index[op_seq].tolist()
        op_dict [index] = { 'proposed':  {k: v for v, k in enumerate(op_list )} }
    except:
        print("Error!!")
        op_dict [index] = { 'proposed':  {k: v for v, k in enumerate( list(stops.keys()) )} }



# Write output data
output_path=path.join(BASE_DIR, 'data/proposed_sequences_step2/proposed_sequences_test.json')
with open(output_path, 'w') as out_file:
    json.dump(op_dict, out_file)
    print("Success: The '{}' file has been saved".format(output_path))

print('Done!')