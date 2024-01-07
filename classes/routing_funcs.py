import numpy as np
import numba as nb
import pandas as pd
import utm
import datetime
import copy
import math
import multiprocessing
from math import radians
from typing import List, Union, Any, Tuple, Dict
from pandera.typing import Series
from .predictors.cts_utils import haversine_nb
#from hgs_vrptw import GenVRP, VRPTWInstance, VRPTWOutput

DF = pd.DataFrame
MINUTES = 60

import pickle as pkl


def _load_test_instance():
    with open("nodes.pickle", "rb") as file:
        nodes = pkl.load(file)

    with open("slot_selections.pickle", "rb") as file:
        slot_selections = pkl.load(file)

    return nodes, slot_selections


def _get_default_instance_parameters(
    slot_selections: DF,
    num_vehicles: Union[int, Any],
    capacity: Union[int, Any],
    partition_by: Union[str, None] = None,
) -> Tuple[int, int]:

    if partition_by is not None:
        nr_customers_largest_instance = (
            slot_selections.groupby(partition_by).count().node.max()
        )
    else:
        nr_customers_largest_instance = slot_selections.shape[0] - 1

    num_vehicles = (
        nr_customers_largest_instance if num_vehicles is None else num_vehicles
    )  # one vehicle per customer
    capacity = (
        nr_customers_largest_instance if capacity is None else capacity
    )  # one vehicle could route all customers

    return num_vehicles, capacity


def compute_cost_matrix(nodes: DF, truck_speed: float) -> np.ndarray:
    global MINUTES

    # build pairs of nodes
    nodes["join_column"] = 1
    nodes = nodes.set_index("join_column")
    costs = nodes.join(
        nodes.rename(
            columns={
                "node": "destination",
                "lat": "lat_destination",
                "long": "long_destination",
            }
        )
    )

    # compute cost between nodes as the distance
    costs["cost"] = costs["long lat long_destination lat_destination".split()].apply(
        lambda x: haversine_nb(
            radians(x.long),
            radians(x.lat),
            radians(x.long_destination),
            radians(x.lat_destination),
        ),
        axis=1,
    )

    # convert distance in kms to travel time in minutes
    costs["cost"] = (costs["cost"] / truck_speed * MINUTES).astype(int)

    # get costs as a matrix with elements < origin node, destination node >
    costs = costs[["node", "destination", "cost"]].pivot(
        index="node", columns="destination", values="cost"
    )

    # generate np.ndarray with the same order as the one in nodes
    costs = costs.loc[nodes.set_index("node").index][
        nodes.set_index("node").index
    ].values

    return costs


def transform_latlong_into_xy_coordinates(nodes: DF) -> List:

    # use utm to convert <lat, lon> to <x, y>
    nodes[["x", "y"]] = pd.DataFrame(
        nodes.apply(
            lambda x: utm.from_latlon(latitude=x.lat, longitude=x.long), axis=1
        ).tolist(),
        index=nodes.index,
    )[[0, 1]]

    # get relative coordinates
    nodes["x"] = (
        nodes["x"] - nodes[lambda x: x.node.str.contains("STORE")]["x"].values.squeeze()
    )
    nodes["y"] = (
        nodes["y"] - nodes[lambda x: x.node.str.contains("STORE")]["y"].values.squeeze()
    )

    # convert coordinates to int
    nodes["x y".split()] = nodes["x y".split()].astype(int)

    # build node_coordinates as a List[Tuple[int, int]]
    xy_coordinates = list(nodes["x y".split()].itertuples(index=False, name=None))

    return xy_coordinates


def get_node_time_windows(slot_selections: DF, duration_matrix: np.ndarray) -> List:
    # get minutes since start of the day
    slot_selections = slot_selections.assign(
        hour_start=lambda x: x.slot_time.str.split(" - ").str[0],
        hour_end=lambda x: x.slot_time.str.split(" - ").str[1],
    )
    slot_selections["hour_start"] = pd.to_timedelta(
        slot_selections["hour_start"] + ":00"
    ) / pd.to_timedelta(1, "m")
    slot_selections["hour_end"] = pd.to_timedelta(
        slot_selections["hour_end"] + ":00"
    ) / pd.to_timedelta(1, "m")

    # get minutes since first day
    first_delivery_day = slot_selections[
        lambda x: ~x.deliverydate.isna()
    ].deliverydate.min()
    slot_selections = slot_selections.assign(
        day_start=lambda x: (x.deliverydate - first_delivery_day)
        / pd.to_timedelta(1, "m")
    )

    # compute slot_start and slot_end
    slot_selections = slot_selections.assign(
        slot_start=lambda x: x.day_start + x.hour_start,
        slot_end=lambda x: x.day_start + x.hour_end,
    )[["node", "slot_start", "slot_end"]]

    # to tighten the bounds of the problem,
    #  slot_end(STORE) = max(slot_end) + max(duration_matrix(arg max(slot_end), STORE))
    max_slot_end = slot_selections.slot_end.max()
    arg_max_slot_end = slot_selections[
        lambda x: x.slot_end == max_slot_end
    ].index.values
    max_duration_to_depot = max(duration_matrix[arg_max_slot_end, -1])
    slot_selections["slot_end"] = np.where(
        slot_selections.node.str.contains("STORE"),
        max_slot_end + max_duration_to_depot,
        slot_selections.slot_end,
    )

    # slot_start(START) = min(slot_start) - min(duration_matrix(arg min(slot_start), STORE))
    min_slot_start = slot_selections.slot_start.min()
    arg_min_slot_start = slot_selections[
        lambda x: x.slot_start == min_slot_start
    ].index.values
    min_duration_to_first_time_slot = min(duration_matrix[0, arg_min_slot_start])
    slot_selections["slot_start"] = np.where(
        slot_selections.node.str.contains("STORE"),
        min_slot_start - min_duration_to_first_time_slot,
        slot_selections.slot_start,
    )
    slot_selections[["slot_start", "slot_end"]] = slot_selections[
        ["slot_start", "slot_end"]
    ] - (
        min_slot_start - min_duration_to_first_time_slot
    )  # change the time reference

    slot_selections["slot_start slot_end".split()] = slot_selections[
        "slot_start slot_end".split()
    ].astype(int)

    return slot_selections["slot_start slot_end".split()].values


def time_window_feasibility_check(
    routes: np.ndarray, node_time_windows: np.ndarray, duration_matrix: np.ndarray
) -> bool:

    arrival_times = np.zeros(shape=routes.shape)

    for v in np.arange(nr_vehicles := routes.shape[0], dtype=int):
        for pos in np.arange(nr_route_positions := routes.shape[1], dtype=int):
            node: int = routes[v][pos]
            if node == -1:
                # end of route reached
                break

            slot_start = node_time_windows[node][0]
            slot_end = node_time_windows[node][1]

            if pos == 0:
                # depot to first customer
                arrival_times[v][pos] = max(duration_matrix[0][node], slot_start)
            else:
                arrival_times[v][pos] = max(
                    (arrival_at_previous_node := arrival_times[v][pos - 1])
                    + duration_matrix[(previous_node := routes[v][pos - 1]), node],
                    slot_start,
                )

            if arrival_times[v][pos] < slot_start or arrival_times[v][pos] > slot_end:
                # the solution is infeasible
                return False

    # solution is feasible
    return True


@nb.jit(nopython=True)
def compute_routing_cost(
    routes: np.ndarray,
    fixed_cost: float,
    variable_cost: float,
    distance_matrix: np.ndarray,
) -> float:
    """
    Computes the operational routing cost as the sum of fixed and variable costs

    Parameters
    ----------
    routes : np.ndarray
        the routing schedule given as < vehicle, route position >
    fixed_cost : float
        the cost of deploying each vehicle
    variable_cost : float
        the cost incurred per km traversed by the vehicle
    distance_matrix: np.ndarray
        number of kms necessary to traverse between any node i and j

    Returns
    -------
    float
        Total transportation cost
    """

    total_fixed_cost = fixed_cost * routes.shape[0]

    total_variable_cost = 0
    for route in routes:
        # distance depot to 1st node
        total_variable_cost += distance_matrix[0, route[0]] * variable_cost

        for pos in np.arange(len(route) - 1):
            origin = route[pos]
            destination = route[pos + 1]
            total_variable_cost += distance_matrix[origin, destination] * variable_cost

        # distance last node to depot
        total_variable_cost += distance_matrix[route[-1], 0] * variable_cost

    return total_fixed_cost + total_variable_cost


def compute_hierarchical_routing_cost(
    instances: List,
    results: List,
    fixed_cost: float,
    variable_cost: float,
    truck_speed: float,
) -> float:
    global MINUTES

    nr_required_vehicles = max(pd.Series(results).apply(lambda x: len(x.routes)))
    total_fixed_cost = fixed_cost * nr_required_vehicles

    total_variable_cost = 0
    for instance_id in np.arange(len(instances)):
        max_customers_per_vehicle = max(
            [len(route.nodes) for route in results[instance_id].routes]
        )
        routes = np.array(
            [
                np.append(
                    np.array(route.nodes),
                    np.repeat(-1, max_customers_per_vehicle - len(route.nodes)),
                )
                for route in results[instance_id].routes
            ]
        )  # populate blank spaces in the routes with -1 nodes to allow using np.ndarray

        distance_matrix = (
            instances[instance_id].duration_matrix * truck_speed / MINUTES
        )  # convert minutes to kms

        total_variable_cost += compute_routing_cost(
            routes=routes,
            fixed_cost=0,
            variable_cost=variable_cost,
            distance_matrix=distance_matrix,
        )

    return total_fixed_cost + total_variable_cost


def create_instances(
    nodes: DF,
    slot_selections: DF,
    partition_by: Union[List, np.ndarray] = "areaid deliverydate".split(),
) -> List[Dict]:

    instances = []
    store_info = slot_selections[lambda x: x.node == "STORE"]

    for group in slot_selections.groupby(partition_by):
        # group is a tuple with (group_by_index, corresponding_df)
        _slot_selections = pd.DataFrame(group[1])
        _nodes = nodes[
            lambda x: x.node.isin(np.append(_slot_selections.node.values, ["STORE"]))
        ]
        instance = {
            "id": group[0],
            "slot_selections": pd.concat([store_info, _slot_selections]),
            "nodes": _nodes,
        }

        instances.append(instance)

    return instances


def convert_into_vrptw_instance(
    instance: Dict,
    num_vehicles: Union[int, Any] = None,
    capacity: Union[int, Any] = None,
    truck_speed: float = 20,  # km/h
) -> VRPTWInstance:
    nodes = instance["nodes"]
    slot_selections = instance["slot_selections"]

    num_vehicles, capacity = _get_default_instance_parameters(
        slot_selections, num_vehicles, capacity
    )

    # ensure same row order between nodes and slot_selections
    nodes = nodes.reset_index(drop=True)
    slot_selections = slot_selections.reset_index(drop=True)

    # save backup of nodes as it is mutated in some functions
    _nodes = copy.deepcopy(nodes)

    node_coordinates = transform_latlong_into_xy_coordinates(nodes)
    nodes = copy.deepcopy(_nodes)

    # demand is unitary, since capacity is measured in number of orders
    node_demands = np.where(nodes["node"].str.contains("STORE"), 0, 1)

    node_service_times = np.zeros(shape=(nodes.shape[0],), dtype=int)

    duration_matrix = compute_cost_matrix(nodes, truck_speed)
    nodes = copy.deepcopy(_nodes)

    node_time_windows = get_node_time_windows(slot_selections, duration_matrix)

    instance_args = {
        "capacity": capacity,
        "num_vehicles": num_vehicles,
        "node_coordinates": node_coordinates,
        "node_demands": node_demands,
        "node_service_times": node_service_times,
        "node_time_windows": node_time_windows,
        "duration_matrix": duration_matrix,
    }

    v = VRPTWInstance(**instance_args)

    return v


def hierarchically_compute_vrptw_routing_cost(
    nodes: DF,
    slot_selections: DF,
    num_vehicles: Union[int, Any] = None,
    capacity: Union[int, Any] = None,
    truck_speed: float = 20,  # km/h
    fixed_cost: float = 500,  # m.u./used vehicle
    variable_cost: float = 0.5,  # m.u./km
    partition_by="areaid deliverydate".split(),
) -> float:
    global MINUTES

    # Partition instance by `partition_by` conditions
    instances = create_instances(nodes, slot_selections)

    # Convert instances into VRPTWInstance objects
    vrptw_instances = []
    for instance in instances:
        vrptw_instances.append(convert_into_vrptw_instance(instance))

    # Solve instances
    results = []
    gv = GenVRP()
    for instance in vrptw_instances:
        results.append(gv.evaluate_instance(instance))

    # Evaluate results
    results = np.array(results)
    if np.any(results == None):
        # if at least one result is None means GenVRP did not
        # find a feasible solution
        return -math.inf

    return compute_hierarchical_routing_cost(
        instances=vrptw_instances,
        results=results,
        fixed_cost=fixed_cost,
        variable_cost=variable_cost,
        truck_speed=truck_speed,
    )


def compute_vrptw_routing_cost(
    nodes: DF,
    slot_selections: DF,
    num_vehicles: Union[int, Any] = None,
    capacity: Union[int, Any] = None,
    truck_speed: float = 20,  # km/h
    fixed_cost: float = 500,  # m.u./used vehicle
    variable_cost: float = 0.5,  # m.u./km
) -> float:
    global MINUTES

    num_vehicles, capacity = _get_default_instance_parameters(
        slot_selections, num_vehicles, capacity
    )

    # ensure same row order between nodes and slot_selections
    nodes = nodes.reset_index(drop=True)
    slot_selections = slot_selections.reset_index(drop=True)

    # save backup of nodes as it is mutated in some functions
    _nodes = copy.deepcopy(nodes)

    node_coordinates = transform_latlong_into_xy_coordinates(nodes)
    nodes = copy.deepcopy(_nodes)

    # demand is unitary, since capacity is measured in number of orders
    node_demands = np.where(nodes["node"].str.contains("STORE"), 0, 1)

    node_service_times = np.zeros(shape=(nodes.shape[0],), dtype=int)

    duration_matrix = compute_cost_matrix(nodes, truck_speed)
    nodes = copy.deepcopy(_nodes)

    node_time_windows = get_node_time_windows(slot_selections, duration_matrix)

    instance_args = {
        "capacity": capacity,
        "num_vehicles": num_vehicles,
        "node_coordinates": node_coordinates,
        "node_demands": node_demands,
        "node_service_times": node_service_times,
        "node_time_windows": node_time_windows,
        "duration_matrix": duration_matrix,
    }

    v = VRPTWInstance(**instance_args)

    gv = GenVRP()
    result = gv.evaluate_instance(v)

    if result is None:
        # result None means GenVRP did not find a feasible solution
        return -math.inf

    # adapt result as np.ndarray
    max_customers_per_vehicle = max([len(route.nodes) for route in result.routes])
    routes = np.array(
        [
            np.append(
                np.array(route.nodes),
                np.repeat(-1, max_customers_per_vehicle - len(route.nodes)),
            )
            for route in result.routes
        ]
    )  # populate blank spaces in the routes with -1 nodes to allow using np.ndarray

    feasible = time_window_feasibility_check(routes, node_time_windows, duration_matrix)

    if not feasible:
        return -math.inf

    distance_matrix = duration_matrix * truck_speed / MINUTES  # convert minutes to kms

    return compute_routing_cost(routes, fixed_cost, variable_cost, distance_matrix)
