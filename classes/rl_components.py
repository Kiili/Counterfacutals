import itertools
import numpy as np
import pandas as pd

from math import radians
from scipy.stats import skew
from pandera.typing import Series
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

from .funcs import get_wtp
from .predictors.cts_utils import haversine_nb
#from .routing_funcs import (
#    compute_vrptw_routing_cost,
#    hierarchically_compute_vrptw_routing_cost,
#)
from .instance import Instance, Order, Slot, Customer
from .predictors.wtp_predictors import WTPPredictor, aq

DF = pd.DataFrame


@dataclass
class Action:
    panel: pd.DataFrame
    # TODO: Derive class
    price_level: float
    assymetry_level: float
    nr_options: Union[float, int]

    @property
    def features(self) -> dict:
        return asdict(self)


@dataclass
class State:
    order: Order
    customer: Customer
    slots: Series[Slot]
    # TODO: Derive class
    basket_value: float
    distance_to_centroids: pd.Series
    basket_value_lift: float

    @property
    def features(self) -> dict:
        return asdict(self)


@dataclass
class Observation:
    selected_slot: Optional[Tuple]
    previous_order: Optional[Order]
    reward: float
    order: Optional[Order]
    # TODO: Should Slot store Orders? Or just the order_id? Have another class with slot assignments?
    slots: Series[Slot]
    customer: Optional[Customer]


class Environment:
    instance: Instance
    wtp_predictor: WTPPredictor
    _stage_id: int
    _nr_orders: int
    current_timestamp: pd.Timestamp

    def __init__(self, instance: Instance, wtp_predictor: WTPPredictor) -> None:
        """
        Initializing an Environment object

        Parameters
        ----------
        instance : Instance
            The set of orders and time slots to simulate

        wtp_predictor : WTPPredictor
            The customer behavior model that simulates customer time slot bookings
        """
        self.og_instance = instance
        self.wtp_predictor = wtp_predictor
        self._nr_orders = instance.orders.shape[0]
        self.reset()

    def reset(self):
        self._stage_id = 0
        self.instance = self.og_instance.get_copy()
        self.current_timestamp = self.instance.orders.iloc[0].timestamp

    def _compute_panel_reward(self, action: Action) -> float:
        # present panel to customer
        panel = np.array([action.panel.values.reshape(-1)])
        order: Order = self.instance.orders.iloc[self._stage_id - 1]
        customer: Customer = self.instance.customers.loc[order.customer_id]
        target_slots = self.instance.slots.loc[action.panel.index]

        # determine selection probabilities
        _, selection_probability, walkaway_probability = get_wtp(
            wtp_predictor=self.wtp_predictor,
            order=order,
            panels=panel,
            customer=customer,
            slots=target_slots,
        )

        # book slot

        # 0 because 1 panel at stake
        order.walkaway_probability = walkaway_probability[0]

        order.walkaway = np.random.rand() <= order.walkaway_probability
        if order.walkaway:
            return 0

        # determine reward
        selected_slot = np.random.choice(
            np.arange(target_slots.index.shape[0]),
            p=selection_probability[0],  # 0 because 1 panel at stake
        )
        selected_slot_id = target_slots.index[selected_slot]
        order.selected_slot = self.instance.slots.loc[selected_slot_id]

        # update slot occupation
        order_id = self.instance.orders.index[self._stage_id - 1]
        self.instance.slots.loc[selected_slot_id].increment_occupation(order_id)

        # compute reward
        basket_value = order.order_df.total.iloc[0]
        delivery_fee = panel[0, selected_slot]  # 0 because 1 panel at stake
        return basket_value + delivery_fee

    def _transition(
        self, action: Action
    ) -> Tuple[Optional[Order], Optional[int], float]:
        """Upon a specific action, applies an environment transition"""

        is_in_booking_period = 0 < self._stage_id < self._nr_orders
        if is_in_booking_period:
            return self._compute_panel_transition(action)

        return (
            previous_order := None,
            selected_slot_index := None,
            reward := 0 if self._stage_id == 0 else 0,  # -self.compute_routing_cost(),
        )

    def _compute_panel_transition(
        self, action: Action
    ) -> Tuple[Order, Optional[int], float]:
        # present panel to customer
        panel = np.array([action.panel.values.reshape(-1)])
        order: Order = self.instance.orders.iloc[self._stage_id - 1]
        customer: Customer = self.instance.customers.loc[order.customer_id]
        target_slots = self.instance.slots.loc[action.panel.index]

        # determine selection probabilities
        _, selection_probability, walkaway_probability = get_wtp(
            wtp_predictor=self.wtp_predictor,
            order=order,
            panels=panel,
            customer=customer,
            slots=target_slots,
        )

        # book slot

        # 0 because 1 panel at stake
        order.walkaway_probability = walkaway_probability[0]

        order.walkaway = np.random.rand() <= order.walkaway_probability
        if order.walkaway:
            return (
                previous_order := order,
                selected_slot_index := None,
                reward := 0,
            )

        # determine reward
        selected_slot = np.random.choice(
            np.arange(target_slots.index.shape[0]),
            p=selection_probability[0],  # 0 because 1 panel at stake
        )
        selected_slot_index = target_slots.index[selected_slot]
        order.selected_slot = self.instance.slots.loc[selected_slot_index]

        # update slot occupation
        order_id = self.instance.orders.index[self._stage_id - 1]
        self.instance.slots.loc[selected_slot_index].increment_occupation(order_id)

        # compute reward
        basket_value = order.order_df.total.iloc[0]
        delivery_fee = panel[0, selected_slot]  # 0 because 1 panel at stake
        return (
            previous_order := order,
            selected_slot_index,
            reward := basket_value + delivery_fee,
        )

    def step(self, action: Action = None) -> Observation:
        """Applies an environment transition upon a given action."""
        previous_order, selected_slot, reward = self._transition(action)

        if not self._stage_id < self._nr_orders:
            return Observation(
                reward=reward,
                order=None,
                slots=None,
                selected_slot=None,
                previous_order=None,
                customer=None,
            )

        # Receives new order
        new_order = self.instance.orders.iloc[
            np.min([self._stage_id, self._nr_orders - 1])
        ]
        self.current_timestamp = new_order.timestamp

        # Check the available time slots
        self.instance.slots.apply(lambda x: x.check_open(self.current_timestamp))
        slots = self.instance.slots.loc[new_order.allowed_slots].loc[
            lambda x: x.apply(lambda y: y.is_open) == 1
        ]

        # Filter order_df for available time slots
        new_order.order_df = new_order.order_df.loc[slots.index]

        self._stage_id += 1

        return Observation(
            reward=reward,
            order=new_order,
            slots=slots,
            previous_order=previous_order,
            selected_slot=selected_slot,
            customer=self.instance.customers.loc[new_order.customer_id],
        )

    def _get_time_slot_selections(self) -> DF:
        # get list of orders for each time slot
        slot_allocations = self.instance.slots.apply(lambda x: x.assigned_orders)
        slot_allocations = slot_allocations.apply(lambda x: pd.Series(x, dtype="str"))

        # get time slot selection for each order
        order_selections = pd.melt(
            slot_allocations.reset_index(),
            id_vars=slot_allocations.index.names,
            value_vars=slot_allocations.columns,
            var_name="order_nr",
            value_name="node",
        )
        order_selections = order_selections[lambda x: ~x.node.isna()]
        order_selections = order_selections.drop(columns="order_nr")[
            "node areaid deliverydate slot_time".split()
        ]

        # create a row for a store's virtual selection
        store_selection = pd.DataFrame(
            {"node": "STORE", "areaid": 140, "deliverydate": None, "slot_time": None},
            index=[0],
        )
        order_selections = pd.concat([order_selections, store_selection]).sort_values(
            by="node", ascending=False
        )

        return order_selections

    def _get_node_coordinates(self, order_selections: pd.DataFrame) -> DF:

        # get customer locations
        demand_locations = self.instance.order_cp7s.loc[
            order_selections[lambda x: ~x.node.str.contains("STORE")].node
        ]["customer_lat customer_long".split()]
        demand_locations = demand_locations.reset_index().rename(
            columns={
                "shippingnumber": "node",
                "customer_lat": "lat",
                "customer_long": "long",
            }
        )

        # get store location
        store_locations = (
            self.instance.order_cp7s.loc[
                order_selections[lambda x: ~x.node.str.contains("STORE")].node
            ]["store_lat store_long".split()]
            .reset_index()
            .drop(columns="shippingnumber")
            .drop_duplicates()
        )
        store_locations = store_locations.assign(node=lambda x: "STORE")
        store_locations = store_locations.rename(
            columns={"store_lat": "lat", "store_long": "long"}
        )

        # concat result and order nodes
        nodes_df = (
            pd.concat([demand_locations, store_locations])
            .sort_values(by="node", ascending=False)
            .reset_index()
            .drop(columns="index")
        )

        return nodes_df

    def compute_random_routing_cost(self) -> float:
        return 1000 * np.random.random()

    def compute_routing_cost(self) -> float:

        slot_selections = self._get_time_slot_selections()
        nodes = self._get_node_coordinates(slot_selections)

        # return compute_vrptw_routing_cost(nodes, slot_selections)
        return hierarchically_compute_vrptw_routing_cost(nodes, slot_selections)


class Agent:
    hyperparameters: Dict[str, Any]
    symb_expr: str
    price_points: np.ndarray

    # TODO: Derive class
    wtp_oracle: WTPPredictor
    slot_centroids: DF
    average_basket_value: float

    def __init__(
        self,
        hyperparameters: Dict[str, Any],
        symb_expr: str,
        # TODO: Derive class
        slot_centroids: DF,
        average_basket_value: float,
        wtp_oracle: WTPPredictor,
    ):
        self.hyperparameters = hyperparameters
        self.symb_expr = symb_expr

        min_price = int(np.ceil(hyperparameters["min_price"]))
        max_price = int(np.floor(hyperparameters["max_price"]))
        delta = hyperparameters["delta"]

        self.price_points = np.arange(min_price, max_price, delta)

        # TODO: Derive class
        self.slot_centroids = slot_centroids
        self.average_basket_value = average_basket_value
        self.wtp_oracle = wtp_oracle

    def _compute_action_space_speedup(self, o: Observation) -> Series[Action]:
        # @nb.jit(nopython=True)
        def _generate_full_action_space(
            price_points: np.ndarray, nr_slots: int
        ) -> np.ndarray:

            panels = np.empty(shape=(0, nr_slots))
            for panel in itertools.product(price_points, repeat=nr_slots):
                panels = np.vstack([panels, np.array(panel)])

            return panels

        # @nb.jit(nopython=True)
        def _generate_sample_action_space(
            price_points: np.ndarray, nr_slots: int, sample_size: int
        ) -> np.ndarray:

            panels = np.empty(shape=(0, nr_slots))
            for _ in np.arange(sample_size):
                panel = np.random.choice(a=price_points, size=nr_slots)
                panels = np.vstack([panels, np.array(panel)])

            return panels

        slot_ids = o.slots.index
        full_space_size = len(self.price_points) ** len(slot_ids)  # P x S

        if full_space_size <= self.hyperparameters["max_solution_space"]:
            panels = _generate_full_action_space(
                price_points=self.price_points, nr_slots=len(slot_ids)
            )
        else:
            panels = _generate_sample_action_space(
                price_points=self.price_points,
                nr_slots=len(slot_ids),
                sample_size=self.hyperparameters["max_solution_space"],
            )

        panel_series = {"panel": list(panels)}

        return (
            pd.DataFrame(panel_series, columns=panel_series.keys())
            .assign(
                _panel=lambda x: x.to_dict(orient="records"),
                panel_df=lambda x: x._panel.apply(
                    lambda y: pd.DataFrame(y, index=slot_ids)
                ),
            )
            .assign(
                price_level=lambda x: x.panel.apply(np.mean),
                assymetry_level=lambda x: x.panel.apply(skew),
                nr_options=lambda x: len(slot_ids),
            )
            .fillna(0.0)
            .drop(columns="panel _panel".split())
            .rename(columns={"panel_df": "panel"})
            .apply(
                lambda x: Action(
                    panel=x.panel,
                    price_level=x.price_level,
                    assymetry_level=x.assymetry_level,
                    nr_options=x.nr_options,
                ),
                axis=1,
            )
        )

    def _full_action_space(self, slot_index: Any) -> List[Action]:
        return [
            Action(
                panel=(
                    prices := pd.DataFrame(
                        np.array(panel),
                        index=slot_index,
                        columns="slotcost".split(),
                    )
                ),
                price_level=prices.slotcost.mean(),
                assymetry_level=prices.slotcost.skew(),
                nr_options=prices.slotcost.count(),
            )
            for panel in itertools.product(self.price_points, repeat=len(slot_index))
        ]

    def _sample_action_space(self, slot_index: Any) -> List[Action]:
        return [
            Action(
                panel=(
                    prices := pd.DataFrame(
                        np.random.choice(a=self.price_points, size=len(slot_index)),
                        index=slot_index,
                        columns="slotcost".split(),
                    )
                ),
                price_level=prices.slotcost.mean(),
                assymetry_level=prices.slotcost.skew(),
                nr_options=prices.slotcost.count(),
            )
            for _ in np.arange(self.hyperparameters["max_solution_space"])
        ]

    def _compute_action_space(self, o: Observation) -> Series[Action]:
        slot_ids = o.slots.index
        full_space_size = len(self.price_points) ** len(slot_ids)  # P x S

        return pd.Series(
            self._full_action_space(slot_ids)
            if full_space_size <= self.hyperparameters["max_solution_space"]
            else self._sample_action_space(slot_ids)
        )

    def _get_state(self, o: Observation) -> State:
        if o.order is None:
            raise ValueError("Order is None!")

        if o.selected_slot is not None:
            self._update_slot_centroids(o)

        distance_to_centroids = self._compute_distance_to_centroids(o)

        # TODO: add basket value as an order attribute
        basket_value_lift = o.order.order_df.total.iloc[0] / self.average_basket_value

        return State(
            order=o.order,
            customer=o.customer,
            slots=o.slots,
            basket_value=o.order.basket_value,
            basket_value_lift=basket_value_lift,
            distance_to_centroids=distance_to_centroids,
        )

    # TODO: Derive class
    def _update_slot_centroids(self, o: Observation) -> None:
        selected_slot = o.selected_slot
        customer_lat, customer_long = o.order.coordinates

        self.slot_centroids[
            ["centroid_lat", "centroid_long", "nr_nodes"]
        ] = self.slot_centroids.assign(
            new_centroid_lat=lambda x: (
                (x.index == selected_slot)
                * ((x.centroid_lat * x.nr_nodes + customer_lat) / (x.nr_nodes + 1))
                + (1 - (x.index == selected_slot)) * x.centroid_lat
            ),
            new_centroid_long=lambda x: (
                (x.index == selected_slot)
                * ((x.centroid_long * x.nr_nodes + customer_long) / (x.nr_nodes + 1))
                + (1 - (x.index == selected_slot)) * x.centroid_long
            ),
            new_nr_nodes=lambda x: ((x.index == selected_slot) * 1 + x.nr_nodes),
        )[
            ["new_centroid_lat", "new_centroid_long", "new_nr_nodes"]
        ]

    def _compute_distance_to_centroids(self, o: Observation) -> pd.Series:
        # get incoming customers' coordinates
        coordinates = o.order.coordinates

        return self.slot_centroids.assign(
            customer_lat=coordinates[0], customer_long=coordinates[1]
        )["centroid_long centroid_lat customer_long customer_lat".split()].apply(
            lambda x: haversine_nb(
                radians(x.centroid_long),
                radians(x.centroid_lat),
                radians(x.customer_long),
                radians(x.customer_lat),
            ),
            axis=1,
        )

    def _determine_action_old(
        self, state: State, action_space: Series[Action]
    ) -> Action:
        def _score_based_on_state_action_features(s: State, a: Action) -> float:
            features = {**s.features, "nan": np.NaN, "aq": aq}
            return eval(self.symb_expr, features)

        # Choose the action with the maximal score for the given state
        best_action, _ = max(
            (
                (action, _score_based_on_state_action_features(state, action))
                for action in action_space
            ),
            key=lambda x: x[1],
        )
        return best_action

    def _get_state_action_pairs(self, state, action_space) -> DF:
        state_features = pd.DataFrame.from_dict(
            # For from_dict to work, we need to wrap the dict values in a list
            {k: [v] for k, v in asdict(state).items()}
        )
        state_features["state"] = state
        return pd.merge(state_features, action_space.rename("action"), how="cross")

    def _calculate_slot_selection_prob_based_on_wtp(self, state_action_features):
        def _get_selection_walkaway_prob(x):
            _, selection_probability, walkaway_probability = get_wtp(
                wtp_predictor=self.wtp_oracle,
                order=x["state"].order,
                panels=np.array([x["action"].panel.values.reshape(-1)]),
                customer=x["state"].customer,
                slots=x["state"].slots,
            )
            return selection_probability[0], walkaway_probability

        state_action_features[
            ["slot_selection_prob", "slot_walkaway_prob"]
        ] = state_action_features.apply(_get_selection_walkaway_prob, axis=1).apply(
            pd.Series
        )

    def _compute_state_action_pairs(
        self, state: State, action_space: Series[Action]
    ) -> DF:
        def _get_selection_walkaway_prob(
            state_action_features: DF, wtp_predictor: WTPPredictor
        ) -> DF:
            state = state_action_features.iloc[0]["state"]
            _, selection_probability, walkaway_probability = get_wtp(
                wtp_predictor=wtp_predictor,
                order=state.order,
                panels=state_action_features.apply(
                    lambda x: x.action.panel.slotcost, axis=1
                ).values,
                customer=state.customer,
                slots=state.slots,
            )

            def _2darray_into_column_of_arrays(arr: np.ndarray) -> pd.Series:
                # TODO: Is there a more pythonic way of doing this?
                assert len(arr.shape) == 2
                cols = [f"__{i}" for i in range(arr.shape[1])]
                df = pd.DataFrame(arr, columns=cols)
                return df.apply(lambda x: np.array([x[cols]]).reshape(-1), axis=1)

            state_action_features[
                "slot_selection_prob"
            ] = _2darray_into_column_of_arrays(selection_probability)
            state_action_features["slot_walkaway_prob"] = walkaway_probability
            return state_action_features

        def _calc_top_slot_indices(x):
            indices = [(i, p) for i, p in enumerate(x["slot_selection_prob"])]
            indices.sort(key=lambda x: x[1], reverse=True)
            return [i for i, _ in indices]

        def _get_feature_of_top_i_slot(x, feature, i):
            # Sometimes, we might only have two slots available, for example.
            # In that case, we can't simply populate the top 3rd slot features.
            # So, we guard against that edge case.
            _safe_index = min(i, len(x["top_slot_indices"]) - 1)
            top_i_slot_index = x["top_slot_indices"][_safe_index]

            col = None
            if feature == "distance_to_centroids":
                col = x[feature]
            if feature == "slotcost":
                col = x["action"].panel

            if col is None:
                raise ValueError()

            return col.iloc[top_i_slot_index]

        # logger.info("_compute_state_action_pairs")
        state_action_features = self._get_state_action_pairs(state, action_space)

        state_action_features = _get_selection_walkaway_prob(
            state_action_features, self.wtp_oracle
        )

        state_action_features["top_slot_indices"] = state_action_features.apply(
            _calc_top_slot_indices, axis=1
        )

        # TODO: Could be optimized, probably not worth it, though?
        NUM_TOP_SLOTS = 3
        for feature in ["distance_to_centroids", "slotcost"]:
            for i in range(NUM_TOP_SLOTS):
                state_action_features[
                    f"{feature}_of_top_{i+1}_slot"
                ] = state_action_features.apply(
                    lambda x: _get_feature_of_top_i_slot(x, feature, i), axis=1
                )

        return state_action_features

    def _determine_action(self, state_action_pairs: DF) -> Action:
        def _score_based_on_features(x) -> float:
            features = {**x.to_dict(), "nan": np.NaN, "aq": aq}
            return eval(self.symb_expr, features)

        state_action_pairs["score"] = state_action_pairs.apply(
            _score_based_on_features, axis=1
        )
        # TODO: No need to sort, get the top one only
        state_action_pairs.sort_values("score", ascending=False, inplace=True)
        return state_action_pairs["action"].iloc[0]

    def decide_on_observation(self, o: Observation) -> Action:
        # TODO: Get state should incorporate the action space
        state = self._get_state(o)
        action_space = self._compute_action_space(o)
        state_action_pairs = self._compute_state_action_pairs(state, action_space)

        return self._determine_action(state_action_pairs)


class RLSimulation:
    agent: Agent
    environment: Environment
    _environment_original: Environment
    observations: List[Observation]

    def __init__(self, agent: Agent, environment: Environment) -> None:
        self.agent = agent
        self._environment_original = environment

        self.reset()

        # FIXME: Maybe this needs to be a deepcopy?
        # The way it is now, it's probably just a pointer
        self._environment_original = environment

    def run(self) -> None:
        action = None
        while True:
            o: Observation = self.environment.step(action)
            self.observations.append(o)
            if o.order is None:
                break

            print(self.environment._stage_id)
            action = self.agent.decide_on_observation(o)

    def profit(self) -> float:
        return sum(o.reward for o in self.observations)

    def update_agent(self, agent: Agent) -> None:
        self.agent = agent

    def reset(self) -> None:
        self.observations = []
        self.environment = self._environment_original
