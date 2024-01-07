import datetime
import itertools
import numpy as np
import pandas as pd
import pickle as pkl

from classes import (
    Agent,
    Action,
    Customer,
    Environment,
    Instance,
    InstanceData,
    Order,
    #RLSimulation,
    Slot,
    WTPPredictor,
    WTPGPPredictor,
)
from classes.funcs import get_wtp
from typing import Any, Dict, List, Optional, Tuple, Union
from pandera.typing import Series
from functools import partial
from decimal import Decimal

from dice_cf_demo import initialize_dice, find_cf

DF = pd.DataFrame

def load_wtp_predictor()->WTPPredictor:
    wtp_predictor = WTPGPPredictor(
            model_path="models/gp/runFinalTests_Results_trust-ai-wtp-gp-7ls79_iter1.json"
    )
    wtp_predictor.load()
    return wtp_predictor

def get_representative_instance_data(
    instance_data: InstanceData,
    num_booking_days: int = 3,
    num_orders: int = 20,
    num_slots: int = 5,
    area_id: Union[int, None] = 140,
) -> InstanceData:
    """Downsizes the input instance to consider a representative instance with `num_booking_days`
    each with `num_orders` orders incoming per day that can book any of the same `num_slots`"""

    # Read each DataFrame in Instance
    order_df = instance_data.orders
    customer_df = instance_data.customers
    slot_df = instance_data.slots

    # Filter area_id
    if area_id is not None:
        order_df = order_df.loc[:, [area_id], :, :]

    # Filter num_booking_days
    if num_booking_days < len(order_df.orderdate.unique()):
        start_timestamp = order_df.orderinstant.min()
        start_timestamp = pd.DatetimeIndex([start_timestamp]).normalize()[
            0
        ]  # set time to 00:00:00
        end_timestamp = start_timestamp + datetime.timedelta(days=num_booking_days)

        order_df = order_df[
            lambda x: (x.orderinstant >= start_timestamp)
            & (x.orderinstant < end_timestamp)
        ]

    # Sample orders
    orders = order_df.reset_index()[
        "shippingnumber orderdate".split()
    ].drop_duplicates()

    def sample_orders(group:Tuple, num_orders: int)->np.ndarray:
        if len(group)>=num_orders:
            return (sample := group.sample(n=num_orders)['shippingnumber'].unique())
        
        return group['shippingnumber'].values

    order_list = (orders
        .groupby('orderdate')
        .apply(sample_orders, num_orders)
        .explode()
        .reset_index(drop=True)
    )

    # Filter DataFrames
    order_df = order_df.loc[order_list, :, :, :]
    customer_df = customer_df.loc[order_df.uqicustomerid.unique(), :]
    slot_list = (
        slot_df[
            lambda x: (x.is_open) & (x.index.get_level_values("areaid").isin([area_id]))
        ]
        .groupby("slot_time")
        .sum("occupation")
        .occupation.sort_values(ascending=False)
        .iloc[:num_slots]
        .index.values
    )
    slot_df = slot_df.loc[area_id, :, slot_list].sort_index()

    return InstanceData(order_df, customer_df, slot_df, instance_data.order_cp7s)


def update_instance_data_capacity(
    instance: InstanceData, capacity_type: str = "max"
) -> InstanceData:
    """
    Adjusts the capacity of the slots based on the
    slots offered and several orders coming at specific instants.

    - Rationale: Equally divide the no. of customers that see the slot by
    the minimum/mean/maximum no. of slots competing with the slot under analysis
    """

    # read order and slot information
    order_df = instance.orders
    slot_df = instance.slots

    # join relevant information to see how many customers see each slot
    df = order_df["orderinstant".split()].join(
        slot_df["capacity occupation opening_time cutoff".split()]
    )

    # assuming infinite capacity, determine which slots the customer can see
    df = df.assign(
        is_open=lambda x: np.where(
            (x.orderinstant <= x.cutoff) & (x.orderinstant >= x.opening_time), 1, 0
        )
    )[lambda x: x.is_open == 1].drop(columns="is_open")

    # compute the number of customers passing by each slot
    df_customers = df.groupby(df.reset_index("shippingnumber").index.names).agg(
        nr_passing_customers=("capacity", "count")
    )

    # compute the number of alternative slots each customer would see
    ## computing this value means we know how many competing slots there were at each point in time
    # (each point being the arrival of a customer)
    df_slots = df.groupby("shippingnumber").agg(
        nr_available_slots=("capacity", "count")
    )

    # statistical measures on the distribution of competing slots for each individual slot over time
    df_dist_alt_slots = (
        df.join(df_slots)
        .reset_index("shippingnumber", drop=True)
        .set_index("orderinstant", append=True)["nr_available_slots".split()]
        .groupby(slot_df.index.names)
        .agg(
            min=("nr_available_slots", "min"),
            mean=("nr_available_slots", "mean"),
            max=("nr_available_slots", "max"),
            stdev=("nr_available_slots", "std"),
        )
    )

    # compute the possible capacity measures
    df_dist_alt_slots = df_dist_alt_slots.join(df_customers).assign(
        max_capacity=lambda x: x.nr_passing_customers / x["min"],
        mean_capacity=lambda x: x.nr_passing_customers / x["mean"],
        min_capacity=lambda x: x.nr_passing_customers / x["max"],
    )
    df_dist_alt_slots = df_dist_alt_slots[f"{capacity_type}_capacity".split()].assign(
        capacity=lambda x: np.ceil(x[f"{capacity_type}_capacity"]).astype(int)
    )

    slot_df = slot_df.drop(columns="capacity").join(df_dist_alt_slots[["capacity"]])
    slot_df["capacity"] = slot_df["capacity"].fillna(
        0
    )  # for those slots that won't open in the simulation
    slot_df["capacity"] += slot_df["occupation"]

    return InstanceData(
        instance.orders,
        instance.customers,
        slot_df.copy(),
        instance.order_cp7s,
    )


def generate_realistic_instance(
    instance_data: InstanceData, abt_shape: str = "l"
) -> Instance:
    """
    Creates an instance based on sampled data.
    """

    slot_df = prepare_slot_df(instance_data.slots, instance_data.orders)

    _slot_time_limits = get_slot_time_limits(slot_df)
    abt_curves = get_abt_curves(abt_shape, slot_df, _slot_time_limits)
    slots = get_slot_series(slot_df, abt_curves)

    customers = get_customer_series(instance_data.customers)
    orders = get_order_series(instance_data.orders, slots)

    return Instance(orders, customers, slots, instance_data.order_cp7s)


def get_order_series(order_df: DF, slots: Series[Slot]) -> Series[Order]:
    order_dfs = {
        shippingnumber: table.droplevel("shippingnumber")
        for shippingnumber, table in order_df.groupby("shippingnumber")
    }

    return (
        order_df.reset_index()[
            "shippingnumber uqicustomerid orderinstant areaid total customer_lat customer_long".split()
        ]
        .drop_duplicates()
        .rename(
            columns={
                "shippingnumber": "order_id",
                "uqicustomerid": "customer_id",
                "orderinstant": "timestamp",
                "total": "basket_value",
            }
        )
        .assign(
            order_df=lambda x: (x.order_id.T).map(order_dfs),
            # An order can only see slots reserved for its area
            allowed_slots=lambda x: x.apply(
                lambda y: slots.loc[pd.Series(y.areaid).astype(int), :, :].index.values,
                axis=1,
            ).T,
            coordinates=lambda x: tuple(zip(x.customer_lat, x.customer_long)),
        )
        .drop(columns=["areaid", "customer_lat", "customer_long"])
        .assign(
            # wtp_predictor=lambda x: np.repeat(wtp_pred, x.shape[0]),
            # cts_predictor=lambda x: np.repeat(cts_pred, x.shape[0]),
            # panel_generator=lambda x: np.repeat(panel_generator, x.shape[0]),
            _order=lambda x: x.to_dict(orient="records"),
            order=lambda x: x._order.apply(lambda y: Order(**y)),
        )
        .set_index("order_id")
        .order
    )


def get_target(cutoff, current_time, xs, ys, granularity="h"):
    return np.around(
        -np.interp((cutoff - current_time) / np.timedelta64(1, granularity), xs, ys),
        decimals=3,
    )


def get_customer_series(customer_df: DF) -> Series[Customer]:
    customer_dfs = {
        uqicustomerid: table.droplevel("uqicustomerid")
        for uqicustomerid, table in customer_df.groupby("uqicustomerid")
    }

    return (
        pd.Series(customer_df.index.get_level_values("uqicustomerid").unique())
        .to_frame()
        .assign(
            customer_df=lambda x: (x.uqicustomerid.T).map(customer_dfs),
            _customer=lambda x: x.to_dict(orient="records"),
            customer=lambda x: x._customer.apply(lambda y: Customer(**y)),
        )
        .set_index("uqicustomerid")
        .customer
    )


def get_slot_series(slot_df: DF, abt_curves: DF) -> Series[Slot]:
    slots = slot_df.join(abt_curves)

    # abt curve interpolation function
    return slots.assign(
        abt=lambda x: x.apply(
            lambda y: partial(
                get_target,
                cutoff=y.cutoff,
                xs=y.abt_curve["time_to_cutoff"],
                ys=y.abt_curve["target_capacity"],
                granularity="m",
            ),
            axis=1,
        ),
        _slot=lambda x: x.to_dict(orient="records"),
        slot=lambda x: x._slot.apply(lambda y: Slot(**y)),
    )["slot"]


def get_abt_curves(abt_shape: str, slot_df: DF, slots: Series[Slot]) -> Series:
    # abt curves - set of time instants
    nr_points = 40
    abt_curves = {
        slot: np.linspace(
            slots.loc[slot].last_time_to_cutoff,
            slots.loc[slot].first_time_to_cutoff,
            nr_points,
            dtype=int,
        )
        for slot in slots.index
    }

    # add points to plot the curve
    abt_curves = (
        pd.DataFrame.from_dict(abt_curves, orient="index")
        .stack()
        .rename_axis("slot obs".split(), axis=0)
        .rename("time_to_cutoff")
        .sort_index()
    )
    abt_curves = pd.DataFrame(abt_curves).join(
        pd.DataFrame(
            slot_df["capacity occupation".split()].values,
            index=pd.Index(slot_df.index.values, name="slot"),
            columns="capacity occupation".split(),
        )
    )
    abt_curves = abt_curves.join(
        pd.DataFrame(
            slots["first_time_to_cutoff last_time_to_cutoff".split()].values,
            index=pd.Index(slot_df.index.values, name="slot"),
            columns="first_time_to_cutoff last_time_to_cutoff".split(),
        )
    )

    if abt_shape == "l":
        # Linear - l
        # The abt follows a linear function: f(x) = ax + b
        # The desired occupation when x = tf (last time open in the simulation) is 1, i.e., f(tf) = 1.
        abt_curves = abt_curves.assign(
            a_factor=0,
            b_factor=lambda x: ((x.occupation / x.capacity) - 1)
            / (x.first_time_to_cutoff - x.last_time_to_cutoff),
            c_factor=1,
        )
    elif abt_shape == "qd":
        # Quadratic facing downwards - qd
        # The abt curve follows a quadratic behavior: f(x) = ax2 + bx + c
        # Let's assume f(0) = 1, local optimum of the function occurs at the cutoff f'(0) = 0
        abt_curves = abt_curves.assign(
            a_factor=lambda x: (
                (x.occupation / x.capacity - 1)
                / ((x.first_time_to_cutoff - x.last_time_to_cutoff) ** 2)
            ),
            b_factor=0,
            c_factor=1,
        )
    elif abt_shape == "qu":
        # Quadratic facing upwards - qu
        # The abt curve follows a quadratic behavior: f(x) = ax2 + bx + c
        # Let's assume f(0) = 1 and the local optimum of the function occurs at the initial instant f'(ti) = Oi/C
        abt_curves = abt_curves.assign(
            # f'(ti) = Oi/C
            a_factor=lambda x: (x.occupation / x.capacity - 1)
            / (-((x.first_time_to_cutoff - x.last_time_to_cutoff) ** 2)),
            # f'(ti) = Oi/C
            b_factor=lambda x: (2 / (x.first_time_to_cutoff - x.last_time_to_cutoff))
            * (x.occupation / x.capacity - 1),
            # from f(0) = 1
            c_factor=1,
        )

    abt_curves["target_capacity"] = (
        abt_curves.reset_index("obs")
        .assign(
            target_capacity=lambda x: (
                x.a_factor * (x.time_to_cutoff.astype(int) - x.last_time_to_cutoff) ** 2
            )
            + (x.b_factor * (x.time_to_cutoff.astype(int) - x.last_time_to_cutoff))
            + x.c_factor
        )["target_capacity"]
        .values
    )

    abt_curves = (
        abt_curves["time_to_cutoff target_capacity".split()]
        .reset_index()
        .set_index("slot time_to_cutoff".split())
        .drop(columns="obs")
    )

    abt_curves = {
        slot: table.droplevel("slot")
        .reset_index()["time_to_cutoff target_capacity".split()]
        .to_dict(orient="list")
        for slot, table in abt_curves.groupby("slot")
    }
    abt_curves = {
        slot: {col_name: np.array(col_values) for col_name, col_values in col.items()}
        for slot, col in abt_curves.items()
    }

    # data to instantiate slot class
    abt_curves = (
        pd.Series(abt_curves).to_frame().sort_index().rename(columns={0: "abt_curve"})
    )
    abt_curves.index.set_names(list(slot_df.index.names), inplace=True)
    return abt_curves


def get_slot_time_limits(slot_df: DF) -> DF:
    # for slots already open, we will consider the initial instant to be the simulation start
    # for slots that remain open past the cutoff, we will consider the last instant to be the simulation end
    return (
        slot_df["opening_time simulation_start cutoff simulation_end".split()]
        .assign(
            first_time_to_cutoff=lambda x: (
                (x.cutoff - np.maximum(x.opening_time, x.simulation_start))
                / np.timedelta64(1, "m")
            ).astype(int),
            last_time_to_cutoff=lambda x: (
                (x.cutoff - np.minimum(x.cutoff, x.simulation_end))
                / np.timedelta64(1, "m")
            ).astype(int),
        )
        .drop(columns="simulation_start simulation_end".split())
    )


def prepare_slot_df(slot_df: DF, order_df: DF) -> DF:
    slot_df = slot_df.drop(columns="alt")

    # determine the start and end of the simulation
    slot_df["simulation_start"] = order_df.orderinstant.min()
    slot_df["simulation_end"] = order_df.orderinstant.max()
    return slot_df


def get_instance():
    with open('instances/instance_midyear.pkl', 'rb') as f:
        _inst_data = pkl.load(f)

    instance_data = InstanceData(
        _inst_data["order_df"],
        _inst_data["customer_df"],
        _inst_data["slot_df"],
        _inst_data["order_cp7s"],
    )

    # Get a toy instance
    instance_data = get_representative_instance_data(instance_data)

    # To adjust the slot capacities according to instance features
    instance_data = update_instance_data_capacity(instance_data, capacity_type="max")

    # Generate instance
    return generate_realistic_instance(instance_data, abt_shape="l")


def initialize_time_slot_centroids(instance: Instance) -> DF:
    # Initialize time slot centroids beforehand
    slot_centroids = instance.slots.apply(
        lambda x: instance.order_cp7s.loc[x.assigned_orders][
            ["customer_lat", "customer_long"]
        ]
        if x.assigned_orders
        else instance.order_cp7s.iloc[[0]][["store_lat", "store_long"]].rename(
            columns={"store_lat": "customer_lat", "store_long": "customer_long"}
        )
    )
    nr_nodes = instance.slots.apply(
        lambda x: instance.order_cp7s.loc[x.assigned_orders]
    ).apply(lambda x: x.count())[["store_lat"]]
    nr_nodes.columns = ["nr_nodes"]
    slot_centroids = slot_centroids.apply(lambda x: x.mean()).join(nr_nodes)
    slot_centroids.columns = list(
        map(lambda s: s.replace("customer", "centroid"), slot_centroids.columns.values)
    )
    slot_centroids["nr_nodes"] = slot_centroids["nr_nodes"].astype(int)

    return slot_centroids


# Load instance
instance = get_instance()

# Load willigness-to-pay model
wtp_predictor = load_wtp_predictor()


"""
    INPUTS: DEFINE CUSTOMER TO ANALYZE
"""
# Select an order to analyze
order_id = np.random.choice(instance.orders.index) # a random order
order: Order = instance.orders.loc[order_id]

# Get corresponding customer
customer: Customer = instance.customers.loc[order.customer_id]

# Get the slots that the customer is seeing
current_timestamp = order.timestamp
instance.slots.apply(lambda x: x.check_open(current_timestamp))
slots = instance.slots.loc[order.allowed_slots].loc[
    lambda x: x.apply(lambda y: y.is_open) == 1
]

# Filter order_df for available time slots
order.order_df = order.order_df.loc[slots.index]


"""
    DECISION VARIABLES
"""
# Customer features / slot features

## This unveils the features that impact the wtp
print(wtp_predictor.symb_expr)

# If you notice, only exact_selection_customer_perc and slot_start impact the probability
# the remaining features are a consequence of the price panel presented

"""
    PANELS
"""
# Solution space hyperparameters
MIN_PRICE = 2
MAX_PRICE = 9
DELTA = 1
MAX_SOLUTION_SPACE = 1000

price_points = np.arange(MIN_PRICE, MAX_PRICE, DELTA)

slot_ids = slots.index
full_space_size = len(price_points) ** len(slot_ids)  # P x S

def full_action_space(price_points: np.ndarray, slot_index: Any) -> List[Action]:
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
            for panel in itertools.product(price_points, repeat=len(slot_index))
        ]

def sample_action_space(price_points: np.ndarray, slot_index: Any, max_solution_space:int) -> List[Action]:
    return [
        Action(
            panel=(
                prices := pd.DataFrame(
                    np.random.choice(a=price_points, size=len(slot_index)),
                    index=slot_index,
                    columns="slotcost".split(),
                )
            ),
            price_level=prices.slotcost.mean(),
            assymetry_level=prices.slotcost.skew(),
            nr_options=prices.slotcost.count(),
        )
        for _ in np.arange(max_solution_space)
    ]

action_space = pd.Series(
        full_action_space(price_points, slot_ids)
        if (full_space_size <= MAX_SOLUTION_SPACE)
        else sample_action_space(price_points, slot_ids, MAX_SOLUTION_SPACE)
    )

# You can evaluate one panel (at random)
action = action_space.loc[lambda x: np.random.choice(x.index)]
panel = np.array([action.panel.values.reshape(-1)])

# Display customer features
print("customer features ", customer.customer_df.head())
print("list of columns ", customer.customer_df.columns)


_, selection_probability, walkaway_probability = get_wtp(
            wtp_predictor=wtp_predictor,
            order=order,
            panels=panel,
            customer=customer,
            slots=slots,
        )

# Provides the selection_probability of each time slot option
#print("selection probability", selection_probability)
print("the shape of probability distribution ", selection_probability.shape)
print("the sum of selection probabilities ", sum(selection_probability[0]))
print("selected time slot ", slots.index[np.argmax(selection_probability)])
print("probability of max selection probability ", np.max(selection_probability))

# Provides the walkaway probability of each time slot option
print("walkaway proability ", walkaway_probability)


# You can also evaluate several panels at once, you just need to ensure that you
# provide a matrix nr_panels x nr_available_slots
panels = action_space.apply(lambda x: x.panel.slotcost).values
print("how many panels ", panels.shape)

_, selection_probability, walkaway_probability = get_wtp(
            wtp_predictor=wtp_predictor,
            order=order,
            panels=panels,
            customer=customer,
            slots=slots,
        )

# Each row contains the selection probabily distribution of each panel
print("For multiple panels ", selection_probability)
print("probabilities shape ", selection_probability.shape)

# Find selected time slot for every panel
selected_slots = np.argmax(selection_probability, axis=1)
print("selected time slot for each panel ", selected_slots)
# selected slot with datetime
print("selected time slot with datetime ", slots.index[selected_slots])

# Each element corresponds to the walkaway probability of each panel
print(walkaway_probability)


"""
    EXACT_SELECTION_CUSTOMER_PERC AND SLOT_START
"""

# to change this features, you need to change the order_df or customer_df themselves

# slot_start - evaluate the impact of antecipating 1 hour slot 12:00 - 14:30 of day 2017-03-30
order.order_df.loc[(140, datetime.date(2017, 3, 30), '12:00 - 14:30'),'slot_start'] -= 60

# exact_selection_customer_perc - assume customer chose slot  16:00 - 18:30 of the 6 days ahead in 5% more of the cases
customer.customer_df.loc[('6-16:00 - 18:30'), 'exact_selection_customer_perc'] += Decimal(0.05)

# and then you would proceed to see how the selection probabilities changed

