import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Tuple
from pydantic import BaseModel, PrivateAttr
from pandera.typing import Series, DataFrame

DF = pd.DataFrame


class Customer(BaseModel):
    customer_df: pd.DataFrame = None

    class Config:
        arbitrary_types_allowed = True


class Slot(BaseModel):
    capacity: int
    opening_time: pd.Timestamp
    cutoff: pd.Timestamp
    occupation: int = 0
    is_open: bool = False
    abt: Callable  # TODO: Detail this type further
    _occupation_evolution: str = PrivateAttr(default_factory=list)
    occupation_evolution_df: pd.DataFrame = None
    assigned_orders: list = []

    class Config:
        arbitrary_types_allowed = True

    def check_open(self, current_time: np.datetime64) -> bool:
        self.is_open = (self.occupation < self.capacity) and (
            self.opening_time < current_time < self.cutoff
        )
        return self.is_open

    def get_target(self, target_time) -> float:
        return self.abt(current_time=target_time)

    def increment_occupation(self, order_id: str, increment: int = 1):
        self.assigned_orders.append(order_id)
        self.occupation += increment

    def register_occupation(self, timestamp):
        self._occupation_evolution.append([timestamp, self.occupation])

    def parse_occupation_to_df(self):
        self.occupation_evolution_df = (
            pd.DataFrame(self._occupation_evolution)
            .set_axis(["timestamp", "occupation"], axis=1)
            .set_index("timestamp")
            .occupation.loc[self.opening_time : self.cutoff]
        )

    def assess_occupation(self):
        ax = (self.occupation_evolution_df / self.capacity).plot(color="C0")

        indexes_ = list(self.occupation_evolution_df.index.values)
        pd.Series(
            [self.get_target(x) for x in indexes_], indexes_, name="abt_interpolated"
        ).plot(ax=ax, style="--", color="C1")

        indexes_ = list(self.cutoff - self.abt.keywords["xs"].astype("timedelta64[m]"))
        pd.Series(
            [self.get_target(x) for x in indexes_], indexes_, name="abt_curve"
        ).plot(ax=ax, style="o", color="C1")

        return ax


class Order(BaseModel):
    timestamp: datetime
    customer_id: str
    basket_value: float
    order_df: pd.DataFrame = None
    allowed_slots: np.ndarray
    selected_slot: Slot = None
    walkaway: bool = False
    walkaway_probability: np.ndarray = None
    coordinates: Tuple = None

    # This disables some of pydantic's pedantic nature
    # https://stackoverflow.com/questions/69189210/pydantic-validators-py-no-validator-found-for-class-pandas-core-frame-data
    class Config:
        arbitrary_types_allowed = True

    def filter_slots(self, slots: Series[Slot]) -> DataFrame:
        """
        Returns an updated version of order_df based on the target slots
        """
        # TODO: order slots according to the order of target_slots
        # this code assumes that order_df presents all possibilities of available timeslots
        self.order_df = self.order_df[lambda x: x.index.isin(slots.index)].sort_index()
        return self.order_df

    def get_expected_profit(
        self,
        panels: np.ndarray,
        basket_value: float,
        wtp_estimates: np.ndarray,
        cts_estimate: np.ndarray,
        walkaway_estimates: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the immediate reward
        """
        # TODO: atencao ao valor da cesta média aqui
        return (np.multiply(panels + basket_value - cts_estimate, wtp_estimates)).sum(
            axis=1
        ) * (1 - walkaway_estimates)


@dataclass
class InstanceData:
    orders: DF
    customers: DF
    slots: DF
    order_cp7s: DF


@dataclass
class Instance:
    orders: Series[Order]
    customers: Series[Customer]
    slots: Series[Slot]
    order_cp7s: DF

    def get_copy(self):
        return pickle.loads(pickle.dumps(self))
