import numpy as np
import pandas as pd
from pydantic import BaseModel
from pandera.typing import Series

from .instance import Slot, Order, Customer


class PanelGenerator(BaseModel):
    n_panels: int = 1

    class Config:
        arbitrary_types_allowed = True

    def generate(self, **_kwargs):
        pass


class TacticalPanelGenerator(PanelGenerator):
    """
    Generates a panel with the retailer's tactical
    price offerings
    """

    def generate(self, order: Order) -> np.ndarray:
        return np.array([order.order_df.slotcost.values])


class RandomGenerator(PanelGenerator):
    """
    Generates panels by sampling random real numbers.
    Uses a uniform distribution centered around `starting_price`,
    with a width of `price_variation`.
    """

    starting_price: float
    price_variation: float

    def generate(self, slots: Series[Slot], **_kwargs) -> np.ndarray:
        n_slots = slots.shape[0]
        return np.around(
            np.repeat(self.starting_price, (self.n_panels * n_slots)).reshape(
                self.n_panels, -1
            )
            + (np.random.random(size=(self.n_panels, n_slots)) - 0.5)
            * self.price_variation,
            decimals=2,
        )


class RandomWTPBasedGenerator(PanelGenerator):
    """
    Generates panels by sampling random numbers from a discrete distribution of real numbers
    that lie within the interval [`min_price`, `max_price`] for the top `top` slots by incrementing
    a base price by `delta` price units.

    delta : float
        the price interval between sampled price points
    """

    top: int
    delta: float
    min_price: float
    max_price: float

    def generate(
        self,
        baseline_wtp,
        order: Order,
        customer: Customer,
        slots: Series[Slot],
        **_kwargs,
    ) -> np.ndarray:
        # Determine the WTP for the default price offering
        panels = np.array([order.order_df.slotcost.values])

        # Select the slots to be explored
        slots_df = (
            pd.DataFrame(slots)
            .reset_index()
            .assign(baseline_wtp=baseline_wtp)
            .sort_values(by="baseline_wtp", ascending=False)
            .reset_index()
            .rename(columns={"index": "slot_id"})
        )
        threshold = min(self.top, slots.shape[0])
        explored_slots = slots_df.iloc[:threshold].slot_id.values

        # Update Slots to be evaluated
        slots_df = slots_df.iloc[:threshold].set_index(slots.index.names)
        slots = slots[lambda x: x.index.isin(slots_df.index)]

        # Build the base for the panel array
        slot_prices = np.repeat(a=panels, repeats=self.n_panels, axis=0)

        # Generate random prices for selected slots based on a price step
        random_prices = delta_based_panels(
            starting_panel=order.order_df.slotcost.values[explored_slots],
            slots=slots,
            n_panels=self.n_panels,
            delta=self.delta,
            min_price=self.min_price,
            max_price=self.max_price,
        )

        # Allocate newly generated prices to the price panels
        slot_prices[:, explored_slots] = random_prices
        return slot_prices


def delta_based_panels(
    starting_panel: np.ndarray,
    slots: Series[Slot],
    n_panels: int,
    delta=0.5,
    min_price=2.0,
    max_price=9.0,
) -> np.ndarray:
    """
    Provides `n_panels` price panels, which are a
    random iteration of the `starting_panel`.
    """

    assert (
        starting_panel.ndim == 1
    ), "argument <starting_panel> should be one-dimensional"

    n_slots = slots.shape[0]

    # Determine step boundaries
    # The +1 compensates for the fact that randint's upper limit is exclusive
    min_step = -np.floor((starting_panel - min_price) / delta)
    max_step = np.floor((max_price - starting_panel) / delta)

    steps = np.array(
        [
            [
                np.random.randint(min_step[slot_id], max_step[slot_id] + 1)
                for slot_id in range(n_slots)
            ]
            for _ in range(n_panels)
        ]
    )

    # returns the array with the prescribed prices for each (panel, timeslot)
    return np.around(
        np.repeat(np.array([starting_panel]), n_panels, axis=0) + delta * steps,
        decimals=2,
    )
