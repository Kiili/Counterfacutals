import numpy as np
import pandas as pd
from pandera.typing import Series

from .instance import Order, Slot, Customer
from .panel_generator import PanelGenerator
from .predictors import WTPPredictor, CTSPredictor

DF = pd.DataFrame


def predict_cts(
    order: Order,
    cts_predictor: CTSPredictor,
    available_slots: Series[Slot],
    new_order_id: str,
    order_cp7s: pd.DataFrame,
) -> np.ndarray:
    """
    Estimates the delivery cost for each timeslot
    """
    return cts_predictor.predict(
        slots=available_slots,
        new_order_id=new_order_id,
        new_order=order,
        order_cp7s=order_cp7s,
    )


def get_wtp(
    wtp_predictor: WTPPredictor,
    order: Order,
    panels: np.ndarray,
    customer: Customer,
    slots: Series[Slot],
) -> np.ndarray:
    """
    Applies the WTP model
    """
    return wtp_predictor.predict(
        desired_output_shape=panels.shape,
        panels=panels,
        order_df=order.order_df,
        customer_df=customer.customer_df,
        slots=slots,
    )


def generate_panels(
    panel_generator: PanelGenerator,
    wtp_predictor: WTPPredictor,
    order: Order,
    customer: Customer,
    slots: Series[Slot],
) -> np.ndarray:
    """
    Generate panels in random fashion for a given `customer`.
    Returns an array with the prescribed prices for each panel-timeslot pair
    """

    panels = np.array([order.order_df.slotcost.values])
    baseline_wtp = get_wtp(wtp_predictor, order, panels, customer, slots)[0][0]

    return panel_generator.generate(
        wtp_predictor=wtp_predictor,
        baseline_wtp=baseline_wtp,
        order=order,
        customer=customer,
        slots=slots,
    )
