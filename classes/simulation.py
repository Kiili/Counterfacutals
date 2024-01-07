import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from timeit import default_timer as timer

from .instance import Instance, Order
from .panel_generator import PanelGenerator
from .predictors.cts_predictors import CTSPredictor
from .predictors.wtp_predictors import WTPPredictor
from .funcs import get_wtp, predict_cts, generate_panels


@dataclass
class Simulation:
    instance: Instance
    wtp_predictor: WTPPredictor
    cts_predictor: CTSPredictor
    panel_generator: PanelGenerator
    w_expected_profits: int = 1
    w_target_occupations: int = 2
    exec_time: int = None
    output_df: pd.DataFrame = None

    def run(self):
        # TODO: Change this later
        output_dicts = []

        start_simul = timer()

        for order_id, order in tqdm(
            self.instance.orders.items(), total=self.instance.orders.shape[0]
        ):
            order: Order
            order_id: int

            # loop for each timestamp in the orders
            current_timestamp = order.timestamp
            customer = self.instance.customers.loc[order.customer_id]

            # update slot availability
            self.instance.slots.apply(lambda x: x.check_open(current_timestamp))

            # this array has the slots for a given order
            target_slots = (
                self.instance.slots.loc[order.allowed_slots]
                .loc[lambda x: x.apply(lambda y: y.is_open) == 1]
                .sort_index()
            )
            if target_slots.shape[0] == 0:
                print("no slots available")
                continue
            n_slots = target_slots.shape[0]

            # filter target slots in order_df
            order.filter_slots(slots=target_slots)

            panels = generate_panels(
                self.panel_generator, self.wtp_predictor, order, customer, target_slots
            )

            _, selection_probability, walkaway_probability = get_wtp(
                self.wtp_predictor, order, panels, customer, target_slots
            )

            cts = predict_cts(
                order,
                self.cts_predictor,
                available_slots=target_slots,
                new_order_id=order_id,
                order_cp7s=self.instance.order_cp7s,
            )

            # this gives an array of immediate rewards
            immediate_reward = order.get_expected_profit(
                panels=panels,
                basket_value=order.order_df.total.iloc[0],
                wtp_estimates=selection_probability,
                cts_estimate=cts,
                walkaway_estimates=walkaway_probability,
            )

            # Alternative way to compute occupation incentives
            current_occupation = np.tile(
                target_slots.apply(lambda x: x.occupation).values,
                self.panel_generator.n_panels,
            ).reshape(self.panel_generator.n_panels, -1)

            expected_occupation = (
                current_occupation + selection_probability
            )  # need to correct for walkaway probability

            target_occupation = np.tile(
                target_slots.apply(
                    lambda x: x.get_target(current_timestamp) * x.capacity
                ).values,
                self.panel_generator.n_panels,
            ).reshape(self.panel_generator.n_panels, -1)
            likely_occupation_incentives = -np.abs(
                (expected_occupation - target_occupation) / target_occupation
            ).mean(axis=1)

            reward = (
                self.w_target_occupations * likely_occupation_incentives
                + self.w_expected_profits * immediate_reward
            )

            panel_to_select = np.random.choice(np.argsort(reward)[-10:], 1)[0]
            order.walkaway_probability = walkaway_probability[panel_to_select]
            if np.random.rand() > walkaway_probability[panel_to_select]:
                selected_slot = np.random.choice(
                    np.arange(n_slots), p=selection_probability[panel_to_select]
                )
                order.selected_slot = selected_slot
                order.walkaway_probability = walkaway_probability[panel_to_select]

                selected_slot_id = target_slots.index[selected_slot]

                self.instance.slots.loc[selected_slot_id].increment_occupation(order_id)
                delivery_fee = panels[panel_to_select, selected_slot]
                cts = cts[selected_slot]
            else:
                selected_slot_id = None
                order.walkaway = 1
                delivery_fee = 0
                cts = 0
                # print("Customer walked away")

            self.instance.slots.apply(
                lambda x: x.register_occupation(current_timestamp)
            )

            # Save results
            metrics = {
                "time_instant": current_timestamp,
                "customer_id": order.customer_id,
                "target_slots": target_slots.index,
                "price_panel": panels[panel_to_select],
                "selected_slot": selected_slot_id,
                "basket_value": order.order_df.total.iloc[0],
                "delivery_fee": delivery_fee,
                "estimated_cost": cts,
                "profit_objective": immediate_reward,
                "occupation_incentive": likely_occupation_incentives,
            }

            output_dicts.append(metrics)

        self.output_df = pd.DataFrame(output_dicts)
        self.instance.slots.apply(lambda x: x.parse_occupation_to_df())
        self.exec_time = timer() - start_simul
