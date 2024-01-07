# TODO: Rethink the use of evals
import math
import json
import numpy as np
import pandas as pd
import scipy.stats as ss
from typing import Tuple
from functools import partial
from pydantic import BaseModel


def _get_slotcost_vars(panels):
    funcs = {
        "min": partial(np.min, axis=1),
        "q1": partial(np.percentile, q=25, axis=1),
        "median": partial(np.median, axis=1),
        "q3": partial(np.percentile, q=75, axis=1),
        "max": partial(np.max, axis=1),
        "avg": partial(np.mean, axis=1),
        "std": partial(np.std, axis=1),
    }

    applied_funcs = [v(panels) for v in funcs.values()]
    applied_funcs2 = {k: v(panels) for k, v in funcs.items()}

    # perform iqr
    applied_funcs += [applied_funcs[3] - applied_funcs[1]]
    applied_funcs2["iqr"] = applied_funcs2["q3"] - applied_funcs2["q1"]

    # perform cv
    applied_funcs += [applied_funcs[6] / applied_funcs[5]]
    applied_funcs2["cv"] = applied_funcs2["std"] / applied_funcs2["avg"]

    applied_funcs2 = list(funcs.values())

    # TODO: Test if they're the same

    # mutate the data
    mutated_panels = np.tile(
        np.vstack(applied_funcs).T, (panels.shape[-1], 1, 1)
    ).transpose(1, 0, 2)

    # add the rank and slot cost
    return np.dstack(
        [mutated_panels, panels, ss.rankdata(panels, axis=1) / panels.shape[-1]]
    )


def _build_slot_vars_df(panels, slots):
    slot_vars_df = pd.DataFrame(
        _get_slotcost_vars(panels).reshape(-1, 11),
        index=pd.MultiIndex.from_product(
            [np.arange(panels.shape[0]), slots.index.values],
            names="panel slot_id".split(),
        ),
        columns="min q1 median q3 max  avg std iqr cv slotcost rank".split(),
    )

    # Change df index
    slot_vars_df[slots.index.names] = pd.DataFrame(
        slot_vars_df.index.get_level_values("slot_id").tolist(),
        index=slot_vars_df.index,
        columns=slots.index.names,
    )

    return (
        slot_vars_df.reset_index()
        .drop(columns="slot_id")
        .set_index(list(np.append(np.array("panel"), (np.array(slots.index.names)))))
    )


def aq(a, b):
    """
    Analytic quotient, used in evaluating GP expressions
    """
    return a / (1 + b**2) ** 0.5


def plog(a):
    """
    Protected logarithm, used in evaluating GP expressions
    """
    result = np.log(np.abs(a))
    nf = np.isfinite(result)
    return np.where(nf == False, 0, result)


def sigmoid(value: float):
    UPPER_BOUND = 10
    LOWER_BOUND = -10
    value = min(max(value, LOWER_BOUND), UPPER_BOUND)
    return 1 / (1 + np.exp(-value))


class WTPPredictor(BaseModel):
    model_path: str = None
    # model: h2o.model.model_base.ModelBase=None
    df: pd.DataFrame = None
    predictors: np.ndarray = None

    class Config:
        arbitrary_types_allowed = True

    def load(self):
        pass

    def compute_vars(self):
        pass

    def predict(self, desired_shape):
        pass


class WTPRandomPredictor(WTPPredictor):
    def load(self):
        pass

    def predict(self, desired_output_shape: Tuple[int, int]):
        selection_probability = np.random.random(desired_output_shape)
        return selection_probability / selection_probability.sum(axis=1)[:, None]


class WTPGPPredictor(WTPPredictor):
    symb_expr: str = None
    walkaway_expr: str = None

    def load(self):
        """
        Function to populate features model_path, symb_expr and
        predictors based on a json file.
        """

        with open(self.model_path) as f:
            model_info = json.load(f)

        self.symb_expr = model_info["model"]
        self.predictors = model_info["gp_predictors"]
        self.walkaway_expr = model_info["walkaway_expr"]

        conv_dict = {
            f"x{i}": "x." + self.predictors[i] for i in range(len(self.predictors))
        }
        for key in range(len(self.predictors) - 1, -1, -1):
            self.symb_expr = self.symb_expr.replace(f"x{key}", conv_dict[f"x{key}"])

    def predict_walkaway(self, selection_probability: np.ndarray) -> np.ndarray:
        return eval(
            self.walkaway_expr,
            globals(),
            {"selection_probability": selection_probability.max(axis=1)},
        )

    def predict(
        self,
        desired_output_shape: Tuple[int, int],
        panels: np.ndarray,
        order_df: pd.DataFrame,
        customer_df: pd.DataFrame,
        slots: pd.Series,
    ) -> np.ndarray:
        """
        Applies the WTP model, returning an array with the
        selection probabilities for each panel-timeslot pair
        """

        # GP needs both customer and order data
        df = order_df.set_index(customer_df.index.names, append=True).join(
            customer_df[
                np.array(set(customer_df.columns).difference(set(order_df.columns)))
            ]
        )

        # Multiply information for each panel and calculate vars
        slot_vars_df = (
            _build_slot_vars_df(panels, slots)
            .drop(columns="avg std".split())
            .rename(
                columns={
                    "min": "min_cost",
                    "q1": "q1_cost",
                    "median": "median_cost",
                    "q3": "q3_cost",
                    "max": "max_cost",
                    "iqr": "iqr_cost",
                    "cv": "cv_cost",
                    "rank": "rank_cost",
                }
            )
        )

        # Compile information
        df = slot_vars_df.join(df.drop(columns=slot_vars_df.columns))

        # Predict
        selection_prob = (
            df[self.predictors]
            .astype(float)
            .assign(
                predict=lambda x: eval(
                    self.symb_expr,
                    None,
                    {"x": x, "math.e": math.e, "aq": aq, "exp": np.exp, "plog": plog},
                )
            )["predict"]
            .values
        )
        ## Apply logistic regression transformation
        selection_prob = np.vectorize(sigmoid)(selection_prob)
        selection_prob = selection_prob.reshape(
            -1, desired_output_shape[-1], desired_output_shape[0]
        ).transpose(0, 2, 1)[0]

        walkaway_probability = self.predict_walkaway(selection_prob)

        return (
            selection_prob,
            selection_prob / (selection_prob.sum(axis=1))[:, None],
            walkaway_probability,
        )
