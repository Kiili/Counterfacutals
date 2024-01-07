import json
import math
import warnings

import dice_ml
import numpy as np
import pandas as pd
import pickle as pkl
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.4f}'.format

from numpy import ndarray

from classes import WTPPredictor
from classes.predictors.wtp_predictors import sigmoid, aq, plog

from typing import Any
from make_panels import Panel, gen_panel


class WTPGPPredictor(WTPPredictor):
    symb_expr: str = None
    walkaway_expr: str = None

    class Config:
        arbitrary_types_allowed = True

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

        return self

    def predict_walkaway(self, selection_probability: np.ndarray) -> np.ndarray:
        return eval(
            self.walkaway_expr,
            globals(),
            {"selection_probability": selection_probability.max(axis=1)},
        )

    def predict(self, x):
        if type(x) is pd.Series:
            p = Panel().load_from_single_row(x)
            _, selection_probs, walkaway_prob = self.predict_df(p.get_df())
            return walkaway_prob[0]

        if type(x) is pd.DataFrame:
            return [self.predict(row) for row in x.iloc]

    def predict_selection_prbs(self, x):
        if type(x) is pd.Series:
            p = Panel().load_from_single_row(x)
            _, selection_probs, walkaway_prob = self.predict_df(p.get_df())
            return selection_probs

        if type(x) is pd.DataFrame:
            return [self.predict_selection_prbs(row) for row in x.iloc]


    def predict_df(self, df: pd.DataFrame) -> tuple[Any, Any, ndarray | ndarray]:
        """
        Applies the WTP model, returning an array with the
        selection and walkaway probabilities for each timeslot
        """
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

        # Apply logistic regression transformation
        selection_prob = np.vectorize(sigmoid)(selection_prob)
        selection_prob = selection_prob.reshape(-1, len(df), 1).transpose(0, 2, 1)[0]

        walkaway_probability = self.predict_walkaway(selection_prob)

        return (
            selection_prob,
            selection_prob / (selection_prob.sum(axis=1))[:, None],
            walkaway_probability,
        )


if __name__ == '__main__':
    wtp_predictor = WTPGPPredictor(
        model_path="models/gp/runFinalTests_Results_trust-ai-wtp-gp-7ls79_iter1.json"
    ).load()  # load genetic predictor

    out_filename = "counterfactuals.txt"  # output file name (None -> Print to console)
    counterfactuals = 1  # amount of counterfactuals generated for each panel
    desired_range = (0, 0.1)  # desired walkaway probability range
    features_to_vary = ([f'slotcost_{i}' for i in range(1, 31)] +
                        ["expanding_avg_days_to_delivery",
                         "days_since_first_purchase"])  # vary all slotcosts & customer features

    panels = []  # gen_panel(n_panels=5)  # Amount of panels generated # TODO fix making less panels

    while len(panels) < 3:
        n = gen_panel(10)[0]
        prob = wtp_predictor.predict(n.get_single_row_df())
        if prob[0] > desired_range[1]:
            panels.append(n)

    data = pd.concat(p.get_single_row_df() for p in panels)
    features = list(data.columns)
    data["walkaway_probability"] = data.apply(lambda x: wtp_predictor.predict(x), axis=1)

    d = dice_ml.Data(dataframe=data,
                     continuous_features=features,
                     outcome_name='walkaway_probability')

    m = dice_ml.Model(model=wtp_predictor,
                      backend='sklearn',
                      model_type='regressor')

    exp_random = dice_ml.Dice(d, m, method="random")  # change method to "genetic" for genetic algorithm

    data = data[data["walkaway_probability"] > desired_range[1]]  # only look at panels with too big walkaway prob

    dice_exp_random = exp_random.generate_counterfactuals(query_instances=data.drop(columns=["walkaway_probability"]),
                                                          total_CFs=counterfactuals,
                                                          desired_range=desired_range,
                                                          features_to_vary=features_to_vary,
                                                          verbose=0,
                                                          # proximity_weight=0.2,  # some hyperparameters for generic algorithm
                                                          # diversity_weight=1.0,  # some hyperparameters for generic algorithm
                                                          )

    for i, counterfacual_df in enumerate(dice_exp_random.cf_examples_list):
        print(type(counterfacual_df))
        if counterfacual_df.final_cfs_df is None:
            continue
        for j, row in enumerate(counterfacual_df.final_cfs_df.iloc):
            row = Panel().load_from_single_row(row).get_single_row_df().iloc[0]

    # output counterfactuals to file
    # f = open(out_filename, "w") if out_filename else None
    # for i, counterfacual_df in enumerate(dice_exp_random.cf_examples_list):
    #     print(f"{i + 1}: Original panel", file=f)
    #     print("\t" + "\n\t".join(data.iloc[i].__repr__().split("\n")[:-1]), file=f)
    #     if counterfacual_df.final_cfs_df is None:
    #         continue
    #
    #
    #
    #     for j, row in enumerate(counterfacual_df.final_cfs_df.iloc):
    #         del row["walkaway_probability"]
    #         row = Panel().load_from_single_row(row).get_single_row_df().iloc[0]  # recalculate median, q1, etc
    #         print(f"\nCounterfactual {j + 1}: new walkaway_probability: {round(wtp_predictor.predict(row), 4)}", file=f)
    #         r = row - data.iloc[i]
    #         del r["walkaway_probability"]
    #         r = r[r != 0]
    #         print("\t" + "\n\t".join(r.__repr__().split("\n")[:-1]), file=f)
    #     print("-" * 50, file=f)
    # if out_filename:
    #     f.close()

































