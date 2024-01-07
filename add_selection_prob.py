from predict_walkaway import WTPGPPredictor
import numpy as np
import pickle as pkl
import pandas as pd
import tqdm
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.4f}'.format


wtp_predictor = WTPGPPredictor(
    model_path="models/gp/runFinalTests_Results_trust-ai-wtp-gp-7ls79_iter1.json"
).load()  # load genetic predictor

with open('data_mean_panel_250_counterf.pkl', 'rb') as handle:
    all_data: list[dict] = pkl.load(handle)
# print(len(all_data))  # amount of panels
# print(len(all_data[0]["changes"]))  # amount of counterfactuals for each panel
# print(all_data[0].keys())  # for each panel, there is the original panel, changes(counterfactuals), and the new walkaway probabilities after changes
# print(all_data[0]["panel"].columns)  # columns of each panel

c = dict()
for panel in tqdm.tqdm(all_data):
    original_panel: pd.Series = panel["panel"].iloc[0]  # the original panel
    original_walkaway_probability: float = original_panel[
        "walkaway_probability"]  # original panels walkaway probability
    counterfactuals: pd.DataFrame = panel["changes"]  # all counterfactuals for this panel
    new_walkaway_probs: list = panel[
        "new_walkaway_probabilities"]  # walkaway probabilities after applying counterfactuals
    # print(len(counterfactuals))  # amount of counterfactuals for this panel
    # print(original_panel["median_cost"])  # get some feature of panel
    orig_selection_probs = wtp_predictor.predict_selection_prbs(original_panel.drop("walkaway_probability"))[0]

    for counterfactual_idx in range(len(new_walkaway_probs)):  # look through each counterfactual for this panel
        changes: pd.Series = counterfactuals.iloc[counterfactual_idx]  # the changes that were made
        new_walkaway_prob: float = new_walkaway_probs[counterfactual_idx]  # the new walkaway probability after changes

        # print(changes)
        # for i in range(1, 31):
        #     c1 = f"slotcost_{i}"
        #     field = f"days_since_first_purchase"
        #     if c1 in changes:
        #         if pd.notna(changes[c1]):
        #             c[original_panel[field]] = c.get(original_panel[field], []) + [changes[c1]]

        # print(new_walkaway_prob)

        # you can get the changed full panel like this (add the changes to the original panel and set the new walkaway prob)
        # counterfactual_panel: pd.Series = changes.fillna(0) + original_panel.iloc[0]
        counterfactual_panel = original_panel.copy(deep=True)
        for name, val in changes.fillna(0).items():
            if name in counterfactual_panel:
                counterfactual_panel[name] += val

        selection_probs = wtp_predictor.predict_selection_prbs(counterfactual_panel.drop("walkaway_probability"))[0]
        # print(selection_probs, sum(selection_probs))
        # print(orig_selection_probs, sum(orig_selection_probs))
        s = 0
        for i in range(len(orig_selection_probs)):
            delta = selection_probs[i] - orig_selection_probs[i]
            s += selection_probs[i]
            if delta != 0:
                col = f"selection_probability_{i + 1}"
                if col not in counterfactuals.columns:
                    counterfactuals[col] = np.nan
                with warnings.catch_warnings(action="ignore"):
                    ooo = counterfactuals.iloc[counterfactual_idx]
                    ooo[col] = delta
                counterfactuals.iloc[counterfactual_idx] = ooo
        # print(original_panel)
        # print(counterfactual_panel)
        # print(original_panel)
        # print(counterfactual_panel)
        # counterfactual_panel["walkaway_probability"] = new_walkaway_prob

        # print(new_panel["median_cost"])
    for i in range(len(orig_selection_probs)):
        original_panel[f"selection_probability_{i + 1}"] = orig_selection_probs[i]
    panel["changes"] = counterfactuals
    panel["panel"] = pd.DataFrame(original_panel).T
    panel["panel"]["walkaway_probability"] = original_walkaway_probability
#print(all_data[0]["panel"])

with open('data_mean_panel_250_counterf_selection.pkl', 'wb') as handle:
    pkl.dump(all_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
