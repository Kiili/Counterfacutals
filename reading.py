import pickle as pkl
import pandas as pd

with open('data_998.pkl', 'rb') as handle:  # put file name here
    all_data: list[dict] = pkl.load(handle)
# print(len(all_data))  # amount of panels
# print(len(all_data[0]["changes"]))  # amount of counterfactuals for each panel
# print(all_data[0].keys())  # for each panel, there is the original panel, changes(counterfactuals), and the new walkaway probabilities after changes
# print(all_data[0]["panel"].columns)  # columns of each panel

for panel in all_data:
    original_panel: pd.Series = panel["panel"].iloc[0]  # the original panel
    original_walkaway_probability: float = original_panel["walkaway_probability"]  # original panels walkaway probability
    counterfactuals: pd.DataFrame = panel["changes"]  # all counterfactuals for this panel
    new_walkaway_probs: list = panel["new_walkaway_probabilities"]  # walkaway probabilities after applying counterfactuals
    # print(len(counterfactuals))  # amount of counterfactuals for this panel
    # print(original_panel["median_cost"])  # get some feature of panel

    for counterfactual_idx in range(len(new_walkaway_probs)):  # look through each counterfactual for this panel
        changes: pd.Series = counterfactuals.iloc[counterfactual_idx]  # the changes that were made
        new_walkaway_prob: float = new_walkaway_probs[counterfactual_idx]  # the new walkaway probability after changes

        # print(changes)
        # print(new_walkaway_prob)

        # you can get the changed full panel like this (add the changes to the original panel and set the new walkaway prob)
        counterfactual_panel: pd.Series = original_panel.iloc[0] + changes.fillna(0)
        counterfactual_panel["walkaway_probability"] = new_walkaway_prob

        # print(new_panel["median_cost"])
