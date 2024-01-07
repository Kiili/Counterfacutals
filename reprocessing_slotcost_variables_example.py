import numpy as np
import pandas as pd

from numpy.typing import NDArray


def reprocess_slotcost_variables(
    dataset_df: pd.DataFrame,
    group_by_columns: NDArray | list = ["shippingnumber"],
    vars_to_reprocess: NDArray
    | list = [
        "min_cost",
        "q1_cost",
        "median_cost",
        "q3_cost",
        "max_cost",
        "cv_cost",
        "iqr_cost",
        "rank_cost",
    ],
) -> pd.DataFrame:
    grouped_data = dataset_df.slotcost.groupby(group_by_columns)
    dataset_df.loc[:, vars_to_reprocess] = (
        grouped_data.rank(pct=True, method="dense")
        .rename("rank_cost")
        .to_frame()
        .join(
            pd.concat(
                [
                    grouped_data.agg("std mean min max".split()),
                    grouped_data.quantile([0.25, 0.5, 0.75]).unstack(-1),
                ],
                axis=1,
            )
        )
        .assign(cv=lambda x: x["std"] / x["mean"], iqr=lambda x: x[0.75] - x[0.25])
        .set_axis(
            "rank_cost std mean min_cost max_cost q1_cost median_cost q3_cost cv_cost iqr_cost".split(),
            axis=1,
        )
    )[vars_to_reprocess]

    return dataset_df
