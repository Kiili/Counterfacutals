import numpy as np
import pandas as pd
from typing import Any
from pydantic import BaseModel
from dataclasses import dataclass
from catboost import CatBoostRegressor

from .cts_utils import calc_cts_vars


class CTSPredictor(BaseModel):
    model_path: str = None
    df: pd.DataFrame = None

    class Config:
        arbitrary_types_allowed = True

    def load(self):
        pass

    def compute_vars(self):
        pass

    def predict(self):
        pass


@dataclass(frozen=True)
class CTSImmutable:
    predictor: Any


class CTSRandomPredictor(CTSPredictor):
    starting_price: float
    price_variation: float

    def predict(self, slots, **kwargs):
        desired_shape = slots.shape[0]
        return np.around(
            np.repeat(self.starting_price, desired_shape)
            + (np.random.random(size=desired_shape) - 0.5) * self.price_variation,
            decimals=2,
        )


class CTSCatBoostPredictor(CTSPredictor):
    file_path: str
    model: str = None
    features: list = None
    label: str = None

    def __init__(self, **data):
        super().__init__(**data)
        self.load()

    def load(self) -> None:
        self.model = CatBoostRegressor()
        self.model.load_model(fname=self.file_path)
        self._load_feature_names()

    def _load_feature_names(self) -> None:
        self.features = [
            "expanding_mean_haversine_distance",
            "centroid_haversine_distance_to_depot",
            "ncustomers",
            "avg_customer_depot_bearing",
            "std_customer_depot_bearing",
            "hours_to_slot_start",
            "storeid",
            "slot",
            "delivery_dow",
            "areaid",
            "ndistinct_sku",
            "total_requested_qtd",
            "total_requested_amount",
            "lagged_cost",
            "rolling_mean_time_stopped",
            "rolling_mean_total_cost",
            "slot_area_capacity",
            "slot_store_capacity",
            "occupied_area_slot",
            "occupied_store_slot",
            "first_order",
        ]
        self.label = "total_cost"

    def predict(self, slots, new_order, new_order_id, order_cp7s):
        cts_vars = self.compute_vars(
            slots=slots,
            new_order=new_order,
            new_order_id=new_order_id,
            order_cp7s=order_cp7s,
        )
        data_pool = self.create_data_pool(cts_vars)
        return self.model.predict(data_pool)

    def create_data_pool(self, data: pd.DataFrame):
        import catboost

        data = data.assign(
            **{self.label: 0}
        )  # don't know why, but apparently the label is needed
        data_pool = catboost.Pool(
            data.assign(
                **{self.label: 0}
            )  # don't know why, but apparently the label is needed
            .reset_index(drop=True)[self.features]
            .astype(
                {"ndistinct_sku": "int64", "lagged_cost": "float64", "slot": "object"}
            ),
            label=data[self.label],
            cat_features=["areaid", "delivery_dow", "slot", "storeid", "first_order"],
        )
        return data_pool

    def compute_vars(self, slots, new_order_id, new_order, order_cp7s):
        vars_cts = (
            slots.apply(
                lambda x: order_cp7s.loc[x.assigned_orders + [new_order_id]]
                if x.assigned_orders
                else order_cp7s.loc[[new_order_id]]
            )
            .dropna()
            .rename("assigned_orders")
            .to_frame()
            .assign(
                slot_store_capacity=lambda x: slots.apply(lambda y: y.capacity),
                slot_area_capacity=lambda x: x.slot_store_capacity,
                opening_time=lambda x: slots.apply(lambda x: x.opening_time),
                order_time=lambda x: new_order.timestamp,
                time_slot_opening=lambda x: (
                    pd.to_datetime(
                        x.index.get_level_values("deliverydate").astype(str)
                        + " "
                        + x.index.get_level_values("slot_time").str[:5],
                        format="%Y-%m-%d %H:%M",
                    )
                ),
                hours_to_slot_start=lambda x: (
                    x.time_slot_opening - x.order_time
                ).dt.total_seconds()
                / 3600,
                ncustomers=lambda x: x.assigned_orders.apply(lambda y: y.shape[0]),
                occupied_area_slot=lambda x: x.ncustomers / x.slot_area_capacity,
                occupied_store_slot=lambda x: x.ncustomers
                / x.slot_store_capacity,  # TODO: this is not right at the moment, it should be the store capacity for that time of day
                cp4=lambda x: str(new_order.order_df.shippingpostalcode.values[0])[:4],
                delivery_dow=lambda x: x.opening_time.dt.dayofweek.astype(str),
                areaid=lambda x: x.index.get_level_values("areaid").astype(str),
                slot=lambda x: x.index.get_level_values("slot_time").astype(str),
                **new_order.order_df[
                    ["total", "total_requestedqty", "nsku", "storeid", "npurchases"]
                ]
                .iloc[0]
                .to_dict(),
                first_order=lambda x: (x.npurchases == 0) * 1,
                rolling_mean_total_cost=0,  # TODO: see how we can improve here
                rolling_mean_time_stopped=0,  # TODO: see how we can improve here
                lagged_cost=0  # TODO: see how we can improve here
            )
            .assign(storeid=lambda x: x.storeid.astype(int).astype(str))
            .rename(
                columns={
                    "total": "total_requested_amount",
                    "total_requestedqty": "total_requested_qtd",
                    "nsku": "ndistinct_sku",
                }
            )
        )

        vars_cts = vars_cts.join(
            vars_cts.assigned_orders.apply(calc_cts_vars).apply(pd.Series)
        )[self.features].assign(
            expanding_mean_haversine_distance=lambda x: x.expanding_mean_haversine_distance.astype(
                float
            ).fillna(
                0
            )
        )
        return vars_cts
