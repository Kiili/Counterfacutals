import numba as nb
import numpy as np
from sklearn.metrics.pairwise import haversine_distances


@nb.jit(nopython=True)
def haversine_nb(lon1, lat1, lon2, lat2):
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6367 * 2 * np.arcsin(np.sqrt(a))


def _get_expanding_haversine_distances(x):
    distances = haversine_distances(x["lat_radian long_radian".split()].values)
    masked = np.ma.masked_equal(distances, 0)
    KM_CONVERSION_FACTOR = 6371
    return masked.mean() * KM_CONVERSION_FACTOR


def _get_centroid_distance_to_depot(x):
    centroid_position = x["long_radian lat_radian".split()].values.mean(axis=0)
    depot_position = np.radians(x["store_long store_lat".split()].values[0])
    return haversine_nb(*centroid_position, *depot_position)


@nb.jit(nopython=True)
def _get_bearings(a, b):
    d_lon = b[:, -1] - a[:, -1]
    x = np.cos(b[:, 0]) * np.sin(d_lon)
    y = np.cos(a[:, 0]) * np.sin(b[:, 0]) - np.sin(a[:, 0]) * np.cos(b[:, 0]) * np.cos(
        d_lon
    )
    return np.arctan2(x, y)  # *180/math.pi


def _get_bearing_customer_to_depot(x):
    customer_positions = x["customer_lat customer_long".split()].values
    depot_positions = x["store_lat store_long".split()].values[0]
    bearings = _get_bearings(
        customer_positions,
        np.tile(depot_positions, customer_positions.shape[0]).reshape(
            customer_positions.shape
        ),
    )
    return np.mean(bearings), np.std(bearings)


def calc_cts_vars(x):
    """
    This bit calculates the CTS variables that depend on the orders assigned to a given slot
    """
    x = x.assign(
        lat_radian=lambda x: np.radians(x.customer_lat),
        long_radian=lambda x: np.radians(x.customer_long),
    )

    (
        avg_customer_depot_bearing,
        std_customer_depot_bearing,
    ) = _get_bearing_customer_to_depot(x)

    return {
        "expanding_mean_haversine_distance": _get_expanding_haversine_distances(x),
        "avg_customer_depot_bearing": avg_customer_depot_bearing,
        "std_customer_depot_bearing": std_customer_depot_bearing,
        "centroid_haversine_distance_to_depot": _get_centroid_distance_to_depot(x),
    }
