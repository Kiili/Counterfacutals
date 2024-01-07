import random

import pandas as pd
import scipy


class TimeSlot:
    def __init__(self, slotcost, slot_start, slot_width, exact_selection_customer_perc,
                 partial_selection_customer_perc):
        self.slotcost = slotcost
        self.slot_start = slot_start
        self.slot_width = slot_width
        self.exact_selection_customer_perc = exact_selection_customer_perc
        self.partial_selection_customer_perc = partial_selection_customer_perc
        self.rank_cost = None


class Panel:
    def __init__(self, timeslots=None, expanding_avg_days_to_delivery=None, days_since_first_purchase=None):
        self.timeslots: list[TimeSlot] = timeslots
        self.expanding_avg_days_to_delivery = expanding_avg_days_to_delivery
        self.days_since_first_purchase = days_since_first_purchase

    def calc_costs(self):
        costs = [x.slotcost for x in self.timeslots]
        df = pd.DataFrame(costs)
        self.min_cost = min(costs)
        self.max_cost = max(costs)
        self.q1_cost = df.quantile(0.25)[0]
        self.median_cost = df.median(axis=0)[0]
        ranks = df.rank(pct=True, method="dense")
        for i in range(len(costs)):
            self.timeslots[i].rank_cost = ranks.iloc[i][0]

    def get_df(self):
        return pd.DataFrame({"slotcost": [x.slotcost for x in self.timeslots],
                             "slot_start": [x.slot_start for x in self.timeslots],
                             "slot_width": [x.slot_width for x in self.timeslots],
                             "exact_selection_customer_perc": [x.exact_selection_customer_perc for x in self.timeslots],
                             "partial_selection_customer_perc": [x.partial_selection_customer_perc for x in
                                                                 self.timeslots],
                             "rank_cost": [x.rank_cost for x in self.timeslots],
                             "min_cost": [self.min_cost for _ in range(len(self.timeslots))],
                             "max_cost": [self.max_cost for _ in range(len(self.timeslots))],
                             "q1_cost": [self.q1_cost for _ in range(len(self.timeslots))],
                             "median_cost": [self.median_cost for _ in range(len(self.timeslots))],
                             "expanding_avg_days_to_delivery": [self.expanding_avg_days_to_delivery for _ in
                                                                range(len(self.timeslots))],
                             "days_since_first_purchase": [self.days_since_first_purchase for _ in
                                                           range(len(self.timeslots))],
                             })

    def load_from_single_row(self, df: pd.Series):  # reverse of get_single_row_df()
        self.timeslots = []
        df = list(df)
        for n in range(0, len(df) - 6, 6):
            (slotcost, slot_start, slot_width, exact_selection_customer_perc, partial_selection_customer_perc,
             rank_cost) = df[n:n + 6]
            timeslot = TimeSlot(slotcost, slot_start, slot_width, exact_selection_customer_perc,
                                partial_selection_customer_perc)
            self.timeslots.append(timeslot)
        self.calc_costs()
        self.expanding_avg_days_to_delivery = df[-2]
        self.days_since_first_purchase = df[-1]
        return self

    def get_single_row_df(self):  # reverse of load_from_single_row()
        df = pd.DataFrame({"slotcost": [x.slotcost for x in self.timeslots],
                           "slot_start": [x.slot_start for x in self.timeslots],
                           "slot_width": [x.slot_width for x in self.timeslots],
                           "exact_selection_customer_perc": [x.exact_selection_customer_perc for x in self.timeslots],
                           "partial_selection_customer_perc": [x.partial_selection_customer_perc for x in
                                                               self.timeslots],
                           "rank_cost": [x.rank_cost for x in self.timeslots],
                           })
        df.index = df.index + 1
        df_out = df.stack()
        df_out.index = df_out.index.map('{0[1]}_{0[0]}'.format)
        df_out = df_out.to_frame().T
        df_out["min_cost"] = [self.min_cost]
        df_out["max_cost"] = [self.max_cost]
        df_out["q1_cost"] = [self.q1_cost]
        df_out["median_cost"] = [self.median_cost]
        df_out["expanding_avg_days_to_delivery"] = [self.expanding_avg_days_to_delivery]
        df_out["days_since_first_purchase"] = [self.days_since_first_purchase]
        return df_out


def get_distribution(min_val, max_val, mean, std):
    # https://stackoverflow.com/questions/50626710/generating-random-numbers-with-predefined-mean-std-min-and-max
    scale = max_val - min_val
    location = min_val
    unscaled_mean = (mean - min_val) / scale
    unscaled_var = (std / scale) ** 2
    t = unscaled_mean / (1 - unscaled_mean)
    beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
    alpha = beta * t
    if alpha <= 0 or beta <= 0:
        raise ValueError('Cannot create distribution for the given parameters.')
    return scipy.stats.beta(alpha, beta, scale=scale, loc=location)


def get_mean_panel():
    n_weekdays = 5
    n_slots = 6
    timeslots = []
    mean_start = 9873.17 + (-6480 + 120) / 2
    for weekday in range(n_weekdays):
        for slot_id in range(n_slots):
            timeslot = TimeSlot(slotcost=6.21,
                                slot_start=mean_start + slot_id * 120 + weekday * 1440,
                                slot_width=120,
                                exact_selection_customer_perc=0.065,
                                partial_selection_customer_perc=0.116,
                                )
            timeslots.append(timeslot)

    panel = Panel(timeslots=timeslots,
                  expanding_avg_days_to_delivery=1.46,
                  days_since_first_purchase=162.99,
                  )
    panel.calc_costs()
    return panel


def gen_panel(n_panels):
    n_weekdays = 5
    n_slots = 6  # per day
    mincosts = get_distribution(4.42, 6.32, 5.28, 0.3).rvs(size=n_panels)
    maxcosts = get_distribution(6.31, 13.46, 7.37, 0.94).rvs(size=n_panels)
    days_since_first_purchase = get_distribution(1, 363, 162.99, 100.31).rvs(size=n_panels)
    expanding_avg_days_to_delivery = get_distribution(0, 6, 1.46, 0.64).rvs(size=n_panels)
    slot_width = 120
    exact_selection_customer_perc = get_distribution(0, 1, 0.065, 0.142).rvs(size=n_panels * n_weekdays * n_slots)
    partial_selection_customer_perc = get_distribution(0, 1, 0.116, 0.176).rvs(size=n_panels * n_weekdays * n_slots)
    panels = []

    for i in range(n_panels):
        # start of first timeslot of panel. Made so that the last (friday) slot_start will always be <= 22800
        week_start_time = random.randint(840, 22800 - 6480 + 120)
        try:
            slotcost = get_distribution(mincosts[i], maxcosts[i], 6.21, 0.59).rvs(size=n_weekdays * n_slots)
        except:
            continue

        timeslots = []
        for weekday in range(n_weekdays):
            for slot_id in range(n_slots):
                idx = i * n_weekdays * n_slots + n_weekdays * weekday + slot_id
                timeslot = TimeSlot(slotcost=slotcost[n_weekdays * weekday + slot_id],
                                    slot_start=week_start_time + slot_id * 120 + weekday * 1440,
                                    slot_width=slot_width,
                                    exact_selection_customer_perc=exact_selection_customer_perc[idx],
                                    partial_selection_customer_perc=partial_selection_customer_perc[idx],
                                    )
                timeslots.append(timeslot)

        panel = Panel(timeslots=timeslots,
                      expanding_avg_days_to_delivery=expanding_avg_days_to_delivery[i],
                      days_since_first_purchase=days_since_first_purchase[i],
                      )
        panel.calc_costs()
        panels.append(panel)

    return panels


if __name__ == '__main__':
    p = gen_panel(10)[0]
    p2 = Panel().load_from_single_row(p.get_single_row_df().iloc[0])

    assert all(p.get_df() == p2.get_df())
    # print(f.get_df() == p2.get_df())
