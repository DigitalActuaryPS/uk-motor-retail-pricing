# Price change approval template

**Simulated data.** This template is an illustrative project control. Completing it does not demonstrate regulatory compliance. The tables below are filled from the simulated optimisation output so the layout can be reviewed.

## 1. Change record

| Field | Entry |
|---|---|
| Change reference | |
| Product and channel | UK motor, all channels (simulated) |
| Proposed effective date | |
| Proposer (role) | |
| Rationale in one sentence | |
| Evidence base | Simulated data. Replace with real data and a live test before use |

## 2. Proposed changes by segment

| Segment | NB change | Renewal change | NB conversion before | NB conversion after | NB rank before | NB rank after |
|---|---|---|---|---|---|---|
| 17-24|London_SE | -5.0% | +1.5% | 2.6% | 3.9% | 6.79 | 5.67 |
| 17-24|Midlands_East | -5.0% | +1.1% | 4.6% | 6.8% | 5.74 | 4.53 |
| 17-24|North_West_Rest | -5.0% | +1.6% | 4.7% | 7.0% | 5.71 | 4.47 |
| 25-34|London_SE | -5.0% | +1.9% | 5.8% | 8.2% | 5.30 | 4.13 |
| 25-34|Midlands_East | +1.7% | +1.6% | 9.3% | 8.4% | 4.32 | 4.72 |
| 25-34|North_West_Rest | +1.7% | +1.7% | 9.2% | 8.3% | 4.26 | 4.65 |
| 35-49|London_SE | +1.7% | +1.8% | 6.0% | 5.5% | 5.56 | 5.96 |
| 35-49|Midlands_East | +1.6% | +1.8% | 8.5% | 7.8% | 4.67 | 5.05 |
| 35-49|North_West_Rest | +1.7% | +1.7% | 9.0% | 8.3% | 4.39 | 4.78 |
| 50-64|London_SE | +1.6% | +1.7% | 7.8% | 7.2% | 4.63 | 5.02 |
| 50-64|Midlands_East | +1.7% | +1.8% | 11.5% | 10.4% | 3.61 | 4.01 |
| 50-64|North_West_Rest | +1.9% | +1.9% | 11.8% | 10.6% | 3.49 | 3.89 |
| 65+|London_SE | +1.8% | +1.9% | 5.8% | 5.4% | 6.24 | 6.64 |
| 65+|Midlands_East | +1.6% | +1.7% | 7.6% | 7.2% | 5.39 | 5.78 |
| 65+|North_West_Rest | +1.7% | +1.9% | 7.9% | 7.3% | 5.15 | 5.54 |

## 3. Expected impact (model basis, planning book of test-period quotes)

| scenario | nb_policies | nb_gwp | nb_contribution | renewal_policies | renewal_gwp | renewal_contribution | total_gwp | total_contribution | margin |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1,668 | 1,025,876 | 117,012 | 8,045 | 5,380,829 | 1,365,701 | 6,406,705 | 1,482,713 | 23.1% |
| optimised | 1,602 | 1,025,024 | 130,627 | 7,893 | 5,366,918 | 1,428,840 | 6,391,942 | 1,559,467 | 24.4% |

## 4. Constraint check

Optimiser converged: True. Minimum constraint slack after solve: -2.52e-14.
Binding constraints: nb_rank_cap:25-34|Midlands_East, nb_rank_cap:25-34|North_West_Rest, nb_rank_cap:35-49|London_SE, nb_rank_cap:35-49|Midlands_East, nb_rank_cap:35-49|North_West_Rest, nb_rank_cap:50-64|London_SE, nb_rank_cap:50-64|Midlands_East, nb_rank_cap:50-64|North_West_Rest, nb_rank_cap:65+|London_SE, nb_rank_cap:65+|Midlands_East, nb_rank_cap:65+|North_West_Rest, renewal_rank_cap:25-34|London_SE, renewal_rank_cap:25-34|Midlands_East, renewal_rank_cap:25-34|North_West_Rest, renewal_rank_cap:35-49|London_SE, renewal_rank_cap:35-49|Midlands_East, renewal_rank_cap:35-49|North_West_Rest, renewal_rank_cap:50-64|London_SE, renewal_rank_cap:50-64|Midlands_East, renewal_rank_cap:50-64|North_West_Rest, renewal_rank_cap:65+|London_SE, renewal_rank_cap:65+|Midlands_East, renewal_rank_cap:65+|North_West_Rest, renewal_change_cap:17-24|London_SE, renewal_change_cap:17-24|Midlands_East, renewal_change_cap:17-24|North_West_Rest.

## 5. Stress tests

| scenario | gain | nb_policies_vs_baseline | baseline_margin | optimised_margin | nb_volume_floor_met |
|---|---|---|---|---|---|
| central | +5.2% | 96.0% | 23.1% | 24.4% | True |
| elasticity x0.7 | +5.6% | 96.9% | 23.1% | 24.4% | True |
| elasticity x1.3 | +4.8% | 95.5% | 23.1% | 24.4% | True |
| competitor prices -5% | +5.0% | 96.2% | 23.4% | 24.7% | True |
| competitor prices +5% | +5.5% | 95.8% | 22.8% | 24.0% | True |
| cost per policy +5% | +6.5% | 96.0% | 19.4% | 20.7% | True |
| cost per policy +10% | +8.5% | 96.0% | 15.6% | 17.0% | True |
| combined: elasticity x1.3, competitors -5%, cost +10% | +7.7% | 95.6% | 15.9% | 17.3% | True |

## 6. Customer outcome checks (illustrative)

- Renewal price change cap applied per segment: 10260 renewal offers in the planning book.
- Illustrative renewal versus new business check: 7 of 15 segments have an average renewal price above the average new business price. Review each before approval.
- Fair value and conduct review: to be completed by the relevant function. Not performed in this project.

## 7. Known limitations to disclose

- Renewal elasticity is observational, with no randomised renewal test.
- Cost per policy is a fixed assumption, not a loss cost model.
- See the model risk register for the full list.

## 8. Sign-off

| Role | Name | Decision | Date |
|---|---|---|---|
| Pricing actuary | | | |
| Pricing director | | | |
| Risk or compliance | | | |
