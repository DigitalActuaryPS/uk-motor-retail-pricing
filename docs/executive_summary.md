# Executive summary for the Pricing Director

**Simulated data.** All figures come from a simulated UK motor quote market built from illustrative assumptions. They show how the method behaves. They are not real insurer experience and not a forecast.

## Decision

Move new business and renewal prices by segment within plus or minus 5% of baseline. The optimiser discounts some segments and raises the others slightly. Renewal prices rise by +1.1% to +1.9% above baseline, and the average renewal change against last year's premium stays within the 10% cap. On the simulated test-period book, the recommended set of moves:

| Effect (versus baseline prices) | Model view | Simulated true process |
|---|---|---|
| New business policies | -4.0% | -4.8% |
| Total GWP (new business and renewal) | -0.23% | -1.04% |
| Total contribution | +5.2% (about 76,754 in the simulated book) | +4.2% |
| Portfolio margin | 23.1% to 24.4% | not tabulated |

New business conversion falls from 8.13% to 7.81% on the model view. The plan buys margin with volume, inside the volume floor.

By segment: 4 segments get a new business discount (17-24|London SE, 17-24|Midlands East, 17-24|North West Rest, 25-34|London SE), 11 get an increase and 0 hold. Average renewal change against last year's premium runs from 7.8% to 10.0% across segments, and 3 segments sit at the cap, so the cap rather than the economics limits those. Full table below.

## Why the answer is credible and where it is not

- Observational elasticity is understated. The overall elasticity is -6.03 in the simulated truth. A conversion GLM fitted to observational quotes gives -3.70 and the GBM gives -4.05, because our price correlates with unobserved customer intent (correlation 0.28). The randomised price test recovers -5.75 (95% interval -7.07 to -4.30). Optimising on the uncalibrated observational GLM instead would have given a true-process contribution gain of 4.1%, against +4.2% for the calibrated model. The difference is small here, because the rank and margin constraints bind in both cases.
- The optimiser uses the experiment-calibrated model. The truth-based gain (+4.2%) is lower than the model-based gain (+5.2%), which is the honest range.
- The broker experiment is too small to estimate a broker elasticity (95% interval -2.11 to 4.46). Direct and broker were pooled for calibration.
- Cost per policy is a fixed configuration input and never fitted. Margin and contribution depend on it directly.
- 26 constraints bind at the solution, mostly rank-deterioration caps. Changing the cap changes the answer.

## Does a lifetime view change the decision?

Yes. Mean CLV per new policy is 521 against a first-year contribution of 101. 10 of 15 segments take a different new business action when the objective is CLV instead of first-year contribution. The CLV-optimal set raises new business volume by +25.8% and new business CLV by +11.1% (first-year set: +3.4%), at a first-year contribution change of +3.0%. The trade is short-term profit for volume that is expected to renew. It rests on renewal retention assumptions and on premium and cost growth set in the configuration.

## Robustness

- Out-of-time monthly ROC-AUC of the conversion GLM ranges from 0.690 to 0.708.
- Largest population stability index across checked features and the score: 0.099 (below the amber threshold of 0.10).
- Across 8 stress scenarios the optimised plan beats baseline contribution in every case, with a smallest gain of 4.8%. In 3 stress scenarios (the cost shifts and the combined case) the baseline is already below the margin floor, so a floor flag there is not caused by the plan.
- 31 of 31 data quality checks pass on the simulated data.

## Segment decisions

| Segment | NB price | NB action | NB conversion before | NB conversion after | Renewal avg change |
|---|---|---|---|---|---|
| 17-24|London SE | -5.0% | discount | 2.6% | 3.9% | +10.0% |
| 17-24|Midlands East | -5.0% | discount | 4.6% | 6.8% | +10.0% |
| 17-24|North West Rest | -5.0% | discount | 4.7% | 7.0% | +10.0% |
| 25-34|London SE | -5.0% | discount | 5.8% | 8.2% | +9.2% |
| 25-34|Midlands East | +1.7% | increase | 9.3% | 8.4% | +9.0% |
| 25-34|North West Rest | +1.7% | increase | 9.2% | 8.3% | +9.0% |
| 35-49|London SE | +1.7% | increase | 6.0% | 5.5% | +8.1% |
| 35-49|Midlands East | +1.6% | increase | 8.5% | 7.8% | +8.1% |
| 35-49|North West Rest | +1.7% | increase | 9.0% | 8.3% | +7.8% |
| 50-64|London SE | +1.6% | increase | 7.8% | 7.2% | +7.9% |
| 50-64|Midlands East | +1.7% | increase | 11.5% | 10.4% | +7.8% |
| 50-64|North West Rest | +1.9% | increase | 11.8% | 10.6% | +8.0% |
| 65+|London SE | +1.8% | increase | 5.8% | 5.4% | +8.8% |
| 65+|Midlands East | +1.6% | increase | 7.6% | 7.2% | +8.9% |
| 65+|North West Rest | +1.7% | increase | 7.9% | 7.3% | +9.4% |

## Not in scope

No claims, frequency, severity or loss-cost modelling. No risk rating. No compliance demonstration: the renewal cap and the renewal versus new business comparison are illustrative controls and are not an ENBP test. See `docs/regulatory_notes.md`.
