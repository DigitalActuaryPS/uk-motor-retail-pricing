# Competitor and PCW recommendations

**Simulated data.** Numbers below are produced by `retailpricing.competitor` on the simulated quote table.

Portfolio new business conversion is 9.2% on 119,203 quotes, with an average rank of 4.48 and an average price competitiveness index of 102.6 (100 means we match the market median).

## Market position drives volume
Conversion falls from 19.7% at rank 1 to 5.2% at rank 9. Rank 1 to 3 quotes are 35.9% of volume but write 50.9% of policies.

## Channel
- PCW: conversion 6.7%, average rank 4.48, first-year margin 1.2%.
- broker: conversion 12.4%, average rank 4.52, first-year margin 9.0%.
- direct: conversion 15.7%, average rank 4.48, first-year margin 10.4%.

## Overpriced segments (mean gap to market median at or above the flag threshold)
- 17-24|London_SE: +7.1% to median, conversion 3.3%, first-year margin 27.3%. Margin is above portfolio, so a test discount is the first action to try.
- 65+|London_SE: +4.7% to median, conversion 6.3%, first-year margin 17.7%. Margin is above portfolio, so a test discount is the first action to try.

## Underpriced segments
- 50-64|North_West_Rest: -7.2% to median, conversion 13.1%, first-year margin -9.2%. Margin is below portfolio, so this segment is a candidate for an increase.
- 50-64|Midlands_East: -6.3% to median, conversion 12.6%, first-year margin -6.7%. Margin is below portfolio, so this segment is a candidate for an increase.
- 25-34|North_West_Rest: -4.3% to median, conversion 10.6%, first-year margin 3.7%. Margin is below portfolio, so this segment is a candidate for an increase.
- 25-34|Midlands_East: -3.5% to median, conversion 10.4%, first-year margin 5.9%. Margin is at or above portfolio, so protect volume and monitor.
- 35-49|North_West_Rest: -3.3% to median, conversion 10.1%, first-year margin 1.0%. Margin is below portfolio, so this segment is a candidate for an increase.

These are directional. Module 4 shows observational price effects are biased, so size any move with the randomised test, then use Module 5.
