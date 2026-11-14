-- Simulated data. Randomised price test arms.
SELECT treatment_group,
       COUNT(*)                                         AS quotes,
       AVG(converted)                                   AS conversion,
       AVG(our_premium / reference_premium - 1)         AS mean_price_change
FROM quotes
WHERE treatment_group <> 'none'
GROUP BY treatment_group
ORDER BY mean_price_change;
