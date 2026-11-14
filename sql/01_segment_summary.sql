-- Simulated data. New business quotes outside the randomised test, by segment.
SELECT segment,
       COUNT(*)                                        AS quotes,
       SUM(converted)                                  AS policies,
       AVG(converted)                                  AS conversion,
       AVG(our_rank)                                   AS avg_rank,
       AVG(our_premium / comp_median - 1)              AS mean_median_gap,
       SUM(written_premium)                            AS gwp,
       SUM(fixed_cost_contribution)                    AS contribution,
       SUM(fixed_cost_contribution) / NULLIF(SUM(written_premium), 0) AS margin
FROM quotes
WHERE business_type = 'new_business' AND treatment_group = 'none'
GROUP BY segment
ORDER BY segment;
