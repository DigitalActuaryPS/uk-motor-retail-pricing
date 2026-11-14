-- Simulated data. Conversion, volume and contribution by our rank (new business, outside the test).
SELECT our_rank,
       COUNT(*)                         AS quotes,
       SUM(converted)                   AS policies,
       AVG(converted)                   AS conversion,
       SUM(written_premium)             AS gwp,
       SUM(fixed_cost_contribution)     AS contribution,
       SUM(fixed_cost_contribution) / COUNT(*) AS contribution_per_quote
FROM quotes
WHERE business_type = 'new_business' AND treatment_group = 'none'
GROUP BY our_rank
ORDER BY our_rank;
