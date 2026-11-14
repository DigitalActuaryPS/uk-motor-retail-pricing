-- Simulated data. Channel summary (new business, outside the test).
SELECT channel,
       COUNT(*)                         AS quotes,
       SUM(converted)                   AS policies,
       AVG(converted)                   AS conversion,
       AVG(our_rank)                    AS avg_rank,
       SUM(written_premium)             AS gwp,
       SUM(fixed_cost_contribution)     AS contribution
FROM quotes
WHERE business_type = 'new_business' AND treatment_group = 'none'
GROUP BY channel
ORDER BY channel;
