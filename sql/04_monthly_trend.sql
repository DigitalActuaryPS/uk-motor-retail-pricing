-- Simulated data. Monthly quotes, policies and written premium by business type.
SELECT date_trunc('month', quote_date)  AS month,
       business_type,
       COUNT(*)                         AS quotes,
       SUM(converted)                   AS policies,
       SUM(written_premium)             AS gwp,
       SUM(fixed_cost_contribution)     AS contribution
FROM quotes
GROUP BY 1, 2
ORDER BY 1, 2;
