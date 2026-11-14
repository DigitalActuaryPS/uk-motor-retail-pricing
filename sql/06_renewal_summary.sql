-- Simulated data. Renewal rate by tenure and channel.
SELECT tenure, channel,
       COUNT(*)                                         AS offers,
       AVG(converted)                                   AS renewal_rate,
       AVG(our_premium / prior_year_premium - 1)        AS mean_price_change
FROM quotes
WHERE business_type = 'renewal'
GROUP BY tenure, channel
ORDER BY tenure, channel;
