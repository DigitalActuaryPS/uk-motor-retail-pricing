-- Simulated data. Conversion by price ratio to market median (new business, outside the test).
SELECT CASE WHEN price_ratio_to_median < 0.90 THEN '<0.90'
            WHEN price_ratio_to_median < 0.95 THEN '0.90-0.95'
            WHEN price_ratio_to_median < 1.00 THEN '0.95-1.00'
            WHEN price_ratio_to_median < 1.05 THEN '1.00-1.05'
            WHEN price_ratio_to_median < 1.10 THEN '1.05-1.10'
            ELSE '>=1.10' END           AS ratio_bucket,
       COUNT(*)                         AS quotes,
       AVG(converted)                   AS conversion
FROM quotes
WHERE business_type = 'new_business' AND treatment_group = 'none'
GROUP BY 1
ORDER BY MIN(price_ratio_to_median);
