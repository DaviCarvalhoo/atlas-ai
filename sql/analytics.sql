-- Named analytical queries (portable SQLite / PostgreSQL). Loaded by atlas.analytics.
-- Each block starts with "-- name: <id>" and "-- description: <text>".

-- name: tickets_by_category
-- description: Volume, share and resolution KPIs per ticket category
SELECT category,
       COUNT(*)                                              AS tickets,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)    AS share_pct,
       ROUND(CAST(AVG(resolution_hours) AS NUMERIC), 1)      AS avg_resolution_h,
       ROUND(AVG(csat), 2)                                   AS avg_csat
FROM tickets
GROUP BY category
ORDER BY tickets DESC;

-- name: sla_breaches_by_priority
-- description: SLA breach rate per priority (targets: urgent 8h, high 24h, medium 48h, low 120h)
WITH sla AS (
    SELECT priority,
           resolution_hours,
           CASE priority WHEN 'urgent' THEN 8 WHEN 'high' THEN 24
                         WHEN 'medium' THEN 48 ELSE 120 END AS target_h
    FROM tickets
    WHERE resolution_hours IS NOT NULL
)
SELECT priority,
       COUNT(*)                                                         AS resolved,
       SUM(CASE WHEN resolution_hours > target_h THEN 1 ELSE 0 END)     AS breaches,
       ROUND(100.0 * AVG(CASE WHEN resolution_hours > target_h THEN 1.0 ELSE 0 END), 1)
                                                                        AS breach_rate_pct
FROM sla
GROUP BY priority
ORDER BY breach_rate_pct DESC;

-- name: carrier_delay_ranking
-- description: Late-delivery rate per carrier, ranked with a window function
WITH delivered AS (
    SELECT carrier,
           CASE WHEN delivered_at > estimated_delivery THEN 1 ELSE 0 END AS late
    FROM orders
    WHERE status = 'delivered'
)
SELECT carrier,
       COUNT(*)                          AS deliveries,
       ROUND(100.0 * AVG(late), 1)       AS late_pct,
       RANK() OVER (ORDER BY AVG(late) DESC) AS delay_rank
FROM delivered
GROUP BY carrier
ORDER BY delay_rank;

-- name: monthly_volume
-- description: Monthly ticket volume with month-over-month growth (LAG window function)
WITH monthly AS (
    SELECT SUBSTR(CAST(created_at AS TEXT), 1, 7) AS month, COUNT(*) AS tickets
    FROM tickets
    GROUP BY SUBSTR(CAST(created_at AS TEXT), 1, 7)
)
SELECT month,
       tickets,
       ROUND(100.0 * (tickets - LAG(tickets) OVER (ORDER BY month))
             / LAG(tickets) OVER (ORDER BY month), 1) AS mom_growth_pct
FROM monthly
ORDER BY month;

-- name: csat_by_channel
-- description: Customer satisfaction and volume per support channel
SELECT channel,
       COUNT(*)              AS tickets,
       ROUND(AVG(csat), 2)   AS avg_csat,
       ROUND(100.0 * AVG(CASE WHEN csat <= 2 THEN 1.0 ELSE 0 END), 1) AS detractors_pct
FROM tickets
WHERE csat IS NOT NULL
GROUP BY channel
ORDER BY avg_csat DESC;

-- name: top_contacting_customers
-- description: Customers with the most tickets (repeat-contact signal), PII-free
SELECT t.customer_id,
       c.tier,
       c.state,
       COUNT(*)                         AS tickets,
       COUNT(DISTINCT t.category)       AS distinct_categories,
       MAX(t.created_at)                AS last_contact
FROM tickets t
JOIN v_customer_safe c ON c.customer_id = t.customer_id
GROUP BY t.customer_id, c.tier, c.state
ORDER BY tickets DESC
LIMIT 10;
