-- TPC-H Q15 — SQLite3 compatible. Schema: docs/schemas/tpch.ddl
WITH revenue(l_suppkey, sum_revenue) AS (
  SELECT l_suppkey, sum(l_extendedprice * (1 - l_discount))
  FROM lineitem
  WHERE l_shipdate >= DATE '1996-01-01'
    AND l_shipdate < DATE '1996-04-01'
  GROUP BY l_suppkey
),
max_revenue(total_revenue) AS (
  SELECT max(sum_revenue) FROM revenue
)
SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
FROM supplier, revenue r2, max_revenue
WHERE s_suppkey = r2.l_suppkey
  AND total_revenue = r2.sum_revenue
ORDER BY s_suppkey;
