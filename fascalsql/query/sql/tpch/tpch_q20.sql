-- TPC-H Q20 — SQLite3 compatible. Schema: docs/schemas/tpch.ddl
WITH lineitem_sum AS (
  SELECT l_partkey, l_suppkey, sum(l_quantity) AS sum_qty
  FROM lineitem
  WHERE l_shipdate >= DATE '1994-01-01'
    AND l_shipdate < DATE '1995-01-01'
  GROUP BY l_partkey, l_suppkey
)
SELECT s_name, s_address
FROM supplier, nation
WHERE s_suppkey IN (
  SELECT ps_suppkey
  FROM partsupp, lineitem_sum
  WHERE ps_partkey = lineitem_sum.l_partkey
    AND ps_suppkey = lineitem_sum.l_suppkey
    AND ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
    AND ps_availqty > 0.5 * lineitem_sum.sum_qty
)
  AND s_nationkey = n_nationkey
  AND n_name = 'CANADA'
ORDER BY s_name;
