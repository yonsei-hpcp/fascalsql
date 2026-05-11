-- TPC-H Q8 — SQLite3 compatible. Schema: docs/schemas/tpch.ddl
SELECT
  EXTRACT(YEAR FROM o_orderdate) AS o_year,
  sum(CASE WHEN n2.n_name = 'BRAZIL' THEN l_extendedprice * (1 - l_discount) ELSE 0 END) AS brazil_volume,
  sum(l_extendedprice * (1 - l_discount)) AS total_volume
FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
WHERE p_partkey = l_partkey
  AND s_suppkey = l_suppkey
  AND l_orderkey = o_orderkey
  AND o_custkey = c_custkey
  AND c_nationkey = n1.n_nationkey
  AND n1.n_regionkey = r_regionkey
  AND r_name = 'AMERICA'
  AND s_nationkey = n2.n_nationkey
  AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
  AND p_type = 'ECONOMY ANODIZED STEEL'
GROUP BY EXTRACT(YEAR FROM o_orderdate)
ORDER BY o_year;
