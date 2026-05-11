-- TPC-H Q22 — SQLite3 compatible. Schema: docs/schemas/tpch.ddl
SELECT
  substr(c_phone, 1, 2) AS cntrycode,
  count(*) AS numcust,
  sum(c_acctbal) AS totacctbal
FROM customer
WHERE (c_phone LIKE '13%' OR c_phone LIKE '31%' OR c_phone LIKE '23%' OR c_phone LIKE '29%' OR c_phone LIKE '30%' OR c_phone LIKE '18%' OR c_phone LIKE '17%')
  AND c_acctbal > (
    SELECT avg(c_acctbal)
    FROM customer
    WHERE c_acctbal > 0.00
      AND (c_phone LIKE '13%' OR c_phone LIKE '31%' OR c_phone LIKE '23%' OR c_phone LIKE '29%' OR c_phone LIKE '30%' OR c_phone LIKE '18%' OR c_phone LIKE '17%')
  )
  AND NOT EXISTS (
    SELECT 1
    FROM orders
    WHERE o_custkey = c_custkey
  )
GROUP BY cntrycode
ORDER BY cntrycode;
