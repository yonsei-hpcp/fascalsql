-- TPC-H Q9 -- DuckDB/Substrait dialect
-- o_orderdate is DATE type; use EXTRACT for year computation.
-- This file is the DuckDB-compatible canonical used only for Substrait generation
-- in the compiler audit. The FaScalSQL runtime version lives at fascalsql/query/sql/tpch/tpch_q9.sql.

select
	nation,
	o_year,
	sum(amount) as sum_profit
from
	(
		select
			n_name as nation,
			extract(year from o_orderdate) as o_year,
			l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
		from
			part,
			supplier,
			lineitem,
			partsupp,
			orders,
			nation
		where
			s_suppkey = l_suppkey
			and ps_suppkey = l_suppkey
			and ps_partkey = l_partkey
			and p_partkey = l_partkey
			and o_orderkey = l_orderkey
			and s_nationkey = n_nationkey
			and p_name like '%yellow%'
	) as profit
group by
	nation,
	o_year
order by
	nation,
	o_year desc;
