-- using 1433771997 as a seed to the RNG
-- Note: original TPC-H Q15 uses CREATE VIEW / DROP VIEW.
-- Rewritten with MATERIALIZED CTE for DuckDB compatibility
-- (avoids floating-point non-determinism across view re-evaluations).

with revenue0 as materialized (
        select
                l_suppkey as supplier_no,
                sum(l_extendedprice * (1 - l_discount)) as total_revenue
        from
                lineitem
        where
                l_shipdate >= date '1993-01-01'
                and l_shipdate < date '1993-01-01' + interval '3' month
        group by
                l_suppkey
)
select
        s_suppkey,
        s_name,
        s_address,
        s_phone,
        total_revenue
from
        supplier,
        revenue0
where
        s_suppkey = supplier_no
        and total_revenue = (
                select
                        max(total_revenue)
                from
                        revenue0
        )
order by
        s_suppkey;
