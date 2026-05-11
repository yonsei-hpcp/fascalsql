-- Star Schema Benchmark (SSB) DDL
-- PK/FK constraints for fact table LINEORDER and dimension tables.
-- Ref: Star Schema Benchmark, TPC-H derived star schema.

-- Dimension: DATE (DWDATE)
CREATE TABLE date (
    d_datekey      INTEGER NOT NULL,
    d_date         VARCHAR(19),
    d_dayofweek    VARCHAR(10),
    d_month        VARCHAR(10),
    d_year         INTEGER,
    d_yearmonthnum INTEGER,
    d_yearmonth    VARCHAR(8),
    d_daynuminweek INTEGER,
    d_daynuminmonth INTEGER,
    d_daynuminyear INTEGER,
    d_monthnuminyear INTEGER,
    d_weeknuminyear INTEGER,
    d_sellingseason VARCHAR(13),
    d_lastdayinweekfl INTEGER,
    d_lastdayinmonthfl INTEGER,
    d_holidayfl    INTEGER,
    d_weekdayfl    INTEGER,
    PRIMARY KEY (d_datekey)
);

-- Dimension: CUSTOMER
CREATE TABLE customer (
    c_custkey    INTEGER NOT NULL,
    c_name       VARCHAR(25),
    c_address    VARCHAR(40),
    c_city       VARCHAR(10),
    c_nation     VARCHAR(15),
    c_region     VARCHAR(12),
    c_phone      VARCHAR(15),
    c_mktsegment VARCHAR(10),
    PRIMARY KEY (c_custkey)
);

-- Dimension: SUPPLIER
CREATE TABLE supplier (
    s_suppkey INTEGER NOT NULL,
    s_name    VARCHAR(25),
    s_address VARCHAR(40),
    s_city    VARCHAR(10),
    s_nation  VARCHAR(15),
    s_region  VARCHAR(12),
    s_phone   VARCHAR(15),
    PRIMARY KEY (s_suppkey)
);

-- Dimension: PART
CREATE TABLE part (
    p_partkey   INTEGER NOT NULL,
    p_name      VARCHAR(22),
    p_mfgr      VARCHAR(6),
    p_category  VARCHAR(7),
    p_brand1    VARCHAR(9),
    p_color     VARCHAR(11),
    p_type      VARCHAR(25),
    p_size      INTEGER,
    p_container VARCHAR(10),
    PRIMARY KEY (p_partkey)
);

-- Fact: LINEORDER
CREATE TABLE lineorder (
    lo_orderkey       INTEGER NOT NULL,
    lo_linenumber     INTEGER NOT NULL,
    lo_custkey        INTEGER NOT NULL,
    lo_partkey        INTEGER NOT NULL,
    lo_suppkey        INTEGER NOT NULL,
    lo_orderdate      INTEGER NOT NULL,
    lo_orderpriority  VARCHAR(15),
    lo_shippriority   INTEGER,
    lo_quantity       INTEGER,
    lo_extendedprice  INTEGER,
    lo_ordtotalprice  INTEGER,
    lo_discount       INTEGER,
    lo_revenue        INTEGER,
    lo_supplycost     INTEGER,
    lo_tax            INTEGER,
    lo_commitdate     INTEGER,
    lo_shipmode       VARCHAR(10),
    PRIMARY KEY (lo_orderkey, lo_linenumber),
    FOREIGN KEY (lo_custkey)   REFERENCES customer(c_custkey),
    FOREIGN KEY (lo_partkey)   REFERENCES part(p_partkey),
    FOREIGN KEY (lo_suppkey)   REFERENCES supplier(s_suppkey),
    FOREIGN KEY (lo_orderdate) REFERENCES date(d_datekey)
);
