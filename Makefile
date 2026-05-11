.PHONY: all clean ssb tpch

all:
	$(MAKE) -C fascalsql/host all

tpch:
	$(MAKE) -C fascalsql/host tpch_standalone

ssb:
	$(MAKE) -C fascalsql/host ssb_standalone

clean:
	$(MAKE) -C fascalsql/host clean
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
