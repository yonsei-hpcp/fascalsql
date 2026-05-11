"""FaScalSQL Python package.

Provides data preparation, verification, and utility modules.
"""

from importlib import import_module
from typing import Any


_LAZY_MODULES = {
	"encoding_catalog": "fascalsql.python.utils.encoding_catalog",
	"sql_query_registry": "fascalsql.python.utils.sql_query_registry",
	"plan_registry": "fascalsql.python.utils.plan_registry",
	"verify_tpch_results": "fascalsql.python.utils.verify_tpch_results",
	"verify_ssb_results": "fascalsql.python.utils.verify_ssb_results",
}


def __getattr__(name: str) -> Any:
	mod = _LAZY_MODULES.get(name)
	if mod is None:
		raise AttributeError(f"module 'fascalsql.python' has no attribute {name!r}")
	return import_module(mod)


__all__ = list(_LAZY_MODULES.keys())
