from tabrepo.benchmark.models.ag.tabpfnv2.rfpfn.configs import TabPFNRFConfig
from tabrepo.benchmark.models.ag.tabpfnv2.rfpfn.sklearn_based_decision_tree_tabpfn import (
    DecisionTreeTabPFNClassifier,
    DecisionTreeTabPFNRegressor,
)
from tabrepo.benchmark.models.ag.tabpfnv2.rfpfn.sklearn_based_random_forest_tabpfn import (
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)

# Backward compatibility for imports
# These classes were previously in CamelCase files but are now imported from snake_case files

__all__ = [
    "DecisionTreeTabPFNClassifier",
    "DecisionTreeTabPFNRegressor",
    "RandomForestTabPFNClassifier",
    "RandomForestTabPFNRegressor",
    "TabPFNRFConfig",
]
