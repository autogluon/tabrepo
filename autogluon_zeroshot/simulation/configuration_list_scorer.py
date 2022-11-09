from typing import List

from autogluon_zeroshot.simulation.simulation_context import ZeroshotSimulatorContext


class ConfigurationListScorer:
    # todo we could probably factor more common arguments from Single/Ensemble. For now, we only
    #  factor out datasets which is needed by downstream classes.
    def __init__(self, datasets):
        self.datasets = datasets

    @classmethod
    def from_zsc(cls, zeroshot_simulator_context: ZeroshotSimulatorContext, **kwargs):
        raise NotImplementedError()

    def score(self, configs: list) -> float:
        raise NotImplementedError()

    def subset(self, datasets: List[str]) -> "ConfigurationListScorer":
        raise NotImplementedError()