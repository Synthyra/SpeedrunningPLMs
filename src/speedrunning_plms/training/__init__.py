"""Training entry points and explicit model publication."""


__all__ = ["ExperimentConfig", "run_experiment"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from speedrunning_plms.research.engine import ExperimentConfig, run_experiment

        return {"ExperimentConfig": ExperimentConfig, "run_experiment": run_experiment}[name]
    raise AttributeError(name)
