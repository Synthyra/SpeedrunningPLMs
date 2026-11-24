"""The tabular foundation model behind `search.TabularProposer`: context runs and their values in, query quantiles out.

Kumo Tabular (NVIDIA, OpenMDW-1.1, package `sdm`) was the one lead of the replay (`2026-11-03_tabular_surrogate_replay`): on 54
adaptive rungs it ranks at rho 0.801 reading every earlier run, against 0.640 for the GP on the same track. This file adds it to the
proposer's seat for the head-to-head and nothing else, so `search.py` imports no model until an arm asks for one. TabPFN stays out: its
license reaches the models a proposer helps produce (`docs/decisions/representation/speedrunning_plms/2026-11-03_tabular_surrogates.md`).

The package needs the GPU and its own image on the box (`structured-data-models`, imported as `sdm`; `Dockerfile.kumo`), so the image that trains stays the record's.

Symbols: n context rows, m query rows, q = 199 quantile levels, 0.005 to 0.995.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import sdm  # the GPU image's package; imported for real only inside `quantiles`


ESTIMATORS = 8  # as the replay's arms ran it
KUMO_LEVELS = 999  # the model's own grid, 0.001 to 0.999


class KumoSurrogate:
    """Kumo Tabular, large: in-context quantile regression. The model loads once and refits each call, in seconds on the GPU."""

    def __init__(self, seed: int = 0, estimators: int = ESTIMATORS) -> None:
        self.seed = seed
        self.estimators = estimators
        self.model: sdm.models.KumoTabular | None = None  # loaded on the first call

    def quantiles(self, context: pd.DataFrame, values: np.ndarray, query: pd.DataFrame) -> np.ndarray:
        # values: (n,); returns (m, q).
        import sdm
        import torch

        if self.model is None:
            self.model = sdm.models.KumoTabular(task="regression", size="large", device="cuda")
        kinds = sdm.infer_stypes(context)
        x_context = sdm.TableTensor.from_pandas(df=context, stypes=kinds, device="cuda")
        y_context = sdm.TableTensor.from_pandas(df=pd.DataFrame({"target": values}), stypes={"target": "numerical"}, device="cuda")
        x_query = sdm.TableTensor.from_pandas(df=query, stypes=kinds, device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(self.seed)
        with torch.amp.autocast("cuda", torch.float16):
            self.model.fit(x=x_context, y=y_context, num_estimators=self.estimators, generator=generator)
            output = self.model.predict(x=x_query).numerical  # (m, 999), or with a singleton target dimension
        grid = output.float().squeeze().reshape(len(query), KUMO_LEVELS).cpu().numpy()  # (m, 999)
        return np.sort(grid, axis=1)[:, 4::5]  # (m, q)
