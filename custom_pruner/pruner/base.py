from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import pandas as pd


class BenchmarkPruner(ABC):
    """
    Abstract base for benchmark pruners that integrate with evalscope.

    Subclasses implement fit() and select(). The evalscope integration
    point is evalscope_filter(), which returns a predicate for
    evalscope's Dataset.filter(predicate) API.
    """

    @abstractmethod
    def fit(
        self,
        score_matrix: Optional[pd.DataFrame] = None,
        item_texts: Optional[List[str]] = None,
        item_metadata: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Fit the pruner on the full score matrix.

        Args:
            score_matrix: (n_models × n_items) DataFrame of binary scores.
                          Columns are item indices; rows are model names.
                          If None, implementations load from their configured evals_dir.
            item_texts:   Per-item text for embedding (parallel to score_matrix columns).
            item_metadata: Per-item metadata DataFrame indexed by item index.
        """

    @abstractmethod
    def select(self, target_size: Optional[int] = None) -> List[int]:
        """Return sorted list of selected sample indices."""

    @abstractmethod
    def validate(self) -> Dict:
        """Run validation suite and return a JSON-serialisable report dict."""

    def evalscope_filter(self) -> Callable:
        """
        Returns a predicate for evalscope's Dataset.filter() API.

        Usage:
            dataset = dataset.filter(pruner.evalscope_filter())
        """
        selected_set = set(self.select())
        return lambda sample: sample.id in selected_set
