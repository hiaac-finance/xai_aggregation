import time
from typing import Literal, Type, Callable

import numpy as np
import pandas as pd

from .explainers import *
from .tools import *

# MCDM:
import pymcdm

# ranking tools:
import ranx

# Concunrrency:
import concurrent.futures
from pathos.multiprocessing import ProcessingPool as Pool
# from .mp import NoDaemonProcessPool as Pool
# from pathos.multiprocessing import ThreadPool as Pool
# from multiprocessing import Pool as ProcessPool
    
class AggregatedExplainer(ExplainerWrapper):
    """
    This class aggregates multiple feature-importance-based explanation methods to provide a single explanation for a given instance.
    The aggregated explanation is calculated using a weighted rank aggregation algorithm, whose weights are calculated using a MCDM algorithm based on the instance explanation metrics.

    Attributes:
        - explainer_types (list[Type[ExplainerWrapper]]): A list of the explainer classes to be used. The classes must inherit from the ExplainerWrapper class, so that they have the same interface and output format.
        - clf (object): The classifier model whose predictions will be explained.
        - X_train (pd.DataFrame | np.ndarray): The training data used to train the classifier.
        - categorical_feature_names (list[str]): The names of the categorical features that were one-hot-encoded.
        - predict_proba (callable): A function that receives a data row and returns the model's prediction probabilities. If None, the classifier's predict_proba method will be used.
        - explainer_params_list (dict[Type[ExplainerWrapper], dict]): A dictionary containing the parameters to be passed to each explainer class, in case they require additional parameters.
            it should be in the format {ExplainerType: {param1: value1, param2: value2, ...}}
        - aggregation_algorithm (str): The rank aggregation algorithm to be used. Options are "wsum" (Weighted Sum), "w_bordafuse" (Weighted BordaFuse), and "w_condorcet" (Weighted Condorcet).
    """

    def __init__(self, explainer_types: list[Type[ExplainerWrapper]], clf, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str], predict_proba: callable = None,
                 explainer_params_list: dict[Type[ExplainerWrapper], dict] = None, aggregation_algorithm: Literal["wsum", "w_bordafuse", "w_condorcet"] = "wsum", **kwargs):
        super().__init__(clf, X_train, categorical_feature_names, predict_proba)

        self.explainers = []
        for ExplainerType in explainer_types:
            extra_params = explainer_params_list.get(ExplainerType, {}) if explainer_params_list is not None else {}
            self.explainers.append(ExplainerType(clf, X_train, categorical_feature_names, predict_proba=predict_proba, **extra_params))

        if kwargs.get('evaluator', None):
            self.xai_evaluator = kwargs['evaluator']
        else:
            self.xai_evaluator = ExplanationModelEvaluator(clf, X_train, categorical_feature_names, self.predict_proba, kwargs.get('noise_gen_args', {}), **kwargs.get('evaluator_args', {}))
            self.xai_evaluator.init()

        self.aggregation_algorithm = aggregation_algorithm

        self.last_explanation_metrics: pd.DataFrame = None
    
    @staticmethod
    def _ranking_to_run(feature_importance_ranking: pd.DataFrame) -> ranx.Run:
        feature_importance_ranking["query"] = "1"
        return ranx.Run.from_df(feature_importance_ranking, q_id_col="query", doc_id_col="feature", score_col="score")
    
    @staticmethod
    def _get_weights(instance_explanation_metrics: pd.DataFrame, higher_is_better: list[bool]) -> np.ndarray[float]:
        """
        Uses a MCDM algorithm to calculate the weights for each explanation method based on the instance explanation metrics.

        Parameters:
        instance_explanation_metrics (pd.DataFrame): DataFrame containing the instance explanation metrics for each explanation method.
        higher_is_better (list[bool]): A list of booleans indicating whether higher values are preferred for each metric.
        """

        evaluation_matrix = instance_explanation_metrics.to_numpy()

        num_metrics = evaluation_matrix.shape[1]
        mcdm_criteria_weights = pymcdm.weights.equal_weights(evaluation_matrix)

        mcdm_criteria_types = np.array([1 if x else -1 for x in higher_is_better])

        # t = Topsis(evaluation_matrix, mcdm_criteria_weights, higher_is_better, debug=False)
        # t.calc()

        # return t.worst_similarity # TODO: check if this is the correct attribute to return

        mcdm_method = pymcdm.methods.TOPSIS()
        return mcdm_method(evaluation_matrix, mcdm_criteria_weights, mcdm_criteria_types)

    def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> pd.DataFrame:
        runs = []
        for explainer in self.explainers:
            runs.append(self._ranking_to_run(explainer.explain_instance(instance_data_row)))

        instance_explanation_metrics = pd.DataFrame(columns=["method", "faithfullness_correlation", "sensitivity", "complexity"]).set_index("method")
        for explainer in self.explainers:
            instance_explanation_metrics.loc[explainer.__class__.__name__] = [
                self.xai_evaluator.faithfullness_correlation(explainer, instance_data_row, iterations=10),
                self.xai_evaluator.sensitivity(explainer, instance_data_row, iterations=10),
                self.xai_evaluator.complexity(explainer, instance_data_row)
            ]
        
        self.last_explanation_metrics = instance_explanation_metrics

        weights = self._get_weights(instance_explanation_metrics, [True, True, False])

        fused_run = ranx.fuse(runs, method=self.aggregation_algorithm,
                              params={"weights": weights})
        
        return fused_run.to_dataframe().drop(columns=["q_id"]).rename(columns={"doc_id": "feature"})