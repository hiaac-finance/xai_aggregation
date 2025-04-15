import time
from typing import Literal, Type, Callable

import numpy as np
import pandas as pd

from .explainers import *
from .tools import *

# MCDM:
import pymcdm
from pymcdm.methods.mcda_method import MCDA_method

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
    .. class:: AggregatedExplainer(ExplainerWrapper)
        Aggregates multiple feature-importance-based explanation methods to provide a single explanation.
        This class combines various explanation methods using a weighted rank aggregation algorithm. 
        The weights are calculated using a Multi-Criteria Decision Making (MCDM) algorithm based on 
        instance explanation metrics.
        :param explainer_types: A list of the explainer classes to be used. 
                                Classes must inherit from ExplainerWrapper.
        :type explainer_types: list[Type[ExplainerWrapper]]
        :param model: The model whose predictions will be explained.
        :param X_train: The training data used to train the classifier.
        :type X_train: pd.DataFrame | np.ndarray
        :param categorical_feature_names: The names of categorical features that were one-hot-encoded.
        :type categorical_feature_names: list[str], optional
        :param predict_fn: A function that receives a data row and returns the model's prediction probabilities.
                          If None, the classifier's predict_proba method will be used.
        :type predict_fn: callable, optional
        :param explainer_params_list: Parameters to be passed to each explainer class.
                                     Format: {ExplainerType: {param1: value1, param2: value2, ...}}
        :type explainer_params_list: dict[Type[ExplainerWrapper], dict], optional
        :param metrics: Metrics used to evaluate explanations and calculate weights.
        :type metrics: list[Literal['complexity', 'sensitivity_spearman', 'faithfulness_corr', 'nrc', 'rb_faithfulness_corr']], optional
        :param mcdm_method: The MCDM method used to calculate weights.
        :type mcdm_method: MCDA_method, optional
        :param aggregation_algorithm: The rank aggregation algorithm.
                                     Options: "wsum" (Weighted Sum), "w_bordafuse" (Weighted BordaFuse), 
                                     or "w_condorcet" (Weighted Condorcet).
        :type aggregation_algorithm: Literal["wsum", "w_bordafuse", "w_condorcet"], optional
        .. method:: explain_instance(instance_data_row)
            Generates an aggregated explanation for the given instance.
            :param instance_data_row: The data instance to explain.
            :type instance_data_row: pd.Series
            :return: A DataFrame containing the aggregated feature importance scores.
            :rtype: pd.DataFrame
        .. method:: get_last_explanation_info()
            Returns explanation metrics and weights for the last explained instance.
            :return: DataFrame with metrics and weights for each explainer type.
            :rtype: pd.DataFrame
    """

    def __init__(self, explainer_types: list[Type[ExplainerWrapper]], model: Any, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str] = [], predict_fn: callable = None,
                 explainer_params_list: dict[Type[ExplainerWrapper], dict] = None,
                 metrics: list[Literal['complexity', 'sensitivity_spearman', 'faithfulness_corr', 'nrc', 'rb_faithfulness_corr']] = ['complexity', 'sensitivity_spearman', 'faithfulness_corr'],
                 mcdm_method: MCDA_method = pymcdm.methods.TOPSIS(), aggregation_algorithm: Literal["wsum", "w_bordafuse", "w_condorcet"] = "wsum", **kwargs):
        super().__init__(model, X_train, categorical_feature_names, predict_fn=predict_fn)

        self.explainer_types = explainer_types
        self.explainers = []
        for ExplainerType in explainer_types:
            extra_params = explainer_params_list.get(ExplainerType, {}) if explainer_params_list is not None else {}
            self.explainers.append(ExplainerType(model, X_train, categorical_feature_names, predict_fn=predict_fn, **extra_params))

        if kwargs.get('evaluator', None):
            self.xai_evaluator = kwargs['evaluator']
        else:
            self.xai_evaluator = ExplanationModelEvaluator(model, X_train, categorical_feature_names, self.predict_fn, kwargs.get('noise_gen_args', {}), **kwargs.get('evaluator_args', {}))
            self.xai_evaluator.init()
        
        self.metrics = metrics
        self.mcdm_method = mcdm_method
        self.aggregation_algorithm = aggregation_algorithm

        self.last_explanation_metrics: pd.DataFrame = None
        self.last_explanation_components: list[DataFrame[ExplanationModel]] = []
        
        self._metric_functions = {
            "faithfulness_corr": lambda explainer, instance_data_row: self.xai_evaluator.faithfullness_correlation(explainer, instance_data_row, iterations=10),
            "rb_faithfulness_corr": lambda explainer, instance_data_row: self.xai_evaluator.faithfullness_correlation(explainer, instance_data_row, iterations=100, len_subset=1, rank_based=True, rb_alg="inverse"),
            "sensitivity_spearman": self.xai_evaluator.sensitivity,
            "complexity": self.xai_evaluator.complexity,
            "nrc": self.xai_evaluator.nrc,
            "nrc_old": self.xai_evaluator.nrc_old
        }
    
    @staticmethod
    def _ranking_to_run(feature_importance_scores: DataFrame[ExplanationModel]) -> ranx.Run:
        fir = get_ranked_explanation(feature_importance_scores, invert=False, epsilon=0)
        fir["rank"] = fir["rank"].astype(float)
        fir["percentile"] = 1 - (fir['rank'] / fir['rank'].values[-1])
        fir["inverse_sq"] = 1 / fir["rank"] ** 2
        fir["query"] = "1"
        return ranx.Run.from_df(fir, q_id_col="query", doc_id_col="feature", score_col="inverse_sq")
        
        # # Normalize the score column to be between 0 and 1
        # fis = feature_importance_scores.copy()
        # fis["score"] = (fis["score"] - fis["score"].min()) / (fis["score"].max() - fis["score"].min())
        # fis["query"] = "1"
        # return ranx.Run.from_df(fis, q_id_col="query", doc_id_col="feature", score_col="score")
    
    def _get_weights(self, instance_explanation_metrics: np.ndarray, higher_is_better: list[bool]) -> np.ndarray[float]:
        """
            Calculate weights for each explanation method using a MCDM algorithm based on instance metrics.
            
            This method applies a Multi-Criteria Decision Making (MCDM) algorithm to determine 
            the relative importance of each explanation method according to their performance 
            on various metrics.
            
            Parameters
            ----------
            instance_explanation_metrics : np.ndarray
                Array containing the instance explanation metrics for each explanation method.
                Each row represents an explanation method and each column represents a metric.
            
            higher_is_better : list[bool]
                List indicating whether higher values are preferred for each metric.
                True means higher values are better, False means lower values are better.
            
            Returns
            -------
            np.ndarray
                Normalized weights for each explanation method, summing to 1.
                
            Notes
            -----
            The calculated weights are also stored in the `_last_explanation_weights` attribute.
        """

        evaluation_matrix = instance_explanation_metrics
        mcdm_criteria_weights = pymcdm.weights.equal_weights(evaluation_matrix)
        mcdm_criteria_types = np.array([1 if x else -1 for x in higher_is_better])

        weights = self.mcdm_method(evaluation_matrix, mcdm_criteria_weights, mcdm_criteria_types)
        self._last_explanation_weights = weights

        return weights

    def explain_instance(self, instance_data_row: pd.Series) -> pd.DataFrame:
        """
        Generate an explanation for a single instance by aggregating explanations from multiple explainers.
        This method:
        1. Obtains individual explanations from each explainer
        2. Converts explanations to ranking runs
        3. Computes performance metrics for each explainer
        4. Determines aggregation weights based on metrics
        5. Fuses the individual runs using the specified aggregation algorithm
        
        Parameters
        ----------
        instance_data_row : pd.Series
            The instance to explain, represented as a pandas Series.
            
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the aggregated explanation, with features as rows
            and their importance scores as values.
            
        Notes
        -----
        The method stores the individual explainer results in `last_explanation_components`
        and metrics in `_last_explanation_metrics` for later inspection.
        """
        
        runs = []
        self.last_explanation_components = []
        for explainer in self.explainers:
            component_explanation = explainer.explain_instance(instance_data_row)
            self.last_explanation_components.append(component_explanation)
            runs.append(self._ranking_to_run(component_explanation))

        instance_explanation_metrics = []
        for explainer in self.explainers:
            expaliner_metrics_row = []
            for metric in self.metrics:
                expaliner_metrics_row.append(self._metric_functions[metric](explainer, instance_data_row))
                
            instance_explanation_metrics.append(expaliner_metrics_row)

        self._last_explanation_metrics = instance_explanation_metrics

        weights = self._get_weights(np.array(instance_explanation_metrics), [ExplanationModelEvaluator.IS_METRIC_HIGHER_BETTER[metric] for metric in self.metrics])

        fused_run = ranx.fuse(runs, method=self.aggregation_algorithm,
                              params={"weights": weights})
        
        return fused_run.to_dataframe().drop(columns=["q_id"]).rename(columns={"doc_id": "feature"})

    def get_last_explanation_info(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing the explanation metrics and weights for the aggregated explainer types
        for the last explained instance.
        
        The DataFrame's rows are indexed by the explainer class names, with columns for each metric used
        in the aggregation plus a 'weight' column showing the weight assigned to each explainer.
        
        Returns:
        --------
            pd.DataFrame: A DataFrame with explanation metrics and weights where:
                - Each row corresponds to an explainer in self.explainers
                - Columns include all metrics in self.metrics plus a 'weight' column
                - Index consists of the explainer class names
        """

        explanation_info = pd.DataFrame(self._last_explanation_metrics, columns=self.metrics, index=[explainer.__class__.__name__ for explainer in self.explainers])
        explanation_info['weight'] = self._last_explanation_weights
        return explanation_info