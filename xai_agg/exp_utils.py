"""
Experiment Utilities
====================

This module provides utility functions for conducting, analyzing, and visualizing
experiments with aggregated explainer models. It includes tools for evaluating explainer
performance across different metrics, calculating statistics, and presenting results.

Functions
---------

evaluate_aggregate_explainer
    Evaluate aggregated explainers across various configurations, metrics, and algorithms.

get_expconfig_mean_results
    Calculate mean results for a specific experiment configuration.

count_worst_case_avoidances
    Count instances where an explainer avoids worst-case performance scenarios.

get_average_metric_rank
    Calculate average rank of each method across multiple instances for each metric.

present_experiment_run
    Present and analyze comprehensive results from an experiment run.

Classes
-------

ExperimentRun
    Data container for storing experiment metadata and results.
"""

from typing import Literal
import pandas as pd
import pymcdm
from pymcdm.methods.mcda_method import MCDA_method

from .agg_exp import *
from .tools import ExplanationModelEvaluator

from datetime import datetime
from dataclasses import dataclass

def evaluate_aggregate_explainer(
        clf, X_train, X_test, categorical_feature_names, predict_proba=None,
        explainer_components_sets: list[list[Type[ExplainerWrapper]]] = [[LimeWrapper, ShapTabularTreeWrapper, AnchorWrapper]],
        mcdm_algs: list[MCDA_method] = [pymcdm.methods.TOPSIS()],
        aggregation_algs: list[Literal["wsum", "w_bordafuse", "w_condorcet"]] = ["wsum"],
        metrics_sets: list[list[Literal['complexity', 'sensitivity_spearman', 'faithfulness_corr', "rb_faithfulness_corr", 'nrc']]] = [['nrc', 'sensitivity_spearman', 'rb_faithfulness_corr']],
        extra_explainer_params: dict = {},
        n_instances: int = 10, indexes: list[int] = None,
        mp_jobs = 10, **kwargs) -> list[list[pd.DataFrame]]:
    
    """
    Evaluate the aggregate explainer with various settings.
    
    This function evaluates the aggregate explainer by iterating over different combinations of explainer components,
    MCDM algorithms, aggregation algorithms, and metrics. It returns the results as a list of lists of dataframes,
    where each dataframe corresponds to an instance check, and each list of dataframes corresponds to a specific
    setting configuration.

    :param clf: The classifier model to be explained
    :type clf: object
    :param X_train: The training dataset
    :type X_train: pd.DataFrame
    :param X_test: The test dataset
    :type X_test: pd.DataFrame
    :param categorical_feature_names: List of names of categorical features
    :type categorical_feature_names: list[str]
    :param predict_proba: Function to predict probabilities. If None, clf.predict_proba is used
    :type predict_proba: callable, optional
    :param explainer_components_sets: List of lists of explainer components to be used in the aggregate explainer
    :type explainer_components_sets: list[list[Type[ExplainerWrapper]]], optional
    :param mcdm_algs: List of MCDM (Multi-Criteria Decision Making) algorithms to be used
    :type mcdm_algs: list[MCDA_method], optional
    :param aggregation_algs: List of aggregation algorithms to be used
    :type aggregation_algs: list[Literal["wsum", "w_bordafuse", "w_condorcet"]], optional
    :param metrics_sets: List of lists of metrics to be evaluated
    :type metrics_sets: list[list[Literal['complexity', 'sensitivity_spearman', 'faithfulness_corr', 'nrc']]], optional
    :param extra_explainer_params: Additional parameters for explainers
    :type extra_explainer_params: dict, optional
    :param n_instances: Number of instances to be evaluated. Default is 10
    :type n_instances: int, optional
    :param indexes: List of indexes of instances to be evaluated. If None, random instances are selected
    :type indexes: list[int], optional
    :param mp_jobs: Number of parallel jobs to be used. Default is 10
    :type mp_jobs: int, optional
    :param kwargs: Additional keyword arguments
    :type kwargs: dict
    
    :return: A list of lists of dataframes containing the evaluation results for each instance and setting configuration
    :rtype: list[list[pd.DataFrame]]
    """

    if predict_proba is None:
        predict_proba = clf.predict_proba

    if indexes is None:
        indexes = np.random.choice(X_test.index, n_instances, replace=False)
    
    print(f"Selected indexes: {indexes}")
    
    evaluator = ExplanationModelEvaluator(clf, X_train, categorical_feature_names, jobs=mp_jobs,
                                          noise_gen_args=extra_explainer_params.get("noise_gen_args", {}))
    evaluator.init()

    s_lbd: Callable[[AggregatedExplainer, pd.Series | np.ndarray], pd.DataFrame] = lambda explainer, instance_data_row: evaluator._sensitivity_sequential(
        explainer, instance_data_row,
        extra_explainer_params={
            "explainer_types": explainer.explainer_types,
            "evaluator": explainer.xai_evaluator,
            "mcdm_method": explainer.mcdm_method,
            "aggregation_algorithm": explainer.aggregation_algorithm,
            "metrics": explainer.metrics
        })
    
    metrics_functions_setup_rae = {                                     
        "faithfulness_corr": evaluator.faithfullness_correlation,
        "rb_faithfulness_corr": lambda explainer, instance_data_row: evaluator.faithfullness_correlation(explainer, instance_data_row, iterations=10, rank_based=True, rb_alg="percentile"),
        "sensitivity_spearman": s_lbd,
        "complexity": evaluator.complexity,
        "nrc": evaluator.nrc
    }

    metadata = {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "indexes": indexes,
        "configs": []
    }

    results = []
    i = 0
    for explainer_components in explainer_components_sets:
        for metrics in metrics_sets:
            for mcdm_alg in mcdm_algs:
                for aggregation_alg in aggregation_algs:
                    print(f"Running evaluation for settings {i + 1}/{len(explainer_components_sets) * len(metrics_sets) * len(mcdm_algs) * len(aggregation_algs)}")
                    
                    metadata["configs"].append({
                        "explainer_components": explainer_components,
                        "metrics": metrics,
                        "mcdm_alg": mcdm_alg,
                        "aggregation_alg": aggregation_alg
                    })
                    
                    explainer = AggregatedExplainer(model=clf, X_train=X_train, categorical_feature_names=categorical_feature_names, 
                                                    predict_proba=predict_proba, explainer_types=explainer_components, 
                                                    evaluator=evaluator, metrics=metrics, mcdm_method=mcdm_alg, 
                                                    aggregation_algorithm=aggregation_alg, **extra_explainer_params)
                    print(f"Explainer components: {explainer.explainer_types}, Metrics: {explainer.metrics}, MCDM algorithm: {explainer.mcdm_method}, Aggregation algorithm: {explainer.aggregation_algorithm}")
                    i += 1

                    settings_results = []

                    for index in indexes:
                        print("\t Running instance", index)
                        
                        explainer.explain_instance(X_test.loc[index])
                        instance_results = explainer.get_last_explanation_info().drop(columns=['weight'])

                        for metric in metrics:
                            instance_results.at["AggregateExplainer", metric] = metrics_functions_setup_rae[metric](explainer, X_test.loc[index])
                        
                        settings_results.append(instance_results)
                    
                    results.append(settings_results)
    
    return results, metadata

@dataclass
class ExperimentRun:
    """
    A dataclass representing a single experiment run.

    This class is used to store the metadata and results of an experiment run. It can be
    used to keep track of experimental settings, hyperparameters, and outcomes.

    :ivar metadata: A dictionary containing metadata about the experiment run, such as
                    hyperparameters, dataset information, model configurations, etc.
    :vartype metadata: dict
    :ivar results: The results of the experiment run. This can be any data type depending on
                   the specific experiment, such as metrics, model outputs, or evaluation results.
    :vartype results: any
    """
    metadata: dict
    results: any

def get_expconfig_mean_results(exp: ExperimentRun, config: int):
    """
    Calculate the mean results for a specific experiment configuration.
    
    This function aggregates all results for a given configuration in an experiment
    and computes the mean values grouped by the 0-level index.
    
    :param exp: The experiment run object containing results
    :type exp: ExperimentRun
    :param config: The configuration index to retrieve results for
    :type config: int
    :return: A DataFrame containing the mean values of all results for the specified
             configuration, grouped by the 0-level index
    :rtype: pandas.DataFrame
    """
    config_results = exp.results[config]
    return pd.concat(config_results).groupby(level=0).mean()


def count_worst_case_avoidances(config_results: list[pd.DataFrame], is_more_better: list[bool], 
                                not_avoidence_tolerance: int = 0, row_of_interest = "AggregateExplainer"):
    """
    Count the number of dataframes in which the specified row avoids the worst-case scenario 
    across all columns, with varying levels of tolerance.
    
    :param config_results: A list of pandas DataFrames containing the results to be analyzed.
    :type config_results: list[pd.DataFrame]
    :param is_more_better: A list of boolean values indicating whether a higher value is better (True) 
                          or a lower value is better (False) for each column.
    :type is_more_better: list[bool]
    :param not_avoidence_tolerance: The tolerance level for not avoiding the worst case. Default is 0.
    :type not_avoidence_tolerance: int, optional
    :param row_of_interest: The index of the row to be analyzed. Default is "AggregateExplainer".
    :type row_of_interest: str, optional
    :return: A list of counts where each element represents the number of dataframes in which the row 
            of interest avoids the worst-case scenario with the corresponding level of tolerance.
    :rtype: list[int]
    """
    
    counts = [0] * (not_avoidence_tolerance + 1)
    
    for instance_result in config_results:
        idx_max = instance_result.idxmax()
        idx_min = instance_result.idxmin()
        
        not_avoided_count = 0
        for col_i, is_better in enumerate(is_more_better):
            if is_better:
                if instance_result.loc[row_of_interest][col_i] == instance_result.loc[idx_min[col_i]][col_i]:
                    not_avoided_count += 1
            else:
                if instance_result.loc[row_of_interest][col_i] == instance_result.loc[idx_max[col_i]][col_i]:
                    not_avoided_count += 1
        
        for tolerance in range(not_avoidence_tolerance + 1):
            if not_avoided_count <= tolerance:
                counts[tolerance] += 1

    return counts


def get_average_metric_rank(config_results: list[pd.DataFrame], is_more_better: list[bool]):
    """
    Calculate the average rank of each method across multiple instances for each metric.
    
    For each instance in config_results, ranks are assigned to methods based on their performance
    for each metric, considering whether higher or lower values are better. Then, the average
    rank across all instances is calculated.
    
    :param config_results: List of DataFrames where each DataFrame contains the results for one instance.
                          Each DataFrame should have methods as indices and metrics as columns.
    :type config_results: list[pd.DataFrame]
    :param is_more_better: List of boolean values indicating whether higher values are better for each metric.
                          True means higher values are better, False means lower values are better.
                          Length should match the number of columns in each DataFrame in config_results.
    :type is_more_better: list[bool]
    :return: DataFrame containing the average rank of each method across all instances.
             Methods are in the index, and metrics are in the columns.
    :rtype: pd.DataFrame
    """
    ranks = []

    for instance_result in config_results:
        instance_rank = []
        for col_i, is_better in enumerate(is_more_better):
            if is_better:
                instance_rank.append(instance_result.iloc[:, col_i].rank(ascending=False))
            else:
                instance_rank.append(instance_result.iloc[:, col_i].rank(ascending=True))
                
        instance_rank = pd.concat(instance_rank, axis=1)
        ranks.append(instance_rank)
    
    avg_ranks = pd.concat(ranks).groupby(level=0).mean()
    
    return avg_ranks

from IPython.display import display

def present_experiment_run(exp: ExperimentRun, labels: list[Any], is_more_better: list[bool] = [False, True, True]):
    """
    Presents and analyzes results from an experiment run.

    This function displays the results for each method in the experiment, calculates worst-case
    avoidances, shows the average results, and computes average metric rankings.
    
    :param exp: The experiment run object containing results to be presented
    :type exp: ExperimentRun
    :param labels: Labels for the different methods being compared in the experiment
    :type labels: list
    :param is_more_better: List of boolean flags indicating for each metric whether higher values are better.
                          Defaults to [False, True, True]
    :type is_more_better: list[bool], optional
    :return: This function only prints and displays results, it doesn't return any value
    :rtype: None
    
    .. note::
        The function displays various performance metrics and statistics for each method:
        
        - Raw results
        - Worst case avoidance counts (for all metrics and for 2/3 of metrics)
        - Average results across experiment configurations  
        - Average rank for each metric
    """
    for i, method in enumerate(labels):
        print(f"{method}:\n")
        display(exp.results[i])
        wca = count_worst_case_avoidances(exp.results[i], is_more_better, 1)
        print(f"Worst case avoidances:\n\t- for all metrics: {wca[0]}\n\t- for 2/3 metrics: {wca[1]}")
        print("AVG:")
        display(get_expconfig_mean_results(exp, i))
        print("\n")
        print("Avg rank:")
        display(get_average_metric_rank(exp.results[i], is_more_better))