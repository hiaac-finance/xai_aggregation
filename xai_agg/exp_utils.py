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

    Parameters:
    -----------
    clf : object
        The classifier model to be explained.
    X_train : pd.DataFrame
        The training dataset.
    X_test : pd.DataFrame
        The test dataset.
    categorical_feature_names : list[str]
        List of names of categorical features.
    predict_proba : callable, optional
        Function to predict probabilities. If None, clf.predict_proba is used.
    explainer_components_sets : list[list[Type[ExplainerWrapper]]], optional
        List of lists of explainer components to be used in the aggregate explainer.
    mcdm_algs : list[MCDA_method], optional
        List of MCDM (Multi-Criteria Decision Making) algorithms to be used.
    aggregation_algs : list[Literal["wsum", "w_bordafuse", "w_condorcet"]], optional
        List of aggregation algorithms to be used.
    metrics_sets : list[list[Literal['complexity', 'sensitivity_spearman', 'faithfulness_corr', 'nrc']]], optional
        List of lists of metrics to be evaluated.
    n_instances : int, optional
        Number of instances to be evaluated. Default is 10.
    indexes : list[int], optional
        List of indexes of instances to be evaluated. If None, random instances are selected.
    mp_jobs : int, optional
        Number of parallel jobs to be used. Default is 10.
    **kwargs : dict
        Additional keyword arguments.
        
    Returns:
    --------
    list[list[pd.DataFrame]]
        A list of lists of dataframes containing the evaluation results for each instance and setting configuration.
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
    metadata: dict
    results: any

def get_expconfig_mean_results(exp: ExperimentRun, config: int):
    config_results = exp.results[config]
    return pd.concat(config_results).groupby(level=0).mean()


def count_worst_case_avoidances(config_results: list[pd.DataFrame], is_more_better: list[bool], 
                                not_avoidence_tolerance: int = 0, row_of_interest = "AggregateExplainer"):
    """
    Count the number of dataframes in which the specified row avoids the worst-case scenario 
    across all columns, with varying levels of tolerance.
    
    Parameters:
    -----------
    config_results : list[pd.DataFrame]
        A list of pandas DataFrames containing the results to be analyzed.
    is_more_better : list[bool]
        A list of boolean values indicating whether a higher value is better (True) or a lower value is better (False) for each column.
    not_avoidence_tolerance : int, optional
        The tolerance level for not avoiding the worst case. Default is 0.
    row_of_interest : str, optional
        The index of the row to be analyzed. Default is "AggregateExplainer".
        
    Returns:
    --------
    counts : list[int]
        A list of counts where each element represents the number of dataframes in which the row of interest 
        avoids the worst-case scenario with the corresponding level of tolerance.
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