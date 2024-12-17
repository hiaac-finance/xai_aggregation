from typing import Literal
import pandas as pd
import pymcdm
from pymcdm.methods.mcda_method import MCDA_method

from .agg_exp import *
from .tools import ExplanationModelEvaluator

def evaluate_aggregate_explainer(
        clf, X_train, X_test, categorical_feature_names, predict_proba=None,
        explainer_components_sets: list[list[Type[ExplainerWrapper]]] = [[LimeWrapper, ShapTabularTreeWrapper, AnchorWrapper]],
        mcdm_algs: list[MCDA_method] = [pymcdm.methods.TOPSIS()],
        aggregation_algs: list[Literal["wsum", "w_bordafuse", "w_condorcet"]] = ["wsum"],
        metrics_sets: list[list[Literal['complexity', 'sensitivity_spearman', 'faithfulness_corr', 'nrc']]] = [['nrc', 'sensitivity_spearman', 'faithfulness_corr']],
        extra_explainer_params: dict = {},
        n_instances: int = 10, indexes: list[int] = None,
        random_state: int = 42, mp_jobs = 10, **kwargs) -> list[list[pd.DataFrame]]:
    
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
    random_state : int, optional
        Random state for reproducibility. Default is 42.
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
    
    evaluator = ExplanationModelEvaluator(clf, X_train, categorical_feature_names, jobs=mp_jobs)
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
    
    metrics_calls_setup = {
        "faithfulness_corr": evaluator.faithfullness_correlation,
        "sensitivity_spearman": s_lbd,
        "complexity": evaluator.complexity,
        "nrc": evaluator.nrc
    }

    results = []
    i = 0
    for explainer_components in explainer_components_sets:
        for metrics in metrics_sets:
            for mcdm_alg in mcdm_algs:
                for aggregation_alg in aggregation_algs:
                    print(f"Running evaluation for settings {i + 1}/{len(explainer_components_sets) * len(metrics_sets) * len(mcdm_algs) * len(aggregation_algs)}")
                    
                    explainer = AggregatedExplainer(clf=clf, X_train=X_train, categorical_feature_names=categorical_feature_names, 
                                                    predict_proba=predict_proba, explainer_types=explainer_components, 
                                                    evaluator=evaluator, metrics=metrics, mcdm_method=mcdm_alg, 
                                                    aggregation_algorithm=aggregation_alg, **kwargs)
                    print(f"Explainer components: {explainer.explainer_types}, Metrics: {explainer.metrics}, MCDM algorithm: {explainer.mcdm_method}, Aggregation algorithm: {explainer.aggregation_algorithm}")
                    i += 1

                    settings_results = []

                    for index in indexes:
                        print("\t Running instance", index)
                        
                        explainer.explain_instance(X_train.iloc[index])
                        instance_results = explainer.get_last_explanation_info().drop(columns=['weight'])

                        for metric in metrics:
                            instance_results.at["AggregateExplainer", metric] = metrics_calls_setup[metric](explainer, X_train.iloc[index])
                        
                        settings_results.append(instance_results)
                    
                    results.append(settings_results)
    
    return results