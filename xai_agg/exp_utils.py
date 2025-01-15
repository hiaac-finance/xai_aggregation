from dataclasses import dataclass
import pandas as pd

from xai_agg import *

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