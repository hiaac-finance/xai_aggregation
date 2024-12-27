from dataclasses import dataclass
import pandas as pd

@dataclass
class ExperimentRun:
    metadata: dict
    results: any

def get_expconfig_mean_results(exp: ExperimentRun, config: int):
    config_results = exp.results[config]
    return pd.concat(config_results).groupby(level=0).mean()


def count_worst_case_avoidances(config_results: list[pd.DataFrame], is_more_better: list[bool], row_of_interest = "AggregateExplainer"):
    """Count in how many dataframes the columns for the row whose index is "AggregateExplainer" avoid the worst case among all the other rows"""
    count = 0
    for instance_result in config_results:
        idx_max = instance_result.idxmax()
        idx_min = instance_result.idxmin()
        
        avoids_worst_case_in_all_columns = True
        for col_i, is_better in enumerate(is_more_better):
            
            if is_better:
                if instance_result.loc[row_of_interest][col_i] == instance_result.loc[idx_min[col_i]][col_i]:
                    avoids_worst_case_in_all_columns = False
                    break
            else:
                if instance_result.loc[row_of_interest][col_i] == instance_result.loc[idx_max[col_i]][col_i]:
                    avoids_worst_case_in_all_columns = False
                    break
        
        if avoids_worst_case_in_all_columns:
            count += 1

    return count
    