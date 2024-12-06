import pandas as pd

def get_ranking_df(ranking_df: pd.DataFrame, epsilon: float = 0.01) -> pd.DataFrame:
    # If two features have a score difference smaller than epsilon, they are considered to have the same rank
    rank = pd.DataFrame(columns=['Feature', 'Rank'])
    rank['Feature'] = ranking_df["feature"]
    
    # Sort the ranking dataframe by score in descending order
    ranking_df = ranking_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    current_rank = 1
    rank['Rank'] = 0
    rank.at[0, 'Rank'] = current_rank
    
    for i in range(1, len(ranking_df)):
        if abs(ranking_df.at[i, 'score'] - ranking_df.at[i-1, 'score']) < epsilon:
            rank.at[i, 'Rank'] = current_rank
        else:
            current_rank += 1
            rank.at[i, 'Rank'] = current_rank
    
    return rank