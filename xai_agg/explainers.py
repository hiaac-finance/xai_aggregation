import shap
from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular # why not used the original anchor package?

import numpy as np
import pandas as pd

"""
This script defines the ExplainerWrapper abstract base class and its subclasses, which are used to wrap the explainer classes,
such as the ones from the LIME, SHAP, and Anchor libraries.
"""

class ExplainerWrapper:
    """
    Wrapper abstract base class for featuer-importance-based explainer classes.
    This class ensures that all explainer classes have the same interface, making it easier to use them interchangeably in the AggregatedExplainer class.
    """

    def __init__(self, clf, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str], predict_proba: callable = None):
        self.clf = clf

        if hasattr(clf, 'predict_proba') and predict_proba is None:
            self.predict_proba = clf.predict_proba
        elif predict_proba is not None:
            self.predict_proba = predict_proba
        else:
            raise ValueError('The classifier does not have a predict_proba method and no predict_proba_function was provided.')

        self.X_train = X_train
        self.categorical_feature_names = categorical_feature_names

    
    def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> pd.DataFrame:
        """
        Explains the prediction of a single instance. Must return a DataFrame with two columns: 'feature' and 'score',
        where 'feature' is the name of the feature and 'score' is the absolute feature importance score.
        """
        pass

class LimeWrapper(ExplainerWrapper):

    def __init__(self, clf, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str], predict_proba: callable = None):
        super().__init__(clf, X_train, categorical_feature_names, predict_proba)
        
        self.explainer = LimeTabularExplainer(self.X_train.values, feature_names=self.X_train.columns, discretize_continuous=False)
    
    def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> pd.DataFrame:
        lime_exp = self.explainer.explain_instance(instance_data_row, self.predict_proba, num_features=len(self.X_train.columns))
        
        ranking = pd.DataFrame(lime_exp.as_list(), columns=['feature', 'score'])    
        ranking['score'] = ranking['score'].apply(lambda x: abs(x))
        return ranking

class ShapTabularTreeWrapper(ExplainerWrapper):
    
        def __init__(self, clf, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str], predict_proba: callable = None, **additional_explainer_args):
            super().__init__(clf, X_train, categorical_feature_names, predict_proba)
            
            self.explainer = shap.TreeExplainer(clf, self.X_train, **additional_explainer_args)
        
        def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> pd.DataFrame:
            shap_values = self.explainer.shap_values(instance_data_row)
    
            ranking = pd.DataFrame(list(zip(self.X_train.columns, shap_values[:, 0])), columns=['feature', 'score'])
            ranking = ranking.sort_values(by='score', ascending=False, key=lambda x: abs(x)).reset_index(drop=True)
            ranking['score'] = ranking['score'].apply(lambda x: abs(x))
            return ranking

class AnchorWrapper(ExplainerWrapper):
    """
    Anchor is not a feature-importance-based explainer, but it can be used to generate feature importance scores based on the rules that it generates.
    This is done by calculating the coverage of each rule and assigning a score to each feature based on the coverage of the rules that don't reference it:
    the lower the coverage, the more impactful the feature is, thus the higher the score.
    """

    def __init__(self, clf, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str], predict_proba: callable = None):
        super().__init__(clf, X_train, categorical_feature_names, predict_proba)
        
        self.explainer = AnchorTabular(predictor=self.predict_proba, feature_names=self.X_train.columns) # TODO: fix parameters
        self.explainer.fit(self.X_train.values)
    
    def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> pd.DataFrame:
        if isinstance(instance_data_row, pd.Series):
            instance_data_row = instance_data_row.to_numpy()

        feature_importances = {feature: 0 for feature in self.X_train.columns}
        explanation = self.explainer.explain(instance_data_row)
        
        for rule in explanation.anchor:
            # Extract the feature name from the rule string
            # This method won't work for column names that have spaces in them or that don't contain any letters
            for expression_element in rule.split():
                if any(c.isalpha() for c in expression_element):
                    referenced_feature = expression_element
                    break

            rule_coverage = self.X_train.query(rule).shape[0] / self.X_train.shape[0]
            feature_importances[referenced_feature] = 1 - rule_coverage
        
        return pd.DataFrame(list(feature_importances.items()), columns=['feature', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)
