import shap
from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular # why not used the original anchor package?

import numpy as np
import pandas as pd

import pandera as pa
from pandera.typing import DataFrame

from sklearn.base import is_classifier, is_regressor
from typing import Literal, Any

class ExplanationModel(pa.DataFrameModel):
    feature: str
    score: float = pa.Field(ge=0)

"""
This script defines the ExplainerWrapper abstract base class and its subclasses, which are used to wrap the explainer classes,
such as the ones from the LIME, SHAP, and Anchor libraries.
"""

class ExplainerWrapper:
    """
    Wrapper abstract base class for featuer-importance-based explainer classes.
    This class ensures that all explainer classes have the same interface, making it easier to use them interchangeably in the AggregatedExplainer class.
    """

    def __init__(self, model: any, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str] = [], predict_fn: callable = None,
                 mode: Literal["classification", "regression", "auto"] = "auto"):
        self.model = model

        if predict_fn is None:
            if hasattr(self.model, 'predict_proba'):
                self.predict_fn = self.model.predict_proba
            elif hasattr(self.model, 'predict'):
                self.predict_fn = self.model.predict
            else:
                raise ValueError('Could not find a predict or predict_proba method in the model. Please provide a value for the predict_fn parameter.')
        else:
            self.predict_fn = predict_fn
        
        if mode == "auto":
            if is_classifier(self.model):
                self.mode = "classification"
            elif is_regressor(self.model):
                self.mode = "regression"
            else:
                raise ValueError("Could not determine the mode of the model. Please provide a value for the mode parameter.")

        self.X_train = X_train
        self.categorical_feature_names = categorical_feature_names
    
    def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> DataFrame[ExplanationModel]:
        """
        Explains the prediction of a single instance. Must return a DataFrame with two columns: 'feature' and 'score',
        where 'feature' is the name of the feature and 'score' is the absolute feature importance score.
        """
        pass

class LimeWrapper(ExplainerWrapper):

    def __init__(self, model: any, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str] = [], predict_fn: callable = None,
                 mode: Literal["classification", "regression", "auto"] = "auto"):
        super().__init__(model, X_train, categorical_feature_names, predict_fn=predict_fn, mode=mode)
        
        self.explainer = LimeTabularExplainer(self.X_train.values, feature_names=self.X_train.columns, discretize_continuous=False, mode=self.mode)
    
    def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> DataFrame[ExplanationModel]:
        lime_exp = self.explainer.explain_instance(np.array(instance_data_row), self.predict_fn, num_features=len(self.X_train.columns))
        
        ranking = pd.DataFrame(lime_exp.as_list(), columns=['feature', 'score'])
        ranking['score'] = ranking['score'].apply(lambda x: abs(x))
        return DataFrame[ExplanationModel](ranking)

class ShapTabularTreeWrapper(ExplainerWrapper):
    
    def __init__(self, model: Any, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str] = [],
                 predict_fn: callable = None, **additional_explainer_args):
        super().__init__(model, X_train, categorical_feature_names, predict_fn=predict_fn)
        
        self.explainer = shap.TreeExplainer(self.model,
                                            data=self.X_train if self.mode == "regression" else None,
                                            **additional_explainer_args)
    
    def explain_instance(self, instance_data_row: np.ndarray) -> DataFrame[ExplanationModel]:
        if isinstance(instance_data_row, pd.Series):
            instance_data_row = instance_data_row.to_numpy()
        
        shap_values = self.explainer.shap_values(instance_data_row)
        if self.mode == "classification":
            predicted_class = np.argmax(self.predict_fn(instance_data_row.reshape(1, -1))) # Only grab shap values for the predicted class, mirroring lime behavior
            attributions = shap_values[:, predicted_class]
        elif self.mode == "regression":
            attributions = shap_values
        
        ranking = pd.DataFrame(list(zip(self.X_train.columns, attributions)), columns=['feature', 'score'])
        ranking = ranking.sort_values(by='score', ascending=False, key=lambda x: abs(x)).reset_index(drop=True)
        ranking['score'] = ranking['score'].apply(lambda x: abs(x))
        return DataFrame[ExplanationModel](ranking)

class AnchorWrapper(ExplainerWrapper):
    """
    Anchor is not a feature-importance-based explainer, but it can be used to generate feature importance scores based on the rules that it generates.
    This is done by calculating the coverage of each rule and assigning a score to each feature based on the coverage of the rules that don't reference it:
    the lower the coverage, the more impactful the feature is, thus the higher the score.
    """

    def __init__(self, model: Any, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str] = [], predict_fn: callable = None,
                 mode: Literal["classification", "regression", "auto"] = "auto"):
        super().__init__(model, X_train, categorical_feature_names, predict_fn=predict_fn, mode=mode)
        
        self.explainer = AnchorTabular(predictor=self.predict_fn, feature_names=self.X_train.columns) # TODO: fix parameters
        self.explainer.fit(self.X_train.values)
    
    def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> DataFrame[ExplanationModel]:
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
            try:
                rule_coverage = self.X_train.query(rule).shape[0] / self.X_train.shape[0]
            except SyntaxError:
                raise ValueError(f"[AnchorWrapper explainer]: Rule '{rule}' could not be parsed. Make sure your data columns are valid python variable names/identifiers" + 
                                 " (i.e. they don't start with a number and don't contain spaces or special characters). Please refer to the usage examples.")
            feature_importances[referenced_feature] = 1 - rule_coverage
        
        ranking = pd.DataFrame(list(feature_importances.items()), columns=['feature', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)
        return DataFrame[ExplanationModel](ranking)
