"""
Explainer Wrappers
==================

This module provides a consistent interface to various explainable AI (XAI) algorithms
by wrapping them in the :class:`ExplainerWrapper` abstract base class and its
subclasses. It offers uniform access to different explanation techniques, making them
interchangeable in higher-level components like the aggregated explainer.

Classes
-------
   - ExplanationModel
   - ExplainerWrapper
   - LimeWrapper
   - ShapTabularTreeWrapper
   - ShapTabularKernelWrapper
   - AnchorWrapper

Supported Explainer Types
-------------------------

- **LIME**: Local Interpretable Model-agnostic Explanations
- **SHAP Tree**: Tree-based implementation of SHAP (SHapley Additive exPlanations)  
- **SHAP Kernel**: Kernel-based implementation of SHAP for any model
- **Anchor**: Rule-based explanations with high precision

Examples
--------

.. code-block:: python

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from xai_agg.explainers import LimeWrapper, ShapTabularTreeWrapper
    
    # Train a model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Create a LIME explainer
    lime_explainer = LimeWrapper(model=model, X_train=X_train)
    
    # Explain a prediction
    instance = X_test.iloc[0]
    explanation = lime_explainer.explain_instance(instance)
    print(explanation)
"""

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

class ExplainerWrapper:
    """
    Abstract base class providing a uniform interface for feature-importance explainers.
    
    This class serves as a foundation for wrapping various explainable AI (XAI) algorithms,
    ensuring they share a common interface. This standardization makes different explanation
    techniques interchangeable in higher-level components like the AggregatedExplainer.
    
    :param model: The machine learning model to be explained
    :type model: Any
    :param X_train: Training data used to initialize the explainer
    :type X_train: pandas.DataFrame or numpy.ndarray
    :param categorical_feature_names: Names of categorical features in the dataset
    :type categorical_feature_names: list[str], optional
    :param predict_fn: Custom prediction function, defaults to model.predict_proba or model.predict
    :type predict_fn: callable, optional
    :param mode: Prediction type ('classification', 'regression', or 'auto'), defaults to 'auto'
    :type mode: Literal["classification", "regression", "auto"], optional
    :param abs_score: Whether to return absolute values for feature importance scores, defaults to True
    :type abs_score: bool, optional
    :param additional_args: Additional arguments to pass to the underlying explainer
    :type additional_args: dict, optional
    
    :ivar model: The machine learning model being explained
    :ivar predict_fn: Function used to make predictions
    :ivar mode: Prediction type ('classification' or 'regression')
    :ivar X_train: Training data used to initialize the explainer
    :ivar categorical_feature_names: Names of categorical features
    :ivar abs_score: Whether absolute values are used for feature importance scores
    :ivar additional_args: Additional arguments passed to the underlying explainer
    
    .. note::
        Subclasses must implement the `explain_instance` method to provide
        specific explanation functionality.
    """

    def __init__(self, model: any, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str] = [], predict_fn: callable = None,
                 mode: Literal["classification", "regression", "auto"] = "auto", abs_score: bool=True, **additional_args):
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
        else:
            self.mode = mode

        self.X_train = X_train
        self.categorical_feature_names = categorical_feature_names
        
        self.abs_score = abs_score
        
        self.additional_args = additional_args
    
    def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> DataFrame[ExplanationModel]:
        """
        Explains the prediction for a single instance.
        
        This abstract method must be implemented by subclasses to provide specific
        explanation functionality. The implementation should process the instance
        and return feature importance scores.
        
        :param instance_data_row: The instance to explain
        :type instance_data_row: pandas.Series or numpy.ndarray
        :return: A DataFrame with two columns: 'feature' (feature names) and 'score' 
                (feature importance values)
        :rtype: DataFrame[ExplanationModel]
        
        .. note::
            If self.abs_score is True, all feature importance scores should be
            returned as absolute values.
        """
        pass

class LimeWrapper(ExplainerWrapper):

    def __init__(self, model: any, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str] = [], predict_fn: callable = None,
                 mode: Literal["classification", "regression", "auto"] = "auto", abs_score: bool=True, **additional_args):
        super().__init__(model, X_train, categorical_feature_names, predict_fn=predict_fn, mode=mode, abs_score=abs_score, additional_agrs=additional_args)
        
        self.explainer = LimeTabularExplainer(self.X_train.values, feature_names=self.X_train.columns, discretize_continuous=False, mode=self.mode)
    
    def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> DataFrame[ExplanationModel]:
        lime_exp = self.explainer.explain_instance(np.array(instance_data_row), self.predict_fn, num_features=len(self.X_train.columns))
        
        ranking = pd.DataFrame(lime_exp.as_list(), columns=['feature', 'score'])
        if self.abs_score:
            ranking['score'] = ranking['score'].apply(lambda x: abs(x))
        return DataFrame[ExplanationModel](ranking)

class ShapTabularTreeWrapper(ExplainerWrapper):
    
    def __init__(self, model: Any, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str] = [],
                 predict_fn: callable = None, abs_score: bool = True, **additional_args):
        super().__init__(model, X_train, categorical_feature_names, predict_fn=predict_fn, abs_score=abs_score, additional_args=additional_args)
        
        self.explainer = shap.TreeExplainer(self.model,
                                            data=self.X_train)
        
        self.on_noise = self.additional_args.get('on_noise', False)
    
    def explain_instance(self, instance_data_row: np.ndarray) -> DataFrame[ExplanationModel]:
        if isinstance(instance_data_row, pd.Series):
            instance_data_row = instance_data_row.to_numpy()
        
        shap_values = self.explainer.shap_values(instance_data_row, check_additivity=False) # when applying on noisy data, additivity check will fail, because the explainer's data wont match the model's
        if self.mode == "classification":
            predicted_class = np.argmax(self.predict_fn(instance_data_row.reshape(1, -1))) # Only grab shap values for the predicted class, mirroring lime behavior
            attributions = shap_values[:, predicted_class]
        elif self.mode == "regression":
            attributions = shap_values
        
        ranking = pd.DataFrame(list(zip(self.X_train.columns, attributions)), columns=['feature', 'score'])
        ranking = ranking.sort_values(by='score', ascending=False, key=lambda x: abs(x)).reset_index(drop=True)
        if self.abs_score:
            ranking['score'] = ranking['score'].apply(lambda x: abs(x))
        return DataFrame[ExplanationModel](ranking)

class ShapTabularKernelWrapper(ExplainerWrapper):
    
    def __init__(self, model: Any, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str] = [],
                 predict_fn: callable = None, abs_score: bool = True, **additional_args):
        super().__init__(model, X_train, categorical_feature_names, predict_fn=predict_fn, abs_score=abs_score, additional_args=additional_args)
        
        self.explainer = shap.KernelExplainer(self.predict_fn, data=self.X_train, **additional_args)
    
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
        if self.abs_score:
            ranking['score'] = ranking['score'].apply(lambda x: abs(x))
        return DataFrame[ExplanationModel](ranking)

class AnchorWrapper(ExplainerWrapper):
    """
    Wrapper for the Anchor explainer, converting rule-based explanations to feature importance scores.
    
    Anchor typically generates rule-based explanations, not feature importance scores. This wrapper
    converts Anchor's rules into a feature importance format by calculating the coverage of each rule
    and assigning scores to features based on rule coverage: the lower the coverage, the more
    impactful the feature is considered to be (thus the higher the score).
    
    :param model: The machine learning model to be explained
    :type model: Any
    :param X_train: Training data used to initialize the explainer
    :type X_train: pandas.DataFrame or numpy.ndarray
    :param categorical_feature_names: Names of categorical features in the dataset
    :type categorical_feature_names: list[str], optional
    :param predict_fn: Custom prediction function, defaults to model.predict_proba or model.predict
    :type predict_fn: callable, optional
    :param mode: Prediction type ('classification', 'regression', or 'auto'), defaults to 'auto'
    :type mode: Literal["classification", "regression", "auto"], optional
    :param abs_score: Whether to return absolute values for feature importance scores, defaults to True
    :type abs_score: bool, optional
    :param additional_args: Additional arguments to pass to the underlying explainer
    :type additional_args: dict, optional
    
    :raises ValueError: If rule parsing fails due to invalid column names in the data
    
    .. note::
       This wrapper requires data column names to be valid Python variable names/identifiers.
       Column names should not start with numbers and should not contain spaces or 
       special characters that would make them invalid in Python syntax.
       
    .. warning::
       The rule extraction method may not work correctly for column names that have
       spaces or don't contain any alphabetic characters.
    """

    def __init__(self, model: Any, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str] = [], predict_fn: callable = None,
                 mode: Literal["classification", "regression", "auto"] = "auto", abs_score: bool=True, **additional_args):
        super().__init__(model, X_train, categorical_feature_names, predict_fn=predict_fn, mode=mode, abs_score=abs_score, additional_args=additional_args)
        
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
