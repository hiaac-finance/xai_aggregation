"""
Internal Tools
==============

This module provides tools and utilities for evaluating explanations and
generating test data used by the AggregatedExplainer class.
"""

import time
from typing import Literal, Type, Callable

import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Explainable AI tools:
from .explainers import *

from scipy.stats import spearmanr, pearsonr

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Concunrrency:
import concurrent.futures
from pathos.multiprocessing import ProcessingPool as Pool


class AutoencoderNoisyDataGenerator():
    """
    A class for generating noisy variations of tabular datasets.

    This class generates noisy data by swapping the values of a small number of features 
    between a sample and a randomly selected close neighbor. The neighbors are determined 
    by reducing the dimensionality of the data using an autoencoder and applying the 
    NearestNeighbors algorithm in the reduced space.
    
    :param X: The original dataset to generate noisy variations from
    :type X: pandas.DataFrame
    :param ohe_categorical_features_names: Names of categorical features that were one-hot encoded
    :type ohe_categorical_features_names: list[str]
    :param encoding_dim: Dimension of the encoded representation, defaults to 5
    :type encoding_dim: int, optional
    :param epochs: Number of epochs to train the autoencoder, defaults to 500
    :type epochs: int, optional
    :param percent_features_to_replace: Fraction of features to replace with noisy values, defaults to 0.1
    :type percent_features_to_replace: float, optional
    """

    def __init__(self, X: pd.DataFrame, ohe_categorical_features_names: list[str], encoding_dim: int = 5, epochs=500, percent_features_to_replace: float = 0.1):
        self.X = X
        self.categorical_features_names = ohe_categorical_features_names
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.replace_features_percent = percent_features_to_replace

        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        
        input_dim = self.X_scaled.shape[1]

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        self.autoencoder = Model(inputs=input_layer, outputs=decoded)
        self.encoder = Model(inputs=input_layer, outputs=encoded)
        self.was_fit = False
        
    
    def fit(self):
        """
        Train the autoencoder model on the dataset.
        
        This method must be called before generating noisy data.
        """
        self.autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
        self.autoencoder.fit(self.X_scaled, self.X_scaled, epochs=self.epochs, batch_size=32, shuffle=True, validation_split=0.2)
        # Extract hidden layer representation:
        self.hidden_representation = self.encoder.predict(self.X_scaled)
        self.was_fit = True


    def generate_noisy_data(self, num_features_to_replace: int = None) -> pd.DataFrame:
        """
        Generate a DataFrame containing a noisy variation of the data.

        Noise is introduced by swapping values of features between a sample and a 
        randomly selected close neighbor. Neighbors are determined using an autoencoder 
        to reduce dimensionality, followed by the NearestNeighbors algorithm.

        :param num_features_to_replace: Number of features to replace with noisy values
                                       If not provided, defaults to a percentage of total features
        :type num_features_to_replace: int, optional
        :return: DataFrame containing the noisy variation of the original data
        :rtype: pandas.DataFrame
        :raises ValueError: If the autoencoder has not been fitted
        """

        if not self.was_fit:
            raise ValueError('The autoencoder has not been fitted yet. Call the fit() method before generating noisy data.')
        
        if not num_features_to_replace:
            num_features_to_replace = math.ceil(self.X.shape[1] * self.replace_features_percent)

        # Compute Nearest Neighbors using hidden_representation
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(self.hidden_representation)
        distances, indices = nbrs.kneighbors(self.hidden_representation)

        X_noisy = self.X.copy()

        # Get id's of columns that belong to the same categorical feature (after being one-hot-encodeded);
        # Columns that belong to the same categorical feature start with the same name, and will be treated as a single feature when adding noise.
        categorical_features_indices = [
            [self.X.columns.get_loc(col_name) for col_name in self.X.columns if col_name.startswith(feature)]
            for feature in self.categorical_features_names
        ]

        # Replace features with random neighbor's features
        for i in range(self.X.shape[0]):  # Iterate over each sample
            available_features_to_replace = list(range(self.X.shape[1]))
            for j in range(num_features_to_replace):
                # Select features to replace; if the feture selected belong to one of the lists in categorical_features_indices, we will replace all the features in that list
                features_to_replace = np.random.choice(available_features_to_replace, 1)
                for feature_indices in categorical_features_indices:
                    if features_to_replace in feature_indices:
                        features_to_replace = feature_indices
                        break
                
                # Remove the selected features from the list of available features to replace
                available_features_to_replace = [f for f in available_features_to_replace if f not in features_to_replace]

                # Choose a random neighbor from the nearest neighbors
                neighbor_idx = np.random.choice(indices[i][1:])

                # Replace the selected features with the neighbor's features
                X_noisy.iloc[i, features_to_replace] = self.X.iloc[neighbor_idx, features_to_replace]
                
        return X_noisy

class ExplanationModelEvaluator:
    """
    A class for evaluating explanation models using multiple performance metrics.
    
    This class defines metrics to evaluate an explanation model's performance on a given data instance.
    This class must be initialized before use by calling the ``init()`` method.

    :param model: The classifier model to be explained
    :type model: object
    :param X_train: The training data used to train the classifier
    :type X_train: pandas.DataFrame or numpy.ndarray
    :param ohe_categorical_feature_names: Names of categorical features that were one-hot encoded
    :type ohe_categorical_feature_names: list[str], optional
    :param predict_fn: Custom prediction function, defaults to model.predict_proba or model.predict
    :type predict_fn: callable, optional
    :param noise_gen_args: Arguments to be passed to the AutoencoderNoisyDataGenerator
    :type noise_gen_args: dict, optional
    :param debug: Whether to enable debug mode, defaults to False
    :type debug: bool, optional
    :param jobs: Number of parallel jobs to use, defaults to None
    :type jobs: int, optional
    
    .. note::
        The class provides the following evaluation metrics:
        
        * **Faithfulness correlation**: Correlation between feature importance and the effect of feature perturbation
        * **Sensitivity**: Stability of explanations when input data is slightly perturbed
        * **Complexity**: Entropy of the explanation's feature importance distribution
        * **Non-redundant contribution (NRC)**: Metric assessing explanation conciseness and coherence
    """

    IS_METRIC_HIGHER_BETTER = {
        "complexity": False,
        "sensitivity_spearman": True,
        "faithfulness_corr": True,
        "nrc": False,
        "nrc_old": False,
        "rb_faithfulness_corr": True
    }

    def __init__(self, model, X_train: pd.DataFrame | np.ndarray, ohe_categorical_feature_names: list[str] = [], predict_fn: callable = None,
                 noise_gen_args: dict = {}, debug=False, jobs = None, **kwargs):
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

        self.X_train = X_train
        self.ohe_categorical_feature_names = ohe_categorical_feature_names

        self.categorical_features_indices = [
            [self.X_train.columns.get_loc(col_name) for col_name in self.X_train.columns if col_name.startswith(feature)]
            for feature in self.ohe_categorical_feature_names
        ]

        self.noisy_data_generator = AutoencoderNoisyDataGenerator(X_train, ohe_categorical_feature_names, **noise_gen_args)
        self.jobs = jobs

        self.debug = debug
        self.was_initialized = False
        
    
    def init(self):
        """
        Initialize the evaluator.
        
        This method trains the noisy data generator and must be called before
        using any evaluation methods.
        """
        self.noisy_data_generator.fit()
        self.was_initialized = True
            
    def faithfullness_correlation(self, explainer: ExplainerWrapper | Type[ExplainerWrapper], instance_data_row: pd.Series, len_subset: int = None,
                                  iterations: int = 100, baseline_strategy: Literal["zeros", "mean"] = "zeros", rank_based = False,
                                  rb_alg: Literal["sum", "percentile", "avg", "inverse"] = "inverse", explanation: DataFrame[ExplanationModel] = None) -> float:
        """
        Measure correlation between feature importance and model output changes when features are perturbed.
        
        This metric evaluates how well the explanation's feature importance scores align with
        the actual impact of those features on the model's predictions.
        
        :param explainer: The explainer object or class to evaluate (must be provided if explanation is None)
        :type explainer: ExplainerWrapper or Type[ExplainerWrapper]
        :param instance_data_row: The instance to explain (must be provided if explanation is None)
        :type instance_data_row: pandas.Series
        :param len_subset: Number of features to perturb in each iteration (default: 25% of features)
        :type len_subset: int, optional
        :param iterations: Number of iterations for metric calculation
        :type iterations: int, optional
        :param baseline_strategy: Strategy for baseline values ("zeros" or "mean")
        :type baseline_strategy: Literal["zeros", "mean"], optional
        :param rank_based: Whether to use rank-based calculation
        :type rank_based: bool, optional
        :param rb_alg: Algorithm for rank-based calculation
        :type rb_alg: Literal["sum", "percentile", "avg", "inverse"], optional
        :param explanation: Pre-computed explanation (if provided, explainer won't be used)
        :type explanation: DataFrame[ExplanationModel], optional
        :return: Absolute Pearson correlation between feature importance and output changes
        :rtype: float
        
        .. note::
            "mean" baseline typically provides higher correlation values, but "zeros" is more conservative.
        """

        if explanation is None:
            if not isinstance(explainer, ExplainerWrapper):
                explainer = explainer(self.model, self.X_train, self.ohe_categorical_feature_names, predict_fn=self.predict_fn)
            
            explanation = explanation if explanation is not None else explainer.explain_instance(instance_data_row)

        importance_sums = []
        delta_fs = []
        
        # Done this way so it'll work both on regression and on classification
        # On sklearn classification, the prediction will be a list of probabilities for each class;
        # On sklearn regression, the prediction will be a single value.
        prediction = self.predict_fn(np.array(instance_data_row).reshape(1, -1))[0]
        predicted_index = np.argmax(prediction)
        f_x = prediction[predicted_index] if isinstance(prediction, (list, np.ndarray)) else prediction

        for _ in range(iterations):
            evaluation = self._evaluate_faithfullness_iteration(instance_data_row, explanation, f_x, predicted_index, len_subset, baseline_strategy, rank_based, rb_alg)
            importance_sums.append(evaluation[0])
            delta_fs.append(evaluation[1])
        
        return abs(pearsonr(importance_sums, delta_fs)[0])

    def _evaluate_faithfullness_iteration(self, instance_data_row, g_x, f_x, predicted_index, len_subset, baseline_strategy, rank_based: bool = False,
                                          rb_alg: Literal["sum", "percentile", "avg", "inverse"] = "inverse") -> tuple[float, float]:
        """
        Helper method to calculate a single faithfulness iteration.
        
        :param instance_data_row: The instance being explained
        :param g_x: The explanation for the instance
        :param f_x: The model's prediction for the instance
        :param predicted_index: Index of the predicted class
        :param len_subset: Number of features to perturb
        :param baseline_strategy: Strategy for generating baseline values
        :param rank_based: Whether to use rank-based calculation
        :param rb_alg: Algorithm for rank-based calculation
        :return: Tuple of (combined_importance, delta_f)
        :rtype: tuple[float, float]
        """
        subset = np.random.choice(instance_data_row.index.values, len_subset if len_subset else len(instance_data_row) // 4, replace=False)
        perturbed_instance = instance_data_row.copy()

        if baseline_strategy == "zeros":
            baseline = np.zeros(len(instance_data_row))
        elif baseline_strategy == "mean":
            baseline = np.mean(self.X_train, axis=0)
            for feature_index in self.categorical_features_indices:
                baseline[feature_index] = 0

        perturbed_instance[subset] = baseline[instance_data_row.index.get_indexer(subset)]

        subset_g_x = g_x[g_x['feature'].isin(subset)]
        subset_feature_importances = subset_g_x['score'].values

        if not rank_based:
            combined_importance = sum(subset_feature_importances)
        else:
            ranked_g_x = get_ranked_explanation(g_x)
            sfi_ranking = ranked_g_x[g_x['feature'].isin(subset)]
            if rb_alg == "sum":
                combined_importance = -sfi_ranking['rank'].sum()
            elif rb_alg == "percentile":
                percentiles = 1 - (sfi_ranking['rank'] / ranked_g_x['rank'].values[-1])
                combined_importance = percentiles.sum()
            elif rb_alg == "avg":
                combined_importance = -sfi_ranking['rank'].mean()
            elif rb_alg == "inverse":
                combined_importance = (1 / sfi_ranking['rank']).sum()

        # Done this way so it'll work both on regression and on classification
        prediction = self.predict_fn(perturbed_instance.to_numpy().reshape(1, -1))[0]
        f_x_perturbed = prediction[predicted_index] if isinstance(prediction, (list, np.ndarray)) else prediction
        delta_f = np.abs(f_x - f_x_perturbed)

        return combined_importance, delta_f

    def sensitivity(self, ExplainerType: ExplainerWrapper | Type[ExplainerWrapper], instance_data_row: pd.Series, iterations: int = 10, method: Literal['mean_squared', 'spearman', 'pearson', 'mean_absolute'] = 'spearman',
                    custom_method: Callable[[pd.DataFrame, pd.DataFrame], float] = None, extra_explainer_params: dict = {}) -> float:
        """
        Measure explanation stability when the input data is slightly perturbed.
        
        This method evaluates how sensitive an explainer is to small changes in the input data
        by comparing explanations of the original instance and noisy versions.
        
        :param ExplainerType: The explainer object or class to be evaluated
        :type ExplainerType: ExplainerWrapper or Type[ExplainerWrapper]
        :param instance_data_row: The instance to be explained
        :type instance_data_row: pandas.Series
        :param iterations: Number of iterations for averaging results
        :type iterations: int, optional
        :param method: Method to calculate sensitivity ("mean_squared", "spearman", "pearson", or "mean_absolute") 
        :type method: Literal['mean_squared', 'spearman', 'pearson', 'mean_absolute'], optional
        :param custom_method: A custom method to calculate sensitivity (overrides method parameter)
        :type custom_method: Callable[[pd.DataFrame, pd.DataFrame], float], optional
        :param extra_explainer_params: Additional parameters for the explainer
        :type extra_explainer_params: dict, optional
        :return: Average sensitivity measure
        :rtype: float
        
        .. note::
            Depending on the method, lower values may be better (mean_squared, mean_absolute) or
            higher values may be better (spearman, pearson). 
        
        .. warning::
            Requires initialization via ``init()`` before use.
        """

        if not self.was_initialized:
            raise ValueError('The XaiEvaluator has not been initialized yet. Call the init() method before evaluating sensitivity.')

        if isinstance(ExplainerType, ExplainerWrapper):
            ExplainerType = ExplainerType.__class__

        original_explainer = ExplainerType(model=self.model, X_train=self.X_train, categorical_feature_names=self.ohe_categorical_feature_names,
                                           predict_fn=self.predict_fn, **extra_explainer_params)

        with Pool(processes = self.jobs) as executor:
            results = executor.map(
                self._evaluate_sensitivity_iteration,
                [original_explainer] * iterations,
                [instance_data_row] * iterations,
                [ExplainerType] * iterations,
                [method] * iterations,
                [custom_method] * iterations,
                [extra_explainer_params] * iterations
            )

        return np.mean(results)

    def _evaluate_sensitivity_iteration(self, original_explainer: ExplainerWrapper, instance_data_row, ExplainerType: Type[ExplainerWrapper], method, custom_method, extra_explainer_params):
        """
        Helper method to evaluate a single sensitivity iteration.
        
        :param original_explainer: The explainer for the original data
        :param instance_data_row: The instance to explain
        :param ExplainerType: The explainer class
        :param method: Metric calculation method
        :param custom_method: Custom metric calculation function
        :param extra_explainer_params: Additional explainer parameters
        :return: Sensitivity score for the iteration
        :rtype: float
        """
        # Obtain the original explanation:
        original_explanation = original_explainer.explain_instance(instance_data_row)

        # Obtain the noisy explanation:
        noisy_data = self.noisy_data_generator.generate_noisy_data()
        noisy_explainer = ExplainerType(model=self.model, X_train=noisy_data, categorical_feature_names=self.ohe_categorical_feature_names, predict_fn=self.predict_fn, on_noise=True, **extra_explainer_params)
        noisy_explanation = noisy_explainer.explain_instance(instance_data_row)

        # Align the two explanations
        noisy_explanation = noisy_explanation.set_index('feature').loc[original_explanation['feature']].reset_index()
        original_explainer.X_train

        if custom_method is not None:
            return custom_method(original_explanation, noisy_explanation)
        elif method == 'mean_squared':
            mean_squared_difference = ((original_explanation['score'] - noisy_explanation['score']) ** 2).mean()
            return mean_squared_difference
        elif method == "mean_absolute":
            mean_absolute_difference = (original_explanation['score'] - noisy_explanation['score']).abs().mean()
            return mean_absolute_difference
        elif method == 'spearman':
            spearman_correlation = spearmanr(original_explanation['score'], noisy_explanation['score']).statistic
            return abs(spearman_correlation)
        elif method == 'pearson':
            pearson_correlation = pearsonr(original_explanation['score'], noisy_explanation['score']).statistic
            return abs(pearson_correlation)
    
    def _sensitivity_sequential(self, ExplainerType: ExplainerWrapper | Type[ExplainerWrapper], instance_data_row: pd.Series, iterations: int = 10, method: Literal['mean_squared', 'spearman', 'pearson', 'mean_absolute'] = 'spearman',
                    custom_method: Callable[[pd.DataFrame, pd.DataFrame], float]=None, extra_explainer_params: dict = {}) -> float:
        """
        Sequential version of the sensitivity method.
        
        :param ExplainerType: The explainer object or class to be evaluated
        :type ExplainerType: ExplainerWrapper or Type[ExplainerWrapper]
        :param instance_data_row: The instance to be explained
        :type instance_data_row: pandas.Series
        :param iterations: Number of iterations for averaging results
        :type iterations: int, optional
        :param method: Method to calculate sensitivity
        :type method: Literal['mean_squared', 'spearman', 'pearson', 'mean_absolute'], optional
        :param custom_method: A custom method to calculate sensitivity
        :type custom_method: Callable[[pd.DataFrame, pd.DataFrame], float], optional
        :param extra_explainer_params: Additional parameters for the explainer
        :type extra_explainer_params: dict, optional
        :return: Average sensitivity measure
        :rtype: float
        
        .. warning::
            Requires initialization via ``init()`` before use.
        """

        if not self.was_initialized:
            raise ValueError('The XaiEvaluator has not been initialized yet. Call the init() method before evaluating sensitivity.')
        
        if isinstance(ExplainerType, ExplainerWrapper):
            ExplainerType = ExplainerType.__class__
        
        original_explainer = ExplainerType(model=self.model, X_train=self.X_train, categorical_feature_names=self.ohe_categorical_feature_names, predict_fn=self.predict_fn, **extra_explainer_params)

        results: list[float] = []
        for _ in range(iterations):
            results.append(
                self._evaluate_sensitivity_iteration(
                    original_explainer, instance_data_row, ExplainerType, method, custom_method, extra_explainer_params
                )
            )
        
        return np.mean(results)

    def complexity(self, explainer: ExplainerWrapper | Type[ExplainerWrapper] = None, instance_data_row: pd.Series = None,
                   explanation: DataFrame[ExplanationModel] = None, **kwargs) -> float:
        """
        Calculate the complexity of an explanation using entropy of feature importance distribution.
        
        Higher entropy indicates higher complexity (more evenly distributed importance).
        
        :param explainer: The explainer object or class (required if explanation is None)
        :type explainer: ExplainerWrapper or Type[ExplainerWrapper], optional
        :param instance_data_row: The instance to explain (required if explanation is None)
        :type instance_data_row: pandas.Series, optional
        :param explanation: Pre-computed explanation 
        :type explanation: DataFrame[ExplanationModel], optional
        :return: Entropy-based complexity score
        :rtype: float
        
        .. note::
            Lower values indicate simpler explanations (more concentrated feature importance).
        """
        
        assert explanation is not None or (explainer is not None and instance_data_row is not None), "Either an explanation or both an explainer and an instance_data_row must be provided."

        if explanation is None:
            if not isinstance(explainer, ExplainerWrapper):
                explainer = explainer(self.model, self.X_train, self.ohe_categorical_feature_names, predict_fn=self.predict_fn)
            
            explanation = explanation if explanation is not None else explainer.explain_instance(instance_data_row)
        
        def frac_contribution(explanation: pd.DataFrame, i: int) -> float:
            abs_score_sum = explanation['score'].abs().sum()
            return explanation['score'].abs()[i] / abs_score_sum

        sum = 0
        for i in range(explanation.shape[0]):
            fc = frac_contribution(explanation, i)
            sum += fc * np.log(fc) if fc > 0 else 0
            
        return -sum
    
    def nrc_old(self, explainer: ExplainerWrapper | Type[ExplainerWrapper] = None, instance_data_row: pd.Series = None, alpha: float = 0.5,
                explanation: DataFrame[ExplanationModel] = None) -> float:
        """
        Legacy version of the NRC (Normalized Ratio of Complexity) metric.
        
        :param explainer: The explainer object or class (required if explanation is None)
        :type explainer: ExplainerWrapper or Type[ExplainerWrapper], optional
        :param instance_data_row: The instance to explain (required if explanation is None)
        :type instance_data_row: pandas.Series, optional
        :param alpha: Dispersion penalty factor
        :type alpha: float, optional
        :param explanation: Pre-computed explanation
        :type explanation: DataFrame[ExplanationModel], optional
        :return: NRC value
        :rtype: float
        
        .. deprecated:: 1.0
            Use :meth:`nrc` instead.
        """
        assert explanation is not None or (explainer is not None and instance_data_row is not None), "Either an explanation or both an explainer and an instance_data_row must be provided."
        
        if explanation is None:
            if not isinstance(explainer, ExplainerWrapper):
                explainer = explainer(self.model, self.X_train, self.ohe_categorical_feature_names, predict_fn=self.predict_fn)
                
            explanation = explainer.explain_instance(instance_data_row)

        attributions = explanation['score'].values
        attributions = np.abs(attributions) / np.sum(np.abs(attributions))

        ranks = get_ranked_explanation(explanation)['rank'].values
        reciprocal_ranks = 1 / ranks
        sum_reciprocal_ranks = sum(reciprocal_ranks)
        sum_attributions = sum(attributions)
        k = np.count_nonzero(attributions)

        if k == 0:
            return 0

        log_weight = math.log(k + 1)
        rank_dispersion = np.std(ranks)
        revised_nrc_value = (sum_reciprocal_ranks / sum_attributions) * log_weight * (1 + alpha * rank_dispersion)
        
        return revised_nrc_value

    def nrc(self, explainer: ExplainerWrapper | Type[ExplainerWrapper] = None, instance_data_row: pd.Series = None, alpha: float = 0.5,
            explanation: DataFrame[ExplanationModel] = None) -> float:
        """
        Calculate NRC (Normalized Ratio of Complexity) with dispersion penalty.
        
        This metric evaluates explanation conciseness and coherence, with higher
        values indicating more concentrated and informative explanations.
        
        :param explainer: The explainer object or class (required if explanation is None)
        :type explainer: ExplainerWrapper or Type[ExplainerWrapper], optional
        :param instance_data_row: The instance to explain (required if explanation is None)
        :type instance_data_row: pandas.Series, optional
        :param alpha: Dispersion penalty factor (higher values penalize dispersed rankings more)
        :type alpha: float, optional
        :param explanation: Pre-computed explanation
        :type explanation: DataFrame[ExplanationModel], optional
        :return: NRC value (lower is better)
        :rtype: float
        
        .. note::
            Lower values indicate more focused explanations with less feature redundancy.
        """
        
        assert explanation is not None or (explainer is not None and instance_data_row is not None), "Either an explanation or both an explainer and an instance_data_row must be provided."

        if explanation is None:
            if not isinstance(explainer, ExplainerWrapper):
                explainer = explainer(self.model, self.X_train, self.ohe_categorical_feature_names, predict_fn=self.predict_fn, abs_score=False)
            
            explanation = explainer.explain_instance(instance_data_row)

        ranks = get_ranked_explanation(explanation)['rank'].tolist()
        ranks = np.array(ranks, dtype=int)
        reciprocal_ranks = 1 / ranks
        sum_reciprocal_ranks = np.sum(reciprocal_ranks)
        n = len(ranks)

        if n == 0:
            return 0

        log_weight = math.log(n + 1)
        rank_dispersion = np.std(ranks)
        revised_nrc_value = sum_reciprocal_ranks * log_weight * (1 + alpha * rank_dispersion)

        return revised_nrc_value  


import pandera as pa
from pandera.typing import DataFrame

class RankedExplanationModel(pa.DataFrameModel):
    """
    Pandera schema for ranked explanations.
    
    :ivar feature: Feature name
    :type feature: str
    :ivar rank: Rank of the feature (higher rank = less important)
    :type rank: int
    """
    feature: str
    rank: int = pa.Field(ge=0)

def get_ranked_explanation(scored_explanation: DataFrame[ExplanationModel],
                           fraction: float = None, method: Literal["std", "spread"] = "std",
                           epsilon: float = None, invert: bool = False) -> DataFrame[RankedExplanationModel]:
    """
    Assign ranks to features based on their scores, grouping similar-scored features.
    
    :param scored_explanation: DataFrame containing feature importance scores
    :type scored_explanation: DataFrame[ExplanationModel]
    :param fraction: Fraction of score range/std to use for epsilon calculation
    :type fraction: float, optional
    :param method: Method to calculate epsilon ("std" or "spread")
    :type method: Literal["std", "spread"], optional
    :param epsilon: Manual epsilon value to determine score similarity
    :type epsilon: float, optional
    :param invert: Whether to invert the ranks (making highest rank = 1)
    :type invert: bool, optional
    :return: DataFrame with features and their ranks
    :rtype: DataFrame[RankedExplanationModel]
    
    .. note::
        Features with score differences less than epsilon receive the same rank.
        When ``invert=True``, the highest-scored feature gets rank 1.
    """

    # Calculating epsilon
    if epsilon is None:
        if method == "std":
            if fraction is None:
                fraction = 0.05

            epsilon = scored_explanation["score"].std() * fraction
        elif method == "spread":
            if fraction is None:
                fraction = 0.025
            epsilon = (scored_explanation["score"].max() - scored_explanation["score"].min()) * fraction

    # Sort the ranking dataframe by score in descending order
    scored_explanation = scored_explanation.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    # If two features have a score difference smaller than epsilon, they are considered to have the same rank
    ranked_explanation = pd.DataFrame(columns=['feature', 'rank'])
    ranked_explanation['feature'] = scored_explanation["feature"]
    
    current_rank = 1
    ranked_explanation['rank'] = 0
    ranked_explanation.at[0, 'rank'] = current_rank
    max_score_in_rank = scored_explanation.at[0, 'score']
    
    for i in range(1, len(scored_explanation)):
        if abs(scored_explanation.at[i, 'score'] - max_score_in_rank) < epsilon:
            ranked_explanation.at[i, 'rank'] = current_rank
        else:
            current_rank += 1
            ranked_explanation.at[i, 'rank'] = current_rank
            max_score_in_rank = scored_explanation.at[i, 'score']
    
    if invert:
        ranked_explanation['rank'] = ranked_explanation['rank'].max() + 1 - ranked_explanation['rank']
    
    return DataFrame[RankedExplanationModel](ranked_explanation)
