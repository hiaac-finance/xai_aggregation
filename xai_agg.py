from typing import Literal, Type, Callable

import numpy as np
import pandas as pd
from tools.topsis import Topsis

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Explainable AI tools:
import shap
from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular # why not used the original anchor package?

from scipy.stats import spearmanr, pearsonr

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam 

# MCDM:
from tools.topsis import Topsis

# ranking tools:
import ranx

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
        pass

class LimeWrapper(ExplainerWrapper):

    def __init__(self, clf, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str], predict_proba: callable = None):
        super().__init__(clf, X_train, categorical_feature_names, predict_proba)
        
        self.explainer = LimeTabularExplainer(self.X_train.values, feature_names=self.X_train.columns, discretize_continuous=False)
    
    def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> pd.DataFrame:
        lime_exp = self.explainer.explain_instance(instance_data_row, self.predict_proba, num_features=len(self.X_train.columns))
        
        ranking = pd.DataFrame(lime_exp.as_list(), columns=['feature', 'score'])
        return ranking

class ShapTabularTreeWrapper(ExplainerWrapper):
    
        def __init__(self, clf, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str], predict_proba: callable = None, **additional_explainer_args):
            super().__init__(clf, X_train, categorical_feature_names, predict_proba)
            
            self.explainer = shap.TreeExplainer(clf, self.X_train, **additional_explainer_args)
        
        def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> pd.DataFrame:
            shap_values = self.explainer.shap_values(instance_data_row)
    
            ranking = pd.DataFrame(list(zip(self.X_train.columns, shap_values[:, 0])), columns=['feature', 'score'])
            ranking = ranking.sort_values(by='score', ascending=False, key=lambda x: abs(x)).reset_index(drop=True)
            
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


class AutoencoderNoisyDataGenerator():
    """
    This class generates noisy data by swapping the values of a small number of features between a sample and a random close neighbor.
    The neighbors are determined using an autoencoder to reduce the dimensionality of the data and then calculate the use the NearestNeightbors algorithm in the reduced space.
    """

    def __init__(self, X: pd.DataFrame, ohe_categorical_features_names: list[str], encoding_dim: int = 5, epochs=500):
        self.X = X
        self.categorical_features_names = ohe_categorical_features_names
        self.encoding_dim = encoding_dim
        self.epochs = epochs

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
        self.autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
        self.autoencoder.fit(self.X_scaled, self.X_scaled, epochs=self.epochs, batch_size=32, shuffle=True, validation_split=0.2)
        # Extract hidden layer representation:
        self.hidden_representation = self.encoder.predict(self.X_scaled)
        self.was_fit = True


    def generate_noisy_data(self, num_features_to_replace: int = 2) -> pd.DataFrame:
        """
        Returns a DataFrame containing a noisy variation of the data.

        The noise is generated by swapping the values of a small number of features between a sample and a random close neighbor.
        To determine the neighbors, we use an autoencoder to reduce the dimensionality of the data and then calculate the use the NearestNeightbors algorithm in the reduced space.
        """

        if not self.was_fit:
            raise ValueError('The autoencoder has not been fitted yet. Call the fit() method before generating noisy data.')

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
    This class defines a set of metrics to evaluate an explantion model's performance on a given data instance. THIS CLASS MUST BE INITIALIZED BEFORE USE.

    The metrics are:
    - Faithfullness correlation: The correlation between the importance of the features in the explanation and the change in the model's output when the features are perturbed.
    - Sensitivity: The relationship (average difference or correlation) between the explanation of the instance and the explanation of a noisy version of the instance.
    - Complexity: The complexity of the explanation, calculated as the entropy of the explanation's feature importance distribution.

    The class can be used to evaluate the performance of different explanation methods, or to evaluate the performance of an aggregated explainer.

    Attributes:
        - clf (object): The classifier model that will be explained.
        - X_train (pd.DataFrame | np.ndarray): The training data used to train the classifier.
        - ohe_categorical_feature_names (list[str]): The names of the categorical features that were one-hot-encoded.
        - predict_proba (callable): A function that receives a data row and returns the model's prediction probabilities.
        - noise_gen_args (dict): A dictionary containing the arguments to be passed to the AutoencoderNoisyDataGenerator class.
    """

    def __init__(self, clf, X_train: pd.DataFrame | np.ndarray, ohe_categorical_feature_names: list[str], predict_proba: callable = None, noise_gen_args: dict = {}):
        self.clf = clf
        if hasattr(clf, 'predict_proba') and predict_proba is None:
            self.predict_proba = clf.predict_proba
        elif predict_proba is not None:
            self.predict_proba = predict_proba
        else:
            raise ValueError('The classifier does not have a predict_proba method and no predict_proba_function was provided.')

        self.X_train = X_train
        self.ohe_categorical_feature_names = ohe_categorical_feature_names

        self.categorical_features_indices = [
            [self.X_train.columns.get_loc(col_name) for col_name in self.X_train.columns if col_name.startswith(feature)]
            for feature in self.ohe_categorical_feature_names
        ]

        self.noisy_data_generator = AutoencoderNoisyDataGenerator(X_train, ohe_categorical_feature_names, **noise_gen_args)

        self.was_initialized = False
    
    # Initialization opeations that take a long time to run
    def init(self):
        self.noisy_data_generator.fit()
        self.was_initialized = True
            
    def faithfullness_correlation(self, explainer: ExplainerWrapper | Type[ExplainerWrapper], instance_data_row: pd.Series, len_subset: int = None,
                                  iterations: int = 100, baseline_strategy: Literal["zeros", "mean"] = "zeros") -> float:
        """
        This metric measures the correlation between the importance of the features in the explanation and the change in the model's output when the features are perturbed.
        Referenced from: https://arxiv.org/abs/2005.00631

        Parameters:
            explainer (ExplainerWrapper | Type[ExplainerWrapper]): The explainer object or class to be evaluated.
            instance_data_row (pd.Series): The instance to be explained.
            len_subset (int): The number of features to perturb in each iteration. If None, the default value is len(instance_data_row)//4 (25% of the features).
            iterations (int): The number of iterations to run the metric calculation. The higher the number of iterations, the more accurate the result.
            baseline_strategy (str): The strategy to be used to generate the baseline values for the perturbed features. Options are "zeros" (all zeros) or "mean" (mean of the training data).
                                     "mean" usually provides hihger correlation values, but "zeros" is more conservative.
        """

        if not isinstance(explainer, ExplainerWrapper):
            explainer = explainer(self.clf, self.X_train, self.ohe_categorical_feature_names, predict_proba=self.predict_proba)
        
        dimension = len(instance_data_row)  

        importance_sums = []
        delta_fs = []

        f_x = self.predict_proba(instance_data_row.to_numpy().reshape(1, -1))[0][1]
        g_x = explainer.explain_instance(instance_data_row)

        for _ in range(iterations):
            # Select a subset of features to perturb
            subset = np.random.choice(instance_data_row.index.values, len_subset if len_subset else dimension//4, replace=False)

            perturbed_instance = instance_data_row.copy()

            if baseline_strategy == "zeros":
                baseline = np.zeros(dimension)  # either mean on all zeros
            elif baseline_strategy == "mean":
                baseline = np.mean(self.X_train, axis=0)
                for feature_index in self.categorical_features_indices:
                    baseline[feature_index] = 0
                
            perturbed_instance[subset] = baseline[instance_data_row.index.get_indexer(subset)]

            importance_sum = 0
            for feature in subset:
                importance_sum += g_x[g_x['feature'] == feature]['score'].values[0] # should I take the abs value here?
            importance_sums.append(importance_sum)

            f_x_perturbed = self.predict_proba(perturbed_instance.to_numpy().reshape(1, -1))[0][1]
            delta_f = np.abs(f_x - f_x_perturbed)
            delta_fs.append(delta_f)
        
        return abs(pearsonr(importance_sums, delta_fs).statistic)
    
    def sensitivity(self, ExplainerType: ExplainerWrapper | Type[ExplainerWrapper], instance_data_row: pd.Series, iterations: int = 10, method: Literal['mean_squared', 'spearman', 'pearson'] = 'spearman',
                    custom_method: Callable[[pd.DataFrame, pd.DataFrame], float]=None, extra_explainer_params: dict = {}) -> float:
        """
        This metric measures the relationship (average difference or correlation) between the explanation of the instance and the explanation of a noisy version of the instance.
        The explainer is instantiated twice: once to explain the original instance and once to explain the noisy instance, since it may need to fit or train itself with the data.

        Beware: depending on the method used, the metric can either be a cost function (the lower the better: mean_squared) or a reward function (the higher the better: spearman, person).

        Parameters:
            - ExplainerType (ExplainerWrapper | Type[ExplainerWrapper]): The explainer object or class to be evaluated.
            - instance_data_row (pd.Series): The instance to be explained.
            - iterations (int): The number of iterations to run the metric calculation. The higher the number of iterations, the more accurate the result.
            - method (str): The method to be used to calculate the sensitivity. Options are "mean_squared", "spearman", and "pearson".
            - custom_method (Callable[[pd.DataFrame, pd.DataFrame], float]): A custom method to calculate the sensitivity. If provided, the method parameter will be ignored.
            - extra_explainer_params (dict): A dictionary containing the parameters to be passed to the explainer class, in case it requires additional parameters.
        """

        if not self.was_initialized:
            raise ValueError('The XaiEvaluator has not been initialized yet. Call the init() method before evaluating sensitivity.')
        
        if isinstance(ExplainerType, ExplainerWrapper):
            ExplainerType = ExplainerType.__class__
        
        original_explainer = ExplainerType(clf=self.clf, X_train=self.X_train, categorical_feature_names=self.ohe_categorical_feature_names, predict_proba=self.predict_proba, **extra_explainer_params)

        results: list[float] = []
        for _ in range(iterations):
            # Obtain the original explanation:
            original_explanation = original_explainer.explain_instance(instance_data_row)

            # Obtain the noisy explanation:
            noisy_data = self.noisy_data_generator.generate_noisy_data(num_features_to_replace=2)
            noisy_explainer = ExplainerType(clf=self.clf, X_train=noisy_data, categorical_feature_names=self.ohe_categorical_feature_names, predict_proba=self.predict_proba, **extra_explainer_params)
            noisy_explanation = noisy_explainer.explain_instance(instance_data_row)

            if custom_method is not None:
                results.append(custom_method(original_explanation, noisy_explanation))
            elif method == 'mean_squared':
                mean_squared_difference = ((original_explanation['score'] - noisy_explanation['score']) ** 2).mean()
                results.append(mean_squared_difference)
            elif method == 'spearman':
                spearman_correlation = spearmanr(original_explanation['score'], noisy_explanation['score']).correlation
                results.append(abs(spearman_correlation))
            elif method == 'pearson':
                pearson_correlation = pearsonr(original_explanation['score'], noisy_explanation['score']).correlation
                results.append(abs(pearson_correlation))
        
        return np.mean(results)

    def complexity(self, explainer: ExplainerWrapper | Type[ExplainerWrapper], instance_data_row: pd.Series, **kwargs) -> float:
        """
        This metric is calculated as the entropy of the explanation's feature importance distribution.
        Referenced from: https://arxiv.org/abs/2005.00631
        """

        if not kwargs.get("bypass_check", False) and not isinstance(explainer, ExplainerWrapper):
            explainer = explainer(self.clf, self.X_train, self.ohe_categorical_feature_names, predict_proba=self.predict_proba)

        explanation = explainer.explain_instance(instance_data_row)

        def frac_contribution(explanation: pd.DataFrame, i: int) -> float:
            abs_score_sum = explanation['score'].abs().sum()
            return explanation['score'].abs()[i] / abs_score_sum

        sum = 0
        for i in range(explanation.shape[0]):
            fc = frac_contribution(explanation, i)
            sum += fc * np.log(fc) if fc > 0 else 0
            
        return -sum
    
class AggregatedExplainer(ExplainerWrapper):
    """
    This class aggregates multiple feature-importance-based explanation methods to provide a single explanation for a given instance.
    The aggregated explanation is calculated using a weighted rank aggregation algorithm, whose weights are calculated using a MCDM algorithm based on the instance explanation metrics.

    Attributes:
        - explainer_types (list[Type[ExplainerWrapper]]): A list of the explainer classes to be used. The classes must inherit from the ExplainerWrapper class, so that they have the same interface and output format.
        - clf (object): The classifier model whose predictions will be explained.
        - X_train (pd.DataFrame | np.ndarray): The training data used to train the classifier.
        - categorical_feature_names (list[str]): The names of the categorical features that were one-hot-encoded.
        - predict_proba (callable): A function that receives a data row and returns the model's prediction probabilities. If None, the classifier's predict_proba method will be used.
        - explainer_params_list (dict[Type[ExplainerWrapper], dict]): A dictionary containing the parameters to be passed to each explainer class, in case they require additional parameters.
            it should be in the format {ExplainerType: {param1: value1, param2: value2, ...}}
        - aggregation_algorithm (str): The rank aggregation algorithm to be used. Options are "wsum" (Weighted Sum), "w_bordafuse" (Weighted BordaFuse), and "w_condorcet" (Weighted Condorcet).
    """

    def __init__(self, explainer_types: list[Type[ExplainerWrapper]], clf, X_train: pd.DataFrame | np.ndarray, categorical_feature_names: list[str], predict_proba: callable = None,
                 explainer_params_list: dict[Type[ExplainerWrapper], dict] = None, aggregation_algorithm: Literal["wsum", "w_bordafuse", "w_condorcet"] = "wsum", **kwargs):
        super().__init__(clf, X_train, categorical_feature_names, predict_proba)

        self.explainers = []
        for ExplainerType in explainer_types:
            extra_params = explainer_params_list.get(ExplainerType, {}) if explainer_params_list is not None else {}
            self.explainers.append(ExplainerType(clf, X_train, categorical_feature_names, predict_proba=predict_proba, **extra_params))

        self.xai_evaluator = ExplanationModelEvaluator(clf, X_train, categorical_feature_names, self.predict_proba, kwargs.get('noise_gen_args', {}))
        self.xai_evaluator.init()

        self.aggregation_algorithm = aggregation_algorithm

        self.last_explanation_metrics: pd.DataFrame = None
    
    @staticmethod
    def _ranking_to_run(feature_importance_ranking: pd.DataFrame) -> ranx.Run:
        feature_importance_ranking["query"] = "1"
        return ranx.Run.from_df(feature_importance_ranking, q_id_col="query", doc_id_col="feature", score_col="score")
    
    @staticmethod
    def _get_weights(instance_explanation_metrics: pd.DataFrame, higher_is_better: list[bool]) -> list[float]:
        """
        Uses a MCDM algorithm to calculate the weights for each explanation method based on the instance explanation metrics.

        Parameters:
        instance_explanation_metrics (pd.DataFrame): DataFrame containing the instance explanation metrics for each explanation method.
        higher_is_better (list[bool]): A list of booleans indicating whether higher values are preferred for each metric.
        """

        evaluation_matrix = instance_explanation_metrics.to_numpy()

        num_metrics = evaluation_matrix.shape[1]
        topsis_weights = [1/num_metrics for _ in range(num_metrics)]

        t = Topsis(evaluation_matrix, topsis_weights, higher_is_better, debug=False)
        t.calc()

        return t.worst_similarity # TODO: check if this is the correct attribute to return

    def explain_instance(self, instance_data_row: pd.Series | np.ndarray) -> pd.DataFrame:
        runs = []
        for explainer in self.explainers:
            runs.append(self._ranking_to_run(explainer.explain_instance(instance_data_row)))

        instance_explanation_metrics = pd.DataFrame(columns=["method", "faithfullness_correlation", "sensitivity", "complexity"]).set_index("method")
        for explainer in self.explainers:
            instance_explanation_metrics.loc[explainer.__class__.__name__] = [
                self.xai_evaluator.faithfullness_correlation(explainer, instance_data_row, iterations=10),
                self.xai_evaluator.sensitivity(explainer, instance_data_row, iterations=10),
                self.xai_evaluator.complexity(explainer, instance_data_row)
            ]
        self.last_explanation_metrics = instance_explanation_metrics

        weights = self._get_weights(instance_explanation_metrics, [True, True, False])

        fused_run = ranx.fuse(runs, method=self.aggregation_algorithm,
                              params={"weights": weights})
        
        return fused_run.to_dataframe().drop(columns=["q_id"]).rename(columns={"doc_id": "feature"})