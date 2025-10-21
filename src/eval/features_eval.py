"""
File: src/eval/features_eval.py

This file contains the FeatureEvaluation class,
which consists of multiple checks for the quality of the
feature set provided.
"""

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.inspection import permutation_importance

import os
import json
from matplotlib_venn import venn3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FeatureEvaluation:
    def __init__(self, data_path: str, outdir: str, df=None):
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(data_path)
        self.X = self.df.drop(["userId"], axis=1)
        self.outdir = outdir
        
    def jaccard_similarity(self, num_iters: int = 10, top_k: int = 50,
                           *args, **kwargs):
        """
        Computes the average jaccard similarity across multiple
        instantiations of a given model over different random seeds.
        
        Jaccard similarity between two sets is defined as follows:
            J(A,B) = | A AND B | / | A OR B |
        i.e., it measures the similarity between a pair of sets
        
        This function computes this value across all pairwise comparisons
        of the num_iterations models.
        
        Args:
            num_iterations (int): The number of models to create; default=10
            top_k (int): The top k anomalies to consider in each model; default=50
            *args, **kwargs: Optional arguments for IsolationForest initialization
            
        Returns:
            tuple(float): The mean, median, min, and max of the jaccard similarities
        """
        # Run iterations of IF models
        top_k_anomalies = []
        for i in range(num_iters):
            if_model = IsolationForest(random_state=i, *args, **kwargs)
            fit_model = if_model.fit(self.X)

            scores = fit_model.decision_function(self.X)

            # Get top k anomalies (lowest scores are most anomalous)
            # argsort returns indices which preserve the original DataFrame index (userIds)
            top_k_indices = np.argsort(scores)[:top_k]
            top_k_anomalies.append(self.df.index[top_k_indices].tolist())
            
        # Compute average jaccard similarities
        jaccard_similarities = []
        for i in range(num_iters):
            for j in range(i+1, num_iters):
                overlap = len(set(top_k_anomalies[i]) & set(top_k_anomalies[j]))
                jaccard = overlap / (2*top_k - overlap)
                jaccard_similarities.append(jaccard)
                
        return np.mean(jaccard_similarities)
    
    def ensemble_methods(self, if_params, lof_params, ocsvm_params) -> float:
        """
        Evaluates the degree of confidence in predictions across
        multiple anomaly detection methods:
            - Isolation forests
            - Local outlier factors
            - One-class SVMs
        
        Returns the proportions of pairwise overlaps in anomaly predictions.
        """
        if_model = IsolationForest(**if_params)
        lof_model = LocalOutlierFactor(contamination=0.13, **lof_params)
        ocsvm_model = OneClassSVM(**ocsvm_params)
        
        logger.info("Fitting Isolation Forest")
        if_predictions = if_model.fit_predict(self.X) == -1
        logger.info("Fitting LOF")
        lof_predictions = lof_model.fit_predict(self.X) == -1
        logger.info("Fitting One-Class SVM")
        ocsvm_predictions = ocsvm_model.fit_predict(self.X) == -1
        
        # Find overlaps between models' anomalies
        if_lof = (if_predictions & lof_predictions).sum() / len(self.X)
        if_ocsvm = (if_predictions & ocsvm_predictions).sum() / len(self.X)
        lof_ocsvm = (lof_predictions & ocsvm_predictions).sum() / len(self.X)
        all_three = (if_predictions & lof_predictions & ocsvm_predictions).sum() / len(self.X)

        return {
            "if_lof": if_lof,
            "if_ocsvm": if_ocsvm,
            "lof_ocsvm": lof_ocsvm,
            "all_three": all_three
        }
        
    def feature_analysis(self, n_repeats=10):
        """
        Performs comparisons in feature statistics
        between anomaly and non-anomaly features.

        Returns standardized mean differences (similar to Cohen's d)
        """
        copy_df = self.df.copy()
        if_model = IsolationForest()
        fit_if_model = if_model.fit(self.X)
        copy_df["labels"] = fit_if_model.predict(self.X)

        # Compute standardized differences
        anomalies = copy_df[copy_df["labels"] == -1].drop(labels=["labels"], axis=1)
        normals = copy_df[copy_df["labels"] != -1].drop(labels=["labels"], axis=1)

        anomaly_means = anomalies.describe().loc["mean"]
        normal_means = normals.describe().loc["mean"]
        std = copy_df.describe().loc["std"]

        standardized_diff = (anomaly_means - normal_means) / std
        
        # Compute permutation importance
        # Note: score_samples already returns negative scores (lower = more anomalous)
        # So we want to maximize the negative score (more negative = better anomaly detection)
        # Scoring function must accept (estimator, X, y) even though y is unused
        result = permutation_importance(
            fit_if_model, X=self.X, y=None,
            scoring=lambda est, X, y: est.score_samples(X).mean(),
            n_repeats=n_repeats,
            random_state=42
        )

        perm_importance = pd.Series(result.importances_mean, index=self.X.columns)
        perm_std = pd.Series(result.importances_std, index=self.X.columns)

        return {
            "permutation_importance": perm_importance,
            "permutation_std": perm_std,
            "standardized_diff": standardized_diff
        }
        
        
        
    def save_report(self, data: dict):
        """
        Saves the values to the specified outdirectory
        """

        os.makedirs(self.outdir, exist_ok=True)

        # 1. Line plot of Jaccard similarities
        plt.figure(figsize=(10, 6))
        plt.plot(data['jaccard']['x'], data['jaccard']['y'], marker='o', linewidth=2)
        plt.xlabel('Top K Anomalies', fontsize=12)
        plt.ylabel('Jaccard Similarity', fontsize=12)
        plt.title('Jaccard Similarity Across Different K Values', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.outdir, 'jaccard_similarity.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Venn diagram for ensemble overlaps
        # if 'ensembles' in data and data['ensembles']:
        #     ensembles = data['ensembles']

        #     # Extract overlap values
        #     # For venn3, we need approximate overlap regions
        #     # Based on pairwise and three-way overlaps
        #     plt.figure(figsize=(10, 8))

        #     # Calculate exclusive regions from overlaps
        #     if_lof_only = ensembles.get('if_lof', 0) - ensembles.get('all_three', 0)
        #     if_ocsvm_only = ensembles.get('if_ocsvm', 0) - ensembles.get('all_three', 0)
        #     lof_ocsvm_only = ensembles.get('lof_ocsvm', 0) - ensembles.get('all_three', 0)
        #     all_three = ensembles.get('all_three', 0)

        #     venn = venn3(
        #         subsets={
        #             '110': max(0, if_lof_only),      # IF & LOF only
        #             '101': max(0, if_ocsvm_only),    # IF & OCSVM only
        #             '011': max(0, lof_ocsvm_only),   # LOF & OCSVM only
        #             '111': max(0, all_three)         # All three
        #         },
        #         set_labels=('Isolation Forest', 'LOF', 'One-Class SVM')
        #     )

        #     plt.title('Ensemble Model Agreement on Anomalies', fontsize=14)
        #     plt.savefig(os.path.join(self.outdir, 'ensemble_venn.png'), dpi=300, bbox_inches='tight')
        #     plt.close()

        # 3. Bar plots for feature analysis
        if 'feature_analysis' in data and data['feature_analysis'] is not None:
            feature_data = data['feature_analysis']

            # feature_data is now a dict with multiple metrics
            if isinstance(feature_data, dict):
                # Plot standardized differences
                if 'standardized_diff' in feature_data:
                    differences = feature_data['standardized_diff'].sort_values(key=abs, ascending=False)

                    plt.figure(figsize=(12, max(6, len(differences) * 0.3)))
                    colors = ['red' if d < 0 else 'green' for d in differences]
                    plt.barh(range(len(differences)), differences.values, color=colors, alpha=0.7)
                    plt.yticks(range(len(differences)), differences.index)
                    plt.xlabel('Standardized Difference (Anomaly - Normal) / σ', fontsize=12)
                    plt.ylabel('Features', fontsize=12)
                    plt.title('Standardized Feature Differences: Anomalies vs Normal', fontsize=14)
                    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.outdir, 'feature_differences.png'), dpi=300, bbox_inches='tight')
                    plt.close()

                # Plot permutation importance
                if 'permutation_importance' in feature_data:
                    perm_imp = feature_data['permutation_importance'].sort_values(ascending=False)
                    perm_std = feature_data.get('permutation_std', pd.Series(0, index=perm_imp.index))

                    plt.figure(figsize=(12, max(6, len(perm_imp) * 0.3)))
                    plt.barh(range(len(perm_imp)), perm_imp.values,
                            xerr=perm_std[perm_imp.index].values,
                            color='steelblue', alpha=0.7)
                    plt.yticks(range(len(perm_imp)), perm_imp.index)
                    plt.xlabel('Permutation Importance', fontsize=12)
                    plt.ylabel('Features', fontsize=12)
                    plt.title('Feature Importance (Permutation-based)', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.outdir, 'permutation_importance.png'), dpi=300, bbox_inches='tight')
                    plt.close()

        # Save numerical results as JSON
        def serialize_value(val):
            """Recursively convert pandas objects to serializable format"""
            if isinstance(val, (pd.DataFrame, pd.Series)):
                return val.to_dict()
            elif isinstance(val, dict):
                return {k: serialize_value(v) for k, v in val.items()}
            elif isinstance(val, (np.integer, np.floating)):
                return float(val)
            else:
                return val

        serializable_data = {k: serialize_value(v) for k, v in data.items()}

        with open(os.path.join(self.outdir, 'evaluation_results.json'), 'w') as f:
            json.dump(serializable_data, f, indent=2)

        
    def evaluate(self, if_params=None, lof_params=None, ocsvm_params=None):
        # Compute jaccards
        top_k = [50, 100, 500, 1000, 2000, 3000]
        logger.info(f"Computing Jaccard Similarities with top_k={top_k}")
        similarities = []
        for k in top_k:
            similarities.append(self.jaccard_similarity(top_k=k))

        # Compute ensemble agreements
        # logger.info("Running ensemble methods")
        # if_params = if_params or {}
        # lof_params = lof_params or {}
        # ocsvm_params = ocsvm_params or {}
        # overlaps = self.ensemble_methods(if_params, lof_params, ocsvm_params)
        
        # Compute feature analysis
        logger.info("Running feature analysis")
        analysis = self.feature_analysis()
        
        logger.info("Saving results...")
        data = {
            "jaccard": {
                "x": top_k,
                "y": similarities
            },
            # "ensembles": overlaps,
            "feature_analysis": analysis
        }
        self.save_report(data)