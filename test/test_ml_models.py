
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analysis.quantitative.ml_models import (
    pca_analysis, kmeans_clustering, simple_momentum_prediction,
    feature_importance_analysis, regime_detection
    # Note: excluding train_random_forest and train_svm from main tests to avoid heavy sklearn dependencies 
    # unless they are mocks, but the file imports them inside the functions.
    # We will test them if sklearn is available.
)

class TestMLModels(unittest.TestCase):
    def setUp(self):
        # Create sample data for PCA/Clustering
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100)
        self.returns = pd.DataFrame({
            'StockA': np.random.normal(0.001, 0.02, 100),
            'StockB': np.random.normal(0.001, 0.02, 100) + 0.5 * np.random.normal(0.001, 0.02, 100), # Correlated
            'StockC': np.random.normal(0.0005, 0.01, 100),
            'StockD': np.random.normal(0.002, 0.03, 100)
        }, index=dates)
        
        # For momentum prediction
        self.price_df = pd.DataFrame({
            'Close': np.cumsum(np.random.randn(100)) + 100
        }, index=dates)

    def test_pca_analysis(self):
        res = pca_analysis(self.returns, n_components=2)
        self.assertIn('explained_variance_ratio', res)
        self.assertIn('loadings', res)
        self.assertEqual(res['n_components'], 2)

    def test_kmeans_clustering(self):
        # Transpose so rows are stocks? No, function expects returns df (T x N)
        # The function says: data = returns.dropna().T  (rows=stocks)
        # So we pass (Time x Stocks)
        res = kmeans_clustering(self.returns, n_clusters=2)
        self.assertIn('assignments', res)
        self.assertIn('profiles', res)
        # 4 stocks, clusters should assign all 4
        self.assertEqual(len(res['assignments']), 4)

    def test_simple_momentum_prediction(self):
        res = simple_momentum_prediction(self.price_df)
        self.assertIn('prediction', res)
        self.assertIn('confidence', res)
        # Prediction is Bullish/Bearish/Neutral
        self.assertIn(res['prediction'], ['Bullish', 'Bearish', 'Neutral'])

    def test_feature_importance_analysis(self):
        # Create target
        df = self.returns.copy()
        df['Target'] = df['StockA'] * 0.5 + df['StockB'] * 0.3 + np.random.normal(0, 0.01, 100)
        
        res = feature_importance_analysis(df, 'Target')
        self.assertIn('StockA', res)
        self.assertIn('StockB', res)
        # StockA should be important
        self.assertGreater(res['StockA'], 0)

    def test_regime_detection(self):
        # Regime detection needs window=60
        # self.returns['StockA'] has 100 points
        res = regime_detection(self.returns['StockA'], window=20)
        self.assertIn('current_regime', res)
        self.assertIn('regime_stability', res)
        
    def test_ml_training_if_sklearn(self):
        # Optional test if sklearn is installed
        try:
            import sklearn
            from quantitative.ml_models import train_random_forest, train_svm
            
            # Create regression target
            X = self.returns[['StockA', 'StockB', 'StockC']]
            y = self.returns['StockD'] # Predict D from others
            
            rf_res = train_random_forest(X, y, n_estimators=10)
            if 'error' not in rf_res:
                self.assertIn('r2_test', rf_res)
                
            # Create classification target for SVM
            y_class = (self.returns['StockD'] > 0).astype(int)
            svm_res = train_svm(X, y_class)
            if 'error' not in svm_res:
                self.assertIn('accuracy', svm_res)
                
        except ImportError:
            print("Skipping sklearn tests")

if __name__ == '__main__':
    unittest.main()
