import os
import unittest
import pandas as pd
import numpy as np
import tempfile
from src.models import FunkSVD, ItemCollaborativeFiltering, UserCollaborativeFiltering, NeuralCFRecommender, ContentBasedRecommender
from src.evaluate import compute_rmse_mae, evaluate_map_at_10

class TestRecommendationSystem(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset for testing
        # 10 users, 10 items, highly sparse
        ratings_data = []
        # Create standard patterns to make recommendation testing meaningful
        # Users 0-4 like items 0-4
        # Users 5-9 like items 5-9
        for u in range(10):
            for i in range(10):
                # Sparsity: only rate some items
                if (u < 5 and i < 5) or (u >= 5 and i >= 5):
                    if (u + i) % 2 == 0:
                        ratings_data.append((i, u, 5.0, "2026-06-12"))
                    else:
                        ratings_data.append((i, u, 4.0, "2026-06-12"))
                elif (u + i) % 5 == 0:
                    ratings_data.append((i, u, 1.0, "2026-06-12"))
                    
        self.df = pd.DataFrame(ratings_data, columns=['MovieID', 'CustomerID', 'Rating', 'Date'])
        self.df['user_idx'] = self.df['CustomerID']
        self.df['movie_idx'] = self.df['MovieID']
        
        # Split into 80% train, 20% test
        n = len(self.df)
        self.train_df = self.df.iloc[:int(n*0.8)].copy()
        self.test_df = self.df.iloc[int(n*0.8):].copy()

    def test_funk_svd(self):
        """Tests that Funk SVD fits and predicts within boundaries."""
        num_users = 10
        num_items = 10
        
        svd = FunkSVD(n_factors=4, lr=0.01, reg=0.02, n_epochs=5)
        svd.fit(self.train_df, num_users, num_items)
        
        # Test single prediction
        pred = svd.predict(0, 0)
        self.assertTrue(1.0 <= pred <= 5.0)
        
        # Test batch prediction
        preds = svd.predict_batch(np.array([0, 1, 2]), np.array([0, 1, 2]))
        self.assertEqual(len(preds), 3)
        self.assertTrue(all(1.0 <= p <= 5.0 for p in preds))

    def test_item_cf(self):
        """Tests that Item-Based CF fits and computes similarities."""
        num_users = 10
        num_items = 10
        
        cf = ItemCollaborativeFiltering(k_neighbors=3, shrinkage=1)
        cf.fit(self.train_df, num_users, num_items)
        
        # Test similarity matrix properties
        self.assertEqual(cf.similarity_matrix.shape, (num_items, num_items))
        self.assertTrue(np.all(cf.similarity_matrix >= 0.0))
        self.assertTrue(np.all(cf.similarity_matrix <= 1.0))
        self.assertEqual(cf.similarity_matrix[0, 0], 0.0) # Self-similarity set to 0
        
        # Test single prediction
        pred = cf.predict(0, 0)
        self.assertTrue(1.0 <= pred <= 5.0)

    def test_user_cf(self):
        """Tests that User-Based CF fits and computes user similarities."""
        num_users = 10
        num_items = 10
        
        cf = UserCollaborativeFiltering(k_neighbors=3, shrinkage=1)
        cf.fit(self.train_df, num_users, num_items)
        
        # Test sparse matrix shapes
        self.assertEqual(cf.R_norm.shape, (num_users, num_items))
        self.assertEqual(cf.R_binary.shape, (num_users, num_items))
        
        # Test single prediction
        pred = cf.predict(0, 0)
        self.assertTrue(1.0 <= pred <= 5.0)

    def test_neural_cf(self):
        """Tests that Neural CF (NeuMF) NumPy wrapper loads and predicts."""
        num_users = 10
        num_items = 10
        n_factors = 8
        
        # Create a mock saved model structure
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "mock_ncf.npz")
            
            save_dict = {
                'P_gmf': np.random.normal(0, 0.1, (num_users, n_factors)),
                'Q_gmf': np.random.normal(0, 0.1, (num_items, n_factors)),
                'P_mlp': np.random.normal(0, 0.1, (num_users, n_factors)),
                'Q_mlp': np.random.normal(0, 0.1, (num_items, n_factors)),
                'final_weight': np.random.normal(0, 0.1, (1, n_factors + 16)),
                'final_bias': np.array([0.0]),
                'num_users': num_users,
                'num_items': num_items,
                'global_mean': 3.5,
                'mlp_w_0': np.random.normal(0, 0.1, (16, n_factors * 2)),
                'mlp_b_0': np.zeros(16)
            }
            np.savez(model_path, **save_dict)
            
            ncf = NeuralCFRecommender()
            self.assertTrue(ncf.load_model(model_path))
            
            # Test predictions
            pred = ncf.predict(0, 0)
            self.assertTrue(1.0 <= pred <= 5.0)
            
            preds = ncf.predict_batch(np.array([0, 1, 2]), np.array([0, 1, 2]))
            self.assertEqual(len(preds), 3)
            self.assertTrue(all(1.0 <= p <= 5.0 for p in preds))

    def test_content_based(self):
        """Tests that Content-Based filtering fits, saves, loads and predicts."""
        num_users = 10
        num_items = 10
        
        # Create mock content features DataFrame
        # movie_id mapping to indices
        movie_to_idx = {i: i for i in range(10)}
        content_data = []
        titles = [
            "Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)", 
            "Waiting to Exhale (1995)", "Father of the Bride Part II (1995)", 
            "Heat (1995)", "Sabrina (1995)", "Tom and Huck (1995)", 
            "Sudden Death (1995)", "GoldenEye (1995)"
        ]
        for i in range(10):
            content_data.append({
                'movie_id': i,
                'title': titles[i],
                'content_string': f"{titles[i]} 1990s"
            })
        content_df = pd.DataFrame(content_data)
        
        cb = ContentBasedRecommender(k_neighbors=3)
        cb.fit(self.train_df, num_users, num_items, content_df, movie_to_idx)
        
        # Test predictions
        pred = cb.predict(0, 0)
        self.assertTrue(1.0 <= pred <= 5.0)
        
        preds = cb.predict_batch(np.array([0, 1, 2]), np.array([0, 1, 2]))
        self.assertEqual(len(preds), 3)
        self.assertTrue(all(1.0 <= p <= 5.0 for p in preds))
        
        # Test save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "mock_cb.npz")
            cb.save_model(model_path)
            
            cb_loaded = ContentBasedRecommender()
            self.assertTrue(cb_loaded.load_model(model_path))
            
            pred_loaded = cb_loaded.predict(0, 0)
            self.assertTrue(1.0 <= pred_loaded <= 5.0)
            
            preds_loaded = cb_loaded.predict_batch(np.array([0, 1, 2]), np.array([0, 1, 2]))
            self.assertEqual(len(preds_loaded), 3)
            self.assertTrue(all(1.0 <= p <= 5.0 for p in preds_loaded))

    def test_evaluation_metrics(self):
        """Tests computation of RMSE, MAE, and MAP@10 metrics."""
        # Test RMSE/MAE
        preds = np.array([4.0, 3.0, 5.0])
        actuals = np.array([5.0, 3.0, 4.0])
        rmse, mae = compute_rmse_mae(preds, actuals)
        
        self.assertAlmostEqual(rmse, np.sqrt(2/3), places=4)
        self.assertAlmostEqual(mae, 2/3, places=4)
        
        # Mock model for MAP@10 testing
        class MockModel:
            def predict_batch(self, users, items):
                preds = []
                for u, i in zip(users, items):
                    if (u < 5 and i < 5) or (u >= 5 and i >= 5):
                        preds.append(4.5)
                    else:
                        preds.append(1.5)
                return np.array(preds)
                
        mock_model = MockModel()
        map_score = evaluate_map_at_10(mock_model, self.train_df, self.test_df, 10, 10, relevance_threshold=3.5)
        self.assertTrue(0.0 <= map_score <= 1.0)

if __name__ == "__main__":
    unittest.main()
