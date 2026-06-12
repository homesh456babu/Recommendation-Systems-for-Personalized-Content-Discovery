import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import time
import os


class ItemCollaborativeFiltering:
    def __init__(self, k_neighbors=40, shrinkage=100):
        """
        Item-Based Collaborative Filtering using Cosine Similarity.
        k_neighbors: number of similar items to consider for predicting a rating.
        shrinkage: regularization parameter for similarity to penalize items with few co-ratings.
        """
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.num_users = 0
        self.num_items = 0
        self.global_mean = 0.0
        self.item_means = None
        self.user_means = None
        self.R = None  # User-Item ratings matrix (sparse)
        self.similarity_matrix = None  # Item-Item similarity matrix

    def fit(self, train_df, num_users, num_items):
        """
        Learns the item-item similarity matrix.
        """
        print("Fitting Item-Based Collaborative Filtering model...")
        start_time = time.time()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        
        self.global_mean = train_df['Rating'].mean()
        
        # Calculate item and user means
        self.item_means = train_df.groupby('movie_idx')['Rating'].mean()
        self.user_means = train_df.groupby('user_idx')['Rating'].mean()
        self.item_means_arr = self.item_means.reindex(range(self.num_items), fill_value=self.global_mean).values
        self.user_means_arr = self.user_means.reindex(range(self.num_users), fill_value=self.global_mean).values
        
        # Create sparse rating matrix R
        self.R = csr_matrix(
            (train_df['Rating'].values, (train_df['user_idx'].values, train_df['movie_idx'].values)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32
        )
        
        # Calculate cosine similarity using sparse matrix dot products
        # Normalize columns (items) of R
        print("Calculating item-item similarities...")
        
        # Col norms
        # We need rating squares sum per column
        col_norms = np.sqrt(np.array(self.R.power(2).sum(axis=0)).flatten())
        col_norms[col_norms == 0] = 1.0  # Prevent division by zero
        
        # Create a normalized ratings matrix
        # Multiply each column by 1 / norm
        R_norm = self.R.multiply(1.0 / col_norms)
        
        # Similarity = R_norm^T * R_norm
        # This yields a sparse similarity matrix of size (num_items, num_items)
        sim = R_norm.T.dot(R_norm).toarray()
        
        # Apply shrinkage (regularization) if specified
        if self.shrinkage > 0:
            # Get co-rated counts
            # Co-rated counts = (R > 0)^T * (R > 0)
            R_binary = self.R.copy()
            R_binary.data = np.ones_like(R_binary.data)
            co_counts = R_binary.T.dot(R_binary).toarray()
            
            # similarity = similarity * co_counts / (co_counts + shrinkage)
            sim = sim * co_counts / (co_counts + self.shrinkage)
            
        # Set self-similarity to 0 to avoid recommending a movie based on itself
        np.fill_diagonal(sim, 0.0)
        
        self.similarity_matrix = np.clip(sim, 0.0, 1.0)
        
        print(f"Similarity matrix computed in {time.time() - start_time:.2f} seconds.")
        return self

    def predict(self, user_idx, movie_idx):
        """
        Predicts rating for a user and movie using sparse indexing.
        """
        # Fallback values
        fallback_movie = self.item_means_arr[movie_idx] if 0 <= movie_idx < self.num_items else self.global_mean
        fallback_user = self.user_means_arr[user_idx] if 0 <= user_idx < self.num_users else self.global_mean
        fallback = 0.6 * fallback_movie + 0.4 * fallback_user
        
        if not (0 <= user_idx < self.num_users) or not (0 <= movie_idx < self.num_items):
            return fallback
            
        # Retrieve movies rated by the user directly from the sparse CSR representation
        start_idx = self.R.indptr[user_idx]
        end_idx = self.R.indptr[user_idx+1]
        rated_movie_indices = self.R.indices[start_idx:end_idx]
        user_ratings_data = self.R.data[start_idx:end_idx]
        
        if len(rated_movie_indices) == 0:
            return fallback
            
        # Get similarities between target movie and movies rated by user
        sims = self.similarity_matrix[movie_idx, rated_movie_indices]
        
        # Find top K similar movies
        if len(sims) > self.k_neighbors:
            top_k_idx = np.argpartition(sims, -self.k_neighbors)[-self.k_neighbors:]
            top_sims = sims[top_k_idx]
            top_ratings = user_ratings_data[top_k_idx]
        else:
            top_sims = sims
            top_ratings = user_ratings_data
            
        # Weighted average
        sim_sum = np.sum(np.abs(top_sims))
        if sim_sum == 0:
            return fallback
            
        pred = np.sum(top_sims * top_ratings) / sim_sum
        return np.clip(pred, 1.0, 5.0)

    def predict_batch(self, user_indices, movie_indices):
        """
        Optimized prediction for a batch of user-movie pairs using sparse indexing.
        """
        unique_users = np.unique(user_indices)
        
        if len(unique_users) == 1:
            u_idx = unique_users[0]
            
            if not (0 <= u_idx < self.num_users):
                return np.array([self.item_means_arr[m] if 0 <= m < self.num_items else self.global_mean for m in movie_indices], dtype=np.float32)
                
            start_idx = self.R.indptr[u_idx]
            end_idx = self.R.indptr[u_idx+1]
            rated_movie_indices = self.R.indices[start_idx:end_idx]
            user_ratings_data = self.R.data[start_idx:end_idx]
            
            preds = []
            fallback_user = self.user_means_arr[u_idx]
            
            for movie_idx in movie_indices:
                fallback_movie = self.item_means_arr[movie_idx] if 0 <= movie_idx < self.num_items else self.global_mean
                fallback = 0.6 * fallback_movie + 0.4 * fallback_user
                
                if len(rated_movie_indices) == 0:
                    preds.append(fallback)
                    continue
                    
                sims = self.similarity_matrix[movie_idx, rated_movie_indices]
                
                if len(sims) > self.k_neighbors:
                    top_k_idx = np.argpartition(sims, -self.k_neighbors)[-self.k_neighbors:]
                    top_sims = sims[top_k_idx]
                    top_ratings = user_ratings_data[top_k_idx]
                else:
                    top_sims = sims
                    top_ratings = user_ratings_data
                    
                sim_sum = np.sum(np.abs(top_sims))
                if sim_sum == 0:
                    preds.append(fallback)
                else:
                    pred = np.sum(top_sims * top_ratings) / sim_sum
                    preds.append(np.clip(pred, 1.0, 5.0))
            return np.array(preds, dtype=np.float32)
            
        # Fallback for general case (multiple users in the batch)
        preds = []
        for u, m in zip(user_indices, movie_indices):
            preds.append(self.predict(u, m))
        return np.array(preds, dtype=np.float32)


class FunkSVD:
    def __init__(self, n_factors=30, lr=0.005, reg=0.02, n_epochs=20, init_mean=0.0, init_std=0.1):
        """
        Funk SVD (Matrix Factorization) using Stochastic Gradient Descent.
        n_factors: dimensionality of the latent space.
        lr: learning rate.
        reg: regularization parameter (lambda).
        n_epochs: number of training iterations.
        """
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std = init_std
        
        self.num_users = 0
        self.num_items = 0
        self.global_mean = 0.0
        
        # Model parameters
        self.user_biases = None
        self.item_biases = None
        self.P = None  # User latent factors (num_users, n_factors)
        self.Q = None  # Item latent factors (num_items, n_factors)

    def fit(self, train_df, num_users, num_items, val_df=None):
        """
        Fits Funk SVD via Stochastic Gradient Descent.
        """
        print("Fitting Funk SVD model...")
        start_time = time.time()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        
        # Calculate global mean
        self.global_mean = train_df['Rating'].mean()
        
        # Initialize biases
        self.user_biases = np.zeros(self.num_users, dtype=np.float32)
        self.item_biases = np.zeros(self.num_items, dtype=np.float32)
        
        # Initialize latent matrices randomly
        self.P = np.random.normal(self.init_mean, self.init_std, (self.num_users, self.n_factors)).astype(np.float32)
        self.Q = np.random.normal(self.init_mean, self.init_std, (self.num_items, self.n_factors)).astype(np.float32)
        
        user_ids = train_df['user_idx'].values
        item_ids = train_df['movie_idx'].values
        ratings = train_df['Rating'].values
        
        num_ratings = len(ratings)
        
        # Checkpoint variables
        best_val_rmse = float('inf')
        best_weights = None
        best_epoch = -1
        
        # SGD Optimization loop
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            
            # Shuffle indices
            indices = np.arange(num_ratings)
            np.random.shuffle(indices)
            
            loss = 0.0
            
            # Fast inner loop running SGD updates
            # Optimized numpy operations
            for idx in indices:
                u = user_ids[idx]
                i = item_ids[idx]
                r = ratings[idx]
                
                # Prediction
                pred = self.global_mean + self.user_biases[u] + self.item_biases[i] + np.dot(self.P[u], self.Q[i])
                err = r - pred
                loss += err ** 2
                
                # Update biases
                self.user_biases[u] += self.lr * (err - self.reg * self.user_biases[u])
                self.item_biases[i] += self.lr * (err - self.reg * self.item_biases[i])
                
                # Update latent vectors
                # Need copies of vectors for parallel update
                P_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * P_u_old - self.reg * self.Q[i])
                
            loss = np.sqrt(loss / num_ratings)
            
            # Check validation score if provided
            val_info = ""
            if val_df is not None:
                val_preds = self.predict_batch(val_df['user_idx'].values, val_df['movie_idx'].values)
                val_rmse = np.sqrt(np.mean((val_df['Rating'].values - val_preds) ** 2))
                val_info = f" | Val RMSE: {val_rmse:.4f}"
                
                # Checkpointing best weights
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_epoch = epoch + 1
                    best_weights = {
                        'user_biases': self.user_biases.copy(),
                        'item_biases': self.item_biases.copy(),
                        'P': self.P.copy(),
                        'Q': self.Q.copy()
                    }
            else:
                # If no validation set, just keep latest as best
                best_weights = {
                    'user_biases': self.user_biases.copy(),
                    'item_biases': self.item_biases.copy(),
                    'P': self.P.copy(),
                    'Q': self.Q.copy()
                }
                
            print(f"Epoch {epoch+1}/{self.n_epochs} - Train RMSE: {loss:.4f}{val_info} - Time: {time.time() - epoch_start:.2f}s")
            
        # Restore best weights
        if best_weights is not None:
            self.user_biases = best_weights['user_biases']
            self.item_biases = best_weights['item_biases']
            self.P = best_weights['P']
            self.Q = best_weights['Q']
            if val_df is not None:
                print(f"Restored best weights from Epoch {best_epoch} (Val RMSE: {best_val_rmse:.4f})")
                
        print(f"Funk SVD trained in {time.time() - start_time:.2f} seconds.")
        return self

    def predict(self, user_idx, movie_idx):
        """
        Predicts rating for user and movie.
        """
        # Handle out of bound indices (cold start fallback)
        has_user = 0 <= user_idx < self.num_users
        has_movie = 0 <= movie_idx < self.num_items
        
        if has_user and has_movie:
            pred = self.global_mean + self.user_biases[user_idx] + self.item_biases[movie_idx] + np.dot(self.P[user_idx], self.Q[movie_idx])
            return np.clip(pred, 1.0, 5.0)
        elif has_movie:
            # Fallback to movie bias
            return np.clip(self.global_mean + self.item_biases[movie_idx], 1.0, 5.0)
        elif has_user:
            # Fallback to user bias
            return np.clip(self.global_mean + self.user_biases[user_idx], 1.0, 5.0)
        else:
            return self.global_mean

    def predict_batch(self, user_indices, movie_indices):
        """
        Vectorized prediction for arrays of user and movie indices.
        """
        preds = np.zeros(len(user_indices), dtype=np.float32)
        for idx in range(len(user_indices)):
            preds[idx] = self.predict(user_indices[idx], movie_indices[idx])
        return preds


class UserCollaborativeFiltering:
    def __init__(self, k_neighbors=40, shrinkage=100):
        """
        User-Based Collaborative Filtering using Cosine Similarity on centered ratings.
        k_neighbors: number of similar users to consider for predicting a rating.
        shrinkage: regularization parameter for similarity to penalize user pairs with few co-rated items.
        """
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.num_users = 0
        self.num_items = 0
        self.global_mean = 0.0
        self.user_means = None
        self.item_means = None
        self.R = None  # User-Item ratings matrix (sparse CSR)
        self.R_csc = None # User-Item ratings matrix (sparse CSC)
        self.similarity_matrix = None  # User-User similarity matrix

    def fit(self, train_df, num_users, num_items):
        """
        Fits User-Based Collaborative Filtering model by storing sparse matrices for dynamic similarity calculation.
        """
        print("Fitting User-Based Collaborative Filtering model...")
        start_time = time.time()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        
        self.global_mean = train_df['Rating'].mean()
        
        # Calculate item and user means
        self.user_means = train_df.groupby('user_idx')['Rating'].mean()
        self.item_means = train_df.groupby('movie_idx')['Rating'].mean()
        self.user_means_arr = self.user_means.reindex(range(self.num_users), fill_value=self.global_mean).values
        self.item_means_arr = self.item_means.reindex(range(self.num_items), fill_value=self.global_mean).values
        
        # Create sparse rating matrix R (CSR)
        self.R = csr_matrix(
            (train_df['Rating'].values, (train_df['user_idx'].values, train_df['movie_idx'].values)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32
        )
        
        # Create CSC version of R for fast column (movie) access
        self.R_csc = self.R.tocsc()
        
        # Center ratings (subtract user mean from each active user's row in R)
        print("Centering ratings for user similarities...")
        self.R_centered = self.R.copy()
        for u in range(self.num_users):
            if u in self.user_means:
                u_mean = self.user_means[u]
                start_idx = self.R_centered.indptr[u]
                end_idx = self.R_centered.indptr[u+1]
                self.R_centered.data[start_idx:end_idx] -= u_mean
                
        # Calculate norms for cosine similarity
        self.row_norms = np.sqrt(np.array(self.R_centered.power(2).sum(axis=1)).flatten())
        self.row_norms[self.row_norms == 0] = 1.0  # Prevent division by zero
        
        # Normalize R_centered
        self.R_norm = self.R_centered.multiply(1.0 / self.row_norms[:, np.newaxis]).tocsr()
        
        # Create binary matrix for co-rated item counts
        self.R_binary = self.R.copy().tocsr()
        self.R_binary.data = np.ones_like(self.R_binary.data)
        
        print(f"User Collaborative Filtering data structures fitted in {time.time() - start_time:.2f} seconds.")
        return self

    def predict(self, user_idx, movie_idx):
        """
        Predicts rating for a user and movie using sparse indexing.
        """
        # Fallback values
        fallback_user = self.user_means_arr[user_idx] if 0 <= user_idx < self.num_users else self.global_mean
        fallback_movie = self.item_means_arr[movie_idx] if 0 <= movie_idx < self.num_items else self.global_mean
        fallback = 0.4 * fallback_user + 0.6 * fallback_movie
        
        if not (0 <= user_idx < self.num_users) or not (0 <= movie_idx < self.num_items):
            return fallback
            
        # Find users who rated this movie using CSC matrix (no dense indexing)
        start_idx = self.R_csc.indptr[movie_idx]
        end_idx = self.R_csc.indptr[movie_idx+1]
        other_users = self.R_csc.indices[start_idx:end_idx]
        movie_ratings = self.R_csc.data[start_idx:end_idx]
        
        if len(other_users) == 0:
            return fallback
            
        # Compute user similarities dynamically for the target user user_idx
        sims_u = self.R_norm[user_idx].dot(self.R_norm.T).toarray().flatten()
        
        # Apply shrinkage dynamically
        if self.shrinkage > 0:
            co_counts_u = self.R_binary[user_idx].dot(self.R_binary.T).toarray().flatten()
            sims_u = sims_u * co_counts_u / (co_counts_u + self.shrinkage)
            
        # Set self-similarity to 0
        sims_u[user_idx] = 0.0
        sims_u = np.clip(sims_u, -1.0, 1.0)
        
        # Get similarities for users who rated the movie
        sims = sims_u[other_users]
        
        # Filter for positive similarities
        pos_idx = np.where(sims > 0)[0]
        if len(pos_idx) == 0:
            return fallback
            
        sims = sims[pos_idx]
        other_users = other_users[pos_idx]
        other_ratings = movie_ratings[pos_idx]
        
        # Keep top K similar users
        if len(sims) > self.k_neighbors:
            top_k = np.argpartition(sims, -self.k_neighbors)[-self.k_neighbors:]
            top_sims = sims[top_k]
            top_users = other_users[top_k]
            top_ratings = other_ratings[top_k]
        else:
            top_sims = sims
            top_users = other_users
            top_ratings = other_ratings
            
        sim_sum = np.sum(np.abs(top_sims))
        if sim_sum == 0:
            return fallback
            
        # Weighted mean of deviations: user_mean + sum(sim * (rating - neighbor_mean)) / sum(|sim|)
        neighbor_means = self.user_means_arr[top_users]
        rating_diffs = top_ratings - neighbor_means
        pred = fallback_user + np.sum(top_sims * rating_diffs) / sim_sum
        return np.clip(pred, 1.0, 5.0)

    def predict_batch(self, user_indices, movie_indices):
        """
        Optimized batch prediction for user-movie pairs using sparse indexing.
        """
        # Optimized for single-user batch evaluations (like in MAP@10)
        unique_users = np.unique(user_indices)
        
        if len(unique_users) == 1:
            u_idx = unique_users[0]
            fallback_user = self.user_means_arr[u_idx] if 0 <= u_idx < self.num_users else self.global_mean
            
            if not (0 <= u_idx < self.num_users):
                return np.array([self.item_means_arr[m] if 0 <= m < self.num_items else self.global_mean for m in movie_indices], dtype=np.float32)
                
            # Compute similarities dynamically for the single user
            sims_u = self.R_norm[u_idx].dot(self.R_norm.T).toarray().flatten()
            if self.shrinkage > 0:
                co_counts_u = self.R_binary[u_idx].dot(self.R_binary.T).toarray().flatten()
                sims_u = sims_u * co_counts_u / (co_counts_u + self.shrinkage)
            sims_u[u_idx] = 0.0
            sims_u = np.clip(sims_u, -1.0, 1.0)
            
            preds = []
            
            for movie_idx in movie_indices:
                fallback_movie = self.item_means_arr[movie_idx] if 0 <= movie_idx < self.num_items else self.global_mean
                fallback = 0.4 * fallback_user + 0.6 * fallback_movie
                
                if not (0 <= movie_idx < self.num_items):
                    preds.append(fallback)
                    continue
                    
                # Find users who rated this movie using CSC matrix (no dense indexing)
                start_idx = self.R_csc.indptr[movie_idx]
                end_idx = self.R_csc.indptr[movie_idx+1]
                other_users = self.R_csc.indices[start_idx:end_idx]
                movie_ratings = self.R_csc.data[start_idx:end_idx]
                
                if len(other_users) == 0:
                    preds.append(fallback)
                    continue
                    
                sims = sims_u[other_users]
                pos_idx = np.where(sims > 0)[0]
                if len(pos_idx) == 0:
                    preds.append(fallback)
                    continue
                    
                sims = sims[pos_idx]
                other_users = other_users[pos_idx]
                other_ratings = movie_ratings[pos_idx]
                
                if len(sims) > self.k_neighbors:
                    top_k = np.argpartition(sims, -self.k_neighbors)[-self.k_neighbors:]
                    top_sims = sims[top_k]
                    top_users = other_users[top_k]
                    top_ratings = other_ratings[top_k]
                else:
                    top_sims = sims
                    top_users = other_users
                    top_ratings = other_ratings
                    
                sim_sum = np.sum(np.abs(top_sims))
                if sim_sum == 0:
                    preds.append(fallback)
                else:
                    neighbor_means = self.user_means_arr[top_users]
                    rating_diffs = top_ratings - neighbor_means
                    pred = fallback_user + np.sum(top_sims * rating_diffs) / sim_sum
                    preds.append(np.clip(pred, 1.0, 5.0))
            return np.array(preds, dtype=np.float32)
            
        # Fallback for general case (multiple users in the batch)
        preds = []
        for u, m in zip(user_indices, movie_indices):
            preds.append(self.predict(u, m))
        return np.array(preds, dtype=np.float32)


class NeuralCFRecommender:
    def __init__(self):
        """
        Neural Collaborative Filtering (NeuMF) NumPy Inference Model.
        Loads embedding parameters and MLP weights to make fast, PyTorch-free predictions.
        """
        self.P_gmf = None
        self.Q_gmf = None
        self.P_mlp = None
        self.Q_mlp = None
        self.mlp_weights = []
        self.mlp_biases = []
        self.final_weight = None
        self.final_bias = None
        self.num_users = 0
        self.num_items = 0
        self.global_mean = 0.0

    def load_model(self, model_path):
        """
        Loads pre-trained NeuMF parameters from a .npz file.
        """
        if not os.path.exists(model_path):
            return False
            
        data = np.load(model_path, allow_pickle=True)
        self.P_gmf = data['P_gmf']
        self.Q_gmf = data['Q_gmf']
        self.P_mlp = data['P_mlp']
        self.Q_mlp = data['Q_mlp']
        self.final_weight = data['final_weight']
        self.final_bias = data['final_bias']
        self.num_users = int(data['num_users'])
        self.num_items = int(data['num_items'])
        self.global_mean = float(data['global_mean'])
        
        # Reconstruct MLP hidden layers
        self.mlp_weights = []
        self.mlp_biases = []
        i = 0
        while f'mlp_w_{i}' in data:
            self.mlp_weights.append(data[f'mlp_w_{i}'])
            self.mlp_biases.append(data[f'mlp_b_{i}'])
            i += 1
            
        return True

    def predict(self, user_idx, movie_idx):
        """
        Predicts rating for a single user-movie pair.
        """
        if not (0 <= user_idx < self.num_users) or not (0 <= movie_idx < self.num_items):
            return self.global_mean
            
        # GMF branch lookup
        p_gmf = self.P_gmf[user_idx]
        q_gmf = self.Q_gmf[movie_idx]
        gmf_out = p_gmf * q_gmf
        
        # MLP branch lookup
        p_mlp = self.P_mlp[user_idx]
        q_mlp = self.Q_mlp[movie_idx]
        mlp_out = np.concatenate([p_mlp, q_mlp])
        
        # Forward pass through MLP layers
        for W, b in zip(self.mlp_weights, self.mlp_biases):
            mlp_out = np.dot(W, mlp_out) + b
            mlp_out = np.maximum(0, mlp_out)  # ReLU
            
        # Concat outputs & final projection
        concat_out = np.concatenate([gmf_out, mlp_out])
        pred = np.dot(self.final_weight, concat_out) + self.final_bias
        return np.clip(pred, 1.0, 5.0)

    def predict_batch(self, user_indices, movie_indices):
        """
        Vectorized prediction for a batch of user and movie indices.
        """
        valid_mask = (user_indices >= 0) & (user_indices < self.num_users) & \
                     (movie_indices >= 0) & (movie_indices < self.num_items)
                     
        preds = np.full(len(user_indices), self.global_mean, dtype=np.float32)
        if not np.any(valid_mask):
            return preds
            
        v_users = user_indices[valid_mask]
        v_movies = movie_indices[valid_mask]
        
        # GMF branch batch lookup
        p_gmf = self.P_gmf[v_users]
        q_gmf = self.Q_gmf[v_movies]
        gmf_out = p_gmf * q_gmf
        
        # MLP branch batch lookup
        p_mlp = self.P_mlp[v_users]
        q_mlp = self.Q_mlp[v_movies]
        mlp_out = np.hstack([p_mlp, q_mlp])
        
        # MLP forward pass
        for W, b in zip(self.mlp_weights, self.mlp_biases):
            mlp_out = np.dot(mlp_out, W.T) + b
            mlp_out = np.maximum(0, mlp_out)
            
        # Combine
        concat_out = np.hstack([gmf_out, mlp_out])
        pred = np.dot(concat_out, self.final_weight.T) + self.final_bias
        
        preds[valid_mask] = np.clip(pred.flatten(), 1.0, 5.0)
        return preds


class ContentBasedRecommender:
    def __init__(self, k_neighbors=20):
        """
        Content-Based Filtering using TF-IDF on movie title words + release year decade.
        Recommends movies with similar title/era characteristics to what the user has enjoyed.
        k_neighbors: number of most similar items to consider for predictions.
        """
        self.k_neighbors = k_neighbors
        self.num_users = 0
        self.num_items = 0
        self.global_mean = 0.0
        self.tfidf_matrix = None       # TF-IDF matrix (sparse or dense)
        self.tfidf_dense = None        # Dense TF-IDF matrix
        self.item_similarity = None    # Content cosine similarity
        self.user_profiles = None      # Dense user content profiles
        self.user_profile_norms = None # Norms for fast cosine computation
        self.user_means_arr = None
        self.item_means_arr = None
        self.movie_norms = None        # Precomputed movie norms

    def fit(self, train_df, num_users, num_items, content_df, movie_to_idx):
        """
        Fits the content-based model using vectorized matrix operations.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("Fitting Content-Based Filtering model...")
        start_time = time.time()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.global_mean = train_df['Rating'].mean()
        
        # User and item means for fallback
        user_means = train_df.groupby('user_idx')['Rating'].mean()
        item_means = train_df.groupby('movie_idx')['Rating'].mean()
        self.user_means_arr = user_means.reindex(range(self.num_users), fill_value=self.global_mean).values
        self.item_means_arr = item_means.reindex(range(self.num_items), fill_value=self.global_mean).values
        
        # Build content strings indexed by movie_idx
        content_strings = [''] * self.num_items
        for _, row in content_df.iterrows():
            mid = row['movie_id']
            if mid in movie_to_idx:
                idx = movie_to_idx[mid]
                if idx < self.num_items:
                    content_strings[idx] = str(row['content_string'])
        
        # TF-IDF Vectorization
        print("Computing TF-IDF vectors for movie content...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(content_strings)
        self.tfidf_dense = self.tfidf_matrix.toarray().astype(np.float32)
        print(f"TF-IDF matrix shape: {self.tfidf_dense.shape}")
        
        # Precompute movie norms
        self.movie_norms = np.linalg.norm(self.tfidf_dense, axis=1)
        self.movie_norms[self.movie_norms == 0] = 1.0
        
        # Item-Item Cosine Similarity in content space
        print("Computing item-item content similarity...")
        self.item_similarity = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        np.fill_diagonal(self.item_similarity, 0.0)
        
        # Vectorized User Profiles construction: user_profiles = R_high * tfidf_matrix
        print("Building user content profiles via matrix multiplication...")
        high_ratings_df = train_df[train_df['Rating'] >= 3.5]
        if len(high_ratings_df) == 0:
            high_ratings_df = train_df
            
        R_high = csr_matrix(
            (high_ratings_df['Rating'].values, (high_ratings_df['user_idx'].values, high_ratings_df['movie_idx'].values)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32
        )
        
        self.user_profiles = R_high.dot(self.tfidf_matrix).toarray().astype(np.float32)
        
        # Divide profiles by the sum of user ratings
        user_rating_sums = np.array(R_high.sum(axis=1)).flatten()
        user_rating_sums[user_rating_sums == 0] = 1.0
        self.user_profiles = self.user_profiles / user_rating_sums[:, np.newaxis]
        
        # Precompute user profile norms
        self.user_profile_norms = np.linalg.norm(self.user_profiles, axis=1)
        self.user_profile_norms[self.user_profile_norms == 0] = 1.0
        
        print(f"Content-Based model fitted in {time.time() - start_time:.2f} seconds.")
        return self

    def predict(self, user_idx, movie_idx):
        """
        Predicts rating affinity based on cosine similarity between user content profile
        and the movie's TF-IDF vector.
        """
        fallback_user = self.user_means_arr[user_idx] if 0 <= user_idx < self.num_users else self.global_mean
        fallback_movie = self.item_means_arr[movie_idx] if 0 <= movie_idx < self.num_items else self.global_mean
        fallback = 0.4 * fallback_user + 0.6 * fallback_movie
        
        if not (0 <= user_idx < self.num_users) or not (0 <= movie_idx < self.num_items):
            return fallback
        
        # Cosine similarity
        movie_vec = self.tfidf_dense[movie_idx]
        movie_norm = self.movie_norms[movie_idx]
        u_norm = self.user_profile_norms[user_idx]
        
        if movie_norm == 0 or u_norm == 0:
            return fallback
        
        cos_sim = np.dot(self.user_profiles[user_idx], movie_vec) / (u_norm * movie_norm)
        pred = 1.0 + 4.0 * max(0.0, cos_sim)
        return np.clip(pred, 1.0, 5.0)

    def predict_batch(self, user_indices, movie_indices):
        """
        Vectorized batch prediction for user-movie pairs.
        """
        preds = np.full(len(user_indices), self.global_mean, dtype=np.float32)
        
        unique_users = np.unique(user_indices)
        
        # Special optimization for single user with many movies (e.g. MAP@10 query)
        if len(unique_users) == 1:
            u_idx = unique_users[0]
            if not (0 <= u_idx < self.num_users):
                return preds
            
            user_prof = self.user_profiles[u_idx]
            u_norm = self.user_profile_norms[u_idx]
            fallback_user = self.user_means_arr[u_idx]
            
            valid_movie_mask = (movie_indices >= 0) & (movie_indices < self.num_items)
            
            if np.any(valid_movie_mask):
                valid_movies = movie_indices[valid_movie_mask]
                movie_vecs = self.tfidf_dense[valid_movies]
                movie_norms = self.movie_norms[valid_movies]
                
                # Batch dot product
                dots = np.dot(movie_vecs, user_prof)
                
                # Cosine similarity
                cos_sims = dots / (u_norm * movie_norms)
                cos_sims = np.clip(cos_sims, 0.0, 1.0)
                
                # Predict
                preds[valid_movie_mask] = 1.0 + 4.0 * cos_sims
                
                # Fallback for movies with zero norm
                zero_norm_mask = (movie_norms == 1.0) & (np.all(movie_vecs == 0.0, axis=1))
                if np.any(zero_norm_mask):
                    zero_norm_movies = valid_movies[zero_norm_mask]
                    zero_norm_indices = np.where(valid_movie_mask)[0][zero_norm_mask]
                    preds[zero_norm_indices] = 0.4 * fallback_user + 0.6 * self.item_means_arr[zero_norm_movies]
            
            for i, movie_idx in enumerate(movie_indices):
                if not (0 <= movie_idx < self.num_items):
                    preds[i] = self.global_mean
            
            return np.clip(preds, 1.0, 5.0)
        
        # General case (multiple users and movies)
        valid_mask = (user_indices >= 0) & (user_indices < self.num_users) & \
                     (movie_indices >= 0) & (movie_indices < self.num_items)
        
        if np.any(valid_mask):
            v_users = user_indices[valid_mask]
            v_movies = movie_indices[valid_mask]
            
            u_profiles = self.user_profiles[v_users]
            m_vectors = self.tfidf_dense[v_movies]
            
            u_norms = self.user_profile_norms[v_users]
            m_norms = self.movie_norms[v_movies]
            
            # Row-wise dot product
            dots = np.sum(u_profiles * m_vectors, axis=1)
            
            # Cosine similarity
            cos_sims = dots / (u_norms * m_norms)
            cos_sims = np.clip(cos_sims, 0.0, 1.0)
            
            # Predictions
            preds[valid_mask] = 1.0 + 4.0 * cos_sims
            
            # Fallback for movies with zero norm
            zero_norm_mask = (m_norms == 1.0) & (np.all(m_vectors == 0.0, axis=1))
            if np.any(zero_norm_mask):
                zero_norm_users = v_users[zero_norm_mask]
                zero_norm_movies = v_movies[zero_norm_mask]
                zero_norm_indices = np.where(valid_mask)[0][zero_norm_mask]
                
                fallback_users = self.user_means_arr[zero_norm_users]
                fallback_movies = self.item_means_arr[zero_norm_movies]
                preds[zero_norm_indices] = 0.4 * fallback_users + 0.6 * fallback_movies
                
        # Fill in out of bounds fallbacks
        for i in range(len(user_indices)):
            if not valid_mask[i]:
                u = user_indices[i]
                m = movie_indices[i]
                fallback_user = self.user_means_arr[u] if 0 <= u < self.num_users else self.global_mean
                fallback_movie = self.item_means_arr[m] if 0 <= m < self.num_items else self.global_mean
                preds[i] = 0.4 * fallback_user + 0.6 * fallback_movie
                
        return np.clip(preds, 1.0, 5.0)

    def save_model(self, path):
        """Saves the content-based model to a .npz file."""
        save_dict = {
            'user_profiles': self.user_profiles,
            'user_profile_norms': self.user_profile_norms,
            'item_similarity': self.item_similarity,
            'user_means_arr': self.user_means_arr,
            'item_means_arr': self.item_means_arr,
            'global_mean': self.global_mean,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'k_neighbors': self.k_neighbors,
            'tfidf_matrix': self.tfidf_dense
        }
        np.savez(path, **save_dict)
        print(f"Content-Based model saved to {path}")

    def load_model(self, path):
        """Loads a pre-trained content-based model from a .npz file."""
        if not os.path.exists(path):
            return False
        
        data = np.load(path, allow_pickle=True)
        self.user_profiles = data['user_profiles']
        self.user_profile_norms = data['user_profile_norms']
        self.item_similarity = data['item_similarity']
        self.tfidf_matrix = data['tfidf_matrix']
        self.tfidf_dense = self.tfidf_matrix
        self.user_means_arr = data['user_means_arr']
        self.item_means_arr = data['item_means_arr']
        self.global_mean = float(data['global_mean'])
        self.num_users = int(data['num_users'])
        self.num_items = int(data['num_items'])
        self.k_neighbors = int(data['k_neighbors'])
        
        # Reconstruct movie norms
        self.movie_norms = np.linalg.norm(self.tfidf_dense, axis=1)
        self.movie_norms[self.movie_norms == 0] = 1.0
        return True
