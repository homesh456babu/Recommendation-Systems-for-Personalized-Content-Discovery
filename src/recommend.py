import os
import numpy as np
import pandas as pd
import json
from src.data_pipeline import get_movie_metadata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

class RecommenderService:
    def __init__(self):
        self.movie_meta = {}
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        
        # Models data
        self.svd_data = None
        self.cf_data = None
        self.user_cf_data = None
        self.ncf_model = None
        self.content_model = None
        self.train_ratings = None
        self.R_sparse = None
        
        self.load_metadata()

    def load_metadata(self):
        """Loads index mappings, metadata, and training ratings."""
        try:
            self.user_to_idx = np.load(os.path.join(PROCESSED_DIR, 'user_to_idx.npy'), allow_pickle=True).item()
            self.movie_to_idx = np.load(os.path.join(PROCESSED_DIR, 'movie_to_idx.npy'), allow_pickle=True).item()
            self.idx_to_movie = np.load(os.path.join(PROCESSED_DIR, 'idx_to_movie.npy'), allow_pickle=True).item()
            
            # Load training data to know what users already watched
            train_path = os.path.join(PROCESSED_DIR, 'train.csv')
            if os.path.exists(train_path):
                self.train_ratings = pd.read_csv(train_path)
                
            # Load movie titles/year metadata
            self.movie_meta = get_movie_metadata()
        except FileNotFoundError:
            print("Metadata files not found. Please run the data pipeline and evaluation first.")

    def load_svd_model(self):
        """Loads trained SVD model weights."""
        svd_path = os.path.join(PROCESSED_DIR, 'svd_model.npz')
        if os.path.exists(svd_path):
            self.svd_data = np.load(svd_path)
            return True
        return False

    def load_cf_model(self):
        """Loads trained Item-CF similarity matrix."""
        cf_path = os.path.join(PROCESSED_DIR, 'item_cf_model.npz')
        if os.path.exists(cf_path):
            self.cf_data = np.load(cf_path)
            return True
        return False

    def get_user_history(self, customer_id, top_n=10):
        """Returns the list of movies a user has rated in the training set, sorted by rating."""
        if self.train_ratings is None or customer_id not in self.user_to_idx:
            return []
            
        u_idx = self.user_to_idx[customer_id]
        u_history = self.train_ratings[self.train_ratings['user_idx'] == u_idx]
        u_history = u_history.sort_values(by='Rating', ascending=False)
        
        history_list = []
        for _, row in u_history.iterrows():
            m_idx = int(row['movie_idx'])
            meta = self.movie_meta.get(m_idx, {'title': f"Movie ID {row['MovieID']}", 'year': ""})
            history_list.append({
                'movie_id': int(row['MovieID']),
                'movie_idx': m_idx,
                'title': meta['title'],
                'year': meta['year'],
                'rating': float(row['Rating']),
                'date': str(row['Date'])[:10]
            })
            
        return history_list[:top_n]

    def get_svd_recommendations(self, customer_id, k=10):
        """Generates Top-K recommendations for a user using Funk SVD."""
        if not self.load_svd_model() or customer_id not in self.user_to_idx:
            return []
            
        u_idx = self.user_to_idx[customer_id]
        
        # Load weights
        global_mean = self.svd_data['global_mean']
        u_bias = self.svd_data['user_biases'][u_idx]
        item_biases = self.svd_data['item_biases']
        P_u = self.svd_data['P'][u_idx]
        Q = self.svd_data['Q']
        num_items = self.svd_data['num_items']
        
        # Get watched movie indices (in index space)
        watched = set()
        if self.train_ratings is not None:
            watched = set(self.train_ratings[self.train_ratings['user_idx'] == u_idx]['movie_idx'].values)
            
        # Predict ratings for all items not watched
        unseen_indices = [i for i in range(num_items) if i not in watched]
        
        # Compute dot products in batch
        preds = global_mean + u_bias + item_biases[unseen_indices] + np.dot(Q[unseen_indices], P_u)
        preds = np.clip(preds, 1.0, 5.0)
        
        # Get Top-K
        top_k_idx = np.argsort(-preds)[:k]
        
        recommendations = []
        for i in top_k_idx:
            m_idx = unseen_indices[i]
            movie_id = self.idx_to_movie[m_idx]
            meta = self.movie_meta.get(m_idx, {'title': f"Movie ID {movie_id}", 'year': ""})
            pred_score = float(preds[i])
            
            # Explain SVD recommendations: find user latent factor overlap
            explanation = self.generate_svd_explanation(u_idx, m_idx, P_u, Q[m_idx])
            
            recommendations.append({
                'movie_id': int(movie_id),
                'movie_idx': m_idx,
                'title': meta['title'],
                'year': meta['year'],
                'score': round(pred_score, 2),
                'explanation': explanation
            })
            
        return recommendations

    def get_cf_recommendations(self, customer_id, k=10):
        """Generates Top-K recommendations using Item-CF."""
        if not self.load_cf_model() or customer_id not in self.user_to_idx:
            return []
            
        u_idx = self.user_to_idx[customer_id]
        
        # Load similarities and historical data
        similarity_matrix = self.cf_data['similarity_matrix']
        global_mean = self.cf_data['global_mean']
        num_items = self.cf_data['num_items']
        
        # Get rated movies and their ratings
        user_ratings_row = self.train_ratings[self.train_ratings['user_idx'] == u_idx]
        if user_ratings_row.empty:
            return self.get_coldstart_recommendations(k)
            
        watched = set(user_ratings_row['movie_idx'].values)
        user_ratings = dict(zip(user_ratings_row['movie_idx'], user_ratings_row['Rating']))
        
        unseen_indices = [i for i in range(num_items) if i not in watched]
        
        preds = []
        for m_idx in unseen_indices:
            # Similarities between m_idx and all items rated by user
            rated_indices = list(user_ratings.keys())
            sims = similarity_matrix[m_idx, rated_indices]
            
            # Take top 40 similar items
            k_neighbors = 40
            if len(sims) > k_neighbors:
                top_k = np.argpartition(sims, -k_neighbors)[-k_neighbors:]
                top_sims = sims[top_k]
                top_ratings = np.array([user_ratings[rated_indices[idx]] for idx in top_k])
            else:
                top_sims = sims
                top_ratings = np.array([user_ratings[idx] for idx in rated_indices])
                
            sim_sum = np.sum(np.abs(top_sims))
            if sim_sum == 0:
                # Fallback to movie mean if saved, else global mean
                # (Simple fallback for prediction here)
                pred = global_mean
            else:
                pred = np.sum(top_sims * top_ratings) / sim_sum
            preds.append((m_idx, np.clip(pred, 1.0, 5.0)))
            
        # Sort recommendations
        preds.sort(key=lambda x: x[1], reverse=True)
        top_k_preds = preds[:k]
        
        recommendations = []
        for m_idx, score in top_k_preds:
            movie_id = self.idx_to_movie[m_idx]
            meta = self.movie_meta.get(m_idx, {'title': f"Movie ID {movie_id}", 'year': ""})
            
            # Explain Item-CF recommendation
            explanation = self.generate_cf_explanation(u_idx, m_idx, user_ratings, similarity_matrix)
            
            recommendations.append({
                'movie_id': int(movie_id),
                'movie_idx': m_idx,
                'title': meta['title'],
                'year': meta['year'],
                'score': round(float(score), 2),
                'explanation': explanation
            })
            
        return recommendations

    def generate_cf_explanation(self, u_idx, m_idx, user_ratings, similarity_matrix):
        """Generates an item-similarity explanation for recommendations."""
        # Find which of the user's highly rated items (>=3.5) has the highest similarity to the recommended item
        best_sim = -1.0
        best_rated_movie_idx = None
        
        for rated_m, rating in user_ratings.items():
            if rating >= 3.5:
                sim = similarity_matrix[m_idx, rated_m]
                if sim > best_sim:
                    best_sim = sim
                    best_rated_movie_idx = rated_m
                    
        if best_rated_movie_idx is not None and best_sim > 0.05:
            meta = self.movie_meta.get(best_rated_movie_idx, {'title': "a movie you liked"})
            pct = int(best_sim * 100)
            return f"Because you rated '{meta['title']}' ({int(user_ratings[best_rated_movie_idx])}★), which has a {pct}% content/rating overlap."
        else:
            return "Based on similarity matching with items in your viewing profile."

    def generate_svd_explanation(self, u_idx, m_idx, P_u, Q_m):
        """Generates a latent factor overlap explanation."""
        # Find which latent factor dimension contributes the most to the dot product
        overlap = P_u * Q_m
        best_factor = np.argmax(np.abs(overlap))
        
        direction = "positive" if overlap[best_factor] > 0 else "negative"
        strength = abs(overlap[best_factor])
        
        if strength > 0.05 and direction == "positive":
            return f"Highly aligns with your profile's preferences for style/theme dimension #{best_factor + 1}."
        else:
            return "Matches your generalized platform rating preferences."

    def get_similar_movies(self, movie_id, k=10):
        """Returns the Top-K most similar movies in the catalog based on rating similarities."""
        if not self.load_cf_model() or movie_id not in self.movie_to_idx:
            return []
            
        m_idx = self.movie_to_idx[movie_id]
        similarity_matrix = self.cf_data['similarity_matrix']
        
        # Get similarities for this movie index
        sims = similarity_matrix[m_idx]
        
        # Sort indices
        top_indices = np.argsort(-sims)[:k+1] # Include k+1 in case self is included (should be 0 anyway)
        
        similar_list = []
        for i in top_indices:
            if i == m_idx:
                continue
                
            sim_score = float(sims[i])
            if sim_score <= 0.0:
                continue
                
            m_id = self.idx_to_movie[i]
            meta = self.movie_meta.get(i, {'title': f"Movie ID {m_id}", 'year': ""})
            
            similar_list.append({
                'movie_id': int(m_id),
                'movie_idx': i,
                'title': meta['title'],
                'year': meta['year'],
                'similarity': round(sim_score, 2)
            })
            
        return similar_list[:k]

    def get_coldstart_recommendations(self, k=10, genres=None):
        """
        Handles cold-start recommendations for new users.
        Returns the overall highest-rated movies with at least 50 ratings.
        If genres are provided, it filters movies by keyword/genre (if available in title).
        """
        if self.train_ratings is None:
            return []
            
        # Compute popular items (ratings count and average rating)
        stats = self.train_ratings.groupby('movie_idx').agg(
            avg_rating=('Rating', 'mean'),
            count=('Rating', 'count')
        )
        
        # Filter items with at least 30 ratings to ensure popularity and quality
        min_ratings = 30
        popular = stats[stats['count'] >= min_ratings].copy()
        
        # Bayesean average rating or weighted sorting: score = avg_rating * (count / (count + min_ratings))
        # This prevents movies with 1 rating of 5.0 from topping the charts
        global_mean = self.train_ratings['Rating'].mean()
        popular['score'] = (popular['avg_rating'] * popular['count'] + global_mean * min_ratings) / (popular['count'] + min_ratings)
        popular = popular.sort_values(by='score', ascending=False)
        
        recommendations = []
        count = 0
        for m_idx, row in popular.iterrows():
            movie_id = self.idx_to_movie[m_idx]
            meta = self.movie_meta.get(m_idx, {'title': f"Movie ID {movie_id}", 'year': ""})
            
            # Simple genre heuristic based on movie titles (MovieLens contains genre in u.item, but raw Netflix format doesn't.
            # In our simulation, we can search keywords or recommend top charts by default.)
            title = meta['title']
            
            if genres:
                # Convert genres to lowercase list
                genre_keywords = [g.lower() for g in genres]
                
                # In MovieLens, titles sometimes contain indicators, or we can use simulated genre tags.
                # For simplicity, we match strings or search popular classics.
                # If genre matches, add it, else skip
                match = False
                # Mock some mappings of keywords to common 90s genres for rich simulation
                title_lower = title.lower()
                sci_fi_keywords = ['star wars', 'jurassic', 'matrix', 'terminator', 'alien', 'star trek', 'fiction']
                action_keywords = ['die hard', 'speed', 'batman', 'goldeneye', 'indiana jones', 'mission', 'fugitive', 'braveheart']
                comedy_keywords = ['toy story', 'liar liar', 'home alone', 'mrs. doubtfire', 'dumb', 'ace ventura', 'clerks']
                romance_keywords = ['titanic', 'forrest gump', 'pretty woman', 'sleepless', 'jerry maguire', 'shakespeare']
                drama_keywords = ['shawshank', 'pulp fiction', 'schindler', 'godfather', 'fargo', 'good will hunting', 'silence']
                
                genre_map = {
                    'action': action_keywords,
                    'sci-fi': sci_fi_keywords,
                    'comedy': comedy_keywords,
                    'romance': romance_keywords,
                    'drama': drama_keywords
                }
                
                for gk in genre_keywords:
                    if gk in genre_map:
                        if any(kw in title_lower for kw in genre_map[gk]):
                            match = True
                            break
                if not match:
                    continue
                    
            recommendations.append({
                'movie_id': int(movie_id),
                'movie_idx': m_idx,
                'title': meta['title'],
                'year': meta['year'],
                'score': round(float(row['avg_rating']), 2),
                'explanation': f"Recommended as a top-rated popular choice ({int(row['count'])} ratings, {row['avg_rating']:.1f}★)."
            })
            count += 1
            if count >= k:
                break
                
        # If genre filtering returned too few results, top up with general popular items
        if len(recommendations) < k and genres:
            additional = self.get_coldstart_recommendations(k - len(recommendations), genres=None)
            for item in additional:
                if item['movie_idx'] not in [r['movie_idx'] for r in recommendations]:
                    recommendations.append(item)
                    
        return recommendations[:k]

    def load_user_cf_model(self):
        """Loads trained User-CF similarities and parameters."""
        user_cf_path = os.path.join(PROCESSED_DIR, 'user_cf_model.npz')
        if os.path.exists(user_cf_path):
            self.user_cf_data = np.load(user_cf_path)
            return True
        return False

    def load_ncf_model(self):
        """Loads trained Neural-CF model weights."""
        ncf_path = os.path.join(PROCESSED_DIR, 'neural_cf_model.npz')
        if os.path.exists(ncf_path):
            from src.models import NeuralCFRecommender
            self.ncf_model = NeuralCFRecommender()
            self.ncf_model.load_model(ncf_path)
            return True
        return False

    def load_content_model(self):
        """Loads trained Content-Based model weights."""
        if self.content_model is not None:
            return True
        content_path = os.path.join(PROCESSED_DIR, 'content_based_model.npz')
        if os.path.exists(content_path):
            from src.models import ContentBasedRecommender
            self.content_model = ContentBasedRecommender()
            self.content_model.load_model(content_path)
            return True
        return False

    def compute_user_similarities_dynamic(self, u_idx):
        """Computes user similarities for user u_idx dynamically on the fly."""
        if self.R_sparse is None:
            from scipy.sparse import csr_matrix
            self.R_sparse = csr_matrix(
                (self.train_ratings['Rating'].values, (self.train_ratings['user_idx'].values, self.train_ratings['movie_idx'].values)),
                shape=(len(self.user_to_idx), len(self.movie_to_idx)),
                dtype=np.float32
            )
            
        if not hasattr(self, 'user_means_dict'):
            user_means = self.train_ratings.groupby('user_idx')['Rating'].mean()
            self.user_means_dict = dict(zip(user_means.index, user_means.values))
            
        u_ratings = self.R_sparse[u_idx].toarray().flatten()
        u_mask = (u_ratings > 0)
        if not np.any(u_mask):
            return np.zeros(len(self.user_to_idx), dtype=np.float32)
            
        u_mean = self.user_means_dict.get(u_idx, self.train_ratings['Rating'].mean())
        u_centered = np.zeros_like(u_ratings)
        u_centered[u_mask] = u_ratings[u_mask] - u_mean
        u_norm = np.linalg.norm(u_centered)
        if u_norm == 0:
            u_norm = 1.0
            
        if not hasattr(self, 'R_norm_rec'):
            from scipy.sparse import csr_matrix
            R_centered = self.R_sparse.copy()
            for u in range(len(self.user_to_idx)):
                if u in self.user_means_dict:
                    mean_val = self.user_means_dict[u]
                    start_idx = R_centered.indptr[u]
                    end_idx = R_centered.indptr[u+1]
                    R_centered.data[start_idx:end_idx] -= mean_val
                    
            row_norms = np.sqrt(np.array(R_centered.power(2).sum(axis=1)).flatten())
            row_norms[row_norms == 0] = 1.0
            self.R_norm_rec = R_centered.multiply(1.0 / row_norms[:, np.newaxis]).tocsr()
            
            self.R_binary_rec = self.R_sparse.copy().tocsr()
            self.R_binary_rec.data = np.ones_like(self.R_binary_rec.data)
            
        sims_u = self.R_norm_rec[u_idx].dot(self.R_norm_rec.T).toarray().flatten()
        
        # Apply shrinkage dynamically (shrinkage=100 for User-CF)
        co_counts_u = self.R_binary_rec[u_idx].dot(self.R_binary_rec.T).toarray().flatten()
        sims_u = sims_u * co_counts_u / (co_counts_u + 100)
        
        sims_u[u_idx] = 0.0
        return np.clip(sims_u, -1.0, 1.0)

    def get_user_cf_recommendations(self, customer_id, k=10):
        """Generates Top-K recommendations using User-Based CF."""
        if not self.load_user_cf_model() or customer_id not in self.user_to_idx:
            return []
            
        u_idx = self.user_to_idx[customer_id]
        
        # Load ratings, means
        user_means = self.user_cf_data['user_means']
        user_means_idx = self.user_cf_data['user_means_idx']
        user_means_dict = dict(zip(user_means_idx, user_means))
        
        global_mean = self.user_cf_data['global_mean']
        num_items = self.user_cf_data['num_items']
        
        # Get watched movie indices
        watched = set()
        if self.train_ratings is not None:
            watched = set(self.train_ratings[self.train_ratings['user_idx'] == u_idx]['movie_idx'].values)
            
        unseen_indices = [i for i in range(num_items) if i not in watched]
        
        if not unseen_indices:
            return []
            
        # Get target user similarities dynamically
        sims_u = self.compute_user_similarities_dynamic(u_idx)
        
        # Find neighbors (users with positive similarity)
        pos_neighbor_indices = np.where(sims_u > 0)[0]
        if len(pos_neighbor_indices) == 0:
            return self.get_coldstart_recommendations(k)
            
        # Select Top-40 neighbors
        k_neighbors = 40
        if len(pos_neighbor_indices) > k_neighbors:
            top_k_indices = pos_neighbor_indices[np.argpartition(sims_u[pos_neighbor_indices], -k_neighbors)[-k_neighbors:]]
        else:
            top_k_indices = pos_neighbor_indices
            
        top_sims = sims_u[top_k_indices]
        
        # Get neighbor ratings and center them
        neighbor_ratings = self.R_sparse[top_k_indices, :].toarray()
        neighbor_means = np.array([user_means_dict.get(v, global_mean) for v in top_k_indices])
        
        neighbor_mask = (neighbor_ratings > 0)
        neighbor_deviations = np.zeros_like(neighbor_ratings)
        neighbor_deviations[neighbor_mask] = neighbor_ratings[neighbor_mask] - neighbor_means[:, np.newaxis][neighbor_mask]
        
        # Weighted sum of deviations per item
        weighted_deviations = np.dot(top_sims, neighbor_deviations)
        sim_sums = np.dot(np.abs(top_sims), neighbor_mask)
        
        # Predict ratings
        mean_u = user_means_dict.get(u_idx, global_mean)
        preds = np.zeros(num_items)
        valid_pred_mask = (sim_sums > 0)
        preds[valid_pred_mask] = mean_u + weighted_deviations[valid_pred_mask] / sim_sums[valid_pred_mask]
        
        # Fallback for items with no rating neighbors
        fallback_item_means = self.train_ratings.groupby('movie_idx')['Rating'].mean().reindex(range(num_items), fill_value=global_mean).values
        fallback_preds = 0.4 * mean_u + 0.6 * fallback_item_means
        preds[~valid_pred_mask] = fallback_preds[~valid_pred_mask]
        preds = np.clip(preds, 1.0, 5.0)
        
        # Sort Top-K unseen
        unseen_preds = [(m, preds[m]) for m in unseen_indices]
        unseen_preds.sort(key=lambda x: x[1], reverse=True)
        top_k_preds = unseen_preds[:k]
        
        recommendations = []
        for m_idx, score in top_k_preds:
            movie_id = self.idx_to_movie[m_idx]
            meta = self.movie_meta.get(m_idx, {'title': f"Movie ID {movie_id}", 'year': ""})
            explanation = self.generate_user_cf_explanation(u_idx, m_idx, top_k_indices, top_sims, neighbor_ratings[:, m_idx])
            
            recommendations.append({
                'movie_id': int(movie_id),
                'movie_idx': m_idx,
                'title': meta['title'],
                'year': meta['year'],
                'score': round(float(score), 2),
                'explanation': explanation
            })
            
        return recommendations

    def generate_user_cf_explanation(self, u_idx, m_idx, neighbor_indices, similarities, ratings):
        """Generates a user-similarity explanation for User-CF."""
        best_contrib = -1.0
        best_neighbor_idx = None
        
        for idx in range(len(neighbor_indices)):
            v_idx = neighbor_indices[idx]
            sim = similarities[idx]
            rating = ratings[idx]
            
            if rating >= 3.5:
                contrib = sim * rating
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_neighbor_idx = v_idx
                    
        if best_neighbor_idx is not None:
            idx_to_user = np.load(os.path.join(PROCESSED_DIR, 'idx_to_user.npy'), allow_pickle=True).item()
            neighbor_cust_id = idx_to_user.get(best_neighbor_idx, best_neighbor_idx)
            rating_val = int(ratings[np.where(neighbor_indices == best_neighbor_idx)[0][0]])
            return f"Because you share tastes with User {neighbor_cust_id}, who rated this movie {rating_val}★."
        else:
            return "Recommended based on preference matching with similar users."

    def get_ncf_recommendations(self, customer_id, k=10):
        """Generates Top-K recommendations using Neural Collaborative Filtering."""
        if not self.load_ncf_model() or customer_id not in self.user_to_idx:
            return []
            
        u_idx = self.user_to_idx[customer_id]
        
        # Get watched movie indices
        watched = set()
        if self.train_ratings is not None:
            watched = set(self.train_ratings[self.train_ratings['user_idx'] == u_idx]['movie_idx'].values)
            
        num_items = self.ncf_model.num_items
        unseen_indices = [i for i in range(num_items) if i not in watched]
        
        if not unseen_indices:
            return []
            
        # Predict in batch
        u_vector = np.full(len(unseen_indices), u_idx, dtype=np.int32)
        m_vector = np.array(unseen_indices, dtype=np.int32)
        
        preds = self.ncf_model.predict_batch(u_vector, m_vector)
        top_k_idx = np.argsort(-preds)[:k]
        
        recommendations = []
        for i in top_k_idx:
            m_idx = unseen_indices[i]
            movie_id = self.idx_to_movie[m_idx]
            meta = self.movie_meta.get(m_idx, {'title': f"Movie ID {movie_id}", 'year': ""})
            pred_score = float(preds[i])
            
            recommendations.append({
                'movie_id': int(movie_id),
                'movie_idx': m_idx,
                'title': meta['title'],
                'year': meta['year'],
                'score': round(pred_score, 2),
                'explanation': "Matches your non-linear preference representations in our Deep Learning model."
            })
            
        return recommendations

    def get_content_recommendations(self, customer_id, k=10):
        """Generates Top-K recommendations using Content-Based Filtering."""
        if not self.load_content_model() or customer_id not in self.user_to_idx:
            return []
            
        u_idx = self.user_to_idx[customer_id]
        
        # Get user rating history
        user_ratings_row = self.train_ratings[self.train_ratings['user_idx'] == u_idx]
        if user_ratings_row.empty:
            return self.get_coldstart_recommendations(k)
            
        watched = set(user_ratings_row['movie_idx'].values)
        user_ratings = dict(zip(user_ratings_row['movie_idx'], user_ratings_row['Rating']))
        
        num_items = self.content_model.num_items
        unseen_indices = [i for i in range(num_items) if i not in watched]
        
        if not unseen_indices:
            return []
            
        # Predict in batch
        u_vector = np.full(len(unseen_indices), u_idx, dtype=np.int32)
        m_vector = np.array(unseen_indices, dtype=np.int32)
        
        preds = self.content_model.predict_batch(u_vector, m_vector)
        top_k_idx = np.argsort(-preds)[:k]
        
        recommendations = []
        for i in top_k_idx:
            m_idx = unseen_indices[i]
            movie_id = self.idx_to_movie[m_idx]
            meta = self.movie_meta.get(m_idx, {'title': f"Movie ID {movie_id}", 'year': ""})
            pred_score = float(preds[i])
            explanation = self.generate_content_explanation(u_idx, m_idx, user_ratings)
            
            recommendations.append({
                'movie_id': int(movie_id),
                'movie_idx': m_idx,
                'title': meta['title'],
                'year': meta['year'],
                'score': round(pred_score, 2),
                'explanation': explanation
            })
            
        return recommendations

    def generate_content_explanation(self, u_idx, m_idx, user_ratings):
        """Generates an explanation based on content similarity with a movie the user liked."""
        if self.content_model is None or self.content_model.item_similarity is None:
            return "Recommended based on matches with your content preferences."
            
        best_sim = -1.0
        best_rated_movie_idx = None
        
        for rated_m, rating in user_ratings.items():
            if rating >= 3.5:
                # Get similarity between recommended movie and watched movie
                sim = self.content_model.item_similarity[m_idx, rated_m]
                if sim > best_sim:
                    best_sim = sim
                    best_rated_movie_idx = rated_m
                    
        if best_rated_movie_idx is not None and best_sim > 0.05:
            meta = self.movie_meta.get(best_rated_movie_idx, {'title': "a movie you liked"})
            pct = int(best_sim * 100)
            return f"Similar title/era style ({pct}% match) to '{meta['title']}' which you rated {int(user_ratings[best_rated_movie_idx])}★."
        else:
            return "Highly aligns with the title keywords and era of movies in your profile."


if __name__ == "__main__":
    recommender = RecommenderService()
    # Test recommendations for user 1
    print("User History for User ID 1:")
    print(recommender.get_user_history(1, top_n=5))
    
    print("\nFunk SVD Recommendations:")
    print(recommender.get_svd_recommendations(1, k=3))
    
    print("\nItem-CF Recommendations:")
    print(recommender.get_cf_recommendations(1, k=3))

    print("\nUser-CF Recommendations:")
    print(recommender.get_user_cf_recommendations(1, k=3))
    
    print("\nNeural-CF Recommendations:")
    print(recommender.get_ncf_recommendations(1, k=3))
    
    print("\nSimilar Movies to Movie ID 1 (Toy Story):")
    print(recommender.get_similar_movies(1, k=3))
