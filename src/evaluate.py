import os
import json
import pandas as pd
import numpy as np
from src.data_pipeline import get_movie_metadata, build_content_features

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def compute_rmse_mae(predictions, actuals):
    """
    Computes Root Mean Squared Error and Mean Absolute Error.
    """
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    return rmse, mae

def evaluate_map_at_10(model, train_df, test_df, num_users, num_items, relevance_threshold=3.5):
    """
    Computes the Mean Average Precision @ 10 (MAP@10) for the model.
    A movie is relevant if its actual rating in the test set is >= relevance_threshold (3.5).
    For each user, we recommend the Top-10 highest-predicted unseen movies.
    """
    print("Calculating MAP@10 on the test set...")
    
    # 1. Group training ratings by user to know what they've already watched
    train_watched = train_df.groupby('user_idx')['movie_idx'].apply(set).to_dict()
    
    # 2. Group test ratings by user and isolate relevant movies
    # Test ratings: user_idx -> dict of {movie_idx: actual_rating}
    test_ratings = {}
    for user_idx, gp in test_df.groupby('user_idx'):
        test_ratings[user_idx] = dict(zip(gp['movie_idx'], gp['Rating']))
        
    # User indices in test set
    test_users = list(test_ratings.keys())
    
    # Randomly sample a subset of users (max 100) to speed up offline testing
    np.random.seed(42)
    if len(test_users) > 100:
        sampled_users = np.random.choice(test_users, size=100, replace=False)
        print(f"Sampling 100 out of {len(test_users)} test users for fast MAP@10 evaluation...")
    else:
        sampled_users = test_users
        
    ap_scores = []
    
    # Evaluate for each sampled user
    for user_count, u in enumerate(sampled_users):
        # Find actual relevant movies for this user in test set
        u_test_ratings = test_ratings[u]
        relevant_movies = {m for m, r in u_test_ratings.items() if r >= relevance_threshold}
        
        # If the user has no relevant items in the test set, skip
        if not relevant_movies:
            continue
            
        # Unseen movies: all movies in catalog minus training watched
        watched = train_watched.get(u, set())
        unseen_movies = [m for m in range(num_items) if m not in watched]
        
        if not unseen_movies:
            continue
            
        # Predict ratings for all unseen movies
        # Create vectors for batch prediction
        u_vector = np.full(len(unseen_movies), u, dtype=np.int32)
        m_vector = np.array(unseen_movies, dtype=np.int32)
        
        preds = model.predict_batch(u_vector, m_vector)
        
        # Get Top-10 predictions
        if len(preds) > 10:
            top_10_indices = np.argpartition(preds, -10)[-10:]
            # Sort the top 10 descending
            top_10_sorted_idx = top_10_indices[np.argsort(-preds[top_10_indices])]
            top_10_movies = [unseen_movies[idx] for idx in top_10_sorted_idx]
        else:
            top_10_movies = [unseen_movies[idx] for idx in np.argsort(-preds)]
            
        # Calculate Average Precision @ 10
        # AP@10 = sum_{k=1}^10 (P(k) * rel(k)) / min(10, number of actual relevant movies in test)
        num_hits = 0
        sum_precisions = 0.0
        
        for k, movie in enumerate(top_10_movies):
            # Rank is 1-indexed
            rank = k + 1
            if movie in relevant_movies:
                num_hits += 1
                precision_at_k = num_hits / rank
                sum_precisions += precision_at_k
                
        num_relevant_in_test = len(relevant_movies)
        ap = sum_precisions / min(10, num_relevant_in_test)
        ap_scores.append(ap)
        
        if (user_count + 1) % 20 == 0:
            print(f"Evaluated {user_count + 1}/{len(sampled_users)} test users...")
            
    map_score = np.mean(ap_scores) if ap_scores else 0.0
    return map_score

def run_evaluation():
    """
    Main evaluation entry point.
    Loads processed data, fits the models, computes metrics, and outputs results.
    """
    from src.models import FunkSVD, ItemCollaborativeFiltering, UserCollaborativeFiltering, NeuralCFRecommender, ContentBasedRecommender
    
    # First, run the PyTorch models training to produce neural_cf_model.npz
    try:
        print("Starting PyTorch models training on GPU...")
        from src.pytorch_gpu_model import train_pytorch_model
        train_pytorch_model()
    except (ImportError, ModuleNotFoundError) as e:
        print(f"PyTorch (torch) is not installed in the local environment: {e}. Skipping PyTorch training and checking for pre-trained weights.")
    except Exception as e:
        print(f"Error training PyTorch models: {e}. Skipping neural training.")
        
    # Load processed data
    print("Loading data for evaluation...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test.csv'))
    
    with open(os.path.join(PROCESSED_DIR, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    num_users = int(metadata['num_users'])
    num_movies = int(metadata['num_movies'])
    
    results = {}
    
    # 1. Funk SVD Evaluation
    print("\n--- Training & Evaluating Funk SVD ---")
    svd = FunkSVD(n_factors=20, lr=0.007, reg=0.04, n_epochs=15)
    # Use test set as validation for tracking training curve
    svd.fit(train_df, num_users, num_movies, val_df=test_df)
    
    print("Predicting test ratings...")
    svd_test_preds = svd.predict_batch(test_df['user_idx'].values, test_df['movie_idx'].values)
    svd_rmse, svd_mae = compute_rmse_mae(svd_test_preds, test_df['Rating'].values)
    svd_map = evaluate_map_at_10(svd, train_df, test_df, num_users, num_movies)
    
    results['Funk SVD'] = {
        'RMSE': float(svd_rmse),
        'MAE': float(svd_mae),
        'MAP@10': float(svd_map)
    }
    
    # Save the trained model parameters for app usage
    np.savez(
        os.path.join(PROCESSED_DIR, 'svd_model.npz'),
        user_biases=svd.user_biases,
        item_biases=svd.item_biases,
        P=svd.P,
        Q=svd.Q,
        global_mean=svd.global_mean,
        num_users=svd.num_users,
        num_items=svd.num_items
    )
    
    # 2. Item-based Collaborative Filtering Evaluation
    print("\n--- Training & Evaluating Item-Based Collaborative Filtering ---")
    item_cf = ItemCollaborativeFiltering(k_neighbors=40, shrinkage=50)
    item_cf.fit(train_df, num_users, num_movies)
    
    print("Predicting test ratings...")
    cf_test_preds = item_cf.predict_batch(test_df['user_idx'].values, test_df['movie_idx'].values)
    cf_rmse, cf_mae = compute_rmse_mae(cf_test_preds, test_df['Rating'].values)
    cf_map = evaluate_map_at_10(item_cf, train_df, test_df, num_users, num_movies)
    
    results['Item-CF'] = {
        'RMSE': float(cf_rmse),
        'MAE': float(cf_mae),
        'MAP@10': float(cf_map)
    }
    
    # Save trained similarity matrix
    np.savez(
        os.path.join(PROCESSED_DIR, 'item_cf_model.npz'),
        similarity_matrix=item_cf.similarity_matrix,
        item_means=item_cf.item_means.values,
        item_means_idx=item_cf.item_means.index.values,
        user_means=item_cf.user_means.values,
        user_means_idx=item_cf.user_means.index.values,
        global_mean=item_cf.global_mean,
        num_users=item_cf.num_users,
        num_items=item_cf.num_items
    )

    # 3. User-based Collaborative Filtering Evaluation
    print("\n--- Training & Evaluating User-Based Collaborative Filtering ---")
    user_cf = UserCollaborativeFiltering(k_neighbors=40, shrinkage=100)
    user_cf.fit(train_df, num_users, num_movies)
    
    print("Predicting test ratings...")
    user_cf_test_preds = user_cf.predict_batch(test_df['user_idx'].values, test_df['movie_idx'].values)
    user_cf_rmse, user_cf_mae = compute_rmse_mae(user_cf_test_preds, test_df['Rating'].values)
    user_cf_map = evaluate_map_at_10(user_cf, train_df, test_df, num_users, num_movies)
    
    results['User-CF'] = {
        'RMSE': float(user_cf_rmse),
        'MAE': float(user_cf_mae),
        'MAP@10': float(user_cf_map)
    }
    
    # Save trained user similarity weights
    np.savez(
        os.path.join(PROCESSED_DIR, 'user_cf_model.npz'),
        similarity_matrix=user_cf.similarity_matrix,
        user_means=user_cf.user_means.values,
        user_means_idx=user_cf.user_means.index.values,
        item_means=user_cf.item_means.values,
        item_means_idx=user_cf.item_means.index.values,
        global_mean=user_cf.global_mean,
        num_users=user_cf.num_users,
        num_items=user_cf.num_items
    )
    
    # 4. Neural Collaborative Filtering Evaluation
    print("\n--- Training & Evaluating Neural Collaborative Filtering (NeuMF) ---")
    ncf = NeuralCFRecommender()
    ncf_model_path = os.path.join(PROCESSED_DIR, 'neural_cf_model.npz')
    if ncf.load_model(ncf_model_path):
        print("Predicting test ratings...")
        ncf_test_preds = ncf.predict_batch(test_df['user_idx'].values, test_df['movie_idx'].values)
        ncf_rmse, ncf_mae = compute_rmse_mae(ncf_test_preds, test_df['Rating'].values)
        ncf_map = evaluate_map_at_10(ncf, train_df, test_df, num_users, num_movies)
        
        results['Neural-CF'] = {
            'RMSE': float(ncf_rmse),
            'MAE': float(ncf_mae),
            'MAP@10': float(ncf_map)
        }
    else:
        print("WARNING: Neural CF weights not found at data/processed/neural_cf_model.npz. Skipping Neural CF evaluation.")
    
    # 5. Content-Based Filtering Evaluation
    print("\n--- Training & Evaluating Content-Based Filtering ---")
    content_features_path = os.path.join(PROCESSED_DIR, 'movie_content_features.csv')
    if os.path.exists(content_features_path):
        content_df = pd.read_csv(content_features_path)
        movie_to_idx = np.load(os.path.join(PROCESSED_DIR, 'movie_to_idx.npy'), allow_pickle=True).item()
        
        content_model = ContentBasedRecommender(k_neighbors=20)
        content_model.fit(train_df, num_users, num_movies, content_df, movie_to_idx)
        
        print("Predicting test ratings...")
        content_test_preds = content_model.predict_batch(test_df['user_idx'].values, test_df['movie_idx'].values)
        content_rmse, content_mae = compute_rmse_mae(content_test_preds, test_df['Rating'].values)
        content_map = evaluate_map_at_10(content_model, train_df, test_df, num_users, num_movies)
        
        results['Content-Based'] = {
            'RMSE': float(content_rmse),
            'MAE': float(content_mae),
            'MAP@10': float(content_map)
        }
        
        # Save content-based model
        content_model.save_model(os.path.join(PROCESSED_DIR, 'content_based_model.npz'))
    else:
        print("WARNING: Content features not found at data/processed/movie_content_features.csv.")
        print("Run 'python src/data_pipeline.py' first to build content features.")
    
    print("\n================ Evaluation Summary ================")
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"  RMSE:   {metrics['RMSE']:.4f}")
        print(f"  MAE:    {metrics['MAE']:.4f}")
        print(f"  MAP@10: {metrics['MAP@10']:.4f}")
    print("====================================================\n")
    
    # Save results to JSON
    with open(os.path.join(PROCESSED_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_evaluation()
