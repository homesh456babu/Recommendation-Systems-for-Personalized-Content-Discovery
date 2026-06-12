import os
import json
from flask import Flask, jsonify, render_template, request
from src.recommend import RecommenderService

app = Flask(__name__, template_folder='templates', static_folder='static')
recommender = RecommenderService()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/system-status')
def system_status():
    """Returns general metadata about the trained models and dataset."""
    status = {
        'ready': False,
        'metadata': {},
        'results': {}
    }
    
    processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
    metadata_path = os.path.join(processed_dir, 'metadata.json')
    results_path = os.path.join(processed_dir, 'results.json')
    
    if os.path.exists(metadata_path) and os.path.exists(results_path):
        try:
            with open(metadata_path, 'r') as f:
                status['metadata'] = json.load(f)
            with open(results_path, 'r') as f:
                status['results'] = json.load(f)
            status['ready'] = True
        except Exception as e:
            app.logger.error(f"Error loading system status: {e}")
            
    return jsonify(status)

@app.route('/api/user-history/<int:customer_id>')
def user_history(customer_id):
    """Retrieves user rating history."""
    # Re-initialize metadata if it wasn't loaded (e.g. if files were created after app started)
    if not recommender.user_to_idx:
        recommender.load_metadata()
        
    history = recommender.get_user_history(customer_id)
    if not history and customer_id not in recommender.user_to_idx:
        return jsonify({'error': 'User ID not found in training set. Use Cold-Start recommendations.'}), 404
        
    return jsonify({
        'customer_id': customer_id,
        'history': history
    })

@app.route('/api/recommendations/<int:customer_id>')
def recommendations(customer_id):
    """Retrieves Top-K recommendations using SVD, Item-CF, User-CF, Neural-CF, and Content-Based."""
    if not recommender.user_to_idx:
        recommender.load_metadata()
        
    if customer_id not in recommender.user_to_idx:
        return jsonify({'error': 'User not found in dataset.'}), 404
        
    svd_recs = recommender.get_svd_recommendations(customer_id, k=10)
    cf_recs = recommender.get_cf_recommendations(customer_id, k=10)
    user_cf_recs = recommender.get_user_cf_recommendations(customer_id, k=10)
    ncf_recs = recommender.get_ncf_recommendations(customer_id, k=10)
    content_recs = recommender.get_content_recommendations(customer_id, k=10)
    
    return jsonify({
        'customer_id': customer_id,
        'svd': svd_recs,
        'cf': cf_recs,
        'user_cf': user_cf_recs,
        'ncf': ncf_recs,
        'content': content_recs
    })

@app.route('/api/similar-movies/<int:movie_id>')
def similar_movies(movie_id):
    """Finds similar movies based on collaborative filtering item similarity."""
    if not recommender.movie_to_idx:
        recommender.load_metadata()
        
    if movie_id not in recommender.movie_to_idx:
        return jsonify({'error': 'Movie ID not found in database.'}), 404
        
    sim_movies = recommender.get_similar_movies(movie_id, k=10)
    return jsonify({
        'movie_id': movie_id,
        'similar_movies': sim_movies
    })

@app.route('/api/coldstart')
def coldstart():
    """Generates recommendations for cold start users."""
    if not recommender.idx_to_movie:
        recommender.load_metadata()
        
    genres_param = request.args.get('genres', '')
    genres = [g.strip() for g in genres_param.split(',')] if genres_param else None
    
    recs = recommender.get_coldstart_recommendations(k=10, genres=genres)
    return jsonify({
        'genres': genres,
        'recommendations': recs
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
