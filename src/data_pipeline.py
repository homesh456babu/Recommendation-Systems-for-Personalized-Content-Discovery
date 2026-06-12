import os
import pandas as pd
import numpy as np
from datetime import datetime

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def parse_netflix_data():
    """
    Parses the raw Netflix Prize formatted files from data/raw/ and returns combined dataframe.
    Expects combined_data_1.txt (and optionally 2-4) + movie_titles.csv in data/raw/.
    
    The rating limit can be controlled via the NETFLIX_RATING_LIMIT environment variable.
    Default: 2,000,000 ratings (safe for Colab RAM). Set to 0 for unlimited.
    """
    print("Parsing Netflix Prize ratings files...")
    ratings_data = []
    
    # Configurable limit via environment variable
    default_limit = 2000000
    env_limit = os.environ.get('NETFLIX_RATING_LIMIT', str(default_limit))
    try:
        limit = int(env_limit)
        if limit <= 0:
            limit = float('inf')
    except ValueError:
        limit = default_limit
    
    if limit != float('inf'):
        print(f"Rating limit set to {int(limit):,} (set NETFLIX_RATING_LIMIT=0 for unlimited)")
    else:
        print("No rating limit — loading all available ratings.")
    
    # Look for combined_data_1.txt up to combined_data_4.txt
    for i in range(1, 5):
        file_path = os.path.join(RAW_DIR, f'combined_data_{i}.txt')
        if not os.path.exists(file_path):
            if i == 1:
                raise FileNotFoundError(
                    f"Missing base raw ratings file: {file_path}\n"
                    f"Please download the Netflix Prize dataset from Kaggle:\n"
                    f"  https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data\n"
                    f"and place the files in {RAW_DIR}/"
                )
            continue
            
        print(f"Reading {os.path.basename(file_path)}...")
        current_movie = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    current_movie = int(line[:-1])
                elif line:
                    parts = line.split(',')
                    if len(parts) == 3:
                        cust_id = int(parts[0])
                        rating = float(parts[1])
                        date = parts[2]
                        ratings_data.append((current_movie, cust_id, rating, date))
                        
                        if len(ratings_data) >= limit:
                            print(f"Reached subset limit of {int(limit):,} ratings. Stopping parsing.")
                            break
        if len(ratings_data) >= limit:
            break
                        
    df = pd.DataFrame(ratings_data, columns=['MovieID', 'CustomerID', 'Rating', 'Date'])
    print(f"Total ratings loaded: {len(df):,}")
    return df

def process_and_split_data(df):
    """
    1. Maps MovieID and CustomerID to contiguous indices starting at 0.
    2. Splits the dataset chronologically (80% train, 20% test per user).
    3. Saves processed data and mappings.
    """
    print("Mapping IDs to contiguous indices...")
    # Unique IDs
    unique_users = df['CustomerID'].unique()
    unique_movies = df['MovieID'].unique()
    
    # Mappings
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    idx_to_user = {i: uid for uid, i in user_to_idx.items()}
    movie_to_idx = {mid: i for i, mid in enumerate(unique_movies)}
    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}
    
    # Add mapped columns
    df['user_idx'] = df['CustomerID'].map(user_to_idx)
    df['movie_idx'] = df['MovieID'].map(movie_to_idx)
    
    # Convert date to datetime for sorting
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Chronological Split (80% Train, 20% Test) per user
    print("Splitting data chronologically per user...")
    train_list = []
    test_list = []
    
    # To speed up grouping, sort by Date first
    df_sorted = df.sort_values('Date')
    
    # Chronological split: split based on timestamp
    # For each user, we take the first 80% as train and the remaining 20% as test
    # If a user has fewer than 5 ratings, we put them all in train to ensure learning
    for user_idx, group in df_sorted.groupby('user_idx'):
        n = len(group)
        if n < 5:
            train_list.append(group)
        else:
            split_idx = int(n * 0.8)
            train_list.append(group.iloc[:split_idx])
            test_list.append(group.iloc[split_idx:])
            
    df_train = pd.concat(train_list).sort_index()
    df_test = pd.concat(test_list).sort_index()
    
    print(f"Train ratings: {len(df_train):,} ({len(df_train)/len(df):.1%})")
    print(f"Test ratings: {len(df_test):,} ({len(df_test)/len(df):.1%})")
    
    # Compute sparsity
    num_users = len(unique_users)
    num_movies = len(unique_movies)
    sparsity = 1.0 - (len(df) / (num_users * num_movies))
    print(f"Sparsity: {sparsity:.4%}")
    
    # Save datasets
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_train.to_csv(os.path.join(PROCESSED_DIR, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(PROCESSED_DIR, 'test.csv'), index=False)
    
    # Save mappings
    np.save(os.path.join(PROCESSED_DIR, 'user_to_idx.npy'), user_to_idx)
    np.save(os.path.join(PROCESSED_DIR, 'idx_to_user.npy'), idx_to_user)
    np.save(os.path.join(PROCESSED_DIR, 'movie_to_idx.npy'), movie_to_idx)
    np.save(os.path.join(PROCESSED_DIR, 'idx_to_movie.npy'), idx_to_movie)
    
    # Save metadata summary
    metadata = {
        'num_users': num_users,
        'num_movies': num_movies,
        'num_ratings': len(df),
        'sparsity': sparsity,
        'train_size': len(df_train),
        'test_size': len(df_test)
    }
    pd.Series(metadata).to_json(os.path.join(PROCESSED_DIR, 'metadata.json'))
    print("Processed datasets and index mappings saved to data/processed/.")
    return df_train, df_test

def build_content_features():
    """
    Builds content-based features from Netflix movie_titles.csv.
    Uses movie title words and release year decade as features.
    
    Returns a DataFrame with columns: movie_id, content_string
    Also saves to data/processed/movie_content_features.csv
    """
    print("Building content features from movie titles...")
    
    movie_titles_path = os.path.join(RAW_DIR, 'movie_titles.csv')
    if not os.path.exists(movie_titles_path):
        movie_titles_path = os.path.join(RAW_DIR, 'movie_titles.txt')
        if not os.path.exists(movie_titles_path):
            print("WARNING: movie_titles.csv not found. Content-based features unavailable.")
            return None
    
    # Load movie_to_idx mapping to only process movies in our dataset
    movie_to_idx_path = os.path.join(PROCESSED_DIR, 'movie_to_idx.npy')
    if os.path.exists(movie_to_idx_path):
        movie_to_idx = np.load(movie_to_idx_path, allow_pickle=True).item()
    else:
        movie_to_idx = None
    
    # Parse movie_titles.csv (format: MovieID,YearOfRelease,Title)
    movies = []
    lines = []
    try:
        with open(movie_titles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(movie_titles_path, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split(',', 2)
        if len(parts) >= 3:
            try:
                movie_id = int(parts[0])
                year_str = parts[1].strip()
                title = parts[2].strip()
                
                # Skip movies not in our dataset if we have the mapping
                if movie_to_idx is not None and movie_id not in movie_to_idx:
                    continue
                
                # Build content string from title words
                # Lowercase and clean the title
                title_clean = title.lower()
                # Remove common punctuation but keep alphanumeric and spaces
                title_clean = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in title_clean)
                # Remove extra whitespace
                title_clean = ' '.join(title_clean.split())
                
                # Add year decade token
                decade_token = ''
                if year_str and year_str != 'NULL':
                    try:
                        year = int(year_str)
                        decade = (year // 10) * 10
                        decade_token = f'decade_{decade}s'
                    except ValueError:
                        pass
                
                # Combined content string
                content_string = f"{title_clean} {decade_token}".strip()
                
                movies.append({
                    'movie_id': movie_id,
                    'title': title,
                    'year': year_str,
                    'content_string': content_string
                })
            except ValueError:
                continue
    
    content_df = pd.DataFrame(movies)
    
    # Save to processed directory
    output_path = os.path.join(PROCESSED_DIR, 'movie_content_features.csv')
    content_df.to_csv(output_path, index=False)
    print(f"Content features built for {len(content_df)} movies. Saved to movie_content_features.csv")
    
    return content_df

def get_movie_metadata():
    """
    Reads the movie titles and years from raw data.
    Returns a dictionary of movie_idx -> {movie_id, title, year}
    """
    movie_titles_path = os.path.join(RAW_DIR, 'movie_titles.csv')
    if not os.path.exists(movie_titles_path):
        movie_titles_path = os.path.join(RAW_DIR, 'movie_titles.txt')
        if not os.path.exists(movie_titles_path):
            return {}
        
    movie_idx_to_title = {}
    movie_to_idx = np.load(os.path.join(PROCESSED_DIR, 'movie_to_idx.npy'), allow_pickle=True).item()
    
    # Try reading as utf-8, fall back to ISO-8859-1 if it fails
    lines = []
    try:
        with open(movie_titles_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(movie_titles_path, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()
            
    for line in lines:
        parts = line.strip().split(',', 2)
        if len(parts) >= 3:
            try:
                movie_id = int(parts[0])
                year = parts[1]
                title = parts[2]
                
                if movie_id in movie_to_idx:
                    m_idx = movie_to_idx[movie_id]
                    movie_idx_to_title[m_idx] = {
                        'movie_id': movie_id,
                        'title': title,
                        'year': year
                    }
            except ValueError:
                continue
                    
    return movie_idx_to_title

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    df = parse_netflix_data()
    process_and_split_data(df)
    build_content_features()
