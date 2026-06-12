import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_idx'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movie_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['Rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

class PyTorchFunkSVD(nn.Module):
    def __init__(self, num_users, num_movies, n_factors=20, global_mean=3.53):
        super().__init__()
        self.global_mean = global_mean
        
        # Latent factor embeddings
        self.P = nn.Embedding(num_users, n_factors)
        self.Q = nn.Embedding(num_movies, n_factors)
        
        # Bias embeddings
        self.user_biases = nn.Embedding(num_users, 1)
        self.item_biases = nn.Embedding(num_movies, 1)
        
        # Initialize embeddings with standard normal weights
        self.P.weight.data.normal_(0, 0.1)
        self.Q.weight.data.normal_(0, 0.1)
        self.user_biases.weight.data.zero_()
        self.item_biases.weight.data.zero_()

    def forward(self, users, movies):
        # dot product between user factors P_u and movie factors Q_i
        interaction = (self.P(users) * self.Q(movies)).sum(dim=1)
        # add biases and global mean
        pred = self.global_mean + self.user_biases(users).squeeze() + self.item_biases(movies).squeeze() + interaction
        return pred

class PyTorchNeuMF(nn.Module):
    def __init__(self, num_users, num_movies, n_factors=20, mlp_hidden_layers=[64, 32, 16], global_mean=3.53):
        super().__init__()
        self.global_mean = global_mean
        
        # GMF embeddings
        self.P_gmf = nn.Embedding(num_users, n_factors)
        self.Q_gmf = nn.Embedding(num_movies, n_factors)
        
        # MLP embeddings
        self.P_mlp = nn.Embedding(num_users, n_factors)
        self.Q_mlp = nn.Embedding(num_movies, n_factors)
        
        # MLP dense layers
        mlp_layers = []
        in_dim = n_factors * 2
        for h in mlp_hidden_layers:
            mlp_layers.append(nn.Linear(in_dim, h))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            in_dim = h
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final output projection
        self.final_layer = nn.Linear(n_factors + mlp_hidden_layers[-1], 1)
        
        # Initialize weights
        self.P_gmf.weight.data.normal_(0, 0.1)
        self.Q_gmf.weight.data.normal_(0, 0.1)
        self.P_mlp.weight.data.normal_(0, 0.1)
        self.Q_mlp.weight.data.normal_(0, 0.1)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, users, movies):
        # GMF branch
        p_gmf = self.P_gmf(users)
        q_gmf = self.Q_gmf(movies)
        gmf_out = p_gmf * q_gmf
        
        # MLP branch
        p_mlp = self.P_mlp(users)
        q_mlp = self.Q_mlp(movies)
        mlp_in = torch.cat([p_mlp, q_mlp], dim=1)
        mlp_out = self.mlp(mlp_in)
        
        # Combine GMF & MLP
        concat = torch.cat([gmf_out, mlp_out], dim=1)
        pred = self.final_layer(concat).squeeze()
        return pred

def train_pytorch_model():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    
    # Load processed splits
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test.csv'))
    
    import json
    with open(os.path.join(PROCESSED_DIR, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    num_users = int(metadata['num_users'])
    num_movies = int(metadata['num_movies'])
    global_mean = train_df['Rating'].mean()

    # Datasets and loaders
    train_dataset = RatingsDataset(train_df)
    test_dataset = RatingsDataset(test_df)
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)

    # 1. Train PyTorch FunkSVD
    print("--------------------------------------------------")
    print("Training PyTorch Funk SVD model on GPU...")
    print("--------------------------------------------------")
    model_svd = PyTorchFunkSVD(num_users, num_movies, n_factors=20, global_mean=global_mean).to(device)
    criterion = nn.MSELoss()
    optimizer_svd = optim.Adam(model_svd.parameters(), lr=0.015, weight_decay=0.002)
    
    start_time = time.time()
    for epoch in range(15):
        model_svd.train()
        train_loss = 0.0
        for users, movies, ratings in train_loader:
            users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
            optimizer_svd.zero_grad()
            predictions = model_svd(users, movies)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer_svd.step()
            train_loss += loss.item() * len(ratings)
            
        train_rmse = np.sqrt(train_loss / len(train_dataset))
        
        model_svd.eval()
        test_loss = 0.0
        with torch.no_grad():
            for users, movies, ratings in test_loader:
                users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
                predictions = model_svd(users, movies)
                loss = criterion(predictions, ratings)
                test_loss += loss.item() * len(ratings)
        test_rmse = np.sqrt(test_loss / len(test_dataset))
        print(f"Epoch {epoch+1}/15 - Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
    print(f"PyTorch SVD trained in {time.time() - start_time:.2f} seconds!")

    # 2. Train PyTorch NeuMF
    print("\n--------------------------------------------------")
    print("Training PyTorch NeuMF (Neural CF) model on GPU...")
    print("--------------------------------------------------")
    model_ncf = PyTorchNeuMF(num_users, num_movies, n_factors=20, mlp_hidden_layers=[64, 32, 16], global_mean=global_mean).to(device)
    optimizer_ncf = optim.Adam(model_ncf.parameters(), lr=0.005, weight_decay=0.0001)
    
    best_test_rmse = float('inf')
    best_weights = None
    
    start_time = time.time()
    for epoch in range(15):
        model_ncf.train()
        train_loss = 0.0
        for users, movies, ratings in train_loader:
            users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
            optimizer_ncf.zero_grad()
            predictions = model_ncf(users, movies)
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer_ncf.step()
            train_loss += loss.item() * len(ratings)
            
        train_rmse = np.sqrt(train_loss / len(train_dataset))
        
        model_ncf.eval()
        test_loss = 0.0
        with torch.no_grad():
            for users, movies, ratings in test_loader:
                users, movies, ratings = users.to(device), movies.to(device), ratings.to(device)
                predictions = model_ncf(users, movies)
                loss = criterion(predictions, ratings)
                test_loss += loss.item() * len(ratings)
        test_rmse = np.sqrt(test_loss / len(test_dataset))
        
        print(f"Epoch {epoch+1}/15 - Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        
        # Save best checkpoint weights
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_weights = {k: v.cpu().clone() for k, v in model_ncf.state_dict().items()}
            
    print(f"PyTorch NeuMF trained in {time.time() - start_time:.2f} seconds! Best Test RMSE: {best_test_rmse:.4f}")
    
    # 3. Export NeuMF parameters as NumPy structures for PyTorch-free fast inference
    if best_weights is not None:
        model_ncf.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        model_ncf.eval()
        with torch.no_grad():
            P_gmf_np = model_ncf.P_gmf.weight.cpu().numpy()
            Q_gmf_np = model_ncf.Q_gmf.weight.cpu().numpy()
            P_mlp_np = model_ncf.P_mlp.weight.cpu().numpy()
            Q_mlp_np = model_ncf.Q_mlp.weight.cpu().numpy()
            
            final_weight_np = model_ncf.final_layer.weight.cpu().numpy()
            final_bias_np = model_ncf.final_layer.bias.cpu().numpy()
            
            save_dict = {
                'P_gmf': P_gmf_np,
                'Q_gmf': Q_gmf_np,
                'P_mlp': P_mlp_np,
                'Q_mlp': Q_mlp_np,
                'final_weight': final_weight_np,
                'final_bias': final_bias_np,
                'num_users': num_users,
                'num_items': num_movies,
                'global_mean': global_mean
            }
            
            layer_idx = 0
            for layer in model_ncf.mlp:
                if isinstance(layer, nn.Linear):
                    save_dict[f'mlp_w_{layer_idx}'] = layer.weight.cpu().numpy()
                    save_dict[f'mlp_b_{layer_idx}'] = layer.bias.cpu().numpy()
                    layer_idx += 1
                    
            np.savez(os.path.join(PROCESSED_DIR, 'neural_cf_model.npz'), **save_dict)
            print("Neural CF (NeuMF) model exported successfully to neural_cf_model.npz!")

if __name__ == "__main__":
    train_pytorch_model()

