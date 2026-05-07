import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

def load_data(client_id: int = 0, total_clients: int = 1):
    df = pd.read_csv("creditcard.csv")

    # Sharding logic: Partition the data based on client_id
    # In a real scenario, this would be done on different physical machines
    if total_clients > 1:
        shard_size = len(df) // total_clients
        start_idx = client_id * shard_size
        end_idx = (client_id + 1) * shard_size if client_id < total_clients - 1 else len(df)
        df = df.iloc[start_idx:end_idx]
        print(f"Client {client_id} loaded shard: {start_idx} to {end_idx} ({len(df)} samples)")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.long),
    )

if __name__ == "__main__":
    print("[TEST] Initializing Data Loader Test...")
    try:
        train_x, train_y, test_x, test_y = load_data()
        print(f"[SUCCESS] Data loaded correctly.")
        print("-" * 40)
        print(f"Total Training Samples: {len(train_x)}")
        print(f"Total Testing Samples:  {len(test_x)}")
        print(f"Feature Vector Size:    {train_x.shape[1]}")
        print(f"Device:                 {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
        print("-" * 40)
    except Exception as e:
        print(f"[ERROR] Loading data: {e}")