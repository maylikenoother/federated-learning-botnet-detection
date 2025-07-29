import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch

def load_and_partition_data(file_path, client_id, num_clients, label_col="label", chunk_size=100_000):
    print(f"📦 Chunked loading for client {client_id}")

    start_row = client_id * chunk_size
    end_row = start_row + chunk_size
    chunk_start = start_row
    chunk_end = end_row

    data_chunks = []

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, skipinitialspace=True):
        if chunk_start >= end_row:
            break
        data_chunks.append(chunk)
        chunk_start += chunk_size

    df = pd.concat(data_chunks)
    print(f"✅ Loaded {len(df)} rows")

    df.dropna(inplace=True)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    y = df[label_col].values.astype(str)

    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=["number", "bool", "float", "int"]).values

    print(f"✅ Data split: X={X.shape}, y={y.shape}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    print("✅ Feature normalization and label encoding complete")

    return X_tensor, y_tensor
