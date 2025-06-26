import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load multi-year combined dataset (e.g. 2020-2025)
df = pd.read_csv('data/f1_dataset_2020_2025.csv')

print("Loaded data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Drop invalid rows
df = df.dropna(subset=['QualifyingPosition', 'Winner', 'Driver', 'Race'])
df = df[df['QualifyingPosition'] > 0]
print(f"\nCleaned dataset length: {len(df)}")

# Manual label encoding for Drivers and Races
drivers = sorted(df['Driver'].unique())
races = sorted(df['Race'].unique())

driver_to_enc = {d: i for i, d in enumerate(drivers)}
race_to_enc = {r: i for i, r in enumerate(races)}

df['Driver_enc'] = df['Driver'].map(driver_to_enc)
df['Race_enc'] = df['Race'].map(race_to_enc)

print(f"\nNumber of unique drivers: {len(drivers)}")
print(f"Number of unique races: {len(races)}")

# Prepare race-wise data: for each race, create feature vector and target vector
processed_races = []

for race_idx, race_name in enumerate(races):
    race_df = df[df['Race_enc'] == race_idx]
    
    # Check exactly one winner per race
    winners = race_df[race_df['Winner'] == 1]
    if len(winners) != 1:
        print(f"Skipping race '{race_name}', found {len(winners)} winners")
        continue

    winner_driver_enc = winners.iloc[0]['Driver_enc']

    # Initialize feature vector for all drivers: 0 = missing / didn't qualify
    features = np.zeros(len(drivers), dtype=np.float32)
    
    # Normalize qualifying positions per race: invert & scale (best=1, worst~0)
    max_q_pos = race_df['QualifyingPosition'].max()
    
    for _, row in race_df.iterrows():
        idx = int(row['Driver_enc'])
        qpos = float(row['QualifyingPosition'])
        features[idx] = (max_q_pos + 1 - qpos) / max_q_pos
    
    # Target vector: one-hot for winner driver
    target_vec = np.zeros(len(drivers), dtype=np.float32)
    target_vec[int(winner_driver_enc)] = 1.0

    processed_races.append({
        'race': race_name,
        'features': features,
        'target': target_vec,
        'winner_enc': winner_driver_enc
    })

print(f"\nProcessed {len(processed_races)} races for training")

if len(processed_races) < 3:
    raise ValueError("Not enough races with exactly one winner to train!")

# Build dataset arrays (shape: drivers x races)
X_data = np.array([race['features'] for race in processed_races]).T  # shape (drivers, races)
y_data = np.array([race['target'] for race in processed_races]).T    # shape (drivers, races)

print(f"\nFeature matrix shape: {X_data.shape}")
print(f"Target matrix shape: {y_data.shape}")

# Shuffle & split data by race columns (samples)
num_samples = X_data.shape[1]
indices = np.arange(num_samples)
np.random.seed(42)
np.random.shuffle(indices)

train_size = int(0.8 * num_samples)
train_idx, val_idx = indices[:train_size], indices[train_size:]

X_train, X_val = X_data[:, train_idx], X_data[:, val_idx]
y_train, y_val = y_data[:, train_idx], y_data[:, val_idx]

print(f"Training samples: {X_train.shape[1]}, Validation samples: {X_val.shape[1]}")

# Activation functions and gradients
def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Initialize weights with He initialization
def init_weights(input_dim, hidden_dim=20, output_dim=None):
    if output_dim is None:
        output_dim = len(drivers)
    np.random.seed(42)
    W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros((hidden_dim, 1))
    W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros((output_dim, 1))
    return W1, b1, W2, b2

# Forward pass
def forward_pass(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    cache = (X, Z1, A1, Z2, A2)
    return A2, cache

# Cross-entropy loss
def compute_loss(A2, Y):
    m = Y.shape[1]
    A2 = np.clip(A2, 1e-10, 1-1e-10)
    loss = -np.sum(Y * np.log(A2)) / m
    return loss

# Backward pass gradients
def backward_pass(cache, W2, Y):
    X, Z1, A1, Z2, A2 = cache
    m = Y.shape[1]
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Update weights
def update_weights(params, grads, lr):
    W1, b1, W2, b2 = params
    dW1, db1, dW2, db2 = grads
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

# Prediction helper
def get_predictions(X, params):
    W1, b1, W2, b2 = params
    A2, _ = forward_pass(X, W1, b1, W2, b2)
    preds = np.argmax(A2, axis=0)
    return preds, A2

# Accuracy calc
def calc_accuracy(preds, Y):
    true_labels = np.argmax(Y, axis=0)
    return np.mean(preds == true_labels) * 100

# Training loop
def train_nn(X_train, Y_train, X_val, Y_val, hidden_units=30, epochs=2000, lr=0.01):
    input_size = X_train.shape[0]
    W1, b1, W2, b2 = init_weights(input_size, hidden_units)
    losses = []
    print(f"\nTraining started: input={input_size}, hidden={hidden_units}")

    for epoch in range(epochs):
        A2, cache = forward_pass(X_train, W1, b1, W2, b2)
        loss = compute_loss(A2, Y_train)
        grads = backward_pass(cache, W2, Y_train)
        W1, b1, W2, b2 = update_weights((W1, b1, W2, b2), grads, lr)

        if epoch % 200 == 0:
            train_preds, _ = get_predictions(X_train, (W1, b1, W2, b2))
            val_preds, _ = get_predictions(X_val, (W1, b1, W2, b2))
            train_acc = calc_accuracy(train_preds, Y_train)
            val_acc = calc_accuracy(val_preds, Y_val)
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Train Acc: {train_acc:5.1f}% | Val Acc: {val_acc:5.1f}%")
            losses.append(loss)

    return (W1, b1, W2, b2), losses

# Predict winner given qualifying positions dict
def predict_winner(qual_pos_dict, params):
    features = np.zeros(len(drivers), dtype=np.float32)
    max_pos = max(qual_pos_dict.values()) if qual_pos_dict else 20

    for drv, pos in qual_pos_dict.items():
        if drv in drivers:
            idx = drivers.index(drv)
            features[idx] = (max_pos + 1 - pos) / max_pos

    X_pred = features.reshape(-1, 1)
    pred_idx, probs = get_predictions(X_pred, params)
    winner_name = drivers[pred_idx[0]]

    prob_vals = probs.flatten()
    top_idxs = np.argsort(prob_vals)[-min(5, len(drivers)):][::-1]

    print("\nRace winner prediction (top 5):")
    for i, idx in enumerate(top_idxs):
        dname = drivers[idx]
        pval = prob_vals[idx] * 100
        if pval > 1:
            print(f"{i+1}. {dname}: {pval:.1f}%")

    return winner_name, prob_vals

# === MAIN TRAINING ===
print("\nStarting neural network training...")
weights, loss_history = train_nn(X_train, y_train, X_val, y_val,
                                 hidden_units=30, epochs=2000, lr=0.01)

# Plot and save loss graph
plt.figure(figsize=(8,5))
plt.plot(np.arange(0, len(loss_history)) * 200, loss_history, marker='o')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('training_loss.png')
plt.show()

# Final accuracy
train_preds, _ = get_predictions(X_train, weights)
val_preds, _ = get_predictions(X_val, weights)

print(f"\nFinal Training Accuracy: {calc_accuracy(train_preds, y_train):.1f}%")
print(f"Final Validation Accuracy: {calc_accuracy(val_preds, y_val):.1f}%")

print("\nValidation set predictions:")
true_winners = [drivers[idx] for idx in np.argmax(y_val, axis=0)]
pred_winners = [drivers[idx] for idx in val_preds]

for i, (true_w, pred_w) in enumerate(zip(true_winners, pred_winners)):
    status = "✓" if true_w == pred_w else "✗"
    print(f"Race {i+1}: True={true_w}, Pred={pred_w} {status}")

# Example prediction (update driver names to ones in your dataset)
example_quals = {
    'VER': 1,
    'HAM': 2,
    'NOR': 3,
    'PER': 4
}

pred_winner, _ = predict_winner(example_quals, weights)
print(f"\nPredicted winner: {pred_winner}")
