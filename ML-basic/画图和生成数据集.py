# Generate and visualize make_moons dataset
import numpy as np
import matplotlib.pyplot as plt
import os

def prepare_data():
    """Prepare dataset"""
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    # Generate 1000 samples of make_moons dataset
    X, y = make_moons(n_samples=1000, random_state=42)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Convert to numpy arrays and reshape y
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).reshape(-1, 1)  # Add dimension to match model output
    y_test = np.array(y_test).reshape(-1, 1)  
    
    # Save dataset to CSV files for C++ program to read
    save_data_to_csv(X_train, y_train, "train_data.csv")
    save_data_to_csv(X_test, y_test, "test_data.csv")
    
    return X_train, X_test, y_train, y_test

def save_data_to_csv(X, y, filename):
    """Save data to CSV file for C++ program to read"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Combine features and labels
    data = np.hstack((X, y))
    
    # Save to CSV file
    with open(filename, 'w') as f:
        # Write header row with feature names
        num_features = X.shape[1]
        header = ",".join([f"feature_{i}" for i in range(num_features)]) + ",label"
        f.write(header + '\n')
        
        # Write data rows
        for row in data:
            # Use space separator for easier C++ reading
            f.write(" ".join([str(x) for x in row]) + '\n')
    
    print(f"Data saved to {filename}")

def plot_dataset(X, y, title="make_moons Dataset", filename="dataset.png"):
    """Plot scatter plot of the dataset"""
    plt.figure(figsize=(10, 8))
    plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], c='blue', marker='o', label='Class 0')
    plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], c='red', marker='s', label='Class 1')
    plt.title(title, fontsize=16)
    plt.xlabel('Feature 1', fontsize=14)
    plt.ylabel('Feature 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Dataset plot saved to {filename}")

if __name__ == "__main__":
    # Generate dataset and save
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Plot full dataset
    X_full = np.vstack((X_train, X_test))
    y_full = np.vstack((y_train, y_test))
    plot_dataset(X_full, y_full, "Complete make_moons Dataset", "full_dataset.png")
    
    # Plot training and test sets
    plot_dataset(X_train, y_train, "Training Set", "train_dataset.png")
    plot_dataset(X_test, y_test, "Test Set", "test_dataset.png")
    
    print("All data processing and plotting completed!")
