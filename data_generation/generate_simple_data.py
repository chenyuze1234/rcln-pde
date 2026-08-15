"""
Generate SIMPLE data - Random Gaussian fields
NOT real NS, but for testing training pipeline
"""
import numpy as np
import h5py
import os

N = 64
correlation_target = 0.92  # Target correlation between input and target

def generate_sample(seed):
    """Generate correlated random fields"""
    np.random.seed(seed)
    
    # Generate input field
    input_field = np.random.randn(3, N, N, N).astype(np.float32)
    
    # Generate target as input + noise (to achieve correlation ~0.92)
    noise = np.random.randn(3, N, N, N).astype(np.float32)
    
    # Mix to get desired correlation
    # correlation = 1/sqrt(1 + (noise_ratio)^2) ≈ 0.92
    # => noise_ratio ≈ 0.4
    target_field = input_field * 0.92 + noise * 0.39
    
    # Normalize
    input_field = input_field / np.std(input_field)
    target_field = target_field / np.std(target_field)
    
    return input_field, target_field

def main():
    print("="*60)
    print("Generating SIMPLE random data")
    print("(For training pipeline testing)")
    print("="*60)
    
    n_train = 100
    n_val = 20
    
    print(f"\nGenerating {n_train} train + {n_val} val samples...")
    
    train_inputs = []
    train_targets = []
    
    for i in range(n_train):
        inp, tgt = generate_sample(i)
        train_inputs.append(inp)
        train_targets.append(tgt)
    
    train_inputs = np.array(train_inputs, dtype=np.float32)
    train_targets = np.array(train_targets, dtype=np.float32)
    
    val_inputs = []
    val_targets = []
    
    for i in range(n_val):
        inp, tgt = generate_sample(n_train + i)
        val_inputs.append(inp)
        val_targets.append(tgt)
    
    val_inputs = np.array(val_inputs, dtype=np.float32)
    val_targets = np.array(val_targets, dtype=np.float32)
    
    # Save
    os.makedirs('data', exist_ok=True)
    with h5py.File('data/simple_random.h5', 'w') as f:
        f.create_dataset('train/input', data=train_inputs)
        f.create_dataset('train/target', data=train_targets)
        f.create_dataset('val/input', data=val_inputs)
        f.create_dataset('val/target', data=val_targets)
        
        mean = train_inputs.mean()
        std = train_inputs.std()
        f.attrs['mean'] = float(mean)
        f.attrs['std'] = float(std)
        f.attrs['N'] = N
        f.attrs['Re'] = 1000
    
    # Verify
    print(f"\nStatistics:")
    print(f"  Train: {train_inputs.shape}")
    print(f"  Val:   {val_inputs.shape}")
    print(f"  mean:  {mean:.4f}")
    print(f"  std:   {std:.4f}")
    print(f"  min:   {train_inputs.min():.4f}")
    print(f"  max:   {train_inputs.max():.4f}")
    
    # Check correlation
    corrs = [np.corrcoef(train_inputs[i].flatten(), train_targets[i].flatten())[0,1]
             for i in range(10)]
    print(f"\nCorrelation (first 10): {np.mean(corrs):.4f}")
    
    if 0.88 <= np.mean(corrs) <= 0.96:
        print("OK: Correlation in target range!")
    
    print(f"\nSaved: data/simple_random.h5")
    print("\nThis is RANDOM data (not real NS)")
    print("For testing training pipeline only!")

if __name__ == '__main__':
    main()
