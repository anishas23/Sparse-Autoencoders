import subprocess
import re

scripts = [
    ("Simple Autoencoder", "standard_autoencoder.py"),
    ("Sparse Autoencoder (L1)", "sparse_autoencoder_l1.py"),
    ("Sparse Autoencoder (KL)", "sparse_autoencoder_kl.py"),
]

results = []

for name, script in scripts:
    print("=" * 60)
    print(f"Running {name} ({script})")
    print("=" * 60)
    result = subprocess.run(
        ["python", script],
        capture_output=True,
        text=True
    )
    print(result.stdout)

    # Extract training time
    time_match = re.search(r"Training finished in ([\d.]+) seconds", result.stdout)
    time_taken = float(time_match.group(1)) if time_match else None

    # Extract final test MSE
    mse_match = re.search(r"Final Test MSE: ([\d.]+)", result.stdout)
    mse_val = float(mse_match.group(1)) if mse_match else None

    results.append((name, time_taken, mse_val))
    print("\n")

# ===== Summary Table =====
print("\n" + "="*60)
print(" Summary of Results ".center(60, "="))
print("="*60)
print(f"{'Model':<30} {'Time (s)':<15} {'Final Test MSE':<15}")
print("-"*60)
for name, t, mse in results:
    print(f"{name:<30} {t:<15.2f} {mse:<15.6f}")
print("="*60)
