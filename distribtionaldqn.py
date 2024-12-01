import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate C51-style categorical distribution
atoms = np.linspace(-10, 10, 51)
probabilities = norm.pdf(atoms, 0, 5)
probabilities /= probabilities.sum()  # Normalize to sum to 1

# Plot C51 categorical distribution
plt.figure(figsize=(8, 4))
plt.bar(atoms, probabilities, width=0.4, color="skyblue")
plt.xlabel("Return (Z)")
plt.ylabel("Probability")
plt.title("Categorical Distribution of Returns (C51)")
plt.savefig("figures/c51.png", dpi=300)
plt.close()

# Generate QR-DQN-style quantile distribution
quantiles = np.linspace(0.01, 0.99, 50)
values = norm.ppf(quantiles, 0, 5)

# Plot QR-DQN quantile distribution
plt.figure(figsize=(8, 4))
plt.plot(quantiles, values, 'bo-', label="Quantile Values")
plt.xlabel("Quantile")
plt.ylabel("Value")
plt.title("Quantile Approximation of Return Distribution (QR-DQN)")
plt.legend()
plt.savefig("figures/qr_dqn.png", dpi=300)
plt.close()
