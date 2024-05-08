import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, coef, exp):
    """Gaussian basis function"""
    return coef * np.exp(-exp * x**2)


# Define basis functions parameters (coefficients and exponents)
# These are illustrative; exact parameters can vary and are usually obtained from quantum chemistry computations
s_orbitals = [(1.24, 0.5), (0.75, 0.2)]  # Coefficients and exponents for s-orbitals
p_orbitals = [
    (1.00, 0.3)
]  # Coefficients and exponents for p-orbitals, simplified for demonstration

x = np.linspace(-1, 1, 200)

# Plot s-orbitals
for coef, exp in s_orbitals:
    plt.plot(x, gaussian(x, coef, exp), label=f"s-orbital: coef={coef}, exp={exp}")

# Plot p-orbital, considering only its radial part
for coef, exp in p_orbitals:
    plt.plot(
        x, gaussian(x, coef, exp), "--", label=f"p-orbital: coef={coef}, exp={exp}"
    )

plt.title("Hydrogen 6-31G* Basis Set Orbitals")
plt.xlabel("Distance from nucleus (arbitrary units)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
