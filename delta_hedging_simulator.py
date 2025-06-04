import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Paramètres de l'option
S0 = 100       # Prix initial du sous-jacent
K = 100        # Strike
T = 1          # Maturité (1 an)
r = 0.01       # Taux sans risque
sigma = 0.2    # Volatilité
N = 252        # Nombre de jours de rebalancement
dt = T / N
np.random.seed(42)

# Fonctions Black-Scholes
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def black_scholes_delta(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)

# Génération du chemin du sous-jacent en simulant un mouvement brownien géométrique 
S = np.zeros(N+1)
S[0] = S0
for t in range(1, N+1):
    z = np.random.normal()
    S[t] = S[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)

# Simulation du delta hedging
portfolio = np.zeros(N+1)
cash = 0
shares = 0

for t in range(N):
    T_remaining = T - t*dt
    delta = black_scholes_delta(S[t], K, T_remaining, r, sigma)

    if t == 0:
        shares = delta
        cash = black_scholes_call_price(S[0], K, T, r, sigma) - shares * S[0]
    else:
        d_shares = delta - shares
        cash -= d_shares * S[t]
        shares = delta
        cash *= np.exp(r * dt)

    portfolio[t] = shares * S[t] + cash

# Valeur finale à maturité
portfolio[-1] = shares * S[-1] + cash
option_payoff = max(S[-1] - K, 0)
PnL = portfolio[-1] - option_payoff
print(f"PnL final du hedge: {PnL:.4f}")

# Graphiques
plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.plot(S, label="Sous-jacent")
plt.title("Évolution du prix du sous-jacent")
plt.xlabel("Jours")
plt.ylabel("Prix")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(portfolio, label="Portefeuille delta-hedgé")
plt.axhline(y=option_payoff, color='r', linestyle='--', label="Payoff option")
plt.title("Valeur du portefeuille vs payoff")
plt.xlabel("Jours")
plt.ylabel("Valeur")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
