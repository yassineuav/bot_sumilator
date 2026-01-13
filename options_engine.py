import numpy as np
from scipy.stats import norm

class OptionsEngine:
    def __init__(self):
        pass

    def black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """
        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
        """
        if T <= 0:
            if option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1

        # Theta (approximate)
        theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
        
        return {
            'price': price,
            'delta': delta,
            'theta_daily': theta / 365.0
        }

    def simulate_contract_price(self, spot_history, strike_pct=1.0, dte=7, initial_vol=0.20):
        """
        Simulates how an option contract price changes over a price history.
        """
        entry_price = spot_history[0]
        strike = entry_price * strike_pct
        r = 0.05 # 5% risk-free rate
        
        contract_prices = []
        
        for i, S in enumerate(spot_history):
            # Remaining time (DTE decreases per step, assuming steps are 15m intervals)
            # 1 day = 6.5 trading hours = 26 intervals of 15m
            current_dte = dte - (i / 26.0) 
            T = max(0, current_dte / 365.0)
            
            # Simple assumption: vol stays same or increases slightly with spot drops
            sigma = initial_vol
            
            option_type = 'call' if strike_pct >= 1.0 else 'put'
            res = self.black_scholes(S, strike, T, r, sigma, option_type)
            contract_prices.append(res['price'])
            
        return contract_prices

if __name__ == "__main__":
    engine = OptionsEngine()
    # Test: Stock at 100, Strike 100, 30 DTE, 20% Vol
    res = engine.black_scholes(100, 100, 30/365.0, 0.05, 0.20, 'call')
    print(f"Call Price: {res['price']:.2f}, Delta: {res['delta']:.2f}, Theta: {res['theta_daily']:.4f}")
