import pandas as pd
from Portfolio import Portfolio, Asset
import marketdata as mkd
import time

def verify():
    coins = ['BTC-USDT', 'ASTER-USDT', 'ONDO-USDT', 'HYPE-USDT', 'KAS-USDT', 'RAY-USDT']
    assets = [Asset(coin) for coin in coins]
    portfolio = Portfolio(assets)
    from datetime import datetime
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    print("Computing current weights...")

    try:
        weights = portfolio.get_current_weights()
        print("\nWeights calculated successfully:")
        print(weights)
        print("\nSum of weights:")
        print(weights.sum())

        pd.DataFrame(weights).to_excel(f"final_weights_{timestamp}.xlsx")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify()
