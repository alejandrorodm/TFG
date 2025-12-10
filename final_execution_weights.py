import pandas as pd
from Portfolio import Portfolio, Asset
import marketdata as mkd
import time

def verify():
    coins = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT', 'ADA-USDT', 'SOL-USDT', 'BNB-USDT', 'DOT-USDT', 'AVAX-USDT', 'DOGE-USDT', 'SHIB-USDT']
    assets = [Asset(coin) for coin in coins]
    portfolio = Portfolio(assets)
    from datetime import datetime
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    print("Computing current weights...")

    try:
        l_riesgo = 0.5
        temporality = '1day'
        delta = 0.01 # As per instruction, default if not specified

        weights = portfolio.get_current_weights(l_riesgo=l_riesgo, delta=delta, temporality=temporality, run_nbeats=True)
        print("\nWeights calculated successfully:")
        print(weights)
        print("\nSum of weights:")
        print(weights.sum())

        with pd.ExcelWriter(f"final_weights_{timestamp}.xlsx") as writer:
            pd.DataFrame(weights).to_excel(writer, sheet_name='Weights')
            params_df = pd.DataFrame({
                'Parameter': ['l_riesgo', 'temporality', 'delta'],
                'Value': [l_riesgo, temporality, delta]
            })
            params_df.to_excel(writer, sheet_name='Parameters', index=False)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify()
