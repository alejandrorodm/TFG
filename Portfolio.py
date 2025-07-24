import pandas as pd
import torch

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

import marketdata as mkd 
import numpy as np
import pandas as pd

from cvxopt import matrix, solvers
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
import os

temporalities = {
        '1min': 1500 * 60,
        '5min': 1500 * 60 * 5,
        '15min': 1500 * 60 * 15,
        '1hour': 1500 * 60 * 60,
        '4hour': 1500 * 60 * 60 * 4,
        '8hour': 1500 * 60 * 60 * 8,
        '1day': 1500 * 60 * 60 * 24,
        '1week': 1500 * 60 * 60 * 24 * 7,
        '1month': 1500 * 60 * 60 * 24 * 30
    }

class Asset:
    
    def __init__(self, name, series=None, scaler=None):
        self.name = name
        self.series = series
        self.scaler = scaler
        self.model = None
        
    def datapricesToDf(self, data):
        """
        Convert the data to a DataFrame and calculate the returns
    
        Args:
            data (list): List of lists with the data of the coin
            
        Returns:
            df (DataFrame): DataFrame with the data of the coin
            df_r (DataFrame): DataFrame with the returns of the coin
        """
        df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'Close', 'High', 'Low', 'Volume'])
  
        # Data preparation and scrubbing 
        df = df[['Timestamp', 'Close']]

        df = df.apply(pd.to_numeric, errors='coerce')
        df['Close'] = df['Close'].interpolate(method="linear") #Interpolate the missing values
        df.dropna(subset=['Close'], inplace=True)
        df.rename(columns={'Close': self.name}, inplace=True)
    
        # Create a new DataFrame for the returns
        df_r = pd.DataFrame(index=df.index, columns=[self.name]) 
        
        # Calculate the return for each row
        df_r[self.name] = np.log(df[self.name] / df[self.name].shift(1)).fillna(0)
        
        return df, df_r
        
        
    def obtainData(self, coin, temporality, startDate, endDate):
        """
        Obtain the data of the coin in the specified temporality and date range
        
        Args:
            coin (str): Name of the coin
            temporality (str): Temporality of the data
            startDate (tuple): Start date of the data
            endDate (tuple): End date of the data
        """
        start_timestamp = mkd.getDatetime_to_timestamp(mkd.getDatetime(*startDate))
        end_timestamp = mkd.getDatetime_to_timestamp(mkd.getDatetime(*endDate))
        
        if start_timestamp > end_timestamp:
            print("Error, the startDate must be lower than the endDate")
            exit(-1)

        if temporalities.get(temporality):
            data, _ = mkd.multi_threading(start_timestamp, end_timestamp, temporality, coin, writeOnExcel=False)
        else:
            print("Error, please select a correct temporality.")
            exit(-1)

        return data
    
    def prepare_series(self, df):
        """
        Prepare the series for the N-BEATS model
        
        Args:
            df (DataFrame): DataFrame with the data prices of the coin
        """
        df = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index, unit='s')  # Si el índice es timestamp en milisegundos

        self.series = TimeSeries.from_dataframe(df, value_cols=[self.name])
        self.scaler = Scaler(MinMaxScaler())
        self.series = self.scaler.fit_transform(self.series)
    
    def get_prices_returns(self, temporality, startDate, endDate):
        """
        Obtain the prices and returns of the coin in the specified temporality and date range
        
        Args:
            temporality (str): Temporality of the data
            startDate (tuple): Start date of the data
            endDate (tuple): End date of the data
        """
        if startDate > endDate:
            print("Error, the startDate must be lower than the endDate")
            exit(-1)
        
        data = self.obtainData(self.name, temporality, startDate, endDate)
        df, df_r = self.datapricesToDf(data)
        self.prepare_series(df) # N-BEATS
        print(f"Data for {self.name} obtained.")
        
        return df, df_r
    
    def train_model(self, folder_name="results", input_chunk_length=60, output_chunk_length=30, epochs=100, val_size=90):
        
        if self.series is None:
            raise ValueError(f"Series for {self.name} has not been prepared.")

        if len(self.series) < input_chunk_length + output_chunk_length:
            raise ValueError(f"Series for {self.name} is too short ({len(self.series)} values) to train any model.")

        # Si hay suficientes datos para separar validación
        if len(self.series) > val_size + input_chunk_length:
            train, val = self.series[:-val_size], self.series[-val_size:]
            use_val = True
        else:
            train = self.series
            val = None
            use_val = False
            print(f"[{self.name}] Not enough data for validation. Training with full dataset.")
            
        early_stopper = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        )

        print(f"Training model for {self.name}...")
        self.model = NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=epochs,
            random_state=42,
            pl_trainer_kwargs={
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "callbacks": [early_stopper],
                "enable_progress_bar": True,
            }        
        )
        
        print(f"Training model for {self.name}...")
        
        if use_val:
            self.model.fit(train, val_series=val)
        else:
            self.model.fit(train)
        print(f"Training completed for {self.name}.")
        
        folder_name = f".//{folder_name}//models"
        os.makedirs(folder_name, exist_ok=True)  
        self.model.save(folder_name + f"/{self.name}_nbeats_model.pth")
        print(f"Model saved for {self.name}.")
    
    def predict(self, n_future=365):
        if self.model is None:
            raise ValueError(f"The model for {self.name} has not been trained.")
        forecast = self.model.predict(n_future)
        forecast = self.scaler.inverse_transform(forecast)
        return forecast

    def evaluate(self, n_future=365):
        actual = self.scaler.inverse_transform(self.series[-n_future:])
        forecast = self.predict(n_future)
        return {
            "MAPE": mape(actual, forecast),
            "RMSE": rmse(actual, forecast)
        }
class Portfolio:
    def __init__(self, assets):
        self.assets = assets
        self.df_markowitz = pd.DataFrame()
        self.df_cvar_markowitz = pd.DataFrame()
        self.df_ew = pd.DataFrame()
        self.df_diversified_markowitz = pd.DataFrame()
        self.df_prices_nbeats = pd.DataFrame()  # Dataframe with the prices of the N-BEATS model
        self.df_nbeats = pd.DataFrame()  # Dataframe with the N-BEATS predictions
        self.df_prices_test = pd.DataFrame() # Dataframe with the prices of the test set
        self.folder_name = "results"
        
    #Guardar como variables los dataframes con los porcentajes de cada modelo??
    def __init__(self, assets):
        self.assets = assets
        
    def get_prices(self, temporality, startDate, endDate, excelFilename="crypto_2024"):
        """
        Obtain the prices of the coins in the specified temporality and date range
        
        Args:
            temporality (str): Temporality of the data
            startDate (tuple): Start date of the data
            endDate (tuple): End date of the data
        """
        df = pd.DataFrame()
        for asset in self.assets:
            coin_df, _ = asset.get_prices_returns(temporality, startDate, endDate)
            df[asset.name] = coin_df[asset.name]
        
        self.df_prices_test = df.copy()  # Save the prices of the test set
        excelFilename = f"{excelFilename}_prices.xlsx"
        df.to_excel(excelFilename, index=False, header=True)
        
        return df
    
    def get_returns(self, temporality, startDate, endDate):
        """
        Obtain the returns of the coins in the specified temporality and date range
        
        Args:
            temporality (str): Temporality of the data
            startDate (tuple): Start date of the data
            endDate (tuple): End date of the data
        """
        df_r = pd.DataFrame()
        for asset in self.assets:
            _, coin_df_r = asset.get_prices_returns(temporality, startDate, endDate)
            df_r[asset.name] = coin_df_r[asset.name]
        
        return df_r
    
    def get_returns_from_prices(self, df_prices):
        """
        Calculate the returns from the prices DataFrame.
        
        Args:
            df_prices (DataFrame): DataFrame with the prices of the coins
            
        Returns:
            df_r (DataFrame): DataFrame with the returns of the coins
        """
        df_r = pd.DataFrame(index=df_prices.index, columns=df_prices.columns)
        
        # Calculate the return for each row
        for column in df_prices.columns:
            df_r[column] = np.log(df_prices[column] / df_prices[column].shift(1)).fillna(0)
        
        return df_r
    
    def get_all_prices_returns(self, temporality, startDate, endDate, excelFilename="crypto_2023"):
        df, df_r = pd.DataFrame(), pd.DataFrame()
        for asset in self.assets:
            coin_df, coin_df_r = asset.get_prices_returns(temporality, startDate, endDate)
            df[asset.name] = coin_df[asset.name]
            df_r[asset.name] = coin_df_r[asset.name]
        
        prices_filename = f"{excelFilename}_prices.xlsx"
        returns_filename = f"{excelFilename}_returns.xlsx"
        
        df.to_excel(prices_filename, index=False, header=True)
        df_r.to_excel(returns_filename, index=False, header=True)
        
        print(f"Data saved in {prices_filename} and {returns_filename}")
        
        return  df, df_r
    
    def compute_optimal_portfolio(self, capital=1000, year=2023, temporality='1day', startDate=(2023, 1, 1, 0, 0, 0), endDate=(2023, 12, 31, 23, 59, 59), excelFilename="crypto_2024", l_riesgo=0.25, delta=0.1, portfolio=None):
        """
        Calculate the optimal weights of the portfolio using different models.
        
        Args:
            temporality (str): Temporality of the data
            startDate (tuple): Start date of the data
            endDate (tuple): End date of the data
            excelFilename (str): Name of the excel file where the data is stored
        """
        if excelFilename.endswith('.xlsx'):
            excelFilename = excelFilename.replace('.xlsx', '')
        
        df_result = pd.DataFrame()
        df, df_r = self.get_all_prices_returns(temporality, startDate, endDate, excelFilename)
        
        if df_r.empty:
            raise ValueError("The returns DataFrame (df_r) is empty. Please check the data retrieval process.")
        
        if portfolio is None:
            raise ValueError("Portfolio object is not provided. Please create a Portfolio instance with assets.")
        
        l_riesgo_map = {
            0.1 : "Conservador",
            0.25: "Moderado",
            0.75 : "Arriesgado",
        }
        l_riesgo_type = l_riesgo_map.get(l_riesgo, "Modificado")   
        
        self.folder_name = f"results_{year}_{l_riesgo_type}"
        
        if os.path.exists(self.folder_name):
            print(f"Folder {self.folder_name} already exists. Overwriting results.")
        else:
            os.makedirs(self.folder_name, exist_ok=True)
        
        coins = df_r.columns.tolist()

        df_markowitz = self.markowitz_function(df_r, l_riesgo)        
        df_cvar_markowitz = self.cvar_markowitz(df_r, l_riesgo, 95)
        df_ew = self.equally_weighted_portfolio(df_r)        
        df_diversified_markowitz = self.diversified_markowitz_function(df_r, l_riesgo, delta)
        
        # Entrenar los modelos N-BEATS para cada activo
        portfolio.train_all()
        
        df_predictions = portfolio.predict_and_export_to_excel(folder_name=self.folder_name, n_future=365)
        results = portfolio.evaluate_all(n_future=365)
        
        # Asignacion de los DataFrames a los atributos de la clase
        self.df_markowitz = df_markowitz
        self.df_cvar_markowitz = df_cvar_markowitz
        self.df_ew = df_ew
        self.df_diversified_markowitz = df_diversified_markowitz
        self.df_prices_nbeats = df_predictions
        
        # Calculamos los retornos de las predicciones de N-BEATS
        df_returns_nbeats = self.get_returns_from_prices(self.df_prices_nbeats)
        
        # Aplicamos Markowitz a las predicciones de N-BEATS
        self.df_nbeats = self.markowitz_function(df_returns_nbeats, l_riesgo)        

        startDate = (startDate[0] + 1, *startDate[1:])
        endDate = (endDate[0] + 1, *endDate[1:])
        
        excelFilename = f"crypto_{startDate[0]}"
        
        self.df_prices_test = self.get_prices(temporality, startDate, endDate, excelFilename=excelFilename)
        
        # Agregamos los resultados al DataFrame final (df_result)
        df_result['Markowitz'] = df_markowitz['Markowitz']        
        df_result['CVaR Markowitz'] = df_cvar_markowitz['CVaR Markowitz']
        df_result['Equally Weighted'] = df_ew['Equally Weighted']
        df_result['Diversified Markowitz'] = df_diversified_markowitz['Diversified Markowitz']
        df_result['N-BEATS'] = self.df_nbeats['Markowitz']
        df_result.reset_index(drop=True, inplace=True)
        df_result['Coins'] = coins
        df_result.set_index('Coins', inplace=True)
        
        excelFilename = f"{excelFilename}_portfolio.xlsx"
        df_result.to_excel(self.folder_name + "//" + excelFilename, index=True, header=True)
        portfolio_values, final_values = self.simulate_portfolio_growth(initial_capital=capital,folder_name=self.folder_name)
        print(f"Data saved in {excelFilename}")
        print(results)
        
        return portfolio_values, final_values, df_result
        
    def markowitz_function(self, df_returns, l_riesgo=1):
        """
        Calculate the optimal weights of the portfolio using the Markowitz model.
            
        Args:
            df_returns (DataFrame): DataFrame with the returns of the coins
            l_riesgo (float): Risk aversion parameter
            excel_filename (str): Name of the excel file where the results will be stored
        
        Returns:
            df_result (DataFrame): DataFrame with the optimal weights of the portfolio    
        """    
        print("\n\nMarkowitz's model.")
        print(f"Lambda (λ): {l_riesgo}")

        cov = df_returns.cov()
        cov_matrix = matrix(cov.values) 
        
        r = df_returns.mean() 
        r_riesgo = matrix(-l_riesgo * r.values)  
        
        #Restriction:
        A = matrix(np.ones(len(self.assets)).reshape(1, -1))
        b = matrix(1.0) 
        
        # Para cumplir w>=0 (no se pueden tener pesos negativos)
        G = matrix(-np.eye(len(self.assets))) # Matriz identidad en negativo 
        h = matrix(np.zeros(len(self.assets))) # Vector de ceros (pesos iguales o mayores a 0)
        
        sol = solvers.qp(P=cov_matrix, q=r_riesgo, G=G, h=h, A=A, b=b)
        
        w_opt = np.round(np.array(sol['x']).flatten(), 4)
        
        print("(Markowitz) Optimal weights:", w_opt)
        print("Sum of weights:", w_opt.sum())
        
        # Crear un DataFrame con los nombres de los activos y los resultados de la optimización
        df_result = pd.DataFrame(w_opt, columns=["Markowitz"])

        return df_result
    

    def cvar_markowitz(self, df_returns, l_riesgo=1, cvar=97):
        """
        Calculate the optimal weights of the portfolio using the Markowitz model.
            
        Args:
            df_returns (DataFrame): the returns of the coins
            l_riesgo (float): Risk aversion parameter
            cvar (int): Value at risk (VaR) at the 95% level
                    
        Returns:
            df_result (DataFrame): the optimal weights of the portfolio
        """    
        print("\n\nMarkowitz model with CVaR.")
        print(f"Risk aversion factor (λ): {l_riesgo}")
        print(f"CVaR level: {cvar}%")
                                
        # Calcular la media de los retornos (μ)
        mu = df_returns.mean()

        port_returns = np.dot(df_returns, np.ones(len(self.assets)))
        VaR = np.percentile(port_returns, (100-cvar))
        cvar_returns = port_returns[port_returns <= VaR]
        CVaR = np.mean(cvar_returns)

        def objective(w, mu, l_riesgo, CVaR):
            term1 = -l_riesgo * np.dot(mu, w)
            term2 = CVaR
            return term2 + term1  # Combinación de CVaR y rentabilidad ajustada

        # Restricciones
        # Los pesos deben sumar 1
        def constraint(w):
            return np.sum(w) - 1

        bounds = [(0, 1) for _ in range(len(self.assets))] # No negatividad
        
        initial_guess = np.ones(len(self.assets)) / len(self.assets) # Inicialización de los pesos

        # Optimización
        result = minimize(objective, initial_guess, args=(mu, l_riesgo, CVaR), method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': constraint})

        # Resultados
        if result.success:
            w_opt = result.x
            print(f"Optimal portfolio weights: {w_opt}")
        else:
            print("Optimization was not successful:", result.message)
        
        df_result = pd.DataFrame(w_opt, columns=["CVaR Markowitz"])
        
        return df_result
    
    def equally_weighted_portfolio(self, df_returns):
        """
        Calculate the optimal weights of the portfolio using the equally weighted model.
        
        Args:
            excel_returns (DataFrame): the returns of the coins
            excel_filename (string): the name of the excel file where the results are stored. If None, the name will be the same as the input file + '_ew.xlsx'.
        """
        print("\n\nEqually Weighted. Modelo de portafolio igualmente ponderado.")
                
        df_result = pd.DataFrame([1/len(self.assets)] * len(self.assets), columns=["Equally Weighted"])
        
        return df_result
    
    def diversified_markowitz_function(self, df_returns, l_riesgo=3, delta=1):
        """
        Calculate the optimal weights of the portfolio using the diversified Markowitz model.
            
        Args:
            df_returns (DataFrame): the returns of the coins
            l_riesgo (float): Risk aversion parameter
            delta (float): Diversification factor
            excel_filename (string): the name of the excel file where the results are stored. If None, the name will be the same as the input file + '_markowitz.xlsx'.
        """    
        
        print("\n\nMarkowitz model with diversification penalty.")
        print(f"Risk aversion factor (λ): {l_riesgo}")
        print(f"Diversification factor (δ): {delta}")
        
        # Media de los retornos (μ)
        mu = df_returns.mean()

        cov = df_returns.cov()
        cov_matrix = matrix(cov.values)  # Matriz de covarianza

        # Vector de rentabilidad esperada (μ)
        mu_vector = np.array(mu.values)

        # Vector de rentabilidad esperada ajustado por λ
        q = matrix(-l_riesgo * mu_vector)  # Rentabilidad ajustada por λ

        # Término de penalización por diversificación: δ * w^T w
        # La matriz de diversificación se agrega aquí como una pequeña constante sobre la identidad
        diversification_penalty = matrix(delta * np.eye(len(mu)))
        
        # Restricción de igualdad: la suma de los pesos debe ser 1 (suma total del portafolio)
        A = matrix(np.ones(len(self.assets)).reshape(1, -1))  # Suma de los pesos
        b = matrix(1.0)  # Suma total debe ser 1

        # Restricción de no negatividad: los pesos deben ser >= 0
        G = matrix(-np.eye(len(self.assets)))  # Matriz identidad en negativo
        h = matrix(np.zeros(len(self.assets)))  

        objective_matrix = cov_matrix + diversification_penalty  # Suma de la matriz de covarianza y penalización por diversificación
        solution = solvers.qp(P=objective_matrix, q=q, G=G, h=h, A=A, b=b)
        
        w_opt = np.round(np.array(solution['x']).flatten(), 4) # Pesos óptimos
        
        df_result = pd.DataFrame(w_opt, columns=["Diversified Markowitz"])

        return df_result
        
    def train_all(self):
        for asset in self.assets:
            asset.train_model(folder_name=self.folder_name)
    
    def predict_all(self):
        forecasts = {}
        for asset in self.assets:
            print(f"Predicting for {asset.name}...")
            forecasts[asset.name] = asset.predict()
        return forecasts
    
    def evaluate_all(self, n_future=365):
        results = {}
        for asset in self.assets:
            print(f"Evaluating {asset.name}...")
            results[asset.name] = asset.evaluate(n_future)
        return results
    
    def predict_and_export_to_excel(self, n_future=365, folder_name="results", filename="predicted_prices.xlsx"):
        self.predict_all()
        
        # Crear un DataFrame para almacenar las predicciones
        df_predictions = pd.DataFrame()

        for asset in self.assets:
            forecast = asset.predict(n_future)
            forecast_df = forecast.to_dataframe()
            df_predictions[asset.name] = forecast_df[asset.name]
            
        df_predictions.to_excel(folder_name + "//" + filename, index=False)
        print(f"Predicciones guardadas en {filename}")
        return df_predictions

    def simulate_portfolio_growth(self, initial_capital=1000, folder_name="results"):
        if self.df_prices_test.empty:
            raise ValueError("df_prices_test está vacío. Ejecuta primero get_prices().")

        model_dfs = {
            'Markowitz': self.df_markowitz,
            'CVaR Markowitz': self.df_cvar_markowitz,
            'Equally Weighted': self.df_ew,
            'Diversified Markowitz': self.df_diversified_markowitz,
            'N-BEATS': self.df_nbeats
        }

        portfolio_values = {}
        final_values = {}

        for model_name, df_weights in model_dfs.items():
            if df_weights.empty:
                continue

            # Indice como los nombres de los activos
            df_weights.index = list(self.df_prices_test.columns)
            weights = df_weights.iloc[:, 0]
            
            prices = self.df_prices_test[weights.index]

            initial_prices = prices.iloc[0]
            quantities = (weights * initial_capital) / initial_prices

            # Calculamos el valor de la cartera en cada momento
            portfolio_value = prices.multiply(quantities, axis=1).sum(axis=1)
            portfolio_values[model_name] = portfolio_value
            
            final_values[model_name] = portfolio_value.iloc[-1]

        # Mostrar valores finales por consola
        print("Valor final de cada cartera:")
        for model_name, final_val in final_values.items():
            print(f"{model_name}: {final_val:.2f} USDT")
            
        # Guardar a Excel
        df_final = pd.DataFrame.from_dict(final_values, orient='index', columns=['Final Portfolio Value'])
        df_final.to_excel(folder_name + "//final_portfolio_values.xlsx")
        
        matplotlib.use('Agg')  # Backend no GUI, solo para crear imágenes

        # Graficamos
        plt.figure(figsize=(12, 6))
        for model_name, values in portfolio_values.items():
            plt.plot(values.index, values.values, label=model_name)

        plt.title("Evolución del valor de la cartera según el modelo")
        plt.xlabel("Fecha")
        plt.ylabel("Valor de la cartera")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(folder_name + "//portfolio_evolution.png", dpi=300)
        plt.close()
        
        return portfolio_values, final_values

def gui_portfolio(capital, year, coins=None, temporality='1day', l_riesgo=0.25, delta=0.1):
    """
    Function to run the portfolio optimization in a GUI.
    
    Args:
        year (int): Year of the data
        l_riesgo (float): Risk aversion parameter
        delta (float): Diversification factor
        coins (list): List of coins to include in the portfolio
        temporality (str): Temporality of the data
    """
    excelfilename = f"crypto_{year}"
    excelfilename_test = f"crypto_{year}"

    if coins is None or coins == '':
        coins = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT', 'ADA-USDT', 'SOL-USDT', 'BNB-USDT', 'DOT-USDT', 'AVAX-USDT', 'DOGE-USDT', 'SHIB-USDT']
    
    assets = []
    for coin in coins:
        asset = Asset(coin)
        assets.append(asset)
    
    portfolio = Portfolio(assets)
    
    startDate = (year, 1, 1, 0, 0, 0)
    endDate = (year, 12, 31, 23, 59, 59)
    
    l_riesgo_map = {
            0.1 : "Conservador",
            0.25: "Moderado",
            0.75 : "Arriesgado",
        }
    l_riesgo_type = l_riesgo_map.get(l_riesgo, "Modificado")
    
    portfolio_values, final_values, df_results = portfolio.compute_optimal_portfolio(capital=capital, year=year, temporality=temporality, startDate=startDate, endDate=endDate, excelFilename=excelfilename, l_riesgo=l_riesgo, delta=delta, portfolio=portfolio)

    colores = {
        "Markowitz": "blue",
        "CVaR Markowitz": "green",
        "Equally Weighted": "orange",
        "Diversified Markowitz": "red"
    }
    
    #simulate_portfolio_growth_from_excel(f"{excelfilename_test}_portfolio.xlsx", f"{excelfilename_test}_prices.xlsx", initial_capital=1000, output_dir=f"results_{year}_{l_riesgo_type}", custom_colors=colores)
    
    return portfolio_values, final_values
  
if __name__ == "__main__":
    coins = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT', 'ADA-USDT', 'SOL-USDT', 'BNB-USDT', 'DOT-USDT', 'AVAX-USDT', 'DOGE-USDT', 'SHIB-USDT']
    temporality = '1day'
    year = 2022
    assets = []
    l_riesgo = 0.25
    delta = 0.01
    capital = 1500
    
    gui_portfolio(capital=capital, year=year, coins=coins, temporality=temporality, l_riesgo=l_riesgo, delta=delta)

    
   
  