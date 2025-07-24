from PySide6.QtWidgets import (
    QApplication, QTableWidgetItem, QTableWidget, QListWidget,
    QPushButton, QComboBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLineEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QRunnable, QThreadPool, Slot, QObject, Signal
import sys
import Portfolio as p

import matplotlib.pyplot as plt
from io import BytesIO
import os 

def resource_path(relative_path):
    """ Devuelve la ruta absoluta para recursos cuando se usa PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
    
class WorkerSignals(QObject):
    final_values = Signal(dict)

class SimulationWorker(QRunnable):
    def __init__(self, year, coins, temporality, l_riesgo, capital):
        super().__init__()
        self.year = year
        self.coins = coins
        self.temporality = temporality
        self.l_riesgo = l_riesgo
        self.capital = capital
        self.signals = WorkerSignals() 

    @Slot()
    def run(self):
        print(f"Running simulation with {self.capital} for year {self.year} with coins {self.coins} and risk level {self.l_riesgo}")
        _, final_values = p.gui_portfolio(int(self.capital), self.year, self.coins, self.temporality, l_riesgo=self.l_riesgo, delta=0.1)
        self.signals.final_values.emit(final_values)
      
class MainApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.threadpool = QThreadPool()
        self.load_ui()
        self.folder_name = "results"
        self.capital = 1000

    def load_ui(self):
        loader = QUiLoader()
        ui_path = resource_path("gui.ui")
        file = QFile(ui_path)
        file.open(QFile.ReadOnly)
        self.window = loader.load(file)
        file.close()
        
        # Poner el icono
        icon_path = resource_path("logo.ico")
        icon = QIcon(icon_path)  
        self.window.setWindowIcon(icon)
        self.app.setWindowIcon(icon) 
        
        self.window.setMinimumSize(400, 300)
        self.window.setMaximumSize(16777215, 16777215)
        self.window.setWindowTitle("Composición de Portfolio y Simulación de Crecimiento")

        self.tabla = self.window.findChild(QTableWidget, "tablaDatos")
        self.coins_list = self.window.findChild(QListWidget, "coins")
        self.boton_simulacion = self.window.findChild(QPushButton, "button_startSimulation")

        self.tabla.setColumnCount(4)
        self.tabla.setHorizontalHeaderLabels(["Modelo", "Capital", "Beneficio", "Rentabilidad"])

        coins = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT', 'ADA-USDT', 'SOL-USDT',
                 'BNB-USDT', 'DOT-USDT', 'AVAX-USDT', 'DOGE-USDT', 'SHIB-USDT']
        self.coins_list.addItems(coins)

        self.boton_simulacion.clicked.connect(self.startSimulation)

        self.window.show()

    def startSimulation(self):
        print("Simulation started")
        year = self.window.findChild(QComboBox, "desplegable_year").currentText().strip()
        if not year.isdigit():
            print("Invalid year")
            return
        year = int(year)
        
        print(f"Selected year: {year}")
        if year is None:
            year = 2023

        l_riesgo_str = self.window.findChild(QComboBox, "desplegable_riesgo").currentText().strip()
        l_riesgo_map = {
            "Conservador": 0.1,
            "Moderado": 0.25,
            "Arriesgado": 0.75
        }
        l_riesgo = l_riesgo_map.get(l_riesgo_str, 0.25)

        coins = ['BTC-USDT', 'ETH-USDT', 'XRP-USDT', 'ADA-USDT', 'SOL-USDT',
                 'BNB-USDT', 'DOT-USDT', 'AVAX-USDT', 'DOGE-USDT', 'SHIB-USDT']
        temporality = '1day'
        
        l_riesgo_inversed_map = {
            0.1: "Conservador",
            0.25: "Moderado",
            0.75: "Arriesgado"
        }
        l_riesgo_type = l_riesgo_inversed_map.get(l_riesgo, "Modificado")
        self.folder_name = f"results_{year}_{l_riesgo_type}"

        capital = self.window.findChild(QLineEdit, "capital").text().strip()
        
        # Si el capital no es un número válido, usar un valor por defecto        
        if not capital.isdigit() or int(capital) <= 0 or capital == '':
            print("Invalid capital value. Using default value of 1000.")
            int_capital = 1000
        else:
            try:
                int_capital = int(capital)
            except ValueError:
                print("Invalid capital value. Using default value of 1000.")
                int_capital = 1000
            
        self.capital = int_capital
            
        self.worker = SimulationWorker(year, coins, temporality, l_riesgo, capital=int_capital)
        self.worker.signals.final_values.connect(self.handle_final_values)
        
        self.boton_simulacion.setEnabled(False)
        self.boton_simulacion.setText("Procesando...")

        print(f"Starting simulation for year {year} with risk level {l_riesgo}")
        self.threadpool.start(self.worker)
    
    def guardar_grafico_barras_rentabilidad(self, data_as_rows):
        # Extraer modelos y rentabilidades 
        modelos = [row[0] for row in data_as_rows]
        rentabilidades = [float(row[3].strip('%')) / 100 for row in data_as_rows]

        plt.figure(figsize=(10, 6))
        barras = plt.bar(modelos, rentabilidades, color='skyblue')

        # Añadir etiquetas de porcentaje encima de cada barra
        for barra, rent in zip(barras, rentabilidades):
            altura = barra.get_height()
            plt.text(
                barra.get_x() + barra.get_width() / 2,
                altura,
                f"{rent:.2%}",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        plt.title('Rentabilidad por Modelo')
        plt.xlabel('Modelo')
        plt.ylabel('Rentabilidad')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(rentabilidades)*1.15)  # un poco más de espacio arriba
        plt.tight_layout()

        nombre_archivo = self.folder_name + "//rentabilidad_por_modelo.png"
        plt.savefig(nombre_archivo, bbox_inches='tight')
        plt.close()

        print(f"Gráfico de barras guardado en {nombre_archivo}")
        
    def handle_final_values(self, data: dict):
        try:
            self.boton_simulacion.setEnabled(True)
            self.boton_simulacion.setText("Iniciar simulación")
            
            capital_inicial = self.capital

            resultados = []
            for modelo, capital_final in data.items():
                beneficio = capital_final - capital_inicial
                rentabilidad = beneficio / capital_inicial
                resultados.append((modelo, capital_final, beneficio, rentabilidad))

            resultados.sort(key=lambda x: x[3], reverse=True)

            data_as_rows = [
                [
                    modelo,
                    f"{capital_final:.2f}",
                    f"{beneficio:.2f}",
                    f"{rentabilidad:.2%}"
                ]
                for modelo, capital_final, beneficio, rentabilidad in resultados
            ]

            self.update_table("tablaDatos", data_as_rows)

            self.graph = self.window.findChild(QGraphicsView, "graph")
            self.mostrar_imagen_en_view(self.graph, resource_path(self.folder_name + "//portfolio_evolution.png"))
            
            self.guardar_grafico_barras_rentabilidad(data_as_rows)
            
            self.graph_bar = self.window.findChild(QGraphicsView, "graph_bar")
            self.mostrar_imagen_en_view(self.graph_bar, resource_path(self.folder_name + "//rentabilidad_por_modelo.png"))

        except Exception as e:
            print(f"Error en handle_final_values: {e}")

    def update_table(self, nombreTabla, data):
        print(f"Updating table {nombreTabla} with data: {data}")
        self.tabla = self.window.findChild(QTableWidget, nombreTabla)
        self.tabla.setRowCount(len(data))
        for row_idx, row_data in enumerate(data):
            for col_idx, item in enumerate(row_data):
                table_item = QTableWidgetItem(str(item))
                self.tabla.setItem(row_idx, col_idx, table_item)
    
    def mostrar_imagen_en_view(self, graphics_view, ruta_imagen):
        print(f"Mostrando imagen en {graphics_view} desde {ruta_imagen}")
        escena = QGraphicsScene()
        pixmap = QPixmap(ruta_imagen)
        
        if pixmap.isNull():
            print(f"No se pudo cargar la imagen {ruta_imagen}")
            return
        print(f"Imagen cargada correctamente: {ruta_imagen}")
        item = QGraphicsPixmapItem(pixmap)
        escena.addItem(item)
        graphics_view.setScene(escena)
        graphics_view.fitInView(item, Qt.KeepAspectRatio)

    def mostrar_grafico_pie_en_view(self, graphics_view, df, columna_activo, columna_valor):
        # Agrupar por activo y sumar la inversión
        datos = df.groupby(columna_activo)[columna_valor].sum()
        
        # Crear figura pie
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(datos, labels=datos.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('Distribución porcentual de la inversión por activo')
        ax.axis('equal')  # Círculo perfecto
        
        # Guardar imagen en buffer memoria
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        # Cargar imagen desde buffer
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())

        # Mostrar en QGraphicsView
        escena = QGraphicsScene()
        item = QGraphicsPixmapItem(pixmap)
        escena.addItem(item)
        graphics_view.setScene(escena)

        # Ajustar vista para mostrar todo el gráfico
        graphics_view.fitInView(item, mode=Qt.KeepAspectRatio)
    
if __name__ == "__main__":
    print("Loading UI...")
    main_app = MainApp()
    sys.exit(main_app.app.exec())

