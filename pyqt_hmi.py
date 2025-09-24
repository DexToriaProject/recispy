# hmi/pyqt_hmi.py
import sys
import numpy as np
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QGridLayout, QTextEdit
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core import RBMKCore
from comms.opcua_server import OPCUAServer

class RBMKOperatorHMI(QMainWindow):
    def __init__(self, reactor_core):
        super().__init__()
        self.core = reactor_core
        self.setWindowTitle("АСУ ТП РБМК-1000 — Пульт оператора")
        self.setGeometry(50, 50, 1400, 900)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")

        # Запуск OPC UA сервера
        self.opc_server = OPCUAServer(self.core)
        self.opc_server.start()

        self.init_ui()
        self.start_simulation_timer()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Верхняя панель — параметры
        top_panel = QGroupBox("Основные параметры реактора")
        top_layout = QGridLayout()
        self.labels = {}

        params = [
            ("Мощность", "%"), ("Темп. топлива", "°C"), ("Темп. на выходе", "°C"),
            ("Давление", "бар"), ("Реактивность", "Δk/k"), ("Паросодержание", "%"),
            ("Глубина СУЗ", "%"), ("Расход ТН", "%"), ("Ксенон-135", "отн. ед.")
        ]

        for i, (name, unit) in enumerate(params):
            label_name = QLabel(f"{name}:")
            label_name.setFont(QFont("Arial", 10, QFont.Bold))
            label_value = QLabel("0.0 " + unit)
            label_value.setFont(QFont("Courier", 11))
            self.labels[name] = label_value
            top_layout.addWidget(label_name, i // 4, (i % 4) * 2)
            top_layout.addWidget(label_value, i // 4, (i % 4) * 2 + 1)

        top_panel.setLayout(top_layout)
        main_layout.addWidget(top_panel)

        # Графики
        chart_panel = QGroupBox("Динамика параметров")
        chart_layout = QHBoxLayout()

        self.figure = Figure(figsize=(12, 4), facecolor='#1e1e2e')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e2e')
        self.ax.tick_params(colors='#cdd6f4')
        for spine in self.ax.spines.values():
            spine.set_color('#45475a')

        self.power_history = []
        self.temp_history = []
        self.xenon_history = []
        self.time_history = []

        chart_layout.addWidget(self.canvas)
        chart_panel.setLayout(chart_layout)
        main_layout.addWidget(chart_panel)

        # Управление
        control_panel = QGroupBox("Управление")
        control_layout = QHBoxLayout()

        suz_group = QGroupBox("Стержни СУЗ")
        suz_layout = QVBoxLayout()
        self.suz_slider = QSlider(Qt.Vertical)
        self.suz_slider.setRange(0, 100)
        self.suz_slider.setValue(100)
        self.suz_slider.valueChanged.connect(self.on_suz_change)
        suz_layout.addWidget(QLabel("Глубина ввода, %", alignment=Qt.AlignHCenter))
        suz_layout.addWidget(self.suz_slider)
        suz_group.setLayout(suz_layout)

        gcn_group = QGroupBox("Расход теплоносителя")
        gcn_layout = QVBoxLayout()
        self.gcn_slider = QSlider(Qt.Vertical)
        self.gcn_slider.setRange(0, 200)
        self.gcn_slider.setValue(100)
        self.gcn_slider.valueChanged.connect(self.on_gcn_change)
        gcn_layout.addWidget(QLabel("Расход, %", alignment=Qt.AlignHCenter))
        gcn_layout.addWidget(self.gcn_slider)
        gcn_group.setLayout(gcn_layout)

        btn_group = QGroupBox("Аварийные системы")
        btn_layout = QVBoxLayout()
        self.az_button = QPushButton("АЗ-5 (Аварийная защита)")
        self.az_button.setStyleSheet("background-color: #f38ba8; font-weight: bold; padding: 10px;")
        self.az_button.clicked.connect(self.trigger_az)
        self.saor_button = QPushButton("САОР (Авар. охлаждение)")
        self.saor_button.setStyleSheet("padding: 10px;")
        self.saor_button.clicked.connect(self.trigger_saor)
        btn_layout.addWidget(self.az_button)
        btn_layout.addWidget(self.saor_button)
        btn_group.setLayout(btn_layout)

        control_layout.addWidget(suz_group)
        control_layout.addWidget(gcn_group)
        control_layout.addWidget(btn_group)
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)

        # Журнал событий
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #313244; color: #a6adc8; font-family: Courier;")
        self.log_box.setMaximumHeight(150)
        main_layout.addWidget(QLabel("Журнал событий:"))
        main_layout.addWidget(self.log_box)

        self.log("✅ Симулятор РБМК-1000 запущен. OPC UA сервер активен.")

    def on_suz_change(self, value):
        depth = value / 100.0
        self.core.insertion_depth = depth
        self.log(f"Установлена глубина СУЗ: {value}%")

    def on_gcn_change(self, value):
        flow = value / 100.0
        self.core.set_coolant_flow(flow)
        self.log(f"Установлен расход теплоносителя: {value}%")

    def trigger_az(self):
        self.core.insertion_depth = 1.0
        self.log("❗ АЗ-5 — все стержни аварийно введены!")

    def trigger_saor(self):
        self.core.set_coolant_flow(1.5)
        self.log("💦 САОР активирована — расход увеличен до 150%")

    def log(self, message):
        self.log_box.append(f"[{self.core.time_elapsed:.1f} с] {message}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def update_display(self):
        status = self.core.get_status()

        # Обновление меток
        self.labels["Мощность"].setText(f"{status['power']:.1f} %")
        self.labels["Темп. топлива"].setText(f"{status['fuel_temp']:.1f} °C")
        self.labels["Темп. на выходе"].setText(f"{status['coolant_out']:.1f} °C")
        self.labels["Давление"].setText(f"{status['pressure']:.1f} бар")
        self.labels["Реактивность"].setText(f"{status['reactivity']:.4f}")
        self.labels["Паросодержание"].setText(f"{(1.0 - status['coolant_flow'])*100:.1f} %")
        self.labels["Глубина СУЗ"].setText(f"{status['insertion_depth']*100:.1f} %")
        self.labels["Расход ТН"].setText(f"{status['coolant_flow']*100:.1f} %")
        self.labels["Ксенон-135"].setText(f"{status['xenon']:.2e}")

        # Цветовая индикация
        danger_style = "color: #f38ba8; font-weight: bold;"
        normal_style = "color: #89b4fa;"

        self.labels["Мощность"].setStyleSheet(danger_style if status['power'] > 120 else normal_style)
        self.labels["Темп. топлива"].setStyleSheet(danger_style if status['fuel_temp'] > 1000 else normal_style)

        # Графики
        self.time_history.append(status['time'])
        self.power_history.append(status['power'])
        self.temp_history.append(status['fuel_temp'])
        self.xenon_history.append(status['xenon'])

        if len(self.time_history) > 100:
            self.time_history.pop(0)
            self.power_history.pop(0)
            self.temp_history.pop(0)
            self.xenon_history.pop(0)

        self.ax.clear()
        self.ax.plot(self.time_history, self.power_history, label="Мощность (%)", color="#89b4fa", linewidth=2)
        self.ax.plot(self.time_history, self.temp_history, label="Темп. топлива (°C)", color="#f9e2af", linewidth=2)
        self.ax.plot(self.time_history, self.xenon_history, label="Ксенон-135 (отн.)", color="#a6e3a1", linewidth=1)
        self.ax.set_title("Динамика параметров реактора", color="#cdd6f4", fontsize=12)
        self.ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=9)
        self.ax.grid(True, color="#45475a", linestyle='--', alpha=0.5)
        self.ax.tick_params(labelsize=9)
        self.canvas.draw()

        # Синхронизация слайдеров (если изменено из SCADA)
        self.suz_slider.blockSignals(True)
        self.suz_slider.setValue(int(status['insertion_depth'] * 100))
        self.suz_slider.blockSignals(False)

        self.gcn_slider.blockSignals(True)
        self.gcn_slider.setValue(int(status['coolant_flow'] * 100))
        self.gcn_slider.blockSignals(False)

        # Аварии
        if status['status'] != "NORMAL" and "ПРЕДУПРЕЖДЕНИЕ" not in status['status']:
            self.log(f"🚨 {status['status']}")

    def start_simulation_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.step_simulation)
        self.timer.start(500)

    def step_simulation(self):
        if self.core.is_running:
            self.core.update_physics()
            self.update_display()
        else:
            self.timer.stop()
            self.log(f"⏹️ СИМУЛЯЦИЯ ЗАВЕРШЕНА. СТАТУС: {self.core.status}")

    def closeEvent(self, event):
        self.opc_server.stop()
        event.accept()

# Запуск приложения
if __name__ == "__main__":
    app = QApplication(sys.argv)
    core = RBMKCore()
    window = RBMKOperatorHMI(core)
    window.show()
    sys.exit(app.exec_())