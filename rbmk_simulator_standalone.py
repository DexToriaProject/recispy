#!/usr/bin/env python3
# rbmk_simulator_pro_fixed.py
# Полностью исправленная профессиональная версия с RK4/BDF, 1D диффузией, CHF, LOCA, GUI, OPC UA, двухоператорной АЗ-5
# Зависимости: pip install PyQt6 matplotlib numpy scipy opcua

import sys
import os
import json
import math
import time
import threading
import random
import numpy as np
from scipy.integrate import solve_ivp

# =============================
# Импорты PyQt (автоматически Qt5/Qt6)
# =============================
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QSlider, QGroupBox, QTextEdit, QListWidget, QComboBox,
        QCheckBox, QLineEdit, QMessageBox, QFrame, QSplitter, QDialog, QDialogButtonBox
    )
    from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject
    from PyQt6.QtGui import QFont, QColor, QPainter, QBrush
    PYQT_VERSION = 6
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QSlider, QGroupBox, QTextEdit, QListWidget, QComboBox,
        QCheckBox, QLineEdit, QMessageBox, QFrame, QSplitter, QDialog, QDialogButtonBox
    )
    from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
    from PyQt5.QtGui import QFont, QColor, QPainter, QBrush
    PYQT_VERSION = 5

# =============================
# Matplotlib — автоматический backend
# =============================
try:
    if PYQT_VERSION == 6:
        from matplotlib.backends.backend_qt6agg import FigureCanvasQTAgg as FigureCanvas
    else:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from matplotlib.figure import Figure

# =============================
# OPC UA (опционально)
# =============================
try:
    from opcua import Server, ua
    OPCUA_AVAILABLE = True
except ImportError:
    OPCUA_AVAILABLE = False
    print("⚠️  opcua не установлен. OPC UA сервер отключен.")

# =============================
# Загрузка конфигурации
# =============================
def load_config(path='config.json'):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

CONFIG = load_config()

# =============================
# ФИЗИКА: Нейтроника — BDF (жёсткий интегратор) + векторизация + 1D диффузия
# =============================
class NeutronicsCore:
    def __init__(self, N_nodes=20):
        self.N_nodes = N_nodes
        self.N_groups = 6
        self.beta_i = np.array([0.00025, 0.0012, 0.0012, 0.0026, 0.0008, 0.00065], dtype=np.float64)
        self.lambda_i = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01], dtype=np.float64)
        self.l = CONFIG.get('prompt_neutron_lifetime', 1e-5)  # сек
        self.D = CONFIG.get('diffusion_coeff', 0.5)  # м
        self.Sigma_a = CONFIG.get('absorption_xsec', 0.1)  # 1/м

        self.phi = np.ones(N_nodes, dtype=np.float64)
        self.Ci = np.zeros((N_nodes, self.N_groups), dtype=np.float64)
        self.rho_local = np.zeros(N_nodes, dtype=np.float64)
        self.time_elapsed = 0.0

        self.diffusion_matrix = self._build_diffusion_matrix()
        self.use_bdf = CONFIG.get('use_bdf_integrator', True)  # переключатель

    def _build_diffusion_matrix(self):
        D = self.D
        dx = 1.0 / (self.N_nodes - 1)
        A = np.zeros((self.N_nodes, self.N_nodes))
        for i in range(self.N_nodes):
            if i == 0:
                A[i, i] = -2 * D / dx**2
                A[i, i+1] = 2 * D / dx**2
            elif i == self.N_nodes - 1:
                A[i, i] = -2 * D / dx**2
                A[i, i-1] = 2 * D / dx**2
            else:
                A[i, i-1] = D / dx**2
                A[i, i] = -2 * D / dx**2
                A[i, i+1] = D / dx**2
        return A

    def rhs(self, t, y):
        # y = [phi_0, phi_1, ..., phi_{N-1}, C0_0, C0_1, ..., C5_{N-1}]
        phi = y[:self.N_nodes]
        Ci_flat = y[self.N_nodes:].reshape((self.N_nodes, self.N_groups))
        total_beta = np.sum(self.beta_i)
        diffusion_term = self.diffusion_matrix @ phi
        dphi_dt = ((self.rho_local - total_beta) / self.l) * phi + np.sum(self.lambda_i * Ci_flat, axis=1) + diffusion_term
        dCi_dt = (self.beta_i / self.l) * phi[:, np.newaxis] - self.lambda_i * Ci_flat
        return np.concatenate([dphi_dt, dCi_dt.flatten()])

    def step(self, rho_local, dt):
        self.rho_local = rho_local
        y0 = np.concatenate([self.phi, self.Ci.flatten()])
        t_span = (0, dt)

        if self.use_bdf:
            sol = solve_ivp(self.rhs, t_span, y0, method='BDF', rtol=1e-4, atol=1e-6)
            if not sol.success:
                print("⚠️  Интегратор BDF не сошелся — переключаемся на RK4")
                return self.rk4_step_fallback(rho_local, dt)
            y_final = sol.y[:, -1]
        else:
            y_final = self.rk4_step_manual(y0, dt)

        self.phi = np.maximum(1e-12, y_final[:self.N_nodes])
        self.Ci = np.maximum(0.0, y_final[self.N_nodes:].reshape((self.N_nodes, self.N_groups)))
        self.time_elapsed += dt
        return float(np.mean(self.phi))

    def rk4_step_manual(self, y0, dt):
        def f(t, y):
            return self.rhs(t, y)
        k1 = f(0, y0)
        k2 = f(0, y0 + 0.5 * dt * k1)
        k3 = f(0, y0 + 0.5 * dt * k2)
        k4 = f(0, y0 + dt * k3)
        return y0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def rk4_step_fallback(self, rho_local, dt):
        # Fallback на ручной RK4 при неудаче BDF
        phi_new, Ci_new = self.rk4_step_manual(np.concatenate([self.phi, self.Ci.flatten()]), dt)
        phi_new = phi_new[:self.N_nodes]
        Ci_new = Ci_new[self.N_nodes:].reshape((self.N_nodes, self.N_groups))
        return np.maximum(1e-12, phi_new), np.maximum(0.0, Ci_new)

# =============================
# ФИЗИКА: Теплогидравлика — с CHF и LOCA, в МВт и Вт/м²
# =============================
class ThermalHydraulics:
    def __init__(self, N_nodes=20):
        self.N_nodes = N_nodes
        self.fuel_temp = np.full(N_nodes, 600.0, dtype=np.float64)
        self.coolant_temp = np.full(N_nodes, 280.0, dtype=np.float64)
        self.pressure = np.full(N_nodes, 70.0, dtype=np.float64)  # бар
        self.void_fraction = np.zeros(N_nodes, dtype=np.float64)
        self.coolant_flow = 1.0
        self.mass_flow = CONFIG.get('mass_flow_kg_s', 1000.0)  # кг/с
        self.C_fuel = CONFIG.get('fuel_heat_capacity', 5e5)  # Дж/К
        self.C_coolant = CONFIG.get('coolant_heat_capacity', 4.2e6)  # Дж/(кг·К)
        self.UA = CONFIG.get('heat_transfer_coeff', 5e5)  # Вт/К
        self.void_coeff_base = CONFIG.get('void_coeff', 0.02)

        self.CHF_base = CONFIG.get('CHF_threshold_W_m2', 2.0e6)  # Вт/м²
        self.CHF_margin = CONFIG.get('CHF_margin', 1.3)
        self.is_CHF = False

        self.is_LOCA = False
        self.LOCA_severity = 0.0

        # Площадь активной зоны для расчёта плотности теплового потока
        self.active_area_m2 = CONFIG.get('active_area_m2', 1000.0)  # м²

    def update_void_fraction(self):
        for i in range(self.N_nodes):
            P_MPa = self.pressure[i] * 0.1
            if P_MPa < 0.1:
                T_sat = 100.0
            elif P_MPa > 22.1:
                T_sat = 374.0
            else:
                T_sat = 100 + 274 * (P_MPa - 0.1) / (22.0 - 0.1)
            delta_T = self.coolant_temp[i] - T_sat
            self.void_fraction[i] = 1.0 / (1.0 + np.exp(-0.1 * delta_T))
            self.void_fraction[i] = np.clip(self.void_fraction[i], 0.0, 0.95)

    def check_CHF(self, q_local):
        # Более реалистичный порог: 80% от CHF_base (без учета margin для чувствительности)
        threshold = self.CHF_base * 0.8
        for i in range(self.N_nodes):
            if q_local[i] > threshold:
                self.is_CHF = True
                return True
        self.is_CHF = False
        return False

    def apply_LOCA(self, severity):
        self.LOCA_severity = max(0.0, min(1.0, severity))
        self.coolant_flow *= (1.0 - 0.8 * self.LOCA_severity)
        self.pressure *= (1.0 - 0.5 * self.LOCA_severity)

    def step(self, power_MW, dt):
        self.update_void_fraction()
        
        # Переводим МВт в Вт, делим на площадь → Вт/м²
        q_local = (power_MW * 1e6) / self.active_area_m2 * np.ones(self.N_nodes)
        self.check_CHF(q_local)

        for i in range(self.N_nodes):
            q = q_local[i]
            q_transfer = self.UA * (self.fuel_temp[i] - self.coolant_temp[i])
            self.fuel_temp[i] += (q - q_transfer) / self.C_fuel * dt
            
            flow = max(0.01, self.coolant_flow)
            self.coolant_temp[i] += (q_transfer / (self.C_coolant * flow)) * dt
            
            self.pressure[i] = 70.0 + (self.coolant_temp[i] - 280.0) * 0.5
            if self.is_LOCA:
                self.pressure[i] *= (1.0 - 0.3 * self.LOCA_severity)

        return np.mean(self.fuel_temp)

# =============================
# ФИЗИКА: Ксенон/Йод
# =============================
class XenonModel:
    def __init__(self, N_nodes=20):
        self.N_nodes = N_nodes
        self.sigma_Xe = CONFIG.get('xenon_absorption_xsec', 2.6e-19)
        self.lambda_I = CONFIG.get('iodine_decay_const', 2.87e-5)
        self.lambda_Xe = CONFIG.get('xenon_decay_const', 2.09e-5)
        self.gamma_I = CONFIG.get('iodine_yield', 0.064)
        self.gamma_Xe = CONFIG.get('xenon_yield', 0.003)
        self.N0 = CONFIG.get('neutron_density_scale', 1e14)
        self.I = np.zeros(N_nodes, dtype=np.float64)
        self.Xe = np.zeros(N_nodes, dtype=np.float64)
        self._I_max = 1e20
        self._Xe_max = 1e18

    def step(self, flux, dt):
        for i in range(self.N_nodes):
            if not np.isfinite(flux[i]) or flux[i] < 0:
                flux[i] = 0.0
            dI = self.gamma_I * flux[i] * self.N0 - self.lambda_I * self.I[i]
            dXe = (self.gamma_Xe * flux[i] * self.N0 + self.lambda_I * self.I[i] -
                   self.lambda_Xe * self.Xe[i] - flux[i] * self.N0 * self.sigma_Xe * self.Xe[i])
            self.I[i] += dI * dt
            self.Xe[i] += dXe * dt
            if not np.isfinite(self.I[i]):
                self.I[i] = self._I_max
            if not np.isfinite(self.Xe[i]):
                self.Xe[i] = self._Xe_max
            self.I[i] = max(0.0, min(self.I[i], self._I_max))
            self.Xe[i] = max(0.0, min(self.Xe[i], self._Xe_max))
        return - (self.sigma_Xe * np.mean(self.Xe)) * 1e3

# =============================
# УПРАВЛЕНИЕ: Стержни
# =============================
class ControlRods:
    def __init__(self, N_rods=211, N_nodes=20):
        self.N_rods = N_rods
        self.N_nodes = N_nodes
        self.insertion = np.ones(N_rods, dtype=np.float64)
        self.max_speed = CONFIG.get('rod_max_speed', 0.02)
        self.target_insertion = np.ones(N_rods, dtype=np.float64)
        depth = np.linspace(0, 1, 100)
        self.efficiency_curve = -0.05 * (1 - depth)**2  # можно вынести в конфиг

    def set_target(self, target):
        if isinstance(target, (int, float)):
            self.target_insertion = np.full(self.N_rods, float(target))
        else:
            self.target_insertion = np.array(target, dtype=np.float64)

    def move(self, dt):
        for i in range(self.N_rods):
            delta = self.target_insertion[i] - self.insertion[i]
            move = np.clip(delta, -self.max_speed * dt, self.max_speed * dt)
            self.insertion[i] += move
            self.insertion[i] = np.clip(self.insertion[i], 0.0, 1.0)

    def get_reactivity(self):
        eff = np.interp(self.insertion, np.linspace(0, 1, 100), self.efficiency_curve)
        return np.sum(eff)

# =============================
# УПРАВЛЕНИЕ: Насосы
# =============================
class CoolantPumps:
    def __init__(self, N_pumps=8):
        self.N_pumps = N_pumps
        self.status = np.ones(N_pumps, dtype=bool)
        self.flow_rate = np.full(N_pumps, 12.0, dtype=np.float64)
        self.power = np.full(N_pumps, 5.5, dtype=np.float64)

    def set_pump(self, pump_id, state):
        self.status[pump_id] = bool(state)

    def get_total_flow(self):
        return sum(self.flow_rate[i] if self.status[i] else 0 for i in range(self.N_pumps))

# =============================
# УПРАВЛЕНИЕ: Защита (АЗ-5) — с двухоператорным подтверждением
# =============================
class ProtectionSystem:
    def __init__(self):
        self.AZ_triggers = {
            "power_over": lambda core: core.power_MW > 150,
            "temp_over": lambda core: np.max(core.thermal.fuel_temp) > 1200,
            "pressure_over": lambda core: np.max(core.thermal.pressure) > 85,
            "flux_rate": lambda core: getattr(core, 'dflux_dt', 0) > 0.1,
            "CHF_risk": lambda core: core.thermal.is_CHF,
            "LOCA_detected": lambda core: core.thermal.is_LOCA and core.thermal.LOCA_severity > 0.3
        }
        self.AZ_activated = False
        self.awaiting_confirmation = False
        self.confirmation_count = 0
        self.last_check = 0.0

    def check(self, core):
        if self.AZ_activated or self.awaiting_confirmation:
            return
        if core.time_elapsed - self.last_check < 0.1:
            return
        self.last_check = core.time_elapsed
        for name, condition in self.AZ_triggers.items():
            if condition(core):
                core.log_event(f"АЗ-5: условие '{name}' — требуется подтверждение")
                self.awaiting_confirmation = True
                self.confirmation_count = 0
                # UI должен вызвать request_confirmation (реализовано в ControlPanel)
                break

    def request_confirmation(self, core):
        if not self.awaiting_confirmation:
            return False
        self.confirmation_count += 1
        core.log_event(f"АЗ-5: подтверждение #{self.confirmation_count}/2")
        if self.confirmation_count >= 2:
            self.trigger_AZ(core, "АЗ-5: активирована по двухоператорному подтверждению")
            return True
        return False

    def trigger_AZ(self, core, reason):
        self.AZ_activated = True
        self.awaiting_confirmation = False
        self.confirmation_count = 0
        core.control_rods.set_target(1.0)
        core.log_event(reason)

# =============================
# ДАТЧИКИ: Шум, отказы, голосование — БЕЗ фиксированного seed
# =============================
class SensorSystem:
    def __init__(self, N_channels=3):
        self.N_channels = N_channels
        self.noise_level = 0.01
        self.failure_prob = 0.001
        self.channels = {}
        # Убрали np.random.seed(42) — шум теперь случайный при каждом запуске

    def add_sensor(self, name, value):
        if name not in self.channels:
            self.channels[name] = [value] * self.N_channels
        for i in range(self.N_channels):
            if np.random.random() < self.failure_prob:
                self.channels[name][i] = value * np.random.uniform(0.5, 1.5)
            else:
                self.channels[name][i] = value + np.random.normal(0, self.noise_level * abs(value) + 1e-3)
        sorted_vals = sorted(self.channels[name])
        return sorted_vals[1]

# =============================
# ЯДРО СИМУЛЯТОРА — с потокобезопасностью
# =============================
class RBMKCore:
    def __init__(self):
        self.N_nodes = CONFIG.get('N_nodes', 20)
        self.neutronics = NeutronicsCore(self.N_nodes)
        self.thermal = ThermalHydraulics(self.N_nodes)
        self.xenon = XenonModel(self.N_nodes)
        self.control_rods = ControlRods(211, self.N_nodes)
        self.pumps = CoolantPumps(8)
        self.protection = ProtectionSystem()
        self.sensors = SensorSystem(3)

        self.power_MW = CONFIG.get('initial_power_MW', 100.0)  # МВт
        self.reactivity = 0.0
        self.time_step = CONFIG.get('time_step', 0.1)
        self.time_elapsed = 0.0
        self.is_running = True
        self.status = "NORMAL"
        self.events = []
        self.dflux_dt = 0.0
        self.last_phi = 1.0

        self.history = {
            'time': [], 'power_MW': [], 'fuel_temp': [], 'xenon': [],
            'reactivity': [], 'coolant_flow': [], 'insertion_depth': [],
            'CHF': [], 'LOCA': []
        }
        self.max_history = CONFIG.get('max_history_points', 1000)

        self._lock = threading.RLock()

    def log_event(self, message):
        with self._lock:
            self.events.append(f"[{self.time_elapsed:.1f}s] {message}")

    def calculate_local_reactivity(self):
        rho_local = np.zeros(self.N_nodes)
        for i in range(self.N_nodes):
            doppler = -5e-4 * math.log(1.0 + max(0.0, (self.thermal.fuel_temp[i] - 600.0) / 100.0) + 1e-9)
            void = self.thermal.void_coeff_base * self.thermal.void_fraction[i]
            xenon_local = - (self.xenon.sigma_Xe * self.xenon.Xe[i]) * 1e3
            rod_eff = self.control_rods.get_reactivity() / self.N_nodes
            rho_local[i] = doppler + void + xenon_local + rod_eff
            rho_local[i] = max(-10.0, min(1.0, rho_local[i]))
        # ✅ Сохраняем среднюю реактивность для UI
        self.reactivity = float(np.mean(rho_local))
        return rho_local

    def step(self):
        with self._lock:
            if not self.is_running:
                return

            self.control_rods.move(self.time_step)
            self.thermal.coolant_flow = self.pumps.get_total_flow() / (8 * 12.0)

            rho_local = self.calculate_local_reactivity()
            phi_avg = self.neutronics.step(rho_local, self.time_step)
            self.dflux_dt = (phi_avg - self.last_phi) / self.time_step
            self.last_phi = phi_avg
            self.power_MW = phi_avg * 100.0  # масштаб: 1.0 phi = 100 МВт

            self.thermal.step(self.power_MW, self.time_step)

            self.xenon.step(self.neutronics.phi, self.time_step)

            self.protection.check(self)

            self.time_elapsed += self.time_step

            self.update_history()
            self.safety_checks()

    def update_history(self):
        for key in self.history:
            if len(self.history[key]) > self.max_history:
                self.history[key].pop(0)
        self.history['time'].append(self.time_elapsed)
        self.history['power_MW'].append(self.power_MW)
        self.history['fuel_temp'].append(np.mean(self.thermal.fuel_temp))
        self.history['xenon'].append(np.mean(self.xenon.Xe))
        self.history['reactivity'].append(self.reactivity)
        self.history['coolant_flow'].append(self.thermal.coolant_flow)
        self.history['insertion_depth'].append(np.mean(self.control_rods.insertion))
        self.history['CHF'].append(1.0 if self.thermal.is_CHF else 0.0)
        self.history['LOCA'].append(self.thermal.LOCA_severity)

    def safety_checks(self):
        if np.max(self.thermal.fuel_temp) > 1200:
            self.status = "ПЛАВЛЕНИЕ АКТИВНОЙ ЗОНЫ"
            self.is_running = False
        elif self.power_MW > 150 and self.thermal.coolant_flow < 0.5:
            self.status = "РИСК ПАРОВОГО ВЗРЫВА"
        elif self.power_MW < 1:
            self.status = "РЕАКТОР ЗАГЛУШЕН"
        elif self.thermal.is_CHF:
            self.status = "РИСК КРИЗИСА ТЕПЛООТДАЧИ (CHF)"
        elif self.thermal.is_LOCA and self.thermal.LOCA_severity > 0.5:
            self.status = "АВАРИЯ С РАЗРЫВОМ КОНТУРА (LOCA)"
        else:
            self.status = "NORMAL"

    def get_status(self):
        with self._lock:
            events_copy = self.events.copy()  # ✅ Безопасное копирование
            alarms = [self.status] if self.status != "NORMAL" else []
            return {
                'time': self.time_elapsed,
                'power_MW': self.sensors.add_sensor('power', self.power_MW),
                'fuel_temp_avg': self.sensors.add_sensor('fuel_temp', np.mean(self.thermal.fuel_temp)),
                'fuel_temp_max': self.sensors.add_sensor('fuel_temp_max', np.max(self.thermal.fuel_temp)),
                'coolant_temp_out': self.sensors.add_sensor('coolant_temp', np.mean(self.thermal.coolant_temp)),
                'pressure': self.sensors.add_sensor('pressure', np.mean(self.thermal.pressure)),
                'reactivity': self.reactivity,
                'coolant_flow': self.thermal.coolant_flow,
                'insertion_depth': np.mean(self.control_rods.insertion),
                'xenon': np.mean(self.xenon.Xe),
                'is_CHF': self.thermal.is_CHF,
                'LOCA_severity': self.thermal.LOCA_severity,
                'status': self.status,
                'fuel_temp_map': self.thermal.fuel_temp.copy(),
                'events': events_copy,
                'alarms': alarms
            }

    def set_rods(self, depth):
        with self._lock:
            self.control_rods.set_target(depth)

    def fail_pump(self, pump_id=0):
        with self._lock:
            self.pumps.set_pump(pump_id, False)

    def trigger_LOCA(self, severity=0.5):
        with self._lock:
            self.thermal.apply_LOCA(severity)
            self.log_event(f"❗ Моделируется LOCA: severity={severity}")

    def confirm_AZ5(self):
        """Вызывается UI для подтверждения АЗ-5"""
        with self._lock:
            return self.protection.request_confirmation(self)

    def export_history(self, filename='history.csv'):
        """Экспорт истории в CSV"""
        import csv
        keys = list(self.history.keys())
        rows = zip(*[self.history[k] for k in keys])
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(keys)
            w.writerows(rows)
        self.log_event(f"✅ История экспортирована в {filename}")

# =============================
# UI: Тепловая карта
# =============================
class ZoneHeatmap(QWidget):
    def __init__(self, N_nodes=20):
        super().__init__()
        self.N_nodes = N_nodes
        self.data = np.full(N_nodes, 600.0)
        self.setMinimumSize(400, 100)

    def update_data(self, data):
        self.data = data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        width = self.width() // self.N_nodes
        height = self.height()
        for i in range(self.N_nodes):
            temp = self.data[i]
            ratio = (temp - 500) / (1200 - 500)
            ratio = max(0.0, min(1.0, ratio))
            r = int(255 * ratio)
            g = int(255 * (1 - ratio) * 0.8)
            b = int(255 * (1 - ratio))
            color = QColor(r, g, b)
            painter.setBrush(QBrush(color))
            painter.drawRect(i * width, 0, width, height)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(i * width + 2, 15, f"{int(temp)}")
            if i % 5 == 0:
                painter.drawText(i * width + 2, 30, f"#{i}")

# =============================
# UI: Цифровой индикатор
# =============================
class DigitalIndicator(QFrame):
    def __init__(self, label, unit):
        super().__init__()
        self.label = label
        self.unit = unit
        self.value = 0.0
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.label_widget = QLabel(self.label)
        self.label_widget.setFont(QFont("Arial", 10))
        self.value_widget = QLabel("0.0 " + self.unit)
        self.value_widget.setFont(QFont("Courier", 16, QFont.Weight.Bold))
        layout.addWidget(self.label_widget)
        layout.addWidget(self.value_widget)
        self.setLayout(layout)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)

    def set_value(self, value):
        self.value = value
        self.value_widget.setText(f"{value:.1f} {self.unit}")
        if "Темп" in self.label and value > 1000:
            self.value_widget.setStyleSheet("color: red;")
        elif "Мощность" in self.label and value > 120:
            self.value_widget.setStyleSheet("color: red;")
        elif "CHF" in self.label and value > 0.5:
            self.value_widget.setStyleSheet("color: orange;")
        elif "LOCA" in self.label and value > 0.3:
            self.value_widget.setStyleSheet("color: red;")
        else:
            self.value_widget.setStyleSheet("color: white;")

# =============================
# UI: График трендов
# =============================
class TrendPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(5, 3), facecolor='#1e1e2e')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e2e')
        self.ax.tick_params(colors='white')
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.data = {'time': [], 'power': [], 'temp': [], 'CHF': [], 'LOCA': []}

    def add_point(self, time, power, temp, CHF, LOCA):
        self.data['time'].append(time)
        self.data['power'].append(power)
        self.data['temp'].append(temp)
        self.data['CHF'].append(CHF)
        self.data['LOCA'].append(LOCA)
        if len(self.data['time']) > 100:
            for key in self.data:
                self.data[key].pop(0)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.data['time'], self.data['power'], label='Power (MW)', color='cyan', linewidth=2)
        self.ax.plot(self.data['time'], self.data['temp'], label='Temp (°C)', color='yellow', linewidth=2)
        self.ax.plot(self.data['time'], [x*1000 for x in self.data['CHF']], label='CHF Risk', color='orange', linewidth=2)
        self.ax.plot(self.data['time'], [x*1000 for x in self.data['LOCA']], label='LOCA', color='red', linewidth=2)
        self.ax.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='white', fontsize=8)
        self.ax.grid(True, color='#45475a', linestyle='--')
        self.ax.set_ylim(0, 1500)
        self.canvas.draw()

# =============================
# UI: Панель тревог
# =============================
class AlarmPanel(QListWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #313244; color: #f38ba8;")

    def add_alarm(self, message):
        if self.findItems(message, Qt.MatchFlag.MatchExactly):
            return
        self.addItem(message)
        self.scrollToBottom()

# =============================
# UI: Диалог подтверждения АЗ-5
# =============================
class AZ5ConfirmationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Подтверждение АЗ-5")
        self.setModal(True)
        layout = QVBoxLayout()
        label = QLabel("Требуется второе подтверждение для активации АЗ-5.\nПодтвердите действие.")
        layout.addWidget(label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

# =============================
# UI: Панель управления (с кнопкой LOCA и двухоператорной АЗ-5)
# =============================
class ControlPanel(QWidget):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        
        rod_group = QGroupBox("Стержни СУЗ")
        rod_layout = QHBoxLayout()
        self.rod_slider = QSlider(Qt.Orientation.Horizontal)
        self.rod_slider.setRange(0, 100)
        self.rod_slider.setValue(100)
        self.rod_slider.valueChanged.connect(self.on_rod_change)
        rod_layout.addWidget(QLabel("Глубина ввода (%):"))
        rod_layout.addWidget(self.rod_slider)
        rod_group.setLayout(rod_layout)
        layout.addWidget(rod_group)

        btn_layout = QHBoxLayout()
        az_btn = QPushButton("АЗ-5 (Аварийная защита)")
        az_btn.setStyleSheet("background-color: #f38ba8; padding: 10px;")
        az_btn.clicked.connect(self.trigger_az)
        fail_btn = QPushButton("Отказ насоса")
        fail_btn.clicked.connect(self.fail_pump)
        loca_btn = QPushButton("Моделировать LOCA")
        loca_btn.setStyleSheet("background-color: #fab387; padding: 10px;")
        loca_btn.clicked.connect(self.trigger_LOCA)
        export_btn = QPushButton("Экспорт в CSV")
        export_btn.clicked.connect(self.export_history)
        btn_layout.addWidget(az_btn)
        btn_layout.addWidget(fail_btn)
        btn_layout.addWidget(loca_btn)
        btn_layout.addWidget(export_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def on_rod_change(self, value):
        self.core.set_rods(value / 100.0)

    def trigger_az(self):
        # Первое подтверждение — сразу
        if self.core.protection.request_confirmation(self.core):
            return  # уже активировано
        # Показываем диалог для второго оператора
        dialog = AZ5ConfirmationDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if self.core.protection.request_confirmation(self.core):
                pass  # успешно активировано

    def fail_pump(self):
        self.core.fail_pump()
        self.core.log_event("Отказ ГЦН #0")

    def trigger_LOCA(self):
        self.core.trigger_LOCA(severity=0.7)
        self.core.log_event("❗ Моделируется LOCA с severity=0.7")

    def export_history(self):
        self.core.export_history()

# =============================
# UI: Главное окно
# =============================
class RBMKOperatorPanel(QMainWindow):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.setWindowTitle("RBMK-1000 Pro — Пульт оператора")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")

        self.init_ui()
        self.start_simulation()

    def init_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout()

        top_panel = QGroupBox("Параметры реактора")
        top_layout = QGridLayout()
        self.indicators = {}
        params = [
            ("Мощность", "МВт"), ("Темп. топлива (ср)", "°C"), ("Темп. топлива (макс)", "°C"),
            ("Темп. на выходе", "°C"), ("Давление", "бар"), ("Реактивность", "Δk/k"),
            ("Расход ТН", "%"), ("Глубина СУЗ", "%"), ("Ксенон", "отн."),
            ("Риск CHF", "усл. ед."), ("LOCA", "усл. ед.")
        ]
        for i, (name, unit) in enumerate(params):
            indicator = DigitalIndicator(name, unit)
            self.indicators[name] = indicator
            top_layout.addWidget(indicator, i // 4, i % 4)
        top_panel.setLayout(top_layout)
        main_layout.addWidget(top_panel)

        mid_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.heatmap = ZoneHeatmap(self.core.N_nodes)
        mid_splitter.addWidget(self.heatmap)
        self.trend_plot = TrendPlot()
        mid_splitter.addWidget(self.trend_plot)
        mid_splitter.setSizes([500, 900])
        main_layout.addWidget(mid_splitter)

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.control_panel = ControlPanel(self.core)
        bottom_splitter.addWidget(self.control_panel)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        alarm_label = QLabel("Тревоги:")
        self.alarm_panel = AlarmPanel()
        event_label = QLabel("Журнал событий:")
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setStyleSheet("background-color: #313244; color: #a6adc8;")
        right_layout.addWidget(alarm_label)
        right_layout.addWidget(self.alarm_panel)
        right_layout.addWidget(event_label)
        right_layout.addWidget(self.event_log)
        right_panel.setLayout(right_layout)
        bottom_splitter.addWidget(right_panel)
        
        bottom_splitter.setSizes([500, 900])
        main_layout.addWidget(bottom_splitter)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def start_simulation(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(1000)

        self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.sim_thread.start()

    def simulation_loop(self):
        while self.core.is_running:
            start_time = time.time()
            self.core.step()
            elapsed = time.time() - start_time
            sleep_time = max(0.0, self.core.time_step - elapsed)
            time.sleep(sleep_time)

    def update_display(self):
        status = self.core.get_status()
        
        indicators_map = {
            "Мощность": status['power_MW'],
            "Темп. топлива (ср)": status['fuel_temp_avg'],
            "Темп. топлива (макс)": status['fuel_temp_max'],
            "Темп. на выходе": status['coolant_temp_out'],
            "Давление": status['pressure'],
            "Реактивность": status['reactivity'],
            "Расход ТН": status['coolant_flow'] * 100,
            "Глубина СУЗ": status['insertion_depth'] * 100,
            "Ксенон": status['xenon'],
            "Риск CHF": 1.0 if status['is_CHF'] else 0.0,
            "LOCA": status['LOCA_severity']
        }

        for name, value in indicators_map.items():
            self.indicators[name].set_value(value)

        self.heatmap.update_data(status['fuel_temp_map'])

        if len(self.core.history['time']) > 1:
            last_idx = -1
            self.trend_plot.add_point(
                self.core.history['time'][last_idx],
                self.core.history['power_MW'][last_idx],
                self.core.history['fuel_temp'][last_idx],
                self.core.history['CHF'][last_idx],
                self.core.history['LOCA'][last_idx]
            )

        for alarm in status['alarms']:
            self.alarm_panel.add_alarm(alarm)

        for event in status['events']:
            if event not in self.event_log.toPlainText():
                self.event_log.append(event)
        self.core.events.clear()  # ✅ Очистка после отображения

        if not self.core.is_running:
            self.timer.stop()
            QMessageBox.critical(self, "Симуляция завершена", f"Статус: {self.core.status}")

# =============================
# OPC UA Server
# =============================
class OPCUAServerWrapper:
    def __init__(self, core):
        if not OPCUA_AVAILABLE:
            self.running = False
            return
        self.core = core
        self.server = Server()
        self.server.set_endpoint("opc.tcp://0.0.0.0:4840/rbmk1000/")
        self.server.set_server_name("RBMK-1000 Pro Simulator OPC UA Server")
        uri = "http://rbmk-simulator.org"
        idx = self.server.register_namespace(uri)
        objects = self.server.get_objects_node()
        self.reactor_obj = objects.add_object(idx, "ReactorCore")
        self.vars = {}
        for name in ['Power_MW', 'FuelTemperature', 'Pressure', 'Reactivity', 'CoolantFlow', 'RodInsertion', 'LOCA_Severity']:
            self.vars[name] = self.reactor_obj.add_variable(idx, name, 0.0)
            if name in ['RodInsertion', 'CoolantFlow']:
                self.vars[name].set_writable()
        self.running = False

    def start(self):
        if not OPCUA_AVAILABLE:
            return
        try:
            self.server.start()
            self.running = True
            threading.Thread(target=self._loop, daemon=True).start()
            print("✅ OPC UA Server запущен на opc.tcp://localhost:4840/rbmk1000/")
        except Exception as e:
            print(f"❌ Ошибка запуска OPC UA: {e}")

    def _loop(self):
        while self.running:
            try:
                status = self.core.get_status()
                self.vars['Power_MW'].set_value(status['power_MW'])
                self.vars['FuelTemperature'].set_value(status['fuel_temp_avg'])
                self.vars['Pressure'].set_value(status['pressure'])
                self.vars['Reactivity'].set_value(status['reactivity'])
                self.vars['CoolantFlow'].set_value(status['coolant_flow'])
                self.vars['RodInsertion'].set_value(status['insertion_depth'])
                self.vars['LOCA_Severity'].set_value(status['LOCA_severity'])
            except Exception:
                pass
            time.sleep(1)

    def stop(self):
        if self.running and OPCUA_AVAILABLE:
            self.server.stop()
            self.running = False

# =============================
# ЗАПУСК
# =============================
if __name__ == "__main__":
    print("🚀 Запуск профессионального симулятора РБМК-1000 Pro...")
    print("⚙️  Установите opcua для OPC UA: pip install opcua")
    print("📄 Создайте config.json для кастомизации параметров.")

    core = RBMKCore()
    opc_server = OPCUAServerWrapper(core)
    opc_server.start()

    app = QApplication(sys.argv)
    window = RBMKOperatorPanel(core)
    window.show()

    def cleanup():
        opc_server.stop()

    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec())