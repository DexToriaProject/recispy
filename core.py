# core.py — гарантированно содержит класс RBMKCore

import math

class RBMKCore:
    def __init__(self):
        self.power = 100.0
        self.neutron_flux = 1.0
        self.fuel_temp = 600.0
        self.coolant_temp_in = 270.0
        self.coolant_temp_out = 280.0
        self.pressure = 70.0
        self.insertion_depth = 1.0
        self.coolant_flow = 1.0
        self.time_step = 1.0
        self.time_elapsed = 0.0
        self.is_running = True
        self.status = "NORMAL"

        self.void_coefficient = +0.005
        self.temp_coefficient = -0.001
        self.control_rod_worth = -0.025
        self.reactivity = 0.0

        # Заглушка для ксенона
        try:
            from physics.xenon_poisoning import XenonPoisoning
            self.xenon = XenonPoisoning()
        except Exception as e:
            print(f"⚠️  Предупреждение: не удалось загрузить XenonPoisoning: {e}")
            self.xenon = None

    def calculate_reactivity(self):
        void_effect = self.void_coefficient * (1.0 - self.coolant_flow)
        temp_effect = self.temp_coefficient * (self.fuel_temp - 600) / 100
        rod_effect = self.control_rod_worth * self.insertion_depth
        xenon_effect = 0.0

        if self.xenon:
            xenon_effect = self.xenon.step(self.neutron_flux, self.time_step)

        self.reactivity = void_effect + temp_effect + rod_effect + xenon_effect

    def update_physics(self):
        self.calculate_reactivity()

        if self.reactivity > -0.1:
            self.neutron_flux *= math.exp(self.reactivity * self.time_step)
        self.neutron_flux = max(0.01, min(self.neutron_flux, 10.0))

        self.power = self.neutron_flux * 100
        self.fuel_temp += (self.power - self.coolant_flow * 100) * 0.01 * self.time_step
        self.coolant_temp_out = self.coolant_temp_in + self.power * 0.1 / max(self.coolant_flow, 0.1)
        self.pressure = 70 + (self.coolant_temp_out - 280) * 0.5

        if self.fuel_temp > 1200:
            self.status = "ПЛАВЛЕНИЕ АКТИВНОЙ ЗОНЫ"
            self.is_running = False
        elif self.power > 150 and self.coolant_flow < 0.5:
            self.status = "РИСК ПАРОВОГО ВЗРЫВА"
        elif self.power < 5:
            self.status = "РЕАКТОР ЗАГЛУШЕН"

        self.time_elapsed += self.time_step

    def insert_rods(self, delta_depth):
        self.insertion_depth = max(0.0, min(1.0, self.insertion_depth + delta_depth))

    def set_coolant_flow(self, flow):
        self.coolant_flow = max(0.0, min(2.0, flow))

    def get_status(self):
        return {
            "time": self.time_elapsed,
            "power": self.power,
            "neutron_flux": self.neutron_flux,
            "fuel_temp": self.fuel_temp,
            "coolant_in": self.coolant_temp_in,
            "coolant_out": self.coolant_temp_out,
            "pressure": self.pressure,
            "insertion_depth": self.insertion_depth,
            "coolant_flow": self.coolant_flow,
            "reactivity": self.reactivity,
            "status": self.status,
            "xenon": getattr(self.xenon, 'Xe_conc', 0.0) if self.xenon else 0.0
        }