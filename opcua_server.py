# comms/opcua_server.py
from opcua import Server
import threading
import time

class OPCUAServer:
    def __init__(self, reactor_core, endpoint="opc.tcp://0.0.0.0:4840/rbmk1000/"):
        self.core = reactor_core
        self.server = Server()
        self.server.set_endpoint(endpoint)
        self.server.set_server_name("RBMK-1000 Simulator OPC UA Server")

        uri = "http://rbmk-simulator.org"
        idx = self.server.register_namespace(uri)

        objects = self.server.get_objects_node()
        self.reactor_obj = objects.add_object(idx, "ReactorCore")

        # Переменные для чтения
        self.power_var = self.reactor_obj.add_variable(idx, "Power", 100.0)
        self.temp_var = self.reactor_obj.add_variable(idx, "FuelTemperature", 600.0)
        self.pressure_var = self.reactor_obj.add_variable(idx, "Pressure", 70.0)
        self.status_var = self.reactor_obj.add_variable(idx, "Status", "NORMAL")

        # Переменные для записи (управление)
        self.insertion_var = self.reactor_obj.add_variable(idx, "RodInsertion", 1.0)
        self.flow_var = self.reactor_obj.add_variable(idx, "CoolantFlow", 1.0)
        self.insertion_var.set_writable()
        self.flow_var.set_writable()

        self.running = False

    def start(self):
        self.server.start()
        self.running = True
        print("✅ OPC UA Server запущен на opc.tcp://localhost:4840/rbmk1000/")

        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def _update_loop(self):
        while self.running:
            status = self.core.get_status()
            self.power_var.set_value(status['power'])
            self.temp_var.set_value(status['fuel_temp'])
            self.pressure_var.set_value(status['pressure'])
            self.status_var.set_value(status['status'])

            # Чтение команд от SCADA
            cmd_insertion = self.insertion_var.get_value()
            cmd_flow = self.flow_var.get_value()

            if abs(cmd_insertion - self.core.insertion_depth) > 0.01:
                self.core.insertion_depth = cmd_insertion
            if abs(cmd_flow - self.core.coolant_flow) > 0.01:
                self.core.set_coolant_flow(cmd_flow)

            time.sleep(0.5)

    def stop(self):
        self.running = False
        self.server.stop()
        print("🛑 OPC UA Server остановлен")