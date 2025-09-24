# physics/xenon_poisoning.py

import numpy as np

class XenonPoisoning:
    def __init__(self):
        self.sigma_Xe = 2.6e6
        self.lambda_I = 2.87e-5
        self.lambda_Xe = 2.09e-5
        self.gamma_I = 0.064
        self.gamma_Xe = 0.003
        self.N0 = 1e20
        self.I_conc = 0.0
        self.Xe_conc = 0.0
        self.xenon_reactivity = 0.0

    def step(self, neutron_flux, dt):
        if neutron_flux < 0:
            neutron_flux = 0

        dI_dt = self.gamma_I * neutron_flux * self.N0 - self.lambda_I * self.I_conc
        dXe_dt = (self.gamma_Xe * neutron_flux * self.N0 +
                  self.lambda_I * self.I_conc -
                  self.lambda_Xe * self.Xe_conc -
                  neutron_flux * self.N0 * self.sigma_Xe * 1e-24 * self.Xe_conc)

        self.I_conc += dI_dt * dt
        self.Xe_conc += dXe_dt * dt
        self.xenon_reactivity = - (self.sigma_Xe * 1e-24 * self.Xe_conc) / 0.0065

        return self.xenon_reactivity
    