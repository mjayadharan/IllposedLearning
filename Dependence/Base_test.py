import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

class Lotka_Volterra():
    def __init__(self,params,t_span,t_eval,z0,noise_level=None):
        self.alpha = params.get("alpha")
        self.beta = params.get("beta")
        self.gamma = params.get("gamma")
        self.delta = params.get("delta")
        self.t_span = t_span
        self.t_eval = t_eval
        self.z0 = z0
        self.noise_level = noise_level

        self.data_sim = self.simulate_lv_condition()

    def lv_rhs(self, t, z):
        x, y = z
        dxdt = self.alpha * x - self.beta * x * y
        dydt = self.delta * x * y - self.gamma * y
        return [dxdt, dydt]
    
    def simulate_lv_condition(self):
        sol = solve_ivp(self.lv_rhs, self.t_span, self.z0, t_eval=self.t_eval)
        if self.noise_level is None:
            dxdt_vals = []
            dydt_vals = []
            for t, z in zip(sol.t, sol.y.T):
                dxdt, dydt = self.lv_rhs(t, z)
                dxdt_vals.append(dxdt)
                dydt_vals.append(dydt)
            df = pd.DataFrame({
                "time": sol.t,
                "x": sol.y[0],
                'y': sol.y[1],
                "dxdt": dxdt_vals,
                "dydt": dydt_vals
            })
        else:
            x_base = sol.y[0]
            y_base = sol.y[1]

            x_sigma = np.abs(x_base) * self.noise_level
            y_sigma = np.abs(y_base) * self.noise_level
            x_noisy = x_base + np.random.normal(0,x_sigma)
            y_noisy = y_base + np.random.normal(0,y_sigma)

            # Computer derivatives
            dxdt_vals = []
            dydt_vals = []
            for t, xn, yn in zip(sol.t, x_noisy, y_noisy):
                dxdt, dydt = self.lv_rhs(t, [xn, yn])
                dxdt_vals.append(dxdt)
                dydt_vals.append(dydt)

            df = pd.DataFrame({
                "time": sol.t,
                "x": x_noisy,
                "y": y_noisy,
                "dxdt": dxdt_vals,
                "dydt": dydt_vals
            })

        return df
