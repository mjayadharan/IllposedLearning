import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, odeint

class Lotka_Volterra:
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


class CRN:
    def __init__(self, k_rates, init_cond, solvedT, noise_level=None):
        self.k_rates = k_rates
        self.init_cond = init_cond
        self.solvedT = solvedT
        self.noise_level = noise_level

        self.data_sim = self.solveToyEnz()

    def toyEnzRHS(self, y, t):
        # Unpack states, params
        S, E, ES, P = y
        k, kr, kcat = self.k_rates.get("k"), self.k_rates.get("kr"), self.k_rates.get("kcat")

        dydt = [kr * ES - k * E * S,
                (kr + kcat) * ES - k * S * E,
                k * E * S - (kr + kcat) * ES,
                kcat * ES]
        return dydt

    def solveToyEnz(self,):
        y0 = self.init_cond
        sol = odeint(lambda y, t: self.toyEnzRHS(y, t), y0, self.solvedT)
        S_base = sol[:,0]
        E_base = sol[:,1]
        ES_base = sol[:,2]
        P_base = sol[:,3]

        if self.noise_level is None:
            dS_dt = []
            dE_dt = []
            dES_dt = []
            dP_dt = []
            for t, z in zip(self.solvedT, sol):
                dSdt, dEdt, dESdt, dPdt = self.toyEnzRHS(z, t)
                dS_dt.append(dSdt)
                dE_dt.append(dEdt)
                dES_dt.append(dESdt)
                dP_dt.append(dPdt)

            df = pd.DataFrame({
                "time": self.solvedT,
                "S": S_base,
                'E': E_base,
                "ES": ES_base,
                "P": P_base,
                "dSdt": dS_dt,
                "dEdt": dE_dt,
                "dESdt": dES_dt,
                "dPdt": dP_dt
            })

        else:
            S_sigma = np.abs(S_base) * self.noise_level
            E_sigma = np.abs(E_base) * self.noise_level
            ES_sigma = np.abs(ES_base) * self.noise_level
            P_sigma = np.abs(P_base) * self.noise_level

            S_noisy = S_base + np.random.normal(0,S_sigma)
            E_noisy = E_base + np.random.normal(0,E_sigma)
            ES_noisy = S_base + np.random.normal(0,ES_sigma)
            P_noisy = S_base + np.random.normal(0,P_sigma)

            # Computer derivatives
            dS_dt = []
            dE_dt = []
            dES_dt = []
            dP_dt = []
            for t, Sn, En, ESn, Pn in zip(self.solvedT, S_noisy, E_noisy, ES_noisy, P_noisy):
                dSdt, dEdt, dESdt, dPdt = self.toyEnzRHS([Sn, En, ESn, Pn], t)
                dS_dt.append(dSdt)
                dE_dt.append(dEdt)
                dES_dt.append(dESdt)
                dP_dt.append(dPdt)

            df = pd.DataFrame({
                "time": self.solvedT,
                "S": S_noisy,
                'E': E_noisy,
                "ES": ES_noisy,
                "P": P_noisy,
                "dSdt": dS_dt,
                "dEdt": dE_dt,
                "dESdt": dES_dt,
                "dPdt": dP_dt
            })

        return df

                                                          
