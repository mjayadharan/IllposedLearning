import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, odeint

class Lotka_Volterra:
    def __init__(self,params,t_span,t_eval,z0,noise_level=None):
        # parameters under the case: 1 prey, 1 predator
        self.alpha = params.get("alpha", None)
        self.beta = params.get("beta", None)
        self.gamma = params.get("gamma", None)
        self.delta = params.get("delta", None)
        # parameters under the case: 1 prey, 2 predators
        self.r = params.get("r", None)
        self.K = params.get("K", None)
        self.a1 = params.get("a1", None)
        self.a2 = params.get("a2", None)
        self.m1 = params.get("m1", None)
        self.m2 = params.get("m2", None)
        self.epsilon1 = params.get("epsilon1", None)
        self.epsilon2 = params.get("epsilon2", None)

        self.t_span = t_span
        self.t_eval = t_eval
        self.z0 = z0
        self.noise_level = noise_level

        self.data_sim = self.simulate_lv_condition_1_2()

    def lv_rhs_1_1(self, t, z):
        """
        Lotka-Volterra equations: 1 prey, 1 predator
        """
        x, y = z
        dxdt = self.alpha * x - self.beta * x * y
        dydt = self.delta * x * y - self.gamma * y
        return [dxdt, dydt]
    
    def lv_rhs_1_2(self,t,z):
        """
        Lotka-Volterra equations: 1 prey, 2 predators
        """
        x, y1, y2 = z
        dxdt = self.r * x * (1.0 - x / self.K) - self.a1 * x * y1 - self.a2 * x * y2
        #dy1dt = y1 * (self.epsilon1 * self.a1 * x - self.m1)
        #dy2dt = y2 * (self.epsilon2 * self.a2 * x - self.m2)
        dy1dt = self.epsilon1 * self.a1 * x * y1 - self.m1 * y1
        dy2dt = self.epsilon2 * self.a2 * x * y2 - self.m2 * y2
        return [dxdt, dy1dt, dy2dt]
    
    def simulate_lv_condition_1_1(self):
        sol = solve_ivp(self.lv_rhs_1_1, self.t_span, self.z0, t_eval=self.t_eval)
        if self.noise_level is None:
            dxdt_vals = []
            dydt_vals = []
            for t, z in zip(sol.t, sol.y.T):
                dxdt, dydt = self.lv_rhs_1_1(t, z)
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

    def simulate_lv_condition_1_2(self):
        #sol = solve_ivp(self.lv_rhs_1_1, self.t_span, self.z0, t_eval=self.t_eval)
        sol = solve_ivp(self.lv_rhs_1_2, self.t_span, self.z0, t_eval=self.t_eval)
        if self.noise_level is None:
            dxdt_vals = []
            dy1dt_vals = []
            dy2dt_vals = []
            for t, z in zip(sol.t, sol.y.T):
                dxdt, dy1dt, dy2dt = self.lv_rhs_1_2(t, z)
                dxdt_vals.append(dxdt)
                dy1dt_vals.append(dy1dt)
                dy2dt_vals.append(dy2dt)
            df = pd.DataFrame({
                "time": sol.t,
                "x": sol.y[0],
                'y1': sol.y[1],
                'y2': sol.y[2]
                #"dxdt": dxdt_vals,
                #"dy1dt": dy1dt_vals,
                #"dy2dt": dy2dt_vals
            })
        else:
            x_base = sol.y[0]
            y1_base = sol.y[1]
            y2_base = sol.y[2]

            x_sigma = np.abs(x_base) * self.noise_level
            y1_sigma = np.abs(y1_base) * self.noise_level
            y2_sigma = np.abs(y2_base) * self.noise_level
            x_noisy = x_base + np.random.normal(0,x_sigma)
            y1_noisy = y1_base + np.random.normal(0, y1_sigma)
            y2_noisy = y2_base + np.random.normal(0, y2_sigma)

            # Computer derivatives
            dxdt_vals = []
            dy1dt_vals = []
            dy2dt_vals = []
            for t, xn, y1n, y2n in zip(sol.t, x_noisy, y1_noisy, y2_noisy):
                dxdt, dy1dt, dy2dt = self.lv_rhs_1_2(t, [xn, y1n, y2n])
                dxdt_vals.append(dxdt)
                dy1dt_vals.append(dy1dt)
                dy2dt_vals.append(dy2dt)

            df = pd.DataFrame({
                "time": sol.t,
                "x": x_noisy,
                "y1": y1_noisy,
                "y2": y2_noisy
                #"dxdt": dxdt_vals,
                #"dy1dt": dy1dt_vals,
                #"dy2dt": dy2dt_vals
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
                (kr + kcat) * ES - k * E * S,
                #(kr + kcat) * G * S,
                k * S * E - (kr + kcat) * ES,
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
                "P": P_base
                #"dSdt": dS_dt,
                #"dEdt": dE_dt,
                #"dESdt": dES_dt,
                #"dPdt": dP_dt
            })

        else:
            S_sigma = np.abs(S_base) * self.noise_level
            E_sigma = np.abs(E_base) * self.noise_level
            ES_sigma = np.abs(ES_base) * self.noise_level
            P_sigma = np.abs(P_base) * self.noise_level

            rng = np.random.default_rng(0)
            S_noisy = S_base + rng.normal(0,S_sigma)
            E_noisy = E_base + rng.normal(0,E_sigma)
            ES_noisy = S_base + rng.normal(0,ES_sigma)
            P_noisy = S_base + rng.normal(0,P_sigma)

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
                "P": P_noisy
                #"dSdt": dS_dt,
                #"dEdt": dE_dt,
                #"dESdt": dES_dt,
                #"dPdt": dP_dt
            })

        return df

                                                          
