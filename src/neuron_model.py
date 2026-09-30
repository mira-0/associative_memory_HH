import math
import numpy as np
import torch
from scipy.signal import find_peaks
from .neuron_parameters import NeuronParameters

class NeuronModel:
    relative_refractory_period = 5.0  # ms
    delta_t = 6.0                     # ms
    spike_threshold = -45.0           # mV
    sigma_noise = 0.01
    eps = 1e-8
    tau_r = 0.5
    tau_d = 8.0

    def __init__(
            self,
            neuron_types: list[int],
            input_neurons,
            model_type: str = "single",
            device: torch.device | str = "cpu",
            dtype: torch.dtype = torch.float32
        ):

        self.device = torch.device(device)
        self.dtype = dtype
        self.model_type = model_type
        self.delay = 2.7  # ms
        self.neuron_types = list(neuron_types)
        self.n_neurons = len(self.neuron_types)
        self.input_neurons = np.asarray(input_neurons)

        if len(self.input_neurons) != self.n_neurons:
            raise ValueError("input_neurons and neuron_types must have the same length.")

        self.tensor_input = torch.as_tensor(self.input_neurons, dtype=self.dtype, device=self.device)
        self.Ach = torch.zeros((self.n_neurons, self.n_neurons), dtype=self.dtype, device=self.device)
        self.Ael = torch.zeros((self.n_neurons, self.n_neurons), dtype=self.dtype, device=self.device)
        self.L = torch.zeros((self.n_neurons, self.n_neurons), dtype=self.dtype, device=self.device)

        self.neuron_parameters = NeuronParameters.from_types(
            self.neuron_types,
            device=self.device,
            dtype=self.dtype,
        )

        self.y0 = self._create_initial_state()
        self.spikes = [[] for _ in range(self.n_neurons)]
        self.current = None
        self.r_history = None
        self.dt = None

    def set_delay(self, delay: float):
        if delay < 0:
            raise ValueError("The synaptic delay cannot be negative.")

        self.delay = delay

    def _injected_current(self, times: torch.Tensor, current_span, current_values):
        if len(current_span) != 2:
            raise ValueError("current_span must contain start and end times.")

        if len(current_values) == 0:
            raise ValueError("current_values must contain at least one value.")

        mask = ((times >= current_span[0]) & (times <= current_span[1]))
        self.current = torch.zeros((self.n_neurons, len(times)), dtype=self.dtype, device=self.device)
        injected = (self.tensor_input * current_values[0]).unsqueeze(1)
        self.current[:, mask] = injected

        return self.current

    def _find_spikes(self, membrane_potential: np.ndarray, time_grid: np.ndarray):
        self.spikes = []

        if len(time_grid) < 2:
            raise ValueError("At least two time points are needed to detect spikes.")

        dt = time_grid[1] - time_grid[0]
        min_distance = max(1, int(self.relative_refractory_period / dt))

        for neuron_id in range(self.n_neurons):
            signal = membrane_potential[neuron_id]

            indices, _ = find_peaks(signal, height=self.spike_threshold, distance=min_distance)

            first_point_is_peak = (
                len(signal) > 1
                and signal[0] >= self.spike_threshold
                and signal[0] > signal[1]
            )

            if first_point_is_peak:
                if len(indices) == 0 or indices[0] != 0:
                    indices = np.insert(indices, 0, 0)

            self.spikes.append(time_grid[indices].tolist())

    def _alpha_n(self, voltage):
        difference = (voltage - self.neuron_parameters.Vt - 15.0)
        return (-0.032 * difference / (torch.exp(-difference / 5.0) - 1.0 + self.eps))

    def _alpha_m(self, voltage):
        difference = (voltage - self.neuron_parameters.Vt - 13.0)
        return (-0.32 * difference / (torch.exp(-difference / 4.0) - 1.0 + self.eps))

    def _alpha_h(self, voltage):
        return 0.128 * torch.exp(-(voltage - self.neuron_parameters.Vt - 17.0) / 18.0)

    def _alpha_q(self, voltage):
        difference = -27.0 - voltage
        return (0.055 * difference / (torch.exp(difference / 3.8) - 1.0 + self.eps))

    def _alpha_s(self, voltage):
        return 0.000457 * torch.exp((-13.0 - voltage) / 50.0)

    def _beta_n(self, voltage):
        return 0.5 * torch.exp(-(voltage - self.neuron_parameters.Vt - 10.0) / 40.0)

    def _beta_m(self, voltage):
        difference = (voltage - self.neuron_parameters.Vt - 40.0)
        return (0.28 * difference / (torch.exp(difference / 5.0) - 1.0 + self.eps))

    def _beta_h(self, voltage):
        difference = (voltage - self.neuron_parameters.Vt - 40.0)
        return (4.0 / (torch.exp(-difference / 5.0) + 1.0 + self.eps))

    def _beta_q(self, voltage):
        return 0.94 * torch.exp((-75.0 - voltage) / 17.0)

    def _beta_s(self, voltage):
        return (0.0065 / (torch.exp((-15.0 - voltage) / 28.0) + 1.0 + self.eps))

    def _p_inf(self, voltage):
        return (1.0 / (torch.exp(-(voltage + 35.0) / 10.0) + 1.0 + self.eps))
    
    def _p_tau(self, voltage):
        denominator = (3.3 * torch.exp((voltage + 35.0) / 20.0) + torch.exp(-(voltage + 35.0) / 20.0) + self.eps)
        return (self.neuron_parameters.TauMax / denominator)

    def _create_initial_state(self, resting_voltage: float = -60.0):
        
        voltage = torch.full(
            (self.n_neurons,),
            resting_voltage,
            dtype=self.dtype,
            device=self.device,
        )
        n = (self._alpha_n(voltage) / (self._alpha_n(voltage) + self._beta_n(voltage)))
        m = (self._alpha_m(voltage) / (self._alpha_m(voltage) + self._beta_m(voltage)))
        h = (self._alpha_h(voltage) / (self._alpha_h(voltage) + self._beta_h(voltage)))
        p = self._p_inf(voltage)
        q = (self._alpha_q(voltage) / (self._alpha_q(voltage) + self._beta_q(voltage)))
        s = (self._alpha_s(voltage) / (self._alpha_s(voltage) + self._beta_s(voltage)))
        r = torch.zeros(self.n_neurons, dtype=self.dtype, device=self.device)

        return torch.cat([voltage, n, m, h, p, q, s, r])

    def solution(self, start_time: float, end_time: float, dt: float, current_span: list, current_values: list):
        if dt <= 0:
            raise ValueError("dt must be greater than zero.")

        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time.")

        self.dt = dt
        times = torch.arange(start_time, end_time, dt, dtype=self.dtype, device=self.device)
        n_steps = len(times)

        if n_steps < 2:
            raise ValueError("The simulation must contain at least two time steps.")

        delay_steps = max(1, int(self.delay / self.dt))
        self.r_history = torch.zeros((delay_steps, self.n_neurons), dtype=self.dtype, device=self.device)
        self.L = -(torch.diag(torch.sum(self.Ael, dim=1)) - self.Ael)
        external_current = self._injected_current(times, current_span, current_values)
        state = self.y0.clone()
        voltage_log = torch.empty((n_steps, self.n_neurons), dtype=self.dtype, device=self.device)

        with torch.no_grad():
            for step in range(n_steps):
                voltage_log[step] = state[:self.n_neurons]
                delayed_r = self.r_history[0].clone()

                state = self._rk4_step(
                    state,
                    external_current[:, step],
                    self.dt,
                    delayed_r,
                )

                self.r_history = torch.roll(self.r_history, shifts=-1, dims=0)
                self.r_history[-1] = state[7 * self.n_neurons: 8 * self.n_neurons]

        solution_y = voltage_log.T.cpu().numpy()
        solution_t = times.cpu().numpy()

        self._find_spikes(solution_y, solution_t)

        return solution_t, solution_y

    def _compute_currents(self, voltage, n, m, h, p, q, s):
        parameters = self.neuron_parameters

        potassium = (parameters.gk * n**4 * (voltage - parameters.Ek))
        m_current = (parameters.gm * p * (voltage - parameters.Ek))
        calcium = (parameters.gca * q**2 * s * (voltage - parameters.Eca))
        sodium = (parameters.gna * m**3 * h * (voltage - parameters.Ena))
        leak = (parameters.gleak * (voltage - parameters.Eleak))

        return (potassium, m_current, calcium, sodium, leak)

    def _derivatives(self, state, external_current, delayed_r):
        voltage, n, m, h, p, q, s, r = torch.split(state, self.n_neurons, dim=0)

        dn_dt = (self._alpha_n(voltage) * (1.0 - n) - self._beta_n(voltage) * n)
        dm_dt = (self._alpha_m(voltage) * (1.0 - m) - self._beta_m(voltage) * m)
        dh_dt = (self._alpha_h(voltage) * (1.0 - h) - self._beta_h(voltage) * h)
        dp_dt = (self._p_inf(voltage) - p) / self._p_tau(voltage)
        dq_dt = (self._alpha_q(voltage) * (1.0 - q) - self._beta_q(voltage) * q)
        ds_dt = (self._alpha_s(voltage) * (1.0 - s) - self._beta_s(voltage) * s)

        spike_detection = torch.sigmoid(voltage + 20.0)

        inverse_tau_d = 1.0 / self.tau_d
        inverse_tau_r = 1.0 / self.tau_r

        dr_dt = ((inverse_tau_r - inverse_tau_d) * (1.0 - r) * spike_detection - r * inverse_tau_d)
        (potassium, m_current, calcium, sodium, leak) = self._compute_currents(voltage, n, m, h, p, q, s)

        electrical_current = torch.relu(self.L @ voltage)

        driving_force = (self.neuron_parameters.Vsyn - voltage.unsqueeze(1))
        chemical_current = torch.sum((self.Ach * driving_force) * delayed_r, dim=1)

        inverse_sqrt_dt = 1.0 / math.sqrt(self.dt)
        noise = (self.sigma_noise * inverse_sqrt_dt * torch.randn_like(voltage))

        dv_dt = (external_current + noise - potassium - m_current - calcium - sodium - leak + chemical_current + electrical_current) / self.neuron_parameters.Cm
        dv_dt = torch.nan_to_num(dv_dt, nan=-70.0, posinf=50.0, neginf=-80.0)

        return torch.cat((dv_dt, dn_dt, dm_dt, dh_dt, dp_dt, dq_dt, ds_dt, dr_dt))

    def _rk4_step(self, state, external_current, dt, delayed_r):
        k1 = self._derivatives( state, external_current, delayed_r)
        k2 = self._derivatives(state + 0.5 * dt * k1, external_current, delayed_r)
        k3 = self._derivatives(state + 0.5 * dt * k2, external_current, delayed_r)
        k4 = self._derivatives(state + dt * k3, external_current, delayed_r)

        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)