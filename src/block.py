import numpy as np
import torch
from .neuron_model import NeuronModel

class Block(NeuronModel):
    def __init__(
        self,
        n_blocks: int,
        input_pattern: list,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if len(input_pattern) != n_blocks:
            raise ValueError("The input pattern must contain one value per block.")

        self.n_blocks = n_blocks
        self.input_pattern = np.asarray(input_pattern)
        input_neurons = np.zeros(n_blocks * 4)
        input_neurons[::4] = self.input_pattern

        super().__init__(
            neuron_types=[4, 2, 2, 1] * n_blocks,
            input_neurons=input_neurons,
            model_type="block",
            device=device,
            dtype=dtype,
        )

        self.stored_patterns = np.empty((0, n_blocks))
        self.rsa_rsa = 0.0
        self.fs_rsa = 0.0
        self.fs_fs = 0.0

        self.update_electrical_connections()

    def update_electrical_connections(self, ib_rsa1: float = 0.2, rsa1_rsa2: float = 0.2, rsa1_fs: float = 0.2):
        for n in range(self.n_blocks):
            self.Ael[1 + 4 * n, 4 * n] = ib_rsa1
            self.Ael[2 + 4 * n, 1 + 4 * n] = rsa1_rsa2
            self.Ael[3 + 4 * n, 1 + 4 * n] = rsa1_fs

    def update_chemical_connections(self, rsa_rsa: float = 0.0, fs_rsa: float = 0.0, fs_fs: float = 0.0):
        if self.stored_patterns.size == 0:
            raise ValueError("Stored patterns must be set before updating chemical connections.")

        self.rsa_rsa = rsa_rsa
        self.fs_rsa = fs_rsa
        self.fs_fs = fs_fs

        self.update_weights(self.stored_patterns)

    def set_stored_patterns(self, patterns: list):
        patterns = self._validate_patterns(patterns)
        self.stored_patterns = patterns

    def update_weights(self, patterns: list):
        patterns = self._validate_patterns(patterns)
        self.stored_patterns = patterns

        for j in range(self.n_blocks):
            for k in range(self.n_blocks):
                if k == j:
                    continue

                excitatory = self._willshaw_rule(patterns, j, k)

                self.Ach[1 + 4 * k, 2 + 4 * j] = self.rsa_rsa * excitatory
                self.Ach[1 + 4 * k, 3 + 4 * j] = self.fs_rsa * (1.0 - excitatory)
                self.Ach[3 + 4 * k, 3 + 4 * j] = self.fs_fs

    def calculate_overlap(self, times: list | np.ndarray | torch.Tensor):
        times = torch.as_tensor(times, device=self.device, dtype=self.dtype)

        stored_patterns = torch.as_tensor(2 * self.stored_patterns - 1, device=self.device, dtype=self.dtype)
        input_pattern = torch.as_tensor(2 * self.input_pattern - 1, device=self.device, dtype=self.dtype)

        n_patterns, n_blocks = stored_patterns.shape
        initial_overlap = (stored_patterns @ input_pattern) / n_blocks

        neurons = torch.arange(n_blocks, device=self.device)
        neuron_activity = self._nj_t_batch(neurons, times)

        time_overlap = (
            stored_patterns[:, :, None]
            * (2 * neuron_activity[None, :, :] - 1)
        ).sum(dim=1) / n_blocks

        time_overlap = torch.round(time_overlap * 1000) / 1000

        return initial_overlap.cpu(), time_overlap.cpu()

    def _validate_patterns(self, patterns):
        patterns = np.asarray(patterns)

        if patterns.ndim != 2:
            raise ValueError("Patterns must be a two-dimensional array.")
        if patterns.shape[0] == 0:
            raise ValueError("At least one stored pattern is required.")
        if patterns.shape[1] != self.n_blocks:
            raise ValueError("Each pattern must contain one value per block.")
        if not np.all(np.isin(patterns, [0, 1])):
            raise ValueError("Patterns must contain only 0 and 1.")

        return patterns

    @staticmethod
    def _willshaw_rule(
        patterns: np.ndarray, j: int, k: int,
    ) -> float:
        return float(np.any(patterns[:, j] * patterns[:, k]))

    def _nj_t_batch(self, neurons: torch.Tensor, times: torch.Tensor):
        n_neurons = len(neurons)
        n_times = len(times)
        activity = torch.zeros((n_neurons, n_times), device=self.device, dtype=self.dtype)

        for index, neuron in enumerate(neurons.tolist()):
            spike_times = torch.as_tensor(self.spikes[2 + 4 * neuron], device=self.device, dtype=self.dtype)

            if spike_times.numel() == 0:
                continue

            differences = torch.abs(spike_times[:, None] - times[None, :])
            activity[index] = (differences <= self.delta_t).any(dim=0).to(self.dtype)

        return activity