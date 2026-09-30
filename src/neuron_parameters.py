import torch

class NeuronParameters:
    def __init__(self, n: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        self.n = n
        self.i = 0
        self.device = torch.device(device)
        self.dtype = dtype
        
        self.name = [""] * n
        self.color = [""] * n
        self.Cm = torch.zeros(n, dtype=dtype, device=self.device)
        self.Vt = torch.zeros(n, dtype=dtype, device=self.device)
        self.gna = torch.zeros(n, dtype=dtype, device=self.device)
        self.Ena = torch.zeros(n, dtype=dtype, device=self.device)
        self.gk = torch.zeros(n, dtype=dtype, device=self.device)
        self.gm = torch.zeros(n, dtype=dtype, device=self.device)
        self.Ek = torch.zeros(n, dtype=dtype, device=self.device)
        self.gca = torch.zeros(n, dtype=dtype, device=self.device)
        self.Eca = torch.zeros(n, dtype=dtype, device=self.device)
        self.gleak = torch.zeros(n, dtype=dtype, device=self.device)
        self.Eleak = torch.zeros(n, dtype=dtype, device=self.device)
        self.TauMax = torch.zeros(n, dtype=dtype, device=self.device)
        self.Vsyn = torch.zeros(n, dtype=dtype, device=self.device)

    @classmethod
    def from_types(cls, neuron_types: list[int], device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32):
        parameters = cls(n=len(neuron_types), device=device, dtype=dtype)

        for neuron_type in neuron_types:
            match neuron_type:
                case 1:
                    parameters.add_neuron(
                        "FS", "blue",
                        0.5, -90.0, 0.0, 50.0, -70.0, -56.2, 10.0, 0.0, 0.0, 56.0, 1.5e-2, 1.0, -80.0
                    )
                case 2:
                    parameters.add_neuron(
                        "RSA", "green",
                        1.0, -90.0, 0.0, 56.0, -70.3, -56.2, 6.0, 0.075, 0.0, 56.0, 2.05e-2, 608.0, 20.0
                    )
                case 3:
                    parameters.add_neuron(
                        "IB", "magenta",
                        1.0, -90.0, 120.0, 50.0, -70.0, -56.2, 5.0, 0.03, 0.2, 50.0, 0.01, 608.0, 20.0
                    )
                case 4:
                    parameters.add_neuron(
                        "IB1", "magenta",
                        1.0, -90.0, 120.0, 50.0, -70.0, -56.2, 5.0, 0.03, 0.15, 50.0, 0.01, 608.0, 20.0
                    )
                case _:
                    raise ValueError(f"Unknown neuron type: {neuron_type}")

        return parameters

    def add_neuron(
        self,
        name,
        color,
        Cm,
        Ek,
        Eca,
        Ena,
        Eleak,
        Vt,
        gk,
        gm,
        gca,
        gna,
        gleak,
        TauMax,
        Vsyn,
    ):
        if self.i >= self.n:
            raise ValueError(f"Cannot add more than {self.n} neurons.")

        self.name[self.i] = name
        self.color[self.i] = color
        self.Cm[self.i] = Cm
        self.Vt[self.i] = Vt
        self.gna[self.i] = gna
        self.Ena[self.i] = Ena
        self.gk[self.i] = gk
        self.gm[self.i] = gm
        self.Ek[self.i] = Ek
        self.gca[self.i] = gca
        self.Eca[self.i] = Eca
        self.gleak[self.i] = gleak
        self.Eleak[self.i] = Eleak
        self.TauMax[self.i] = TauMax
        self.Vsyn[self.i] = Vsyn

        self.i += 1