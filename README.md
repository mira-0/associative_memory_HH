# Remembering Like a Brain: A Bio-Inspired Associative Memory Model Integrating Hodgkin-Huxley Dynamics and Modular Network Architecture

Python implementation of an associative-memory network composed of
Hodgkin–Huxley-type neurons.

The model uses PyTorch for numerical simulations and supports execution on
CPU and compatible GPUs. The repository also contains the notebook used to
generate the retrieval figures presented in the associated article.

## Installation

Python 3.10 or later is required.

PyTorch installations depend on the operating system and compute platform.
For CUDA or AMD ROCm, install the appropriate PyTorch build for your system
before installing the remaining requirements.

## Reproducing the article figures

Open the retrieval notebook:

```bash
jupyter lab notebooks/RetrievalFigures.ipynb
```
The notebook uses explicit random seeds and writes the generated figures to the `figures/` directory.
Numerical differences may occur across PyTorch versions, hardware and accelerator backends.

## License

See the `LICENSE` file for usage and redistribution terms.