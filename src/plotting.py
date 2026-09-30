import matplotlib.pyplot as plt
import numpy as np

def complete_plot_retrieval(network, sol_t, m_t, retrieved_pattern=0, differents=[], name="graph.png", show=True):

    if differents is None:
        differents = []
    if retrieved_pattern < 0 or retrieved_pattern > len(m_t):
        retrieved_pattern = 0

    sol_t = np.asarray(sol_t)
    m_t = np.asarray(m_t)

    if sol_t.ndim != 1 or len(sol_t) == 0:
        raise ValueError("sol_t must be a non-empty one-dimensional array.")
    if m_t.ndim != 2:
        raise ValueError("m_t must be a two-dimensional array.")
    if m_t.shape[1] != len(sol_t):
        raise ValueError("m_t and sol_t have incompatible lengths.")
    if network.current is None:
        raise RuntimeError("The network must be simulated before plotting.")

    initial_current_color = "firebrick"
    wrong_neuron_color = "black"
    spikes_color = "C0"

    plt.close('all')

    plt_overlap = plt.subplot(4, 1, 1)
    plt_current = plt.subplot(4, 1, 4)
    plt_spikes = plt.subplot(4, 1, (2,3))

    N = network.n_blocks
    for ni in range(N):
        y = np.ones_like(network.spikes[(2+(4*ni))]) * ni
        plt_spikes.scatter(network.spikes[(2+(4*ni))], y, marker=".", color=spikes_color)

        if network.input_pattern[ni] == 1:
            plt_spikes.scatter(network.spikes[(2+(4*ni))][0], ni, marker=".", color=initial_current_color)
    for wrong in differents:
        plt_spikes.scatter(network.spikes[(2+(4*wrong))][0], wrong, marker=".", color=wrong_neuron_color)

    inputInx = np.where(network.input_neurons == 1)[0][0]

    plt_spikes.set_title("Spikes", fontsize=14)
    plt_spikes.set_ylim((0,(N+1)))
    plt_spikes.set_xlim((0, sol_t[-1]))
    plt_spikes.set_xticks(np.arange(0,(sol_t[-1]+1),25))
    plt_spikes.set_xlabel("t (msec)")
    plt_spikes.set_ylabel("Block index")

    plt_current.plot(sol_t, network.current.cpu().numpy()[inputInx], color=initial_current_color)
    plt_current.set_xlim((0, sol_t[-1]))
    plt_current.set_xticks(np.arange(0,(sol_t[-1]+1),25))
    plt_current.set_ylabel("$I_{in}$ mV")
    plt_current.set_xlabel("t (msec)")

    plt_overlap.plot(sol_t, m_t[retrieved_pattern])

    plt_overlap.set_xlim((0, sol_t[-1]))
    plt_overlap.set_xticks(np.arange(0,(sol_t[-1]+1),25))
    plt_overlap.set_yticks(np.arange(0.6,1.1,1))
    plt_overlap.set_ylabel("m(t)")
    plt_overlap.set_xlabel("t (msec)")

    plt.tight_layout()
    plt.savefig(name)

    if show:
        plt.show()
    else:
        plt.close()

def plot_retrieval_only(network, sol_t, m_t, retrieved_pattern=0, differents=[], name="graph.png", show=True):

    if differents is None:
        differents = []
    if retrieved_pattern < 0 or retrieved_pattern > len(m_t):
        retrieved_pattern = 0

    sol_t = np.asarray(sol_t)
    m_t = np.asarray(m_t)

    if sol_t.ndim != 1 or len(sol_t) == 0:
        raise ValueError("sol_t must be a non-empty one-dimensional array.")
    if m_t.ndim != 2:
        raise ValueError("m_t must be a two-dimensional array.")
    if m_t.shape[1] != len(sol_t):
        raise ValueError("m_t and sol_t have incompatible lengths.")
    if network.current is None:
        raise RuntimeError("The network must be simulated before plotting.")

    initial_current_color = "firebrick"
    wrong_neuron_color = "black"
    spikes_color = "C0"

    plt.close('all')

    N = network.n_blocks
    for ni in range(N):
        y = np.ones_like(network.spikes[(2+(4*ni))]) * ni
        plt.scatter(network.spikes[(2+(4*ni))], y, marker=".", color=spikes_color)

        if network.input_pattern[ni] == 1:
            plt.scatter(network.spikes[(2+(4*ni))][0], ni, marker=".", color=initial_current_color)
    for wrong in differents:
        plt.scatter(network.spikes[(2+(4*wrong))][0], wrong, marker=".", color=wrong_neuron_color)

    plt.title("Spikes", fontsize=14)
    plt.ylim((0,(N+1)))
    plt.xlim((0, sol_t[-1]))
    plt.xticks(np.arange(0,(sol_t[-1]+1),25))
    plt.xlabel("t (msec)")
    plt.ylabel("Block index")

    plt.tight_layout()
    plt.savefig(name)

    if show:
        plt.show()
    else:
        plt.close()