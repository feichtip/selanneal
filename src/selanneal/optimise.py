import boost_histogram as bh
import numpy as np
from . import annealing


def create_axes(data, n_bins, n_int_axes=0, features=None):
    """
    data: (m, n) array where n is the number of features
    """
    axes = []
    n_axes = data.shape[1] - n_int_axes
    features = features or [f'feature {i+1}' for i in range(n_axes)]
    lower_edges = np.nanquantile(data[:, :n_axes], q=0.001, axis=0)
    upper_edges = np.nanquantile(data[:, :n_axes], q=0.999, axis=0)
    for start, stop, feature in zip(lower_edges, upper_edges, features):
        # move upper edge to include integer values (right edge of interval is open)
        if float(stop).is_integer():
            stop += 0.001
        axes.append(bh.axis.Regular(n_bins, start, stop, metadata=feature))
    return axes


def histogram(data, axes, isSignal, weight=1):
    """
    fill histograms
    """
    h_sig = bh.Histogram(*axes, storage=bh.storage.Double())
    h_bkg = bh.Histogram(*axes, storage=bh.storage.Double())

    if not isSignal.dtype == bool:
        isSignal = isSignal.astype(bool)

    if isinstance(weight, int):
        weight_sig = weight
        weight_bkg = weight
    else:
        weight_sig = weight[isSignal]
        weight_bkg = weight[~isSignal]

    h_sig.fill(*data[isSignal].T, weight=weight_sig)
    h_bkg.fill(*data[~isSignal].T, weight=weight_bkg)

    return h_sig, h_bkg


def roc(data, isSignal, features, weight=1, Nexp=None, roc_points=10, **kwargs):
    Nsig = (isSignal * weight).sum()
    Nbkg = (~isSignal * weight).sum()

    efficiency = Nsig / Nexp
    purity = Nsig / (Nsig + Nbkg)

    efficiencies = np.linspace(efficiency, 0, roc_points)
    purities = [purity]
    print(f'initial purity: {purity}, efficiency: {efficiency}')
    for eff_threshold in efficiencies[1:-1]:
        print(f'\noptimising for efficiency: {eff_threshold}')
        _, _, _, best_energy = iterate(data, isSignal, features, weight=weight, Nexp=Nexp, eff_threshold=eff_threshold, **kwargs)
        purities.append(-best_energy)

    purities.append(1.0)

    return efficiencies, purities


def iterate(data, isSignal, features=None, weight=1, int_axes=[], h_sys_up=None, h_sys_down=None, Nexp=None, eff_threshold=None, new_bins=3, min_iter=5, max_iter=20, rtol=1E-4, eval_function=None,  **kwargs):
    # new_bins has to be at least 3 to create new bins
    max_bins = new_bins * 2 + 3
    new_axes = create_axes(data, max_bins, len(int_axes), features)
    prev_energy = 0
    for i in range(max_iter):
        axes = new_axes
        h_sig, h_bkg = histogram(data, axes + int_axes, isSignal, weight)
        best_state, best_energy = annealing.run(h_sig.values(), h_bkg.values(), h_sys_up, h_sys_down, Nexp, eff_threshold, mode='edges', **kwargs)
        N_sig, N_bkg, sysUp, sysDown = annealing.n_events(h_sig.values(), h_bkg.values(), h_sys_up=h_sys_up, h_sys_down=h_sys_down, state=best_state, mode='edges', n_dim=h_sig.ndim)

        printout = f'iteration {i}:  E={best_energy:>10.3f}'
        if eval_function is not None:
            printout += f' {eval_function(N_sig, N_bkg)}'
        print(printout)

        if (abs((best_energy - prev_energy) / best_energy)) < rtol and (i >= (min_iter - 1)):
            break

        prev_energy = best_energy
        new_axes = []
        for axis, state in zip(axes, best_state):
            lb = [0] * 2
            ub = [0] * 2
            lower_edge = axis.edges[0]
            upper_edge = axis.edges[axis.size]
            for j in range(2):
                current_edge = axis.edges[state[j]]
                if (current_edge != lower_edge) and (current_edge != upper_edge):
                    lower_bin_width = current_edge - axis.edges[state[j] - 1]
                    upper_bin_width = axis.edges[state[j] + 1] - current_edge
                    min_bin_width = min(lower_bin_width, upper_bin_width)
                    lb[j] = current_edge - min_bin_width
                    ub[j] = current_edge + min_bin_width
                else:
                    lb[j] = axis.edges[state[j] - 1] if current_edge != lower_edge else lower_edge
                    ub[j] = axis.edges[state[j] + 1] if current_edge != upper_edge else upper_edge

            # create new bin edges within the new ranges
            if ub[0] < lb[1]:
                bin_edges = np.concatenate((np.linspace(lb[0], ub[0], new_bins + 1), np.linspace(lb[1], ub[1], new_bins + 1)))
            else:
                # for overlapping or touching bin edges
                bin_edges = np.linspace(lb[0], ub[1], 2 * new_bins + 1)

            # add lower and upper edge if needed
            if not np.isclose(bin_edges[0], lower_edge):
                bin_edges = np.insert(bin_edges, 0, lower_edge)
            if not np.isclose(bin_edges[-1], upper_edge):
                bin_edges = np.append(bin_edges, upper_edge)

            assert(lb[0] <= ub[1])
            try:
                new_axes.append(bh.axis.Variable(bin_edges, metadata=axis.metadata))
            except:
                print(bin_edges)

    selection_list = selection(axes + int_axes, best_state)

    return selection_list, axes + int_axes, best_state, best_energy


def selection(axes, best_state):
    selection = []
    for axis, state in zip(axes, best_state):
        if axis.value(state[1]) is None:
            selection.append(f'{axis.value(state[0]):.4f} <= {axis.metadata} <= {axis.value(state[1]-1):.4f}')
        else:
            selection.append(f'{axis.value(state[0]):.4f} <= {axis.metadata} < {axis.value(state[1]):.4f}')

    return selection
