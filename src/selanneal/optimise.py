import boost_histogram as bh
import numpy as np
import itertools
import math
from . import annealing


def create_axes(data, n_bins, n_int_axes=0, features=None, quantile=0.001):
    """
    data: (m, n) array where n is the number of features
    """
    axes = []
    n_axes = data.shape[1] - n_int_axes
    features = features or [f'feature {i+1}' for i in range(n_axes)]
    lower_edges = np.nanquantile(data[:, :n_axes], q=quantile, axis=0)
    upper_edges = np.nanquantile(data[:, :n_axes], q=1 - quantile, axis=0)
    for i, (start, stop, feature) in enumerate(zip(lower_edges, upper_edges, features)):
        # move upper edge to include integer values (right edge of interval is open)
        if float(stop).is_integer() or (data[:, i] == stop).sum() / data.shape[0] > quantile:
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


def roc(data, isSignal, features, weight=1, int_axes=[], Nexp=None, roc_points=25, feat_per_rotation=7, init_feat_weights=None, **kwargs):
    verbosity = kwargs.get('verbosity')
    verbosity = 1 if verbosity is None else verbosity

    Nsig = (isSignal * weight).sum()
    Nbkg = (~isSignal * weight).sum()

    efficiency = Nsig / Nexp
    purity = Nsig / (Nsig + Nbkg)

    # number of features
    n_feat = len(features)
    n_dim = n_feat + len(int_axes)

    # matrix to store selections
    selection_matrix = np.ones(shape=(data.shape[0], n_dim), dtype=bool)

    # lists to store selections
    selection_list = [[]] * n_dim
    selections = [None]

    selected_data = data.copy()  # remove copy?
    selected_isSignal = isSignal.copy()
    selected_weight = weight.copy()

    eff_thresholds = np.linspace(efficiency, 0, roc_points)[1: -1]  # without first and last point
    efficiencies = [efficiency]
    purities = [purity]

    # weights for sampling feature indices
    if init_feat_weights is None:
        feat_weights = np.ones(n_dim)
    else:
        feat_weights = np.asarray(init_feat_weights)

    print(f'efficiency: {efficiency*100:.2f}%, purity: {purity*100:.2f}%')

    for eff_threshold in eff_thresholds:
        # sample indices for random features
        slice_data = np.sort(np.random.choice(range(n_dim), size=feat_per_rotation, replace=False, p=feat_weights / feat_weights.sum()))
        # increase weight for features that were not chosen in this round
        feat_weights[slice_data] *= 0
        feat_weights += 1

        slice_features = slice_data[slice_data < n_feat]
        slice_int_axes = slice_data[slice_data >= n_feat] - n_feat

        selected_features = [features[i] for i in slice_features]
        selected_int_axes = [int_axes[i] for i in slice_int_axes]

        # release selection for variables used in current rotation
        for idx in slice_data:
            selection_matrix[:, idx] = np.ones(data.shape[0], dtype=bool)

        # selection for current rotation (before optimisation)
        selection_before = (selection_matrix.sum(axis=1) == n_dim)

        # take selected data
        selected_data = data[selection_before][:, slice_data]
        selected_isSignal = isSignal[selection_before]
        selected_weight = weight[selection_before]

        pd_selections, axes, best_state, best_energy = iterate(selected_data,
                                                               selected_isSignal,
                                                               selected_features,
                                                               weight=selected_weight,
                                                               int_axes=selected_int_axes,
                                                               Nexp=Nexp,
                                                               eff_threshold=eff_threshold,
                                                               **kwargs)

        # update matrix with currently optimised selection
        np_selections = numpy_selection(axes, best_state, indices=slice_data)
        for idx, np_selection in zip(slice_data, np_selections):
            selection_matrix[:, idx] = eval(np_selection)

        # selection after current optimisation
        selection_after = (selection_matrix.sum(axis=1) == n_dim)
        Nsig = (isSignal * weight)[selection_after].sum()
        Nbkg = (~isSignal * weight)[selection_after].sum()

        efficiency = Nsig / Nexp
        purity = Nsig / (Nsig + Nbkg)

        if verbosity > 0:
            print(pd_selections)
            # print(selected_features + [int_axis.metadata for int_axis in selected_int_axes])
            # print(np_selection)
            # print(axes)
            # print('purity', purity, -best_energy)
            # print('efficiency', efficiency, eff_threshold)

        assert np.isclose(purity, -best_energy), f'{purity} <--> {-best_energy}'
        assert (efficiency > eff_threshold), f'{efficiency} <--> {eff_threshold}'

        purities.append(purity)
        efficiencies.append(efficiency)

        print(f'efficiency: {efficiency*100:.2f}%, purity: {purity*100:.2f}%')

        # save selection of current optimisation
        for idx, pd_selection in zip(slice_data, pd_selections):
            selection_list[idx] = [pd_selection]
        selections.append(list(itertools.chain(*selection_list)))

    purities.append(1.0)
    efficiencies.append(0.0)
    selections.append(None)

    return selections, efficiencies, purities


def iterate(data, isSignal, features=None, weight=1, int_axes=[], Nexp=None, eff_threshold=None, new_bins=3, min_iter=5, max_iter=20, rtol=1E-4, eval_function=None, quantile=0.001, precision=4, roundDownUp=False, verbosity=1, **kwargs):
    # new_bins has to be at least 3 to create new bins
    max_bins = new_bins * 2 + 3
    new_axes = create_axes(data, max_bins, len(int_axes), features, quantile=quantile)
    prev_energy = 0
    for i in range(max_iter):
        axes = new_axes
        h_sig, h_bkg = histogram(data, axes + int_axes, isSignal, weight)
        hists = np.stack([h_sig.values(), h_bkg.values()], axis=-1)
        best_state, best_energy = annealing.run(hists=hists,
                                                n_hists=2,
                                                Nexp=Nexp,
                                                eff_threshold=eff_threshold,
                                                mode='edges',
                                                **kwargs)
        N_sig, N_bkg = annealing.n_events(hists, state=best_state, mode='edges', n_dim=h_sig.ndim)

        if verbosity > 0:
            printout = f'iteration {i}:  E={best_energy:>10.3f}'
            if eval_function is not None:
                printout += f' {eval_function(N_sig, N_bkg)}'
            print(printout)

        if (best_energy != 0) and abs((best_energy - prev_energy) / best_energy) < rtol and (i >= (min_iter - 1)):
            break

        prev_energy = best_energy
        new_axes = []
        for axis, state in zip(axes, best_state):
            lb = [0] * 2  # lower bound of adjacent bins
            ub = [0] * 2  # upper bound of adjacent bins
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
            if (ub[0] < lb[1]) and not np.isclose(ub[0], lb[1]):
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
            assert(lb[0] < ub[0])
            assert(lb[1] < ub[1])

            new_axes.append(bh.axis.Variable(bin_edges, metadata=axis.metadata))

    selection = pd_selection(axes + int_axes, best_state, precision=precision, roundDownUp=roundDownUp)

    return selection, axes + int_axes, best_state, best_energy


def pd_selection(axes, best_state, precision=4, roundDownUp=False):
    """
    precision [int]: floating point precision to which the bounds are rounded
    roundDownUp [bool]: round lower bound down and upper bound up
    """
    selection = []
    for axis, state in zip(axes, best_state):
        if axis.value(state[1]) is None:
            lb = axis.value(state[0])
            ub = axis.value(state[1] - 1)
            if roundDownUp:
                lb = math.floor(lb * 10**precision) / 10**precision
                ub = math.ceil(ub * 10**precision) / 10**precision
            selection.append(f'{lb:.{precision}f} <= {axis.metadata} <= {ub:.{precision}f}')
        else:
            lb = axis.value(state[0])
            ub = axis.value(state[1])
            if roundDownUp:
                lb = math.floor(lb * 10**precision) / 10**precision
                ub = math.ceil(ub * 10**precision) / 10**precision
            selection.append(f'{lb:.{precision}f} <= {axis.metadata} < {ub:.{precision}f}')

    return selection


def numpy_selection(axes, best_state, indices=None, array_name='data'):
    selection = []

    indices = range(len(axes)) if indices is None else indices
    assert(len(indices) == len(axes))
    assert(len(indices) == len(best_state))

    for i, axis, state in zip(indices, axes, best_state):
        if axis.value(state[1]) is None:
            selection.append(f'({axis.value(state[0])} <= {array_name}[:, {i}]) & ({array_name}[:, {i}] <= {axis.value(state[1]-1)})')
        else:
            selection.append(f'({axis.value(state[0])} <= {array_name}[:, {i}]) & ({array_name}[:, {i}] < {axis.value(state[1])})')

    return selection
