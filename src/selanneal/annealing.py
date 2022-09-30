import math
import random

import numba as nb
import numpy as np


@nb.njit
def choose_state(n_dim):
    # change state
    a = random.randint(0, n_dim - 1)
    b = random.randint(0, 1)
    choose_from = [-2, -1, 1, 2]
    c = choose_from[random.randint(0, len(choose_from) - 1)]
    return a, b, c


@nb.njit
def valid_state(state, bins, a, b, c):
    # check boundaries and order of edges
    if b == 0:
        if state[a, 0] + c >= state[a, 1]:
            return False
        if (state[a, 0] + c) < 0:
            return False
    elif b == 1:
        if state[a, 0] >= state[a, 1] + c:
            return False
        if (state[a, 1] + c) > bins[a]:
            return False

    assert(state[a, 0] < state[a, 1])
    assert(state[a, b] <= bins[a])
    assert(state[a, b] >= 0)

    return True


@nb.njit
def get_edge_slice(state, a, b, c):
    # calculate delta numerator, denominator
    sl_state = state.copy()
    lb_a, ub_a = sl_state[a]
    if b == 0:
        if c > 0:
            # when shrinking below
            sl_state[a] = [lb_a, lb_a + c]
            sign = -1
        else:
            # when expanding below
            sl_state[a] = [lb_a + c, lb_a]
            sign = +1
    elif b == 1:
        if c < 0:
            # when shrinking above
            sl_state[a] = [ub_a + c, ub_a]
            sign = -1
        else:
            # when expanding above
            sl_state[a] = [ub_a, ub_a + c]
            sign = +1

    return sign, get_slice(sl_state)


@nb.njit
def get_bin_slice(state, x, y):
    # calculate delta numerator, denominator
    sign = -state[x, y] + ~state[x, y]
    select_bin = np.asarray([[x, x + 1], [y, y + 1]])
    return sign, get_slice(select_bin)


@nb.njit
def energy(Neve, Nexp, eff_threshold, state, sparse_indices, sparse_data, cov_weights):
    if sparse_data is not None:
        """
        joining rows of state grid for covarince matrix
        for specific problem, modify here if needed
        """
        Nsig_1 = Neve[0]
        Nbkg_1 = Neve[1]
        Nsig_2 = Neve[2]
        Nbkg_2 = Neve[3]

        if Nsig_1 == 0 or Nsig_2 == 0:
            return 100
        stat_error = (Nsig_1 + Nbkg_1) / Nsig_1**2 + (Nsig_2 + Nbkg_2) / Nsig_2**2

        N_sigs_1 = Neve[4::4]
        N_bkgs_1 = Neve[5::4]
        N_sigs_2 = Neve[6::4]
        N_bkgs_2 = Neve[7::4]

        numerator = (Nsig_1 + Nbkg_1 - N_bkgs_1) / N_sigs_1
        denominator = (Nsig_2 + Nbkg_2 - N_bkgs_2) / N_sigs_2
        ratio = numerator / denominator
        variance = ((ratio - ratio.mean())**2).sum() / (len(ratio) - 1)
        # return np.sqrt(stat_error + variance) * 100

        states = state.flatten()
        assert len(cov_weights) == len(states)

        # nan_states = (states & ~sparse_positions).sum()
        # sparse_indices = np.where(sparse_positions)[0]

        sum = 0
        not_nan = 0
        data_idx = 0
        weight_sum = 0
        for i in sparse_indices:
            for j in sparse_indices:
                if j > i:
                    break
                if states[i] and states[j]:
                    multiplicity = 1 + int(i != j)
                    entry = sparse_data[data_idx]
                    weight = multiplicity * cov_weights[i] * cov_weights[j]
                    sum += entry * weight
                    weight_sum += weight
                    not_nan += multiplicity
                data_idx += 1

        nan_states = states.sum() - np.sqrt(not_nan)  # penalty term
        sys_error = sum / weight_sum
        # return np.sqrt(stat_error + sys_error) * 100 + nan_states
        return np.sqrt(stat_error + (0.1 * sys_error + 0.9 * variance)) * 100 + nan_states
    else:
        Nsig, Nbkg = Neve
        if (Nsig + Nbkg) != 0:
            if Nexp is None and eff_threshold is None:
                # significance FOM
                return -Nsig / math.sqrt(Nsig + Nbkg)
            else:
                # purity
                return -Nsig / (Nsig + Nbkg) * (Nsig / Nexp > eff_threshold)
        else:
            return 0


@nb.jit
def round_digits(m, d):
    """Round m to d digits."""
    if m == 0:
        return 0
    return round(m, d - 1 - int(math.log10(abs(m))))


@nb.jit
def count_digits(m):
    """count digits of a float, before and after '.' """
    before = 1

    # digits before .
    if int(m) != 0:
        before += int(math.log10(abs(m)))

    # digits after .
    after = 0
    if int(m) == m:
        after += 2
    else:
        while m != 0:
            m = round(m - int(m), 8) * 10
            after += 1

    return before, after - 1


@nb.njit
def print_progress(step, steps,  T, E, accept, improve):
    space = ' '
    progress_r = round_digits((step + 1) / steps, 3)
    T_r = round_digits(T, 4)
    E_r = round_digits(E, 4)
    accept_r = round_digits(accept, 3)
    improve_r = round_digits(improve, 3)
    if step == 0:
        print('\n progress     temperature     energy     acceptance     improvement')
        print('--------------------------------------------------------------------')
    else:
        print(space * (2 - count_digits(progress_r)[0]), progress_r,
              space * (12 - count_digits(progress_r)[1] - count_digits(T_r)[0]), T_r,
              space * (11 - count_digits(T_r)[1] - count_digits(E_r)[0]), E_r,
              space * (8 - count_digits(E_r)[1] - count_digits(accept_r)[0]), accept_r,
              space * (12 - count_digits(accept_r)[1] - count_digits(improve_r)[0]), improve_r)


@nb.njit
def bin_coupling(state, x, y, bins, strength, moore=True):
    """
    return bin coupling difference: -(new state - old state)
    """
    if moore:
        not_border_x = np.array([x != 0, x != 0, x != 0, True, x != (bins[0] - 1), x != (bins[0] - 1), x != (bins[0] - 1), True])
        not_border_y = np.array([True, y != 0, y != (bins[1] - 1), y != 0, True, y != 0, y != (bins[1] - 1), y != (bins[1] - 1)])

        x_not_border = (x + np.array([-1, -1, -1, 0, 1, 1, 1, 0]))[not_border_x & not_border_y]
        y_not_border = (y + np.array([0, -1, 1, -1, 0, -1, 1, 1]))[not_border_x & not_border_y]
    else:
        not_border_x = np.array([x != 0, True, x != (bins[0] - 1), True])
        not_border_y = np.array([True, y != 0, True, y != (bins[1] - 1)])

        x_not_border = (x + np.array([-1, 0, 1, 0]))[not_border_x & not_border_y]
        y_not_border = (y + np.array([0, -1, 0, 1]))[not_border_x & not_border_y]

    # same_state = state[x_not_border, y_not_border] == state[x, y] # does not work with numba
    same_state = np.asarray([state[a, b] for a, b in zip(x_not_border, y_not_border)]) == state[x, y]

    return -strength * (same_state.sum() - (~same_state).sum())


@nb.njit
def start_anneal(initial_state, hists, Neve, Nexp, eff_threshold, sparse_indices, sparse_data, cov_weights, Tmin, Tmax, steps, meshg, bins, n_dim, coupling, mode, moore, verbose):
    state = initial_state.copy()

    E = energy(Neve, Nexp, eff_threshold, state, sparse_indices, sparse_data, cov_weights)
    if verbose:
        print('inital temperature:', Tmax)
        print('final temperature:', Tmin)
        print('steps:', steps)
        print('inital energy:', E)

    T_scaling = (Tmin / Tmax) ** (1 / steps)
    T = Tmax

    prev_E = E
    prev_Neve = Neve.copy()

    best_energy = E
    best_state = state.copy()

    accepted = 0
    improved = 0
    n_prints = 50
    print_every = int(steps / n_prints)

    for step in range(steps):
        for a, b in zip(*meshg):
            if mode == 'edges':
                valid = False
                while not valid:
                    a, b, c = choose_state(n_dim)
                    valid = valid_state(state, bins, a, b, c)
                sign, sl = get_edge_slice(state, a, b, c)
                state[a, b] = state[a, b] + c  # new state
            elif mode == 'bins':
                sign, sl = get_bin_slice(state, a, b)
                state[a, b] = ~state[a, b]  # new state

            dNeve = sign * hists[sl].copy().reshape(-1, hists.shape[-1]).sum(axis=0)  # sum over all except last axis
            Neve += dNeve

            E = energy(Neve, Nexp, eff_threshold, state, sparse_indices, sparse_data, cov_weights)

            # calculate delta e
            dE = E - prev_E

            # add coupling to delta E
            if (mode == 'bins') and (coupling != 0):
                dE += bin_coupling(state, a, b, bins, coupling, moore=moore)

            # always accept if dE < 0
            if dE < 0 or math.exp(-dE / T) > random.random():
                # accept new state

                accepted += 1
                if dE < 0.0:
                    improved += 1

                prev_E = E
                prev_Neve = Neve.copy()

                if E < best_energy:
                    best_state = state.copy()
                    best_energy = E
            else:
                # revert to previous state
                if mode == 'edges':
                    state[a, b] = state[a, b] - c
                elif mode == 'bins':
                    state[a, b] = ~state[a, b]
                E = prev_E
                Neve = prev_Neve.copy()

        if (step == 0) and (accepted / len(meshg[0]) < 0.9):
            accepted_r = round_digits(accepted / len(meshg[0]), 3) * 100
            print('Only', accepted_r, '% of new states were accepted in the first iteration. Consider to increase the maximum temperature.')

        if (step // print_every) > ((step - 1) // print_every):
            if verbose:
                # for step==0: prints only header
                print_progress(step, steps, T, E, accepted / print_every / len(meshg[0]), improved / print_every / len(meshg[0]))
            accepted = 0
            improved = 0

        T *= T_scaling

    norm = (steps - 1) % print_every
    if verbose:  # for last step
        print_progress(step, steps, T, E, accepted / norm / len(meshg[0]), improved / norm / len(meshg[0]))
    if (improved / norm / len(meshg[0]) > 0.01):
        improved_r = round_digits(improved / norm / len(meshg[0]), 3) * 100
        print('Still', improved_r, '% of new states improved in the last iteration. Consider to decrease the minimum temperature.')

    return best_state, best_energy


def run(h_signal=None, h_background=None, hists=None, n_hists=2, Nexp=None, eff_threshold=None, cov_matrix=None, cov_weights=None, Tmin=0.001, Tmax=10, steps=1_000, coupling=0, moore=True, verbose=True, mode='bins'):

    if hists is None:
        if h_signal is None or h_background is None:
            raise Exception('You need to either pass <h_signal> and <h_background>, or <n_hists> other number of histogrmas as <hists>')

        hists = np.stack([h_signal, h_background], axis=-1)

    if (n_hists == 1) and (hists.shape[-1] != 1):
        hists = hists[..., np.newaxis]

    assert hists.shape[-1] == n_hists, 'last axis of hists should represent (n_hists) different histograms (e.g. signal and background)'
    bins = hists.shape[:-1]

    n_dim = len(bins)
    if (mode == 'edges') and (len(bins) == 1):
        bins = bins + (0, )  # to have at least 2 dimensions, otherwise doesn't compile for 1d

    sparse_indices = None
    sparse_data = None

    if mode == 'bins':
        assert(n_dim == 2)
        initial_state = np.random.randint(0, 2, size=bins, dtype='bool')
        meshg = np.meshgrid(range(initial_state.shape[0]), range(initial_state.shape[1]))
        meshg = (meshg[0].flatten(), meshg[1].flatten())
        if cov_matrix is not None:
            sparse_positions = ~np.isnan(np.diag(cov_matrix))
            sparse_indices = np.where(sparse_positions)[0]
            sparse_data = cov_matrix[~np.isnan(cov_matrix) & np.tril(np.ones_like(cov_matrix)).astype(bool)]
            assert sparse_positions.sum() == (-1 / 2 + np.sqrt(1 / 4 + len(sparse_data) * 2))
            assert (~np.isnan(cov_matrix)).sum() == (1 / 2 + len(sparse_data) * 2 - np.sqrt(1 / 4 + len(sparse_data) * 2))
    elif mode == 'edges':
        initial_state = np.array([[0, bins[i]] for i in range(n_dim)], dtype='int')
        meshg = ([0], [0])

    code = f"""global get_slice\n@nb.njit\ndef get_slice(state):
    return ({", ".join(f"slice(state[{i}, 0], state[{i}, 1])" for i in range(n_dim))})
    """
    exec(code)

    Neve = n_events(hists, initial_state, mode, n_dim)

    best_state, best_energy = start_anneal(initial_state,
                                           hists,
                                           Neve,
                                           Nexp,
                                           eff_threshold,
                                           sparse_indices,
                                           sparse_data,
                                           cov_weights,
                                           Tmin,
                                           Tmax,
                                           steps,
                                           meshg,
                                           bins,
                                           n_dim,
                                           coupling,
                                           mode,
                                           moore,
                                           verbose)

    if verbose:
        print('\nbest energy:', best_energy)

    # calculate the energy again based on the best state and check if it is consistent
    Neve = n_events(hists, best_state, mode, n_dim)
    energy_check = energy(Neve, Nexp, eff_threshold, best_state, sparse_indices, sparse_data, cov_weights)
    assert np.isclose(best_energy, energy_check), f'{best_energy:.4f} <--> {energy_check:.4}'

    if mode == 'edges':
        # returns the index for the bin edges that gives the correct cut value
        best_state = [(state[0], state[1]) for state in best_state]

    return best_state, best_energy


def n_events(hists, state, mode, n_dim):
    if mode == 'edges':
        sl = ()
        for i in range(n_dim):
            sl += (slice(*state[i]), )
        state = sl

    Neve = hists[state].reshape(-1, hists.shape[-1]).sum(axis=0)  # sum over all except last axis

    return Neve
