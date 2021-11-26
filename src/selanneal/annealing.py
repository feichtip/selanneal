import numpy as np
import random
import math
import numba as nb


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
def energy(Neve, Nexp, eff_threshold, state, cov_matrix, mode):
    if (mode == 'bins') and (cov_matrix is not None):
        """
        joining rows of state grid for covarince matrix
        for specific problem, modify here if needed
        """
        Nsig_1, Nbkg_1, Nsig_2, Nbkg_2 = Neve
        stat_error = (Nsig_1 + Nbkg_1) / Nsig_1**2 + (Nsig_2 + Nbkg_2) / Nbkg_2**2
        cov_mask = state.flatten()[np.newaxis].T @ state.flatten()[np.newaxis]
        sys_error = np.nansum(cov_matrix[cov_mask]) / (~np.isnan(cov_matrix[cov_mask])).sum()
        return stat_error + sys_error  # variance, still need to take square root
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
def print_progress(progress,  T, E, accept, improve):
    space = ' '
    progress_r = round_digits(progress, 4)
    T_r = round_digits(T, 4)
    E_r = round_digits(E, 4)
    accept_r = round_digits(accept, 3)
    improve_r = round_digits(improve, 3)
    if progress == 0:
        print('\n progress     temperature     energy     acceptance     improvement')
        print('--------------------------------------------------------------------')
    else:
        print(space * (2 - count_digits(progress_r)[0]), progress_r,
              space * (12 - count_digits(progress_r)[1] - count_digits(T_r)[0]), T_r,
              space * (11 - count_digits(T_r)[1] - count_digits(E_r)[0]), E_r,
              space * (8 - count_digits(E_r)[1] - count_digits(accept_r)[0]), accept_r,
              space * (12 - count_digits(accept_r)[1] - count_digits(improve_r)[0]), improve_r)


@nb.njit
def bin_coupling(state, x, y, bins, strength):
    not_border_x = np.array([x != 0, x != 0, x != 0, True, x != (bins[0] - 1), x != (bins[0] - 1), x != (bins[0] - 1), True])
    not_border_y = np.array([True, y != 0, y != (bins[1] - 1), y != 0, True, y != 0, y != (bins[1] - 1), y != (bins[1] - 1)])

    x_not_border = (x + np.array([-1, -1, -1, 0, 1, 1, 1, 0]))[not_border_x & not_border_y]
    y_not_border = (y + np.array([0, -1, 1, -1, 0, -1, 1, 1]))[not_border_x & not_border_y]

    # same_state = state[x_not_border, y_not_border] == state[x, y] # does not work with numba
    same_state = np.asarray([state[a, b] for a, b in zip(x_not_border, y_not_border)]) == state[x, y]

    return strength * (same_state.sum() - (~same_state).sum())


@nb.njit
def start_anneal(initial_state, hists, Neve, Nexp, eff_threshold, cov_matrix, Tmin, Tmax, steps, meshg, bins, n_dim, coupling, mode, verbose):
    state = initial_state.copy()

    E = energy(Neve, Nexp, eff_threshold, state, cov_matrix, mode)
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
    print_every = int(steps / 50)

    for step in range(steps):
        for a, b in zip(*meshg):
            if mode == 'edges':
                valid = False
                while not valid:
                    a, b, c = choose_state(n_dim)
                    valid = valid_state(state, bins, a, b, c)
                sign, sl = get_edge_slice(state, a, b, c)
            elif mode == 'bins':
                c = -1
                sign, sl = get_bin_slice(state, a, b)

            dNeve = sign * hists[sl].copy().reshape(-1, hists.shape[-1]).sum(axis=0)  # sum over all except last axis
            Neve += dNeve

            E = energy(Neve, Nexp, eff_threshold, state, cov_matrix, mode)

            # calculate delta e
            dE = E - prev_E

            # add coupling to delta E
            if (mode == 'bins') and (coupling != 0):
                dE += bin_coupling(state, a, b, bins, coupling)

            # always accept if dE < 0
            if dE < 0 or math.exp(-dE / T) > random.random():
                # accept new state
                state[a, b] = abs(state[a, b] + c)

                accepted += 1
                if dE < 0.0:
                    improved += 1

                prev_E = E
                prev_Neve = Neve.copy()

                if E < best_energy:
                    best_state = state.copy()
                    best_energy = E
            else:
                # keep previous state
                E = prev_E
                Neve = prev_Neve.copy()

        if (step // print_every) > ((step - 1) // print_every) and verbose:
            print_progress(step / steps, T, E, accepted / print_every / len(meshg[0]), improved / print_every / len(meshg[0]))
            accepted = 0
            improved = 0

        T *= T_scaling

    return best_state, best_energy


def run(h_signal=None, h_background=None, hists=None, n_hists=2, Nexp=None, eff_threshold=None, cov_matrix=None, Tmin=0.001, Tmax=10, steps=1_000, coupling=0, verbose=True, mode='bins'):

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

    if mode == 'bins':
        assert(n_dim == 2)
        initial_state = np.random.randint(0, 2, size=bins, dtype='bool')
        meshg = np.meshgrid(range(initial_state.shape[0]), range(initial_state.shape[1]))
        meshg = (meshg[0].flatten(), meshg[1].flatten())
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
                                           cov_matrix,
                                           Tmin,
                                           Tmax,
                                           steps,
                                           meshg,
                                           bins,
                                           n_dim,
                                           coupling,
                                           mode,
                                           verbose)

    if verbose:
        print('\nbest energy:', best_energy)

    # calculate the energy again based on the best state and check if it is consistent
    Neve = n_events(hists, best_state, mode, n_dim)
    energy_check = energy(Neve, Nexp, eff_threshold, best_state, cov_matrix, mode)
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
