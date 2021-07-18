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
def energy(Nsig, Nbkg, sysUp, sysDown):
    if (Nsig + Nbkg) != 0:
        return -Nsig / math.sqrt(Nsig + Nbkg + ((sysUp + sysDown) / 2)**2)
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
def start_anneal(initial_state, h_signal, h_background, h_sys_up, h_sys_down, Nsig, Nbkg, sysUp, sysDown, Tmin, Tmax, steps, meshg, bins, n_dim, coupling, mode, verbose):
    state = initial_state.copy()

    E = energy(Nsig, Nbkg, sysUp, sysDown)
    if verbose:
        print('inital temperature:', Tmax)
        print('final temperature:', Tmin)
        print('steps:', steps)
        print('inital energy:', E)

    T_scaling = (Tmin / Tmax) ** (1 / steps)
    T = Tmax

    prev_E = E
    prev_Nsig = Nsig
    prev_Nbkg = Nbkg
    prev_sysUp = sysUp
    prev_sysDown = sysDown

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

            dNsig = sign * h_signal[sl].sum()
            dNbkg = sign * h_background[sl].sum()

            dSysUp = 0 if h_sys_up is None else sign * h_sys_up[sl].sum()
            dSysDown = 0 if h_sys_down is None else sign * h_sys_down[sl].sum()

            Nsig += dNsig
            Nbkg += dNbkg
            sysUp += dSysUp
            sysDown += dSysDown
            E = energy(Nsig, Nbkg, sysUp, sysDown)

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
                prev_Nsig = Nsig
                prev_Nbkg = Nbkg
                prev_sysUp = sysUp
                prev_sysDown = sysDown

                if E < best_energy:
                    best_state = state.copy()
                    best_energy = E
            else:
                # keep previous state
                E = prev_E
                Nsig = prev_Nsig
                Nbkg = prev_Nbkg
                sysUp = prev_sysUp
                sysDown = prev_sysDown

        if (step // print_every) > ((step - 1) // print_every) and verbose:
            print_progress(step / steps, T, E, accepted / print_every / len(meshg[0]), improved / print_every / len(meshg[0]))
            accepted = 0
            improved = 0

        T *= T_scaling

    return best_state, best_energy


def run(h_signal, h_background, h_sys_up=None, h_sys_down=None, Tmin=0.001, Tmax=10, steps=1_000, coupling=0, verbose=True, mode='bins'):

    bins = h_signal.shape
    n_dim = len(bins)
    # bins = bins + (0, )  # to have at least 2 dimensions

    assert(n_dim == len(h_background.shape))
    for i in range(n_dim):
        assert(h_signal.shape[i] == h_background.shape[i])

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

    Nsig, Nbkg, sysUp, sysDown = n_events(h_signal, h_background, h_sys_up, h_sys_down, initial_state, mode, n_dim)

    best_state, best_energy = start_anneal(initial_state,
                                           h_signal,
                                           h_background,
                                           h_sys_up,
                                           h_sys_down,
                                           Nsig,
                                           Nbkg,
                                           sysUp,
                                           sysDown,
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
    Nsig, Nbkg, sysUp, sysDown = n_events(h_signal, h_background, h_sys_up, h_sys_down, best_state, mode, n_dim)
    energy_check = energy(Nsig, Nbkg, sysUp, sysDown)
    assert(np.isclose(best_energy, energy_check))

    if mode == 'edges':
        # returns the index for the bin edges that gives the correct cut value
        best_state = [(state[0], state[1]) for state in best_state]

    return best_state, best_energy


def n_events(h_signal, h_background, h_sys_up, h_sys_down, state, mode, n_dim):
    if mode == 'edges':
        sl = ()
        for i in range(n_dim):
            sl += (slice(*state[i]), )
        state = sl

    Nsig = h_signal[state].sum()
    Nbkg = h_background[state].sum()

    sysUp = 0 if h_sys_up is None else h_sys_up[state].sum()
    sysDown = 0 if h_sys_down is None else h_sys_down[state].sum()

    return Nsig, Nbkg, sysUp, sysDown
