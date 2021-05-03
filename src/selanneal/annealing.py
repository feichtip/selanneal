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
        if state[a, 0] + c > state[a, 1]:
            return False
        if (state[a, 0] + c) < 0:
            return False
    elif b == 1:
        if state[a, 0] > state[a, 1] + c:
            return False
        if (state[a, 1] + c) >= bins[a]:
            return False

    assert(state[a, 0] <= state[a, 1])
    assert(state[a, b] < bins[a])
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
            sl_state[a] = [lb_a, lb_a - 1 + c]
            sign = -1
        else:
            # when expanding below
            sl_state[a] = [lb_a + c, lb_a - 1]
            sign = +1
    elif b == 1:
        if c < 0:
            # when shrinking above
            sl_state[a] = [ub_a + 1 + c, ub_a]
            sign = -1
        else:
            # when expanding above
            sl_state[a] = [ub_a + 1, ub_a + c]
            sign = +1

    return sign, get_slice(sl_state)


@nb.njit
def get_bin_slice(state, x, y):
    # calculate delta numerator, denominator
    sign = -state[x, y] + ~state[x, y]
    select_bin = np.asarray([[x, x], [y, y]])
    return sign, get_slice(select_bin)


@nb.njit
def energy(numerator, denominator):
    if denominator != 0:
        return -numerator / math.sqrt(denominator)
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
def start_anneal(initial_state, h_signal, h_background,  numerator, denominator, Tmin, Tmax, steps, meshg,  bins, n_dim, mode,  verbose):
    state = initial_state.copy()

    E = energy(numerator, denominator)
    print('inital temperature', Tmax)
    print('final temperature', Tmin)
    print('steps', steps)
    print('inital energy', E)

    # E = full_energy(h_signal, h_background,  state)
    # print(E)

    T_scaling = (Tmin / Tmax) ** (1 / steps)
    T = Tmax

    prevEnergy = E
    prevNumerator = numerator
    prevDenominator = denominator

    best_energy = E

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

            dNum = sign * h_signal[sl].sum()
            dDen = sign * h_background[sl].sum() + dNum
            numerator += dNum
            denominator += dDen
            E = energy(numerator, denominator)

            # calculate delta e
            dE = E - prevEnergy

            # always accept if dE < 0
            if dE < 0 or math.exp(-dE / T) > random.random():
                # accept new state
                state[a, b] = abs(state[a, b] + c)

                accepted += 1
                if dE < 0.0:
                    improved += 1

                prevEnergy = E
                prevNumerator = numerator
                prevDenominator = denominator

                if E < best_energy:
                    best_state = state.copy()
                    best_energy = E
            else:
                # keep previous state
                E = prevEnergy
                numerator = prevNumerator
                denominator = prevDenominator

        if (step // print_every) > ((step - 1) // print_every) and verbose:
            print_progress(step / steps, T, E, accepted / print_every / len(meshg[0]), improved / print_every / len(meshg[0]))
            accepted = 0
            improved = 0

        T *= T_scaling

    return best_state, best_energy


def selanneal(h_signal, h_background,  Tmin=0.001, Tmax=10, steps=1_000, verbose=True, mode='bins'):

    bins = h_signal.shape
    n_dim = len(bins)

    assert(n_dim == len(h_background.shape))
    for i in range(n_dim):
        assert(h_signal.shape[i] == h_background.shape[i])

    if mode == 'bins':
        initial_state = np.random.randint(0, 2, size=bins, dtype='bool')
        meshg = np.meshgrid(range(initial_state.shape[0]), range(initial_state.shape[1]))
        meshg = (meshg[0].flatten(), meshg[1].flatten())
    elif mode == 'edges':
        initial_state = np.array([[0, bins[i] - 1] for i in range(n_dim)], dtype='int')
        meshg = ([0], [0])

    code = f"""global get_slice\n@nb.njit\ndef get_slice(state):
    return ({", ".join(f"slice(state[{i}, 0], state[{i}, 1] + 1)" for i in range(n_dim))})
    """
    exec(code)

    numerator, denominator = getNumDen(h_signal, h_background, initial_state, mode)

    best_state, best_energy = start_anneal(initial_state,
                                           h_signal,
                                           h_background,
                                           numerator,
                                           denominator,
                                           Tmin,
                                           Tmax,
                                           steps,
                                           meshg,
                                           bins,
                                           n_dim,
                                           mode,
                                           verbose)

    print('best state\n', best_state)

    numerator, denominator = getNumDen(h_signal, h_background, best_state, mode)
    energy_check = energy(numerator, denominator)
    print('best energy', best_energy, energy_check)
    return best_state, best_energy


def getNumDen(h_signal, h_background, state, mode):
    if mode == 'edges':
        n_dim = len(h_signal.shape)
        sl = ()
        for i in range(n_dim):
            sl += (slice(state[i, 0], state[i, 1] + 1), )
        state = sl

    n_sig = h_signal[state].sum()
    n_bkg = h_background[state].sum()

    return n_sig, n_sig + n_bkg
