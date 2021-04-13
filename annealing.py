import numpy as np
import random
import math
import numba as nb


@nb.njit
def update(h_signal, h_background, n_bins, state, n_dim):

    # change state
    a = random.randint(0, n_dim - 1)
    b = random.randint(0, 1)
    c = 0

    while (c == 0) or (state[a, 1 - b] == state[a, b] + c):
        c = random.randint(-1, 1)

    if ((b == 0) and (c == 1)) or ((b == 1) and (c == -1)):
        # when shrinking
        prevState = state[a, b]
        sign = -1
    if ((b == 1) and (c == 1)) or ((b == 0) and (c == -1)):
        # when expanding
        prevState = state[a, b] + c
        sign = +1

    state[a, b] = state[a, b] + c

    if state[a, b] > n_bins - 1:
        state[a, b] = n_bins - 1
        return 0, 0
    elif state[a, b] < 0:
        state[a, b] = 0
        return 0, 0

    assert(state[a, 0] < state[a, 1])
    assert(c == -1 or c == 1)

    # calculate delta numerator, denominator
    lb_a, ub_a = state[a]
    state[a] = [prevState, prevState]
    sl = get_slice(state)
    dNum = sign * h_signal[sl].sum()
    dDen = sign * h_background[sl].sum() + dNum
    state[a] = [lb_a, ub_a]

    return dNum, dDen


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
def full_energy(h_signal, h_background, n_bins, state):
    sl = get_slice(state)
    n_sig = h_signal[sl].sum()
    n_bkg = h_background[sl].sum()

    if n_sig + n_bkg != 0:
        return -n_sig / math.sqrt(n_sig + n_bkg)
    else:
        return 0


@nb.njit
def start_anneal(initial_state, h_signal, h_background, n_bins, n_dim, Tmin=0.05, Tmax=1_000, steps=500_000, verbose=True):
    state = initial_state.copy()

    numerator = h_signal.sum()
    denominator = (h_background + h_signal).sum()

    E = energy(numerator, denominator)
    print('inital temperature', Tmax)
    print('final temperature', Tmin)
    print('steps', steps)
    print('inital energy', E)

    # E = full_energy(h_signal, h_background, n_bins, state)
    # print(E)

    T_scaling = (Tmin / Tmax) ** (1 / steps)
    T = Tmax

    prevState = state.copy()
    prevEnergy = E
    prevNumerator = numerator
    prevDenominator = denominator

    best_energy = E

    accepted = 0
    improved = 0
    print_every = int(steps / 50)

    for step in range(steps):

        dNum, dDen = update(h_signal, h_background, n_bins, state, n_dim)
        numerator += dNum
        denominator += dDen
        # E = full_energy(h_signal, h_background, n_bins, state)
        E = energy(numerator, denominator)

        # calculate delta e
        dE = E - prevEnergy
        # print(numerator, denominator, E, dE)

        # always accept if dE < 0
        if dE < 0 or math.exp(-dE / T) > random.random():
            # accept new state
            accepted += 1
            if dE < 0.0:
                improved += 1
            prevState = state.copy()

            prevEnergy = E
            prevNumerator = numerator
            prevDenominator = denominator

            if E < best_energy:
                best_state = state.copy()
                best_energy = E
        else:
            # restore previous state
            state = prevState.copy()
            E = prevEnergy
            numerator = prevNumerator
            denominator = prevDenominator

        if (step // print_every) > ((step - 1) // print_every) and verbose:
            print_progress(step / steps, T, E, accepted / print_every, improved / print_every)
            accepted = 0
            improved = 0

        T *= T_scaling

    print('best state\n', best_state.T)
    print('best energy', best_energy, full_energy(h_signal, h_background, n_bins, best_state))

    return best_state, best_energy


def selanneal(initial_state, h_signal, h_background, n_bins, Tmin=0.05, Tmax=1_000, steps=500_000, verbose=True):
    n_dim = len(initial_state)

    code = f"""global get_slice\n@nb.njit\ndef get_slice(state):
    return ({", ".join(f"slice(state[{i}, 0], state[{i}, 1] + 1)" for i in range(n_dim))})
    """
    exec(code)

    best_state, best_energy = start_anneal(initial_state, h_signal, h_background, n_bins, n_dim, Tmin=0.05, Tmax=1_000, steps=500_000, verbose=True)

    return best_state, best_energy
