import numpy as np
import random
import math
import numba as nb


@nb.njit
def update(h_signal, h_background, n_bins, state, n_dim, tup):

    # change state
    a = random.randint(10 - n_dim, 10 - 1)
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
    # print(a, b, state[a, b])

    if state[a, b] > n_bins - 1:
        state[a, b] = n_bins - 1
        return 0, 0
    elif state[a, b] < 0:
        state[a, b] = 0
        return 0, 0

    assert(state[a, 0] < state[a, 1])
    assert(c == -1 or c == 1)

    # calculate delta numerator, denominator

    # sel_signal = h_signal
    # sel_background = h_background
    #
    # lb_a, ub_a = state[a]
    # state[a] = [prevState, prevState]
    #
    # for lb, ub in state:
    #     sel_signal = np.expand_dims(np.sum(sel_signal[lb:ub + 1], axis=0), axis=-1)
    #     sel_background = np.expand_dims(np.sum(sel_background[lb:ub + 1], axis=0), axis=-1)
    #
    # dNum = sign * sel_signal.item()
    # dDen = sign * sel_background.item() + dNum
    # state[a] = [lb_a, ub_a]

    # transpose solution, faster
    sel_signal = h_signal
    sel_background = h_background

    lb_a, ub_a = state[a]
    state[a] = [prevState, prevState]

    for i, (lb, ub) in enumerate(state):
        sel_signal = np.transpose(sel_signal, axes=tup[i])[lb:ub + 1]
        sel_background = np.transpose(sel_background, axes=tup[i])[lb:ub + 1]

    dNum = sign * sel_signal.sum()
    dDen = sign * sel_background.sum() + dNum
    state[a] = [lb_a, ub_a]

    # # 2D only
    # if a == 0:
    #     lb, ub = state[1]
    #     sel_signal = h_signal[prevState, lb:ub + 1]
    #     sel_background = h_background[prevState, lb:ub + 1]
    # elif a == 1:
    #     lb, ub = state[0]
    #     sel_signal = h_signal[lb:ub + 1, prevState]
    #     sel_background = h_background[lb:ub + 1, prevState]
    # dNum = sign * np.sum(sel_signal)
    # dDen = sign * np.sum(sel_background) + dNum

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
def full_energy(h_signal, h_background, n_bins, state, tup):

    # sel_signal = h_signal
    # sel_background = h_background
    # for lb, ub in state:
    #     sel_signal = np.expand_dims(np.sum(sel_signal[lb:ub + 1], axis=0), axis=-1)
    #     sel_background = np.expand_dims(np.sum(sel_background[lb:ub + 1], axis=0), axis=-1)
    #
    # n_sig = sel_signal.item()
    # n_bkg = sel_background.item()

    sel_signal = h_signal
    sel_background = h_background
    for i, (lb, ub) in enumerate(state):
        sel_signal = np.transpose(sel_signal, axes=tup[i])[lb:ub + 1]
        sel_background = np.transpose(sel_background, axes=tup[i])[lb:ub + 1]
    n_sig = sel_signal.sum()
    n_bkg = sel_background.sum()

    if n_sig + n_bkg != 0:
        return -n_sig / math.sqrt(n_sig + n_bkg)
    else:
        return 0


@nb.njit
def selanneal(initial_state, h_signal, h_background, n_bins, n_dim, Tmin=0.05, Tmax=1_000, steps=500_000, verbose=True):
    tup = [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
           (1, 0, 2, 3, 4, 5, 6, 7, 8, 9),
           (2, 1, 0, 3, 4, 5, 6, 7, 8, 9),
           (3, 1, 2, 0, 4, 5, 6, 7, 8, 9),
           (4, 1, 2, 3, 0, 5, 6, 7, 8, 9),
           (5, 1, 2, 3, 4, 0, 6, 7, 8, 9),
           (6, 1, 2, 3, 4, 5, 0, 7, 8, 9),
           (7, 1, 2, 3, 4, 5, 6, 0, 8, 9),
           (8, 1, 2, 3, 4, 5, 6, 7, 0, 9),
           (9, 1, 2, 3, 4, 5, 6, 7, 8, 0)]

    state = initial_state.copy()

    numerator = h_signal.sum()
    denominator = (h_background + h_signal).sum()

    E = energy(numerator, denominator)
    print('inital temperature', Tmax)
    print('final temperature', Tmin)
    print('steps', steps)
    print('inital energy', E)

    # E = full_energy(h_signal, h_background, n_bins, state, tup)
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

        dNum, dDen = update(h_signal, h_background, n_bins, state, n_dim, tup)
        numerator += dNum
        denominator += dDen
        # E = full_energy(h_signal, h_background, n_bins, state, tup)
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
    print('best energy', best_energy, full_energy(h_signal, h_background, n_bins, best_state, tup))

    return best_state, best_energy
