import numpy as np
from numba import jit

''' 関数の定義 '''
@jit
def calc_I(N, gamma):
    T = N.shape[0]
    if N[0] != 0:
        I = np.ones(T) * N[0]
    else:
        I = np.ones(T)
        
    for i in range(T-1):
        I[i+1] = (1 - gamma)*I[i] + N[i]
    return I

def calc_beta(S, I, N, P):
    T = N.shape[0]
    beta = np.zeros(T)
    beta[1:] = N[1:] / (S[:-1] * I[:-1]) * P
    return beta

def pred_linear(y, W, T):
    y = y[-W:]
    x = np.arange(-W+1, 1)
    fit = np.polyfit(x, y, 1)
    f = np.poly1d(fit)
    x_pred = np.arange(0, T+1)
    y_pred = f(x_pred)
    return y_pred

def calc_beta_pred(beta, beta_target, W, T):
    if (W == 0) | (beta_target == 0):
        beta_pred = np.ones(T+1) * beta_target
    else:
        beta_bin = (beta_target - beta[-1]) / W
        if beta[-1] < beta_target:
            beta_pred = np.arange(beta[-1], 10, beta_bin)
            beta_pred = beta_pred[:T+1]
            beta_pred[beta_pred > beta_target] = beta_target
        else:
            beta_pred = np.arange(beta[-1], -10, beta_bin)
            beta_pred = beta_pred[:T+1]
            beta_pred[beta_pred < beta_target] = beta_target
    return beta_pred

@jit
def SIR(S, I, N, V_pred, P, beta, gamma, T):
    S_pred = np.zeros(T+1)
    I_pred = np.zeros(T+1)
    N_pred = np.zeros(T+1)
    S_pred[0] = S[-1]
    I_pred[0] = I[-1]
    N_pred[0] = N[-1]

    for i in range(T):
        S_pred[i+1] = S_pred[i] - N_pred[i] - (V_pred[i+1] - V_pred[i])
        I_pred[i+1] = (1 - gamma)*I_pred[i] + N_pred[i]
        N_pred[i+1] = beta[i+1] * S_pred[i+1] / P * I_pred[i+1]
    return S_pred, I_pred, N_pred

@jit
def pred_H_weekly(N_weekly_pred, H_weekly, gamma_H, delta_H_weekly_pred, T):
    H_weekly_pred = np.ones(T+1)
    H_weekly_pred[0] = H_weekly[-1]
    for i in range(T):
        H_weekly_pred[i+1] = H_weekly_pred[i] \
        + N_weekly_pred[i] * delta_H_weekly_pred[i] - H_weekly_pred[i] * gamma_H * 7
    return H_weekly_pred

@jit
def pred_ICU_weekly(ICU_weekly, H_weekly_pred, gamma_ICU, delta_ICU_weekly_pred, T):
    ICU_weekly_pred = np.ones(T+1)
    ICU_weekly_pred[0] = ICU_weekly[-1]
    for i in range(T):
        ICU_weekly_pred[i+1] = ICU_weekly_pred[i] \
        + H_weekly_pred[i] * delta_ICU_weekly_pred[i] - ICU_weekly_pred[i] * gamma_ICU * 7
    return ICU_weekly_pred

@jit
def pred_dD_weekly(dD_weekly, H_weekly_pred, delta_D_weekly_pred, T):
    dD_weekly_pred = np.ones(T+1)
    dD_weekly_pred[0] = dD_weekly[-1]
    for i in range(T):
        dD_weekly_pred[i+1] = H_weekly_pred[i] * delta_D_weekly_pred[i]
    return dD_weekly_pred