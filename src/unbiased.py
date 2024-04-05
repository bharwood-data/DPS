"""
unbiased.py

This module contains functions for conducting unbiased scenario simulations
in the context of Markov chain simulations with missing data.


Author: Ben Harwood
Contact: bharwood@syr.edu
Website: https://github.com/bharwood-data/Markov-Chains

Functions:
- unbiasedAppend(states_list, states, missing_list, pct, time_list, chi, obs_norm_list, norm, p_list, p, ss_ratio_list, r,
                 agents_list, N, obs_list, T, imputed_list, imputed, em_list, em, emTime_list, emTime, imputed_time_list, imputeTime):
    Appends unbiased scenario simulation results to lists.
- process_unbiased(states, N, T, ss_range, pct_range, imputation=False, optimization=False, em_iterations=None, tol=None):
    Processes unbiased scenario runs for agent-unbiased simulations.

Dependencies:
- pandas (pd)
- numpy (np)
- time (time)
- itertools
- bmcUtils (custom module)
- bmcSpecial (custom module)
"""

import pandas as pd
import numpy as np
import time as time
import itertools

from bmcUtils import generate_standard_transition_matrix, generate_initial_states, generate_markov_chains, \
    apply_self_selection_bias, introduce_mcar_missing_data, extract_transition_matrix, kl_divergence
from bmcSpecial import forward_algorithm, em_algorithm

def unbiasedAppend(states_list, states, missing_list, pct, time_list, chi, obs_norm_list, norm, p_list, p, ss_ratio_list, r,
                   agents_list, N, obs_list, T, imputed_list, imputed, em_list, em, emTime_list, emTime, imputed_time_list, imputeTime):
    """Appends unbiased scenario simulation results to lists.

    The function appends states, missing data %, RMSE, self-selection noise, 
    self-selection ratio, number of agents, and number of observations to their 
    respective lists after each run.

    Args:
    states_list (list): List to append states to.
    states (int): Number of states.
    missing_list (list): List to append missing data percentages to.
    pct (float): Percent of missing data.
    time_list (list): List to append simulation times to.
    chi (float): Time taken for simulation.
    obs_norm_list (list): List to append KL divergence values to.
    norm (float): KL divergence.
    p_list (list): List to append inaccuracies to.
    p (float): Inaccuracy.
    ss_ratio_list (list): List to append self-selection ratios to.
    r (float): Self-selection ratio.
    agents_list (list): List to append number of agents to.
    N (int): Number of agents.
    obs_list (list): List to append number of observations to.
    T (int): Number of observations.
    imputed_list (list): List to append imputed data to.
    imputed (array-like): Imputed data.
    em_list (list): List to append EM algorithm results to.
    em (array-like): EM algorithm results.
    emTime_list (list): List to append EM algorithm execution times to.
    emTime (float): EM algorithm execution time.
    imputed_time_list (list): List to append imputation execution times to.
    imputeTime (float): Imputation execution time.
    
    Returns: 
    tuple: Tuple of updated lists.
    
    Raises:
    ValueError: If any of the input arguments are of incorrect type.
    """
    # Type checks
    if not isinstance(states_list, list) or not isinstance(missing_list, list) or not isinstance(time_list, list) \
            or not isinstance(obs_norm_list, list) or not isinstance(p_list, list) or not isinstance(ss_ratio_list, list) \
            or not isinstance(agents_list, list) or not isinstance(obs_list, list) or not isinstance(imputed_list, list) \
            or not isinstance(em_list, list) or not isinstance(emTime_list, list) or not isinstance(imputed_time_list, list):
        raise ValueError("All list arguments must be lists.")
    
    if not isinstance(states, int) or not isinstance(pct, float) or not isinstance(chi, float) \
            or not isinstance(norm, float) or not isinstance(p, float) or not isinstance(r, float) \
            or not isinstance(N, int) or not isinstance(T, int) or not isinstance(emTime, float) \
            or not isinstance(imputeTime, float):
        raise ValueError("states must be an integer, pct, chi, norm, p, r, N, T, emTime, and imputeTime must be floats.")
    
    if not isinstance(imputed, (list, np.ndarray)) or not isinstance(em, (list, np.ndarray)):
        raise ValueError("imputed and em must be lists or NumPy arrays.")
    
    states_list.append(states)
    missing_list.append(pct)
    time_list.append(chi)
    obs_norm_list.append(norm)
    p_list.append(p)
    ss_ratio_list.append(r)
    agents_list.append(N)
    obs_list.append(T)
    imputed_list.append(imputed)
    em_list.append(em)
    emTime_list.append(emTime)
    imputed_time_list.append(imputeTime)
    
    return states_list, missing_list, time_list, obs_norm_list, p_list, ss_ratio_list, agents_list, obs_list, imputed_list, em_list, emTime_list, imputed_time_list

def process_unbiased(states, N, T, ss_range, pct_range, imputation = False, optimization = False, em_iterations = None, tol = None):  
    """
    Processes unbiased scenario runs for agent-unbiased simulations.

    Generates transition matrices, Markov chains, and introduces missing data. 
    Calculates RMSE and appends results to lists after each run.

    Args:
    states (int): Number of states.
    N (int): Number of agents.
    T (int): Number of time steps.
    ss_range (tuple): Range of self-selection ratios (default: (0, 0.8)).
    pct_range (tuple): Range of missing data percentages (default: (0, 0.8)).
    imputation (bool): Flag to perform imputation (default: True).
    optimization (bool): Flag to perform optimization (default: True).
    em_iterations (int): Number of iterations for EM algorithm (default: 1000).
    tol (float): Tolerance for EM algorithm convergence (default: None).

    Returns:
    Tuple of updated lists.
    
    Raises:
    ValueError: If any of the input arguments are of incorrect type.
    """

    # Type checks
    if not isinstance(states, int) or not isinstance(N, int) or not isinstance(T, int) \
            or not isinstance(ss_range, tuple) or not isinstance(pct_range, tuple) \
            or not isinstance(imputation, bool) or not isinstance(optimization, bool) \
            or (em_iterations is not None and not isinstance(em_iterations, int)) \
            or (tol is not None and not isinstance(tol, float)):
        raise ValueError("Incorrect type for one or more arguments.")
    
    # Unpack ranges
    if len(ss_range) != 2 or len(pct_range) != 2:
        raise ValueError("ss_range and pct_range must be tuples of length 2.")
    
    # Check the elements in the ranges
    for val in ss_range + pct_range:
        if not isinstance(val, (int, float)):
            raise ValueError("Elements in ss_range and pct_range must be integers or floats.")


    states_list = []
    missing_list = []
    emTime_list = []
    obs_norm_list = []
    p_list = []
    ss_ratio_list = []
    agents_list = []
    obs_list = []
    imputed_list = []
    em_list = []
    time_list = []
    imputed_time_list = []
    
    observed = generate_standard_transition_matrix(states)
    initial_probs = np.sum(observed, axis=0)
    initial_probs /= sum(initial_probs)
    initial_states = generate_initial_states(observed, N, initial_probs)
    chains = generate_markov_chains(observed, initial_states, T, N)
    
    for r, pct in list(itertools.product(np.linspace(ss_range[0], ss_range[1]), np.linspace(pct_range[0], pct_range[1]))):
        if not (pct == 0 and r > 0):
            ss = apply_self_selection_bias(chains, r, N, T, pct)    
            result = pd.DataFrame(introduce_mcar_missing_data(ss, pct))
            start = time.time()  
            estimated = extract_transition_matrix(result, states)
            end = time.time()
            final = forward_algorithm(result, estimated, T, states) if imputation else None
            estimated_imputed = extract_transition_matrix(final, states) if imputation else None
            end_impute = time.time() if imputation else None
            estimated_em = em_algorithm(result, N, T, states, em_iterations, tol) if optimization else None
            end_em = time.time() if optimization else None
            
            states_list, missing_list, emTime_list, obs_norm_list, p_list, ss_ratio_list, agents_list, obs_list, imputed_list, \
                em_list, time_list, imputed_time_list = unbiasedAppend(
                    states_list, states, missing_list, pct, time_list, end - start, obs_norm_list,
                    kl_divergence(estimated, observed, states), p_list,
                    np.linalg.norm(estimated - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)), ss_ratio_list, r,
                    agents_list, N, obs_list, T, imputed_list if imputation else None,
                    np.linalg.norm(estimated_imputed - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)) if imputation else None,
                    em_list if optimization else None,
                    np.linalg.norm(estimated_em - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)) if optimization else None,
                    emTime_list, end_em - start if optimization else None, imputed_time_list, end_impute - start if imputation else None)
                    
    return states_list, missing_list, emTime_list, obs_norm_list, p_list, ss_ratio_list, agents_list, obs_list, imputed_list, em_list, time_list, imputed_time_list
