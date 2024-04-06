"""
Module: unbiased.py

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
    apply_self_selection_bias, introduce_mcar_missing_data, extract_transition_matrix, kl_divergence, is_valid_transition_matrix
import bmcSpecial as bmc

def unbiasedAppend(states_list, states, missing_list, pct, time_list, standard_time, KL_list, norm, inaccuracy_list, inaccuracy, ss_ratio_list, r,
                   agents_list, N, obs_list, T, imputed_list=None, imputed=None, em_list=None, em=None, emTime_list=None, emTime=None, 
              imputed_time_list=None, imputedTime=None):
    
    """
    Appends simulation results to respective lists for unbiased simulation.

    Args:
        states_list (list): List to store the number of states for each simulation.
        states (int): Number of states.
        missing_list (list): List to store the percentage of missing data for each simulation.
        pct (float): Percentage of missing data.
        time_list (list): List to store the execution time for each simulation.
        standard_time (float): Time taken for the simulation.
        KL_list (list): List to store the Kullback-Leibler divergence value for each simulation.
        norm (float): Norm value representing accuracy.
        inaccuracy_list (list): List to store the inaccuracy value for each simulation.
        inaccuracy (float): Inaccuracy value.
        ss_ratio_list (list): List to store the self-similarity ratio for each simulation.
        r (float): Self-similarity ratio.
        agents_list (list): List to store the number of agents for each simulation.
        N (int): Number of agents.
        obs_list (list): List to store the number of observations for each simulation.
        T (int): Number of observations.
        imputed_list (list, optional): List to store the imputation results for each simulation. Defaults to None.
        imputed (list or numpy.ndarray, optional): Imputation results. Defaults to None.
        em_list (list, optional): List to store the EM algorithm results for each simulation. Defaults to None.
        em (list or numpy.ndarray, optional): EM algorithm results. Defaults to None.
        emTime_list (list, optional): List to store the execution time of the EM algorithm for each simulation. Defaults to None.
        emTime (float, optional): Execution time of the EM algorithm. Defaults to None.
        imputed_time_list (list, optional): List to store the execution time of the imputation for each simulation. Defaults to None.
        imputedTime (float, optional): Execution time of the imputation. Defaults to None.

    Returns:
        tuple: Tuple containing updated lists of simulation results.
    """
    # Type checks
    if not isinstance(states_list, list) or not isinstance(missing_list, list) or not isinstance(time_list, list) \
            or not isinstance(KL_list, list) or not isinstance(inaccuracy_list, list) or not isinstance(ss_ratio_list, list) \
            or not isinstance(agents_list, list) or not isinstance(obs_list, list) or not isinstance(imputed_list, list) \
            or not isinstance(em_list, list) or not isinstance(emTime_list, list) or not isinstance(imputed_time_list, list):
        raise ValueError("All list arguments must be lists.")
    
    if not isinstance(states, int) or not isinstance(pct, float) or not isinstance(standard_time, float) \
            or not isinstance(norm, float) or not isinstance(p, float) or not isinstance(r, float) \
            or not isinstance(N, int) or not isinstance(T, int) or not isinstance(emTime, float) \
            or not isinstance(imputedTime, float):
        raise ValueError("states must be an integer, pct, chi, norm, p, r, N, T, emTime, and imputeTime must be floats.")
    
    if not isinstance(imputed, (list, np.ndarray)) or not isinstance(em, (list, np.ndarray)):
        raise ValueError("imputed and em must be lists or NumPy arrays.")
    
    states_list.append(states)
    missing_list.append(pct)
    time_list.append(standard_time)
    KL_list.append(norm)
    inaccuracy_list.append(inaccuracy)
    ss_ratio_list.append(r)
    agents_list.append(N)
    obs_list.append(T)
    # Append imputation result if available
    if imputed_list is not None:
        imputed_list.append(imputed)
    # Append EM result if available
    if em_list is not None:
        em_list.append(em)
    # Append EM execution time if available
    if emTime_list is not None:
        emTime_list.append(emTime)
    # Append imputation execution time if available
    if imputed_time_list is not None:
        imputed_time_list.append(imputedTime)
    
    return states_list, missing_list, time_list, KL_list, inaccuracy_list, ss_ratio_list, agents_list, obs_list, imputed_list, em_list, emTime_list, imputed_time_list

def process_unbiased(states, N, T, ss_range, missing_range, imputation = False, optimization = False, em_iterations = None, tol = None):  
    """
    Runs unbiased scenario simulations and appends results.

    The function generates Markov chains with self-selection bias, introduces missing data,
    estimates transition matrices, performs imputation (if enabled), and optimization (if enabled),
    and appends the results to respective lists after each simulation.

    Args:
        states (int): Number of states.
        N (int): Number of agents.
        T (int): Number of time steps.
        ss_range (list): Range of self-selection ratios.
        missing_range (list): Range of missing data percentages.
        imputation (bool, optional): Flag indicating whether to perform imputation. Defaults to False.
        optimization (bool, optional): Flag indicating whether to perform optimization. Defaults to False.
        em_iterations (int, optional): Number of iterations for the EM algorithm. Defaults to None.
        tol (float, optional): Tolerance parameter for optimization algorithms. Defaults to None.

    Returns:
        tuple: Tuple containing lists of simulation results.
    """

    # Type checks
    if not isinstance(states, int) or not isinstance(N, int) or not isinstance(T, int) \
            or not isinstance(ss_range, list) or not isinstance(missing_range, list) \
            or not isinstance(imputation, bool) or not isinstance(optimization, bool) \
            or (em_iterations is not None and not isinstance(em_iterations, int)) \
            or (tol is not None and not isinstance(tol, float)):
        raise ValueError("Incorrect type for one or more arguments.")
        
    # Check the elements in the ranges
    for val in ss_range + missing_range:
        if not isinstance(val, (int, float)):
            raise ValueError("Elements in ss_range and pct_range must be integers or floats.")


    states_list = []
    missing_list = []
    emTime_list = []
    KL_list = []
    inaccuracy_list = []
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
    
    for r, pct in list(itertools.product(ss_range, missing_range)):
        if not (pct == 0 and r > 0):
            ss = apply_self_selection_bias(chains, r, N, T, pct)    
            result = pd.DataFrame(introduce_mcar_missing_data(ss, pct))
            start = time.time()  
            estimated = extract_transition_matrix(result, states)
            end = time.time()
            final = bmc.forward_algorithm(result, estimated, T, states) if imputation else None
            estimated_imputed = extract_transition_matrix(final, states) if imputation else None
            end_impute = time.time() if imputation else None
            estimated_em = bmc.em_algorithm(result, N, T, states, em_iterations, tol) if optimization else None
            end_em = time.time() if optimization else None
            
            states_list, missing_list, emTime_list, obs_norm_list, inaccuracy_list, ss_ratio_list, agents_list, obs_list, imputed_list, \
                em_list, time_list, imputed_time_list = unbiasedAppend(
                    states_list, states, missing_list, pct, time_list, end - start, KL_list,
                    kl_divergence(estimated, observed, states), inaccuracy_list,
                    np.linalg.norm(estimated - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)), ss_ratio_list, r,
                    agents_list, N, obs_list, T, imputed_list if imputation else None,
                    np.linalg.norm(estimated_imputed - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)) if imputation else None,
                    em_list if optimization else None,
                    np.linalg.norm(estimated_em - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)) if optimization else None,
                    emTime_list, end_em - start if optimization else None, imputed_time_list, end_impute - start if imputation else None)
                    
    return states_list, missing_list, emTime_list, obs_norm_list, inaccuracy_list, ss_ratio_list, agents_list, obs_list, imputed_list, em_list, time_list, imputed_time_list
