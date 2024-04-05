"""
Module: process_bias_simulation.py

This module contains functions to simulate and analyze bias in Markov Chains,
specifically focusing on status quo bias.

Author: Ben Harwood
Contact: bharwood@syr.edu
Website: https://github.com/bharwood-data/Markov-Chains

Dependencies:
    - pandas
    - numpy
    - time
    - itertools
    - random
    - bmcUtils (custom module)
    - bmcSpecial (custom module)

Usage:
    Import the module and use the provided functions to simulate and analyze status quo bias in Markov Chains.

"""

import pandas as pd
import numpy as np
import time as time
import itertools
import random

from bmcUtils import generate_markov_chains, apply_self_selection_bias \
    , introduce_mcar_missing_data, extract_transition_matrix, kl_divergence
import bmcSpecial as bmc

def introduce_status_quo_bias(transition_matrix: np.ndarray, num_states: int, num_agents: int, cbf: float, obs: int):
    """Introduces status quo bias into a transition matrix.

    The function modifies a transition matrix by increasing 
    the probability of staying in the same state and decreasing 
    the probability of transitioning to other states unbiasedd on 
    the status quo bias factor. 

    It generates initial states and Markov chain sequences using 
    the biased transition matrix.

    Args:
    transition_matrix (np.ndarray): The original transition matrix.
    num_states (int): Number of states.
    num_agents (int): Number of agents.
    cbf (float): Status quo bias factor controlling bias (0-1).
    obs (int): Number of observations.

    Returns:
    tuple: Tuple of biased transition matrix and generated Markov chains.
    """
    # Validate input arguments
    if not is_valid_transition_matrix(transition_matrix):
        raise ValueError("Invalid transition matrix")
    
    if not isinstance(transition_matrix, np.ndarray) or transition_matrix.ndim != 2:
        raise ValueError("transition_matrix must be a 2D numpy array")

    if not isinstance(num_states, int) or num_states <= 0:
        raise ValueError("num_states must be a positive integer")

    if not isinstance(num_agents, int) or num_agents <= 0:
        raise ValueError("num_agents must be a positive integer")

    if not isinstance(cbf, float) or not 0 <= cbf <= 1:
        raise ValueError("cbf must be a float between 0 and 1")

    if not isinstance(obs, int) or obs <= 0:
        raise ValueError("obs must be a positive integer")

    remaining_attractors = int(num_states * cbf) if num_states * cbf > 1 else 1

    # Apply status quo bias within the transition matrix
    for i, j in itertools.product(range(num_states), range(num_states)):
        if i == j and remaining_attractors > 0:
            # Adjust self-transition probability unbiasedd on status_quo bias factor
            transition_matrix[i][j] *= random.uniform(1.2, 1.5)  # High self-transition probability
            remaining_attractors -= 1
        else:
            # Lower probability of moving to different states
            transition_matrix[i][j] *= random.uniform(0, 0.2 / num_states)  # Adjust disstatus_quo bias factor

    # Normalize the transition matrix
    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)

    initial_states = np.random.choice(num_states, size=num_agents)

    return transition_matrix, generate_markov_chains(transition_matrix, initial_states, obs, num_agents)

def status_quo_Append(states_list, states, missing_list, pct, time_list, standard_time, KL_list, KL, inaccuracy_list, inaccuracy, ss_ratio_list, r,
                   agents_list, N, obs_list, T, imputed_list, cbf_list, cbf, imputed, em_list, em, emTime_list, emTime, imputed_time_list, imputeTime):
    """Appends status_quo bias results to shared lists.

    The function appends states, missing %, RMSE, agents,  
    observations, self-selection type, self-selection ratio,
    status_quo bias factor to their respective shared lists.

    Args:
    shared_lists: Shared lists to append results to
    states: Number of states
    pct: Percent missing data
    rmse: Root mean squared error 
    N: Number of agents
    T: Number of observations
    ss: Self-selection type
    ss_ratio: Self-selection ratio
    cbf: status_quo bias factor
    
    Returns:
    Tuple of updated shared lists
    """
    
    # Type checks
    if not isinstance(states_list, list) or not isinstance(missing_list, list) or not isinstance(time_list, list) \
            or not isinstance(KL_list, list) or not isinstance(inaccuracy_list, list) or not isinstance(ss_ratio_list, list) \
            or not isinstance(agents_list, list) or not isinstance(obs_list, list) or not isinstance(imputed_list, list) \
            or not isinstance(em_list, list) or not isinstance(emTime_list, list) or not isinstance(imputed_time_list, list) \
            or not isinstance(cbf_list, list):
        raise ValueError("All list arguments must be lists.")
    
    if not isinstance(states, int) or not isinstance(pct, float) or not isinstance(standard_time, float) \
            or not isinstance(KL, float) or not isinstance(p, float) or not isinstance(r, float) \
            or not isinstance(N, int) or not isinstance(T, int) or not isinstance(emTime, float) \
            or not isinstance(imputeTime, float):
        raise ValueError("states must be an integer, pct, chi, norm, p, r, N, T, emTime, and imputeTime must be floats.")
    
    if not isinstance(imputed, (list, np.ndarray)) or not isinstance(em, (list, np.ndarray)):
        raise ValueError("imputed and em must be lists or NumPy arrays.")
    
    states_list.append(states)
    missing_list.append(pct)
    time_list.append(standard_time)
    KL_list.append(KL)
    inaccuracy_list.append(inaccuracy)
    ss_ratio_list.append(r)
    agents_list.append(N)
    obs_list.append(T)
    imputed_list.append(imputed)
    em_list.append(em)
    emTime_list.append(emTime)
    imputed_time_list.append(imputeTime)
    cbf_list.append(cbf)
    
    return states_list, missing_list, time_list, KL_list, inaccuracy_list, ss_ratio_list, agents_list, obs_list, cbf_list, imputed_list, em_list, emTime_list, imputed_time_list


def process_status_quo(states, N, T, ss_range, missing_range, cbf_range, imputation = False, optimization = False, em_iterations = None, tol = None):
    """
    Runs status_quo bias simulations and appends results.

    The function introduces status_quo bias, missing data, 
    and self-selection, calculates RMSE, and appends results 
    to shared lists after each run.

    Args:
    states (int): Number of states.
    N (int): Number of agents.
    T (int): Number of time steps.
    ss_range (tuple): Tuple containing the range of self-selection factors.
    missing_range (tuple): Tuple containing the range of missing data percentages.
    cbf_range (tuple): Tuple containing the range of status_quo bias factors.
    imputation (bool, optional): Flag indicating whether to perform imputation. Defaults to False.
    optimization (bool, optional): Flag indicating whether to perform optimization. Defaults to False.
    em_iterations (int, optional): Number of iterations for the EM algorithm. Defaults to None.
    tol (float, optional): Tolerance parameter for optimization algorithms. Defaults to None.

    Returns:
    tuple: Tuple of updated shared lists.
    """

    # Type checks for function arguments
    if not isinstance(states, int) or not isinstance(N, int) or not isinstance(T, int) \
            or not isinstance(ss_range, tuple) or not isinstance(missing_range, tuple) \
            or not isinstance(cbf_range, tuple) or not isinstance(imputation, bool) \
            or not isinstance(optimization, bool) \
            or (em_iterations is not None and not isinstance(em_iterations, int)) \
            or (tol is not None and not isinstance(tol, float)):
        raise ValueError("Incorrect type for one or more arguments.")

    # Check the elements in the ranges
    for val in ss_range + missing_range + cbf_range:
        if not isinstance(val, (int, float)):
            raise ValueError("Elements in ss_range and pct_range must be integers or floats.")
        
    # Initialize lists to store simulation results
    states_list = []
    missing_list = []
    emTime_list = []
    KL_list = []
    inaccuracy_list = []
    ss_ratio_list = []
    agents_list = []
    obs_list = []
    cbf_list = []
    imputed_list = []
    em_list = []
    time_list = []
    imputed_time_list = []
    
    # Initialize transition matrix
    initial_matrix = np.ones([states, states])
    
    # Iterate over combinations of self-selection factor, missing data percentage, and status_quo bias factor
    for r, pct, cbf in list(itertools.product(np.linspace(ss_range[0], ss_range[1]), np.linspace(missing_range[0], missing_range[1]), np.linspace(cbf_range[0], cbf_range[1]))):
        
        # Exclude cases where self-selection is non-zero but missing data percentage is 0
        if not (pct == 0 and r > 0):
            # Introduce status_quo bias into the transition matrix
            observed, data = introduce_status_quo_bias(initial_matrix, states, N, cbf, T)
            # Apply self-selection bias to the data
            ss = apply_self_selection_bias(data, r, N, T, pct)
            # Introduce missing data using MCAR mechanism
            result = pd.DataFrame(introduce_mcar_missing_data(pd.DataFrame(ss), pct))
            # Start time for simulation
            start = time.time()
            # Estimate transition matrix from the data
            estimated = extract_transition_matrix(pd.DataFrame(result), states)
            # End time for simulation
            end = time.time()
            # Perform imputation if flag is set
            final = bmc.forward_algorithm(result, estimated, T, states) if imputation else None
            # Calculate estimated transition matrix after imputation
            estimated_imputed = extract_transition_matrix(final, states) if imputation else None
            # End time for imputation
            end_impute = time.time() if imputation else None
            # Perform optimization using EM algorithm if flag is set
            estimated_em = bmc.em_algorithm(result, N, T, states, em_iterations, tol) if optimization else None
            # End time for EM algorithm
            end_em = time.time() if optimization else None
            
            # Append simulation results to lists
            states_list, missing_list, emTime_list, time_list, inaccuracy_list, ss_ratio_list, agents_list, obs_list, imputed_list, \
                em_list, imputed_time_list = status_quo_Append(
                    states_list, states, missing_list, pct, time_list, end - start, KL_list,
                    kl_divergence(estimated, observed, states), inaccuracy_list,
                    np.linalg.norm(estimated - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)), ss_ratio_list, r,
                    agents_list, N, obs_list, T, cbf_list, cbf, imputed_list if imputation else None,
                    np.linalg.norm(estimated_imputed - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)) if imputation else None,
                    em_list if optimization else None,
                    np.linalg.norm(estimated_em - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)) if optimization else None,
                    emTime_list, end_em - start if optimization else None, imputed_time_list, end_impute - start if imputation else None)
