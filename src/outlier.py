'''
    Module: outlier_bias.py

    This module contains functions for introducing outlier bias in Markov chain simulations.

    Author: Ben Harwood
    Contact: bharwood@syr.edu
    Website: https://github.com/bharwood-data/Markov-Chains

    Functions:
    - introduce_missing_data_less_vocal(data, less_vocal_group, missing_prob):
        Introduces missing values into a dataset for a specified subset of agents.

    - introduce_outlier_bias(chains, outliers, group4, loud, missing_prob):
        Introduces biased initial states and transitions in a Markov chain simulation.

    - generate_outlier_matrix(num_states, outlier_state1, outlier_state2):
        Generates a transition probability matrix with biased probabilities for specified outlier states.

    - outlier_append(shared_lists, states, missing_percentage, standard_time, KL, 
                    inaccuracy, N, T, imputed_data, em_data, em_time, imputation_time, vocal_group):
        Appends outlier scenario results to shared lists.

    - process_outlier(states, N, T, unaff, missing_range, imputation=False, optimization=False, em_iterations=None, tol=None):
        Runs outlier scenario simulations and appends results.

    Dependencies:
    - pandas (pd)
    - numpy (np)
    - time (time)
    - itertools
    - random
    - bmcUtils (custom module)
    - bmcSpecial (custom module)
'''

import pandas as pd
import numpy as np
import time as time
import itertools
import random

from bmcUtils import generate_standard_transition_matrix, generate_markov_chains, \
     extract_transition_matrix, kl_divergence
import bmcSpecial as bmc

def introduce_missing_data_less_vocal(data, less_vocal_group, missing_prob):
    """
    Introduces missing values into a dataset for a specified subset of agents.

    Randomly replaces values with NaN based on a missing probability.

    Args:
        data (pd.DataFrame): Input dataset.
        less_vocal_group (list): Subset of agents to introduce missing values for.
        missing_prob (float): Probability of replacing a value with NaN.

    Returns:
        pd.DataFrame: Dataset with missing values introduced.

    Raises:
        ValueError: If the data is not a DataFrame, less_vocal_group is not a list,
                    missing_prob is not a float, or any unsupported operations are encountered.
    """
    try:
        # Validate data type
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a DataFrame")

        # Validate less_vocal_group type
        if not isinstance(less_vocal_group, list):
            raise ValueError("Less_vocal_group must be a list of agent indices")

        # Validate missing_prob type and value
        if not isinstance(missing_prob, float) or not 0 <= missing_prob <= 1:
            raise ValueError("Missing_prob must be a float between 0 and 1")

        # Select subset of agents from the dataset
        subset_data = data.iloc[less_vocal_group]

        # Calculate the number of missing values needed
        N = len(less_vocal_group)
        T = data.shape[1]
        needed_missing = int(missing_prob * N * T)

        # Flatten the subset data into a 1D array
        flat = subset_data.values.flatten().astype(float)

        # Randomly select indices to mark as NaN
        indices = np.random.choice(len(flat), size=needed_missing, replace=False)

        # Mark the selected indices as NaN in the flattened array
        np.put(flat, indices, np.nan)

        # Reshape the modified array and create a new DataFrame
        new_df = pd.DataFrame(flat.reshape(-1, len(subset_data.columns)), columns=subset_data.columns)

        # Combine the modified subset with the rest of the data
        final = pd.concat([new_df, data.iloc[~data.index.isin(less_vocal_group)]])

        # Reset the index of the final DataFrame
        final.reset_index(drop=True, inplace=True)

        return final

    except Exception as e:
        raise ValueError(f"Error in introducing missing data: {e}")
    
def introduce_outlier_bias(chains, outliers, group4, loud, missing_prob):
    """
    Introduces biased initial states and transitions in a Markov chain simulation. 

    Agents belonging to specified groups start in particular outlier states. 
    The transition matrix introduces sticky self-transitions for the outlier states.

    A Markov chain simulation is run using the biased initial states and transition matrix.

    Args:
        chains (DataFrame): Input Markov chains simulation.
        outliers (list): List of agents belonging to groups with outlier states.
        group4 (list): List of agents belonging to the fourth group.
        loud (str): Specifies whether to introduce missing data for loud or less vocal groups.
        missing_prob (float): Probability of introducing missing data.

    Returns:
        DataFrame: Simulation output from generate_markov_chains() with introduced missing data.
    """
    try:
        # Check if chains is a DataFrame
        if not isinstance(chains, pd.DataFrame):
            raise ValueError("Chains must be a DataFrame")

        # Check if outliers, group4 are lists
        if not isinstance(outliers, list) or not isinstance(group4, list):
            raise ValueError("Outliers and group4 must be lists of agent indices")

        # Check if loud is a string
        if not isinstance(loud, str):
            raise ValueError("Loud must be a string specifying the group type")

        # Check if missing_prob is a float
        if not isinstance(missing_prob, float) or not 0 <= missing_prob <= 1:
            raise ValueError("Missing_prob must be a float between 0 and 1")

        if missing_prob == 0:
            return chains
        elif loud == 'min':
            return introduce_missing_data_less_vocal(chains, group4, missing_prob)
        else:
            return introduce_missing_data_less_vocal(chains, outliers, missing_prob)
    except Exception as e:
        raise ValueError(f"Error in introducing outlier bias: {e}")   

def generate_outlier_matrix(num_states, outlier_state1, outlier_state2):
    """
    Generates a transition probability matrix with biased probabilities for specified outlier states.

    The transition probability matrix introduces high self-transition probabilities for the specified outlier states,
    and low transition probabilities from other states to the outlier states. This creates a bias where the outlier states are sticky.

    The matrix is normalized so each row sums to 1.

    Args:
        num_states (int): Total number of states.
        outlier_state1 (int): First outlier state index.
        outlier_state2 (int): Second outlier state index.

    Returns:
        numpy.ndarray: num_states x num_states transition probability matrix.

    Raises:
        ValueError: If num_states is not a positive integer,
                    or if outlier_state1 or outlier_state2 are out of bounds.
    """
    try:
        # Validate num_states
        if not isinstance(num_states, int) or num_states <= 0:
            raise ValueError("num_states must be a positive integer")

        # Validate outlier_state1 and outlier_state2
        if outlier_state1 < 0 or outlier_state1 >= num_states or outlier_state2 < 0 or outlier_state2 >= num_states:
            raise ValueError("outlier_state1 and outlier_state2 must be within the range of num_states")

        p = generate_standard_transition_matrix(num_states)
        
        # Modify transition probabilities for outlier states, high values ensure that self-transitions remain high after normalization
        p[outlier_state1, outlier_state1] = 5
        p[outlier_state2, outlier_state2] = 5
        p[outlier_state1, outlier_state2] = 0
        p[outlier_state2, outlier_state1] = 0
        
        # Normalize transition matrix
        p /= np.sum(p, axis=1)

        return p
    except Exception as e:
        raise ValueError(f"Error in generating outlier matrix: {e}")
    
def outlier_append(shared_lists: tuple, states: int, missing_percentage: float, standard_time: float, KL: float, 
                   inaccuracy: float, N: int, T: int, imputed_data: pd.DataFrame, em_data: pd.DataFrame, 
                   em_time: float, imputation_time: float, vocal_group: str) -> tuple:
    """
    Appends outlier scenario results to shared lists.

    Args:
        shared_lists (tuple): Tuple of shared lists to append results to.
        states (int): Number of states.
        missing_percentage (float): Percentage of missing data.
        standard_time (float): Time taken for the simulation.
        KL (float): Kullback-Leibler divergence value.
        inaccuracy (float): Inaccuracy value.
        N (int): Number of agents.
        T (int): Number of observations.
        imputed_data (pd.DataFrame): DataFrame containing imputed data.
        em_data (pd.DataFrame): DataFrame containing EM algorithm results.
        em_time (float): Time taken for EM algorithm.
        imputation_time (float): Time taken for imputation.
        vocal_group (str): Vocal group type.

    Returns:
        tuple: Tuple of updated shared lists.
    """
    try:
        # Unpack shared lists
        states_list, missing_list, time_list, KL_list, inaccuracy_list, agents_list, obs_list, imputed_list, \
            em_list, emTime_list, imputed_time_list, vocal_list = shared_lists
        
        # Type checking for shared_lists
        if not isinstance(shared_lists, tuple):
            raise TypeError("shared_lists must be a tuple")

        # Type checking and value validation for each argument
        if not isinstance(states, int) or states <= 0:
            raise ValueError("states must be a positive integer")
        if not isinstance(missing_percentage, float) or not 0 <= missing_percentage <= 100:
            raise ValueError("missing_percentage must be a float between 0 and 100")
        if not isinstance(standard_time, float) or standard_time < 0:
            raise ValueError("standard_time must be a non-negative float")
        if not isinstance(KL, float) or KL < 0:
            raise ValueError("KL must be a non-negative float")
        if not isinstance(inaccuracy, float) or inaccuracy < 0:
            raise ValueError("inaccuracy must be a non-negative float")
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer")
        if not isinstance(T, int) or T <= 0:
            raise ValueError("T must be a positive integer")
        if not isinstance(imputed_data, pd.DataFrame):
            raise TypeError("imputed_data must be a pandas DataFrame")
        if not isinstance(em_data, pd.DataFrame):
            raise TypeError("em_data must be a pandas DataFrame")
        if not isinstance(em_time, float) or em_time < 0:
            raise ValueError("em_time must be a non-negative float")
        if not isinstance(imputation_time, float) or imputation_time < 0:
            raise ValueError("imputation_time must be a non-negative float")
        if not isinstance(vocal_group, str):
            raise TypeError("vocal_group must be a string")

        # Append results to shared lists
        states_list.append(states)
        missing_list.append(missing_percentage)
        time_list.append(standard_time)
        KL_list.append(KL)
        inaccuracy_list.append(inaccuracy)
        agents_list.append(N)
        obs_list.append(T)
        imputed_list.append(imputed_data)
        em_list.append(em_data)
        emTime_list.append(em_time)
        imputed_time_list.append(imputation_time)
        vocal_list.append(vocal_group)

        # Return updated shared lists
        return states_list, missing_list, time_list, KL_list, inaccuracy_list, agents_list, \
            obs_list, imputed_list, em_list, emTime_list, imputed_time_list, vocal_list
    
    except Exception as e:
        raise ValueError(f"Error in appending outlier scenario results: {e}")

def process_outlier(states, N, T, unaff, missing_range, imputation=False, optimization=False, em_iterations=None, tol=None):
    """
    Runs outlier scenario simulations and appends results.

    The function assigns agents to groups, generates Markov chains, 
    introduces outlier bias, calculates RMSE, and appends results to 
    shared lists after each run.

    Args:
        states (int): Number of states
        N (int): Number of agents
        T (int): Number of time steps
        unaff (float): Proportion of unaffiliated agents
        missing_range (tuple): Range of missing data percentages
        imputation (bool, optional): Flag indicating whether to perform imputation. Defaults to False
        optimization (bool, optional): Flag indicating whether to perform optimization. Defaults to False
        em_iterations (int, optional): Number of iterations for the EM algorithm. Defaults to None
        tol (float, optional): Tolerance parameter for optimization algorithms. Defaults to None
    
    Returns:
        tuple: Tuple of updated shared lists
    """
    try:
        # Type checking and value validation for input arguments
        if not isinstance(states, int) or states <= 0:
            raise ValueError("states must be a positive integer")
        if not isinstance(N, int) or N <= 0:
            raise ValueError("N must be a positive integer")
        if not isinstance(T, int) or T <= 0:
            raise ValueError("T must be a positive integer")
        if not isinstance(unaff, float) or not 0 <= unaff <= 1:
            raise ValueError("unaff must be a float between 0 and 1")
        if not isinstance(missing_range, tuple) or len(missing_range) != 2 or not all(isinstance(val, float) for val in missing_range):
            raise ValueError("missing_range must be a tuple of two floats")
        if not isinstance(imputation, bool):
            raise TypeError("imputation must be a boolean")
        if not isinstance(optimization, bool):
            raise TypeError("optimization must be a boolean")
        if em_iterations is not None and (not isinstance(em_iterations, int) or em_iterations <= 0):
            raise ValueError("em_iterations must be a positive integer or None")
        if tol is not None and (not isinstance(tol, float) or tol < 0):
            raise ValueError("tol must be a non-negative float or None")
        
        # Initialize shared lists to store simulation results
        states_list = []
        missing_list = []
        emTime_list = []
        KL_list = []
        inaccuracy_list = []
        vocal_list = []
        agents_list = []
        obs_list = []
        imputed_list = []
        em_list = []
        time_list = []
        imputed_time_list = []
        
        # Generate random agents and groups
        agents = list(range(N))
        outliers = np.random.choice(agents, int(N*unaff), replace = False)
        group2 = np.random.choice(outliers, len(outliers) // 2, replace = False)
        group3 = np.random.choice([x for x in outliers if x not in group2], len(outliers) - len(group2), replace = False)
        group4 = [x for x in agents if x not in outliers]

        # Generate outlier states
        outlier_state1, outlier_state2 = np.random.choice(range(states), 2, replace=False)

        # Generate transition matrix for observed data
        observed = generate_outlier_matrix(states, outlier_state1, outlier_state2)

        # Generate initial states for each agent
        initial_states = []
        for i in range(N):
            if i in group2:
                initial_states.append(outlier_state1)
            elif i in group3:
                initial_states.append(outlier_state2)
            else:
                initial_states.append(random.choice([j for j in range(states) if j not in [outlier_state1, outlier_state2]]))

        # Generate Markov chains
        chains = generate_markov_chains(observed, initial_states, T, N)
        
        # Iterate over vocal groups and missing percentage ranges
        for loud, pct in list(itertools.product(["min", 'maj'], np.linspace(missing_range[0], missing_range[1]))):
            # Introduce outlier bias and record time
            result = pd.DataFrame(introduce_outlier_bias(chains, outliers, group4, loud, pct))
            start = time.time()
            
            # Estimate transition matrix
            estimated = extract_transition_matrix(result, states)
            end = time.time()
            
            # Perform imputation if enabled
            final = bmc.forward_algorithm(result, estimated, T, states) if imputation else None
            estimated_imputed = extract_transition_matrix(final, states) if imputation else None
            end_impute = time.time() if imputation else None
            
            # Perform optimization if enabled
            estimated_em = bmc.em_algorithm(result, N, T, states, em_iterations, tol) if optimization else None
            end_em = time.time() if optimization else None
            
            # Append results to shared lists
            states_list, missing_list, time_list, KL_list, inaccuracy_list, agents_list, obs_list, imputed_list, \
                em_list, emTime_list, imputed_time_list, vocal_list = outlier_append(
                        states_list, states, missing_list, pct, time_list, end - start, KL_list,
                        kl_divergence(estimated, observed, states), inaccuracy_list,
                        np.linalg.norm(estimated - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)), 
                        agents_list, N, obs_list, T, imputed_list if imputation else None,
                        np.linalg.norm(estimated_imputed - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)) if imputation else None,
                        em_list if optimization else None,
                        np.linalg.norm(estimated_em - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)) if optimization else None,
                        emTime_list, end_em - start if optimization else None, imputed_time_list, end_impute - start if imputation else None,
                        vocal_list, loud)
                        
        return states_list, missing_list, time_list, KL_list, inaccuracy_list, agents_list, obs_list, imputed_list, em_list, emTime_list, imputed_time_list, vocal_list

    except Exception as e:
        raise ValueError(f"Error in processing outlier scenario: {e}")