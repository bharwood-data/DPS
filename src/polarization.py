import pandas as pd
import numpy as np
import time
import itertools
import random 

from bmcUtils import generate_standard_transition_matrix, generate_initial_states, generate_markov_chains, \
    apply_self_selection_bias, introduce_mcar_missing_data, extract_transition_matrix, kl_divergence, is_valid_transition_matrix
import bmcSpecial as bmc


def divide_list_into_groups(input_list, clusters):
    """
    Randomly divides the input list into multiple groups.

    Args:
        input_list (list): The input list to be divided.
        clusters (int): The number of groups to create.

    Returns:
        list: A list of sublists where the input list is randomly divided into clusters.

    Raises:
        ValueError: If input_list is not a list or if clusters is not a positive integer.
    """
    # Validate input_list
    if not isinstance(input_list, list):
        raise ValueError("input_list must be a list")

    # Validate clusters
    if not isinstance(clusters, int) or clusters <= 0:
        raise ValueError("clusters must be a positive integer")
    if clusters > len(input_list):
        raise ValueError("can't have more clusters than states")

    try:
        shuffled_states = np.random.permutation(input_list)  # Shuffle the input list randomly
        extra = len(shuffled_states) % clusters
        sublist_size = len(shuffled_states) // clusters

        states_per_group = [sublist_size] * clusters
        for i in range(extra):
            states_per_group[i] += 1

        state_groups = []  # List to store state groups
        start_index = 0  # Start index for slicing the shuffled list

        # Loop through the number of states per group
        for num_states in states_per_group:
            # Slice the shuffled list to get the states for the current group
            group_states = shuffled_states[start_index:start_index + num_states]

            # Append the current group to the list of state groups
            state_groups.append(group_states)

            # Update the start index for the next group
            start_index += num_states

        return state_groups
    except Exception as e:
        raise ValueError(f"An error occurred during list division: {e}")
    
def divide_agents(agents, unaff_pct):
    """
    Divides a list of agents into polarized and unaffiliated groups unbiasedd on the unaffiliated percentage.

    Args:
        agents (list): The list of all agents.
        unaff_pct (float): The percentage of agents that should be unaffiliated.

    Returns:
        tuple: A tuple containing the list of polarized agents and the list of unaffiliated agents.

    Raises:
        ValueError: If agents is not a list or if unaff_pct is not a float between 0 and 1.
    """
    # Validate agents
    if not isinstance(agents, list):
        raise ValueError("agents must be a list")

    # Validate unaff_pct
    if not isinstance(unaff_pct, (int, float)) or not 0 <= unaff_pct <= 1:
        raise ValueError("unaff_pct must be a float between 0 and 1")

    try:
        num_unaff = int(len(agents) * unaff_pct)
        unaff_agents = np.random.choice(agents, num_unaff, replace=False)
        polar_agents = [a for a in agents if a not in unaff_agents]
        return polar_agents, unaff_agents
    except Exception as e:
        raise ValueError(f"An error occurred during agent division: {e}")

def sample_states(remaining, num):
    """
    Randomly samples unique states from a list without replacement.

    Args:
        remaining (list): List of remaining states to sample from.
        num (int): Number of states to sample.

    Returns:
        list: List of randomly sampled unique states.

    Raises:
        ValueError: If remaining is not a list or if num is not a positive integer.
        ValueError: If num is greater than the length of remaining.
    """
    # Validate remaining
    if not isinstance(remaining, list):
        raise ValueError("remaining must be a list")

    # Validate num
    if not isinstance(num, int) or num <= 0:
        raise ValueError("num must be a positive integer")
    if num > len(remaining):
            raise ValueError("num must be less than or equal to the length of remaining")
    try:
        return np.sort(np.random.choice(remaining, num, replace=False))
    except Exception as e:
        raise ValueError(f"An error occurred during state sampling: {e}")
    
def inter_polar(inter_states, polar_matrix, inter_prob):
    """
    Adds inter-group connections to a polarization matrix.

    Args:
        inter_states (list): List of states to connect between groups.
        polar_matrix (np.array): Polarization transition matrix.
        inter_prob (float): Probability of inter-group transitions.

    Returns:
        np.array: Updated matrix with inter-group connections.

    Raises:
        ValueError: If inter_states is not a list, polar_matrix is not a NumPy array,
                    or inter_prob is not a float between 0 and 1.
    """
    if not isinstance(inter_states, list):
        raise ValueError("Inter_states must be a list.")
    if not isinstance(polar_matrix, np.ndarray):
        raise ValueError("Polar_matrix must be a NumPy array.")
    if not is_valid_transition_matrix(polar_matrix):
        raise ValueError("Polar_matrix is not a valid transition matrix.")
    if not isinstance(inter_prob, (int, float)) or not 0 <= inter_prob <= 1:
        raise ValueError("Inter_prob must be a float between 0 and 1.")

    try:
        # Update inter-group connections in the polarization matrix
        for i, j in itertools.product(inter_states, repeat=2):
            if i != j:
                polar_matrix[i, j] = inter_prob / len(inter_states)

        # Normalize rows of the matrix
        row_sums = np.sum(polar_matrix, axis=1, keepdims=True)
        return np.divide(polar_matrix, row_sums,
                         out=np.zeros_like(polar_matrix), where=row_sums != 0)
    except Exception as e:
        raise ValueError(f"An error occurred during inter-group connection addition: {e}")
    
def insert_polarized(states, p_matrix):
    """
    Inserts a group transition matrix into a polarization matrix.

    Args:
        states (list): The states belonging to the group.
        p_matrix (np.array): The polarization transition matrix.

    Returns:
        np.array: The updated polarization matrix with the group matrix inserted.

    Raises:
        ValueError: If states is not a list or if p_matrix is not a NumPy array.
        ValueError: If the length of states does not match the dimensions of p_matrix.
    """
    if not isinstance(states, list):
        raise ValueError("States must be a list.")
    if not isinstance(p_matrix, np.ndarray):
        raise ValueError("P_matrix must be a NumPy array.")
    if not is_valid_transition_matrix(p_matrix):
        raise ValueError("P_matrix is not a valid transition matrix.")
    
    try:
        # Generate the group transition matrix
        group_matrix = generate_standard_transition_matrix(len(states))

        # Determine the row and column indices in p_matrix
        row_indices = states
        col_indices = states

        # Insert the group_matrix into the corresponding positions in p_matrix
        p_matrix[np.ix_(row_indices, col_indices)] = group_matrix
        return p_matrix
    except Exception as e:
        raise ValueError(f"An error occurred during group transition matrix insertion: {e}")

def group_states(remaining_states, num_states, clusters):
    """
    Samples states for a group unbiasedd on the number of states and clusters.

    Args:
        remaining_states (list): List of remaining unassigned states.
        num_states (int): Total number of states.
        clusters (int): Number of groups.

    Returns:
        list: Sampled states for the group.

    Raises:
        ValueError: If remaining_states is not a list, num_states is not an integer, or clusters is not an integer greater than 0.
    """
    if not isinstance(remaining_states, list):
        raise ValueError("Remaining_states must be a list.")
    if not isinstance(num_states, int):
        raise ValueError("Num_states must be an integer.")
    if not isinstance(clusters, int) or clusters <= 0:
        raise ValueError("Clusters must be an integer greater than 0.")
    if clusters > num_states:
        raise ValueError("Clusters cannot exceed the number of states.")

    try:
        # Determine the number of states to sample for the group
        num_states_per_group = int(num_states / clusters)

        # Check if there are enough remaining states to sample
        if len(remaining_states) >= num_states_per_group:
            # Sample states for the group
            return sample_states(remaining_states, num_states_per_group)
        else:
            # If there are not enough remaining states, return all remaining states
            return remaining_states
    except Exception as e:
        raise ValueError(f"An error occurred during state grouping: {e}")

def generate_polar_matrix(num_states, groups, clusters, inter_prob):
    """
    Generates a polarized transition matrix.

    Args:
        num_states (int): Total number of states.
        groups (list): List of state groups.
        clusters (int): Number of groups.
        inter_prob (float): Inter-group transition probability.

    Returns:
        tuple: A tuple containing the list of grouped states and the polarized transition matrix.

    Raises:
        ValueError: If the number of states is less than the number of groups.
        ValueError: If the number of groups is less than 2.
        ValueError: If the inter-group transition probability is not between 0 and 1.
    """

    if not isinstance(num_states, int) or num_states <= 0:
        raise ValueError("Number of states must be a positive integer.")
    if not isinstance(clusters, int) or clusters < 2:
        raise ValueError("Number of groups must be an integer greater than or equal to 2.")
    if not 0 <= inter_prob <= 1:
        raise ValueError("Inter-group transition probability must be between 0 and 1.")

    try:
        matrix = np.zeros((num_states, num_states))

        state_clusters = divide_list_into_groups(list(range(num_states)), clusters)
        unbiased_states = state_clusters[len(state_clusters) - 1]
        biased_states = [state for cluster in state_clusters for state in cluster if state not in unbiased_states]
        inter_states = []

        for idx, cluster in enumerate(state_clusters):
            grp_states = state_clusters[idx]
            if idx != len(groups) - 1:
                matrix = insert_polarized(grp_states, matrix)
                inter_states.append(np.random.choice([state for state in biased_states if state not in grp_states],1)[0])
            else:
                matrix = inter_polar(inter_states, matrix, inter_prob)
                polar_matrix = insert_polarized(grp_states, matrix)

        row_sums = np.sum(polar_matrix, axis=1, keepdims=True)

        return state_clusters, np.divide(polar_matrix, row_sums, out=np.zeros_like(polar_matrix), where=row_sums != 0)
    except Exception as e:
        raise ValueError(f"An error occurred during polar matrix generation: {e}")
    
def assign_states_to_agents(groups, state_clusters, num_agents):
    """
    Assigns states to agents unbiasedd on the given groups and state clusters.

    Args:
        groups (list): List of groups containing agents.
        state_clusters (list): List of state clusters.
        num_agents (int): Number of agents.

    Returns:
        numpy.ndarray: Array of assigned states for each agent.

    Raises:
        ValueError: If the number of agents is less than the number of groups.
    """
    # Validate input types
    if not isinstance(groups, list) or not isinstance(state_clusters, list) or not isinstance(num_agents, int):
        raise TypeError("groups and state_clusters must be lists, and num_agents must be an integer")

    # Validate the number of agents
    if num_agents < len(groups):
        raise ValueError("Number of agents cannot be less than the number of groups")

    try:
        assigned_states = np.zeros(num_agents, dtype=int)

        for i in range(num_agents):
            for idx, group in enumerate(groups):
                if i in group:
                    group_states = state_clusters[idx]
                    assigned_states[i] = np.random.choice(group_states)
                    break  # Once the agent is assigned a state from its group, exit the loop

        return assigned_states
    except Exception as e:
        raise ValueError(f"An error occurred during state assignment: {e}")
        
def introduce_polarization(agents, num_states, clusters, unaff_pct, inter_prob, obs):
    """
    Introduces polarization to a list of agents and generates Markov chains.

    Args:
        agents (list): List of agents.
        num_states (int): Number of states.
        clusters (int): Number of clusters.
        unaff_pct (float): Percentage of unaffiliated agents.
        inter_prob (float): Inter-cluster transition probability.
        obs (int): Number of observations.

    Returns:
        tuple: Tuple containing the polar matrix and generated Markov chains.

    Raises:
        ValueError: If any argument has an invalid data type or value.
    """
    # Validate input types and values
    if not isinstance(agents, list):
        raise ValueError("Agents must be a list.")
    if not all(isinstance(agent, int) for agent in agents):
        raise ValueError("All elements of agents must be integers.")
    if not isinstance(num_states, int):
        raise ValueError("Num_states must be an integer.")
    if not isinstance(clusters, int):
        raise ValueError("Clusters must be an integer.")
    if clusters <= 2:
        raise ValueError("The number of clusters must be greater than 2.")
    if clusters > num_states:
        raise ValueError("Number of clusters cannot exceed number of states.")
    if not isinstance(unaff_pct, float):
        raise ValueError("Unaff_pct must be a float.")
    if not 0 <= unaff_pct <= 1:
        raise ValueError("Unaff_pct must be between 0 and 1.")
    if not isinstance(inter_prob, float):
        raise ValueError("Inter_prob must be a float.")
    if not 0 <= inter_prob <= 1:
        raise ValueError("Inter_prob must be between 0 and 1.")
    if not isinstance(obs, int) or obs <= 0:
        raise ValueError("Obs must be a positive integer.")

    try:
        polar_agents, unaff_agents = divide_agents(agents, unaff_pct)
        groups = divide_list_into_groups(polar_agents, clusters - 1)
        groups.append(list(unaff_agents))
        
        state_clusters, polar_matrix = generate_polar_matrix(num_states, groups, clusters, inter_prob)
        
        initial_states = assign_states_to_agents(groups, state_clusters, len(agents))
        
        return polar_matrix, generate_markov_chains(polar_matrix, initial_states, obs, len(agents))
    except Exception as e:
        raise ValueError(f"An error occurred during polarization introduction: {e}")

def polar_append(states_list, states, missing_list, pct, time_list, standard_time, KL_list, KL, inaccuracy_list, 
                 inaccuracy, agents_list, N, obs_list, T, clusters_list, clusters, unaff_list, unaff_pct, 
                 inter_list, inter_prob, ss_ratio_list, ss_ratio, imputed_list, imputed, em_list, em, emTime_list, 
                 emTime, imputed_time_list, imputeTime):
    

    states_list.append(states)
    missing_list.append(pct)
    time_list.append(standard_time)
    KL_list.append(KL)
    inaccuracy_list.append(inaccuracy)
    agents_list.append(N) 
    obs_list.append(T)
    clusters_list.append(clusters)
    unaff_list.append(unaff_pct)
    inter_list.append(inter_prob)
    ss_ratio_list.append(ss_ratio)
    imputed_list.append(imputed)
    em_list.append(em)
    emTime_list.append(emTime)
    imputed_time_list.append(imputeTime)
    
    return states_list, missing_list, time_list, \
            KL_list, inaccuracy_list, agents_list, obs_list, \
            clusters_list, unaff_list, inter_list, \
            ss_ratio_list, imputed_list, em_list, emTime_list, imputed_time_list
    
def process_polar(states, N, T, inter_prob, clusters, unaff_range, ss_range, missing_range, imputation = False, optimization = False, em_iterations = None, tol = None):
    
    
    
    
    try:
        agents = list(range(N))
        for cluster, unaff_pct in list(itertools.product(clusters, unaff_range)):
            observed, chains = introduce_polarization(agents, states, int(cluster), unaff_pct, inter_prob, T)
            for pct, r in list(itertools.product(missing_range, ss_range)):
                if not (pct == 0 and r > 0):
                    # Introduce self-selection bias
                    ss = apply_self_selection_bias(chains, r, N, T, pct)
                    result = pd.DataFrame(introduce_mcar_missing_data(pd.DataFrame(ss), pct))
                    start = time.time()  # Start time of execution
                    estimated = extract_transition_matrix(result, states)  # Estimate transition matrix
                    end = time.time()  # End time of execution
                    
                    # Perform imputation if requested
                    final = bmc.forward_algorithm(result, estimated, T, states) if imputation else None
                    estimated_imputed = extract_transition_matrix(final, states) if imputation else None
                    end_impute = time.time() if imputation else None
                    
                    # Perform optimization if requested
                    estimated_em = bmc.em_algorithm(result, N, T, states, em_iterations, tol) if optimization else None
                    end_em = time.time() if optimization else None
                    
                    # Append results to shared lists
                    states_list, missing_list, time_list, KL_list, inaccuracy_list, agents_list, obs_list, clusters_list, \
                        unaff_list, inter_list, ss_ratio_list, imputed_list, em_list, emTime_list, imputed_time_list = polar_append(
                        states_list, states, missing_list, pct, time_list, end - start, KL_list,
                        kl_divergence(estimated, observed, states), inaccuracy_list,
                        np.linalg.norm(estimated - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)), 
                        agents_list, N, obs_list, T, clusters_list, cluster, unaff_list, unaff_pct, inter_list, inter_prob, ss_ratio_list,
                        r, imputed_list if imputation else None, 
                        np.linalg.norm(estimated_imputed - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)) if imputation else None,
                        em_list if optimization else None,
                        np.linalg.norm(estimated_em - observed)/(np.sqrt(2*states)*np.linalg.norm(observed)) if optimization else None,
                        emTime_list, end_em - start if optimization else None, imputed_time_list, end_impute - start if imputation else None)
                        
        return states_list, missing_list, time_list, KL_list, inaccuracy_list, agents_list, obs_list, clusters_list, unaff_list, inter_list, ss_ratio_list, imputed_list, em_list, emTime_list, imputed_time_list

    except Exception as e:
        raise ValueError(f"Error in processing outlier scenario: {e}")
                        
                    
                    