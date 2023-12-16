#############################################
#                                           #
#   Title:      Bised MC Transition Sim     #
#   Author:     Ben Harwood                 #
#   Email:      bharwood@syr.edu            #
#   Version:    Final                       #
#   Date:       December 15, 2023           #
#                                           #
#############################################

import pandas as pd
import numpy as np
import time
import itertools
from itertools import chain
import random 
import multiprocessing
from multiprocessing import Manager

def calculate_log_likelihood(data, T, transition_matrix, pi):
    """
    Calculates the log likelihood for a hidden Markov model given observed data and a transition matrix.

    The log likelihood is calculated by iterating through each time step, getting the current and next states from the observed data, and adding the log of the transition probability from the current state to the next state based on the provided transition matrix.

    Args:
    data: Observed sequence data with states as row index and time as column index
    transition_matrix: Square transition matrix where element i,j is p(j|i) 

    Returns:
    log_likelihood: The log likelihood of the observed data given the transition matrix  
    """
    
    log_likelihood = 0.0

    for t in range(1, T):
        i = data.iloc[:, t - 1]
        j = data.iloc[:, t]
        valid_indices = ~np.isnan(i) & ~np.isnan(j)
        i_valid = i[valid_indices].astype(int)
        j_valid = j[valid_indices].astype(int)

        if len(i_valid) > 0 and len(j_valid) > 0:
            probs = np.array([transition_matrix[x, y] if transition_matrix[x, y] > 0 else 1e-9 for x, y in zip(i_valid, j_valid)])
            log_likelihood += np.sum(np.log(probs)) + np.log(pi[j_valid.iloc[0]] if pi[j_valid.iloc[0]] > 0 else 1e-9)
    
    return log_likelihood

def generate_pi(Y, states):
    """Calculates the initial probability vector from observed data.

    Generates the initial probability vector by calculating the proportion 
    of observations in each state.

    Args:
    Y: Observed sequence data with states as row index and time as column index
    states: List of possible state values
    
    Returns:
    pi: Initial probability vector where pi[i] is probability of starting in state i
    """

    return [
        np.sum(Y == state).sum() / (Y.shape[0] * Y.shape[1] - Y.isna().sum().sum())
        for state in states
    ]
  
def initialize_pi_P(Y, states):
    """
    Calculates the initial probability vector and transition matrix.

    Args:
        Y (numpy.ndarray): The input data.
        weights (numpy.ndarray): The weights for each state.

    Returns:
        tuple: A tuple containing the initial probability vector and transition matrix.

    Example:
        ```python
        Y = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        weights = np.array([0.2, 0.3, 0.5])
        pi_init, P_init = initialize_pi_P(Y, weights)
        print("Initial probability vector:")
        print(pi_init)
        print("Transition matrix:")
        print(P_init)
        ```
    """
    k = len(states)
    
    pi_init = generate_pi(pd.DataFrame(Y), states)
    P_init = extract_transition_matrix(pd.DataFrame(Y), k)
    
    return pi_init, P_init

def forward_algorithm(data, P, T, states):
        
    imputed_data = data.copy()
    
    for t in range(1, T):
        if np.sum(imputed_data.iloc[:, t].isna()) > 0:
            tempP = P if t == 1 else extract_transition_matrix(imputed_data.iloc[: , : t].astype(int), states)
            
            missing_data = np.isnan(imputed_data.iloc[:,t])

            # For missing data, impute values based on the transition matrix
            new_states = np.argmax(tempP[imputed_data.iloc[:,t-1].astype(int), :], axis = 1)
            
            # Impute missing values in column t + 1 with the most likely states
            imputed_data.loc[missing_data, t] = new_states[missing_data]
    
    return imputed_data
   
def em_algorithm(data, T, num_states, num_iterations, tol=0.000001):
    """Estimates a transition matrix from data using the EM algorithm.

    Iteratively performs an E-step to impute missing data and an M-step 
    to re-estimate the transition matrix until convergence.

    Args:
    data: Observed sequence data
    num_states: Number of hidden states
    num_iterations: Maximum number of EM iterations
    tol: Convergence tolerance based on log-likelihood change

    Returns: 
    transition_matrix: Estimated transition matrix  
    """

    prev_log_likelihood = float('-inf')
    pi_init, transition_matrix = initialize_pi_P(data,range(num_states))
    
    for _ in range(num_iterations):
        # E-step: Impute missing data using the Forward Algorithm
        imputed_data = forward_algorithm(data, transition_matrix, T, num_states)

        # M-step: Update the transition matrix
        transition_matrix = extract_transition_matrix(imputed_data, num_states)

        # Calculate log-likelihood for convergence check
        log_likelihood = calculate_log_likelihood(imputed_data, T, transition_matrix, pi_init)

        # Check for convergence based on log-likelihood change
        if abs(log_likelihood - prev_log_likelihood) < tol:
            break

        prev_log_likelihood = log_likelihood
        pi_init = initialize_pi_P(imputed_data, list(range(num_states)))[0]
        
        
    return transition_matrix[:num_states, :num_states]

def apply_self_selection_bias(df, ratio, N, T, pct):
    """
    Applies self-selection bias to a DataFrame across time steps.

    A subset of agents are selected to be self-selecting. For each time step, 
    these agents are randomly masked based on their posting probability. 

    The masked values are used to update the DataFrame. Masks can be consistent
    across time steps or regenerated each step.

    Args:
    df: Input DataFrame
    ratio: Ratio of agents that are self-selecting
    N: Total number of agents
    T: Number of time steps
    consistent: Whether to use consistent masks ('Y') or not ('N') 

    Returns:
    df: DataFrame updated with self-selection bias  
    """

    # Initialize masks if consistent self-selection
    masks = []
    if ratio > 0:
        ssAgentCount = int(ratio * N)
        ssAgents = np.random.choice(range(N), ssAgentCount, replace= False)
        posting_probs = np.random.uniform(0.8, 1, size = ssAgentCount)
            
        for t in range(1, T):
            masks = [np.random.rand(1) < p for p in posting_probs]
            
            # Update the DataFrame using masks
            if np.sum(df.isna().values)/df.size < pct:
                df.loc[ssAgents, t] = [np.nan if mask[0] else df.loc[agent, t] for agent, mask in zip(ssAgents, masks)]
       
    return df

def generate_standard_transition_matrix(num_states):
    """
    Generates a standard transition matrix with random values.

    Args:
        num_states (int): The number of states.

    Returns:
        numpy.ndarray: The generated standard transition matrix.

    Example:
        ```python
        import numpy as np

        # Defining the number of states
        num_states = 3

        # Generating the standard transition matrix
        matrix = generate_standard_transition_matrix(num_states)
        print(matrix)  # Output: <generated standard transition matrix>
        ```
    """
    return np.random.dirichlet(np.ones(num_states), size = num_states)

def introduce_mcar_missing_data(markov_chain, missing_prob):
    """
    Introduces missing data to a given Markov chain based on a specified missing probability.

    Args:
        markov_chain (pandas.DataFrame): The Markov chain.
        missing_prob (float): The probability of introducing missing data.

    Returns:
        numpy.ndarray: The Markov chain with missing data.

    Example:
        ```python
        import numpy as np
        import pandas as pd

        # Creating a Markov chain DataFrame
        markov_chain = pd.DataFrame({'State1': [0, 1, 1, 2], 'State2': [1, 2, 0, 1]})

        # Defining the missing probability
        missing_prob = 0.2

        # Introducing missing data to the Markov chain
        missing_data = introduce_mcar_missing_data(markov_chain, missing_prob)
        print(missing_data)  # Output: <Markov chain with missing data>
        ```
    """
    missing_data = markov_chain.to_numpy().astype(float)
    N, T = markov_chain.shape
    needed_missing = missing_prob * N * T
    for i, j in itertools.product(range(N), range(1, T)):
        if np.count_nonzero(np.isnan(missing_data)) == needed_missing:
            break 
        if missing_data[i, j] != np.nan and np.random.rand() < missing_prob:
            missing_data[i, j] = np.nan
         
            
    return missing_data

def extract_transition_matrix(Y, states):
    """
    Extracts the transition matrix from a given DataFrame of states.

    Args:
        Y (pandas.DataFrame): The DataFrame of states.
        num_states (int): The number of states.

    Returns:
        numpy.ndarray: The extracted transition matrix.

    Example:
        ```python
        import numpy as np
        import pandas as pd

        # Creating a DataFrame of states
        Y = pd.DataFrame({'State': [0, 1, 1, 2, 2, 0]})

        # Defining the number of states
        num_states = 3

        # Extracting the transition matrix
        matrix = extract_transition_matrix(Y, num_states)
        print(matrix)  # Output: <extracted transition matrix>
        ```
    """
    result = Y.values
    N, T = result.shape
    state_counts = [np.nansum(result == i) for i in range(states)]

    # Create an array to count transitions between states
    transition_counts = np.zeros((states, states), dtype=int)

    # Loop through each agent and each time step to count occurrences and transitions
    for n, t in itertools.product(range(N), range(T)):
        if t < T - 1 and np.isnan(result[n,t]) == False and np.isnan(result[n,t+1]) == False:
            transition_counts[int(result[n, t]), int(result[n, t + 1])] += 1
            
    non_zero_total = [state_counts[state] if state_counts[state] > 0 else 1 for state in range(states)]
    

    # Create the transition probability matrix
    P = np.zeros((states, states), dtype=float)
    P = (transition_counts.T/non_zero_total).T
    row_sums = np.sum(P, axis=1, keepdims=True)
    
    return np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums != 0)

def generate_initial_states(P, num_agents):
    """
    Generate initial states for multiple agents using a transition matrix.

    Parameters:
    P (numpy.ndarray): The transition matrix.
    num_agents (int): The number of agents.

    Returns:
    list: A list of initial states for each agent.
    """
    num_states = P.shape[0]
    initial_states = []

    # Calculate the steady-state distribution
    stationary_distribution = calculate_steady_state(P)

    for _ in range(num_agents):
        initial_state = np.random.choice(num_states, p=stationary_distribution)
        initial_states.append(initial_state)

    return initial_states

def calculate_steady_state(P):
    """
    Calculate the steady-state distribution of a Markov chain.

    Parameters:
    P (numpy.ndarray): The transition matrix of the Markov chain.

    Returns:
    numpy.ndarray: The steady-state distribution.
    """
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    stationary_vector = np.real(eigenvectors[:, np.argmax(eigenvalues)])

    return stationary_vector / np.sum(stationary_vector)

def generate_outlier_chain(transition_matrix, initial_state, num_steps):
    """
    Generates a Markov chain of agent states based on a given transition matrix, initial state, and number of steps.

    Args:
        transition_matrix (numpy.ndarray): The transition matrix of the Markov chain.
        initial_state (int): The initial state of the Markov chain.
        num_steps (int): The number of steps to generate in the Markov chain.

    Returns:
        List[int]: The generated Markov chain of agent states.

    Example:
        ```python
        import numpy as np

        # Creating a transition matrix
        transition_matrix = np.array([[0.2, 0.8], [0.6, 0.4]])

        # Defining the initial state and number of steps
        initial_state = 0
        num_steps = 5

        # Generating the Markov chain
        chain = generate_agent_chain(transition_matrix, initial_state, num_steps)
        print(chain)  # Output: <generated Markov chain>
        ```
    """
    markov_chain = [initial_state]
    current_state = initial_state
    
    for _ in range(1, num_steps):
        next_state = np.random.choice(len(transition_matrix), p = transition_matrix[current_state])
        markov_chain.append(next_state)
        current_state = next_state
    
    return markov_chain

def generate_markov_chains(transition_matrix, initial_states, num_steps, num_agents):
    """
    Generate Markov chains for multiple agents given a transition matrix.

    Parameters:
    transition_matrix (numpy.ndarray): The transition matrix.
    initial_states (list or numpy.ndarray): Initial states for each agent.
    num_steps (int): The number of steps to generate in each Markov chain.
    num_agents (int): The number of agents.

    Returns:
    list: A list of Markov chains for each agent.
    """
    markov_chains = []

    for agent in range(num_agents):
        markov_chain = [initial_states[agent]]
        current_state = initial_states[agent]

        for _ in range(num_steps - 1):
            next_state = np.random.choice(len(transition_matrix), p=transition_matrix[current_state])
            markov_chain.append(next_state)
            current_state = next_state

        markov_chains.append(markov_chain)

    return markov_chains

def generate_outlier_matrix(num_states, outlier_state1, outlier_state2):
    """
    Generates a transition probability matrix with biased probabilities for specified outlier states.

    The transition probability matrix introduces high self-transition probabilities for the specified outlier states, and low transition probabilities from other states to the outlier states. This creates a bias where the outlier states are sticky. 

    The matrix is normalized so each row sums to 1.

    Args:
    num_states: Total number of states 
    outlier_state1: First outlier state index
    outlier_state2: Second outlier state index

    Returns:
    p: num_states x num_states transition probability matrix 

    """
    p = np.zeros((num_states, num_states))

    for i in range(num_states):
        if i in [outlier_state1, outlier_state2]:
            row = np.random.rand(num_states-1)
            row = row/np.sum(row) * 0.02
            p[i, :i] = row[:len(p[i,:i])]
            p[i,i] = 0.98
            p[i, i + 1 :] = row[i:]
        else:
            row = np.random.rand(num_states)
            row[outlier_state1] = 0.0001
            row[outlier_state2] = 0.0001
            row = row/np.sum(row)
            p[i] = row
    
    return p

def divide_list_into_groups(input_list, clusters):
    """
    Randomly divides the input list into multiple groups.

    Args:
    input_list (list): The input list to be divided.
    num_groups (int): The number of groups to create.

    Returns:
    list: A list of sublists where the input list is randomly divided into num_groups.
    """
    random.shuffle(input_list)  # Shuffle the input list randomly
    extra = len(input_list) % clusters
    sublist_size = len(input_list) // clusters
    if sublist_size > 0:
        return [input_list[i:i + sublist_size + max(1, extra)] for i in range(0, len(input_list), sublist_size)]
    else:
        return [input_list[i:i + sublist_size + max(1, extra)] for i in range(len(input_list))]
    
def divide_agents(agents, unaff_pct):
    """
    Divides a list of agents into polarized and unaffiliated groups based on the unaffiliated percentage.

    Args:
    agents (list): The list of all agents
    unaff_pct (float): The percentage of agents that should be unaffiliated

    Returns:
    polar_agents (list): The list of polarized agents
    unaff_agents (list): The list of unaffiliated agents
    """
    num_unaff = int(len(agents) * unaff_pct)
    unaff_agents = np.random.choice(agents, num_unaff, replace=False)
    polar_agents = [a for a in agents if a not in unaff_agents]
    return polar_agents, unaff_agents

def sample_states(remaining, num):
    """
    Randomly samples unique states from a list without replacement.

    Args:
    remaining (list): List of remaining states to sample from  
    num (int): Number of states to sample

    Returns:
    sampled (list): List of randomly sampled unique states  
    """

    return np.sort(np.random.choice(remaining, num, replace=False))

def inter_polar(inter_states, polar_matrix, inter_prob):
    """
    Adds inter-group connections to a polarization matrix.

    Args:
    inter_states (list): List of states to connect between groups
    polar_matrix (np.array): Polarization transition matrix
    inter_prob (float): Probability of inter-group transitions

    Returns:
    polar_matrix (np.array): Updated matrix with inter-group connections
    """

    for i in inter_states:
        for j in inter_states:
            if i != j:
                polar_matrix[i, j] = inter_prob
    
    row_sums = np.sum(polar_matrix, axis=1, keepdims=True)
    
    return np.divide(polar_matrix, row_sums, out=np.zeros_like(polar_matrix), where=row_sums != 0)
    
def insert_polarized(states, p_matrix):
    """
    Inserts a group transition matrix into a polarization matrix.

    Args:
    states (list): The states belonging to the group 
    p_matrix (np.array): The polarization transition matrix
    
    Returns:
    p_matrix (np.array): The updated polarization matrix  
                            with the group matrix inserted
    """

    group_matrix = generate_standard_transition_matrix(len(states))
    # Determine the row and column indices in polar_matrix
    row_indices = states
    col_indices = states

    # Insert the group_matrix into the corresponding positions in polar_matrix
    p_matrix[np.ix_(row_indices, col_indices)] = group_matrix
    return p_matrix

def update_state_lists(inter_states, initial_states, group_states, group, remaining_states):
    """
    Updates state lists by sampling from a group's states.
    
    Args:
    inter_states (list): List of inter-group states
    initial_states (list): List of initial states
    group_states (list): List of states in the group
    group (list): The current group
    remaining_states (list): List of remaining unassigned states

    Returns:
    inter_states (list): Updated inter-group states
    initial_states (list): Updated initial states  
    remaining_states (list): Updated remaining states
    """

    inter_states.append(np.random.choice(group_states))
    initial_states.append(np.random.choice(group_states, len(group), replace = True))
    # Remove the used states from remaining_states
    remaining_states = [state for state in remaining_states if state not in group_states]
    return inter_states, initial_states, remaining_states

def group_states(remaining_states, num_states, clusters):
    """
    Samples states for a group based on the number of states and clusters.

    Args:
    remaining_states (list): List of remaining unassigned states
    num_states (int): Total number of states 
    clusters (int): Number of groups

    Returns:
    sampled_states (list): Sampled states for the group  
    """

    if len(remaining_states) >= int(num_states / clusters):
        return sample_states(remaining_states, int((num_states / clusters)))
    else:
        return remaining_states

def generate_polar_matrix(num_states, groups, clusters, inter_prob):
    """
    Generates a polarized transition matrix.

    Args:
    num_states (int): Total number of states
    groups (list): List of state groups 
    clusters (int): Number of groups
    inter_prob (float): Inter-group transition probability

    Returns:
    initial_states (list): List of initial states
    matrix (np.array): Polarized transition matrix
    """

    matrix = np.zeros((num_states, num_states))
    remaining_states = list(range(num_states))
    
    initial_states = []
    inter_states = []
    for group in groups:
        if len(remaining_states) == 0:
            break
        grp_states = group_states(remaining_states, num_states, clusters)
        
        matrix = insert_polarized(grp_states, matrix)

        inter_states, initial_states, remaining_states = update_state_lists(inter_states, initial_states, grp_states, group, remaining_states)
        
        
    polar_matrix = inter_polar(inter_states, matrix, inter_prob)
    
    return initial_states, polar_matrix

def introduce_polarization(observed, agents, num_states, clusters, unaff_pct, inter_prob, obs):
    """
    Introduces polarization into a base transition matrix.

    Args:
    observed (np.array): Base transition matrix 
    agents (list): List of agents
    num_states (int): Number of states
    clusters (int): Number of groups
    unaff_pct (float): Percentage of unaffiliated agents
    inter_prob (float): Inter-group transition probability
    obs (int): Number of observations
    
    Returns:
    rmse (float): RMSE between original and polarized matrices
    """
        
    polar_agents, unaff_agents = divide_agents(agents, unaff_pct)
    polar_count = len(polar_agents)
    groups = divide_list_into_groups(polar_agents, clusters)
    
    initial_states, polar_matrix = generate_polar_matrix(num_states, groups, clusters, inter_prob)
    
    initial_states_polar = list(chain.from_iterable(initial_states))
    initial_states_noPolar = np.random.choice(list(range(num_states)) , len(unaff_agents), replace = True)
    
    polar_chains = generate_markov_chains(polar_matrix, initial_states_polar, obs, polar_count)
    
    unaff_chains = generate_markov_chains(observed, initial_states_noPolar, obs, len(unaff_agents))
    
    # Combine polarized and unaffiliated chains
    return pd.DataFrame(np.concatenate((polar_chains, unaff_chains), axis=0))
    
def introduce_missing_data_less_vocal(data, less_vocal_group, missing_prob):
    """
    Introduces missing values into a dataset.

    Randomly replaces values with NaN based on a missing probability, for a specified subset of agents.

    Args:
    data: Input dataset 
    less_vocal_group: Subset of agents to introduce missing values for
    missing_prob: Probability of replacing a value with NaN 

    Returns: 
    data: Dataset with missing values introduced

    """
    N, T = data.shape
    needed_missing = missing_prob * N * T
    for agent in less_vocal_group:
        for t in range(T):
            if np.random.rand() < missing_prob:
                data.at[agent, t] = np.nan
            if np.count_nonzero(np.isnan(data)) == needed_missing:
                break  
    
    return data

def introduce_outlier_bias(chains, group2, group3, group4, loud, missing_prob):
    """
    Introduces biased initial states and transitions in a Markov chain simulation. 

    Agents belonging to specified groups start in particular outlier states. 
    The transition matrix introduces sticky self-transitions for the outlier states.

    A Markov chain simulation is run using the biased initial states and transition matrix.

    Args:
    transition_matrix: Transition matrix 
    group1: Agents belonging to first group
    outlier_state1: Outlier state for first group
    group2: Agents belonging to second group 
    outlier_state2: Outlier state for second group
    num_agents: Total number of agents
    num_obs: Number of observations in simulation
    
    Returns:
    Simulation output from generate_markov_chains()
    """
    if loud == 'min':
        return introduce_missing_data_less_vocal(chains, group4, missing_prob)
    else:
        return introduce_missing_data_less_vocal(introduce_missing_data_less_vocal(chains, group3, missing_prob), group2, missing_prob)
 
def introduce_popularity_bias(d1, state_probabilities, desired_missing_pct, N, T, asc):
    """
    Introduces missing data based on state popularity.

    Processes data to introduce missing values based on specified state probabilities.
    More popular states will have fewer missing values.

    Args:
    d1: Original data 
    state_probabilities: DataFrame of state probabilities  
    desired_missing_pct: Desired percentage of missing values
    N: Number of rows in data
    T: Number of columns in data

    Returns:
    d2: Data with missing values introduced
    
    """

    # Create a copy of the original data
    d2 = d1.iloc[:,1:].copy()
    states = state_probabilities.shape[0]
    
    # Sort the states by probability in ascending order
    sorted_states = state_probabilities.sort_values(by='Probability', ascending = asc)

    # Calculate the desired number of missing values
    desired_missing_count = round(desired_missing_pct * N * T)

    # Create a list to store state indices
    state_indices_list = [[] for _ in range(states)]

    # Store the indices of each state's observations
    for state in range(states):
        state_indices_list[state] = np.where(d1.values == state)

    # Initialize a variable to track the remaining missing count
    remaining_missing_count = desired_missing_count

    # Loop through the states in ascending order of probability
    for i in range(len(sorted_states)):
        if remaining_missing_count <= 0:
            break
        state = sorted_states['State'].values[i]
        
        occurrences = (d2 == state).sum().sum()
        
        # Calculate the number of occurrences to mark as NaN
        needed = int(occurrences * 0.9)
        
        # Randomly generate observations for the current state
        if needed > 0:
            # Identify the indices of occurrences for the current state
            state_indices = np.where(d2.values == state)
            
            # Randomly select indices for the current state
            selected_indices = random.sample(list(zip(state_indices[0], state_indices[1])), k = needed)
            
            # Mark occurrences as NaN using the selected indices
            for index in selected_indices:
                d2.iloc[index] = np.nan
                # Update the remaining missing count
                remaining_missing_count -= 1
                if remaining_missing_count == 0:
                    break

         
        

    return d2   

def introduce_confirmation_bias(num_states, num_agents, cbf, obs):
    """Introduces confirmation bias into a transition matrix.

    The function modifies a transition matrix by increasing 
    the probability of staying in the same state and decreasing 
    the probability of transitioning to other states based on 
    the confirmation bias factor. 

    It generates initial states and Markov chain sequences using 
    the biased transition matrix.

    Args:
    num_states: Number of states
    num_agents: Number of agents
    confirmation_bias_factor: Factor controlling bias (0-1)
    obs: Number of observations
    
    Returns:
    Tuple of biased transition matrix and generated Markov chains 
    """
    transition_matrix = np.full((num_states, num_states), 1.0)
    remaining_attractors = int(num_states * cbf) if num_states * cbf > 1 else 1
    
    # Apply confirmation bias within the transition matrix
    for i in range(num_states):
        for j in range(num_states):
            if i == j and remaining_attractors > 0:
                # Adjust self-transition probability based on confirmation bias factor
                transition_matrix[i][j] *= random.uniform(1.2,1.5)  # High self-transition probability
            else:
                # Lower probability of moving to different states
                transition_matrix[i][j] *= random.uniform(0, 0.2 / num_states)  # Adjust disconfirmation bias factor
            remaining_attractors -= 1
        
    # Normalize the transition matrix
    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
    
    initial_states = generate_initial_states(transition_matrix, num_agents)
        
    return transition_matrix, generate_markov_chains(transition_matrix, initial_states, obs, num_agents)

def cAppend(shared_confirmation_states_list, states, shared_confirmation_missing_list, pct, shared_confirmation_emTime_list, chi, 
        shared_confirmation_norm_list, norm, shared_confirmation_p_list, p , shared_confirmation_agents_list, N, shared_confirmation_obs_list, 
        T, shared_confirmation_ss_ratio_list, ss_ratio, shared_bias_factor_list, cbf):
    """Appends confirmation bias results to shared lists.

    The function appends states, missing %, RMSE, agents,  
    observations, self-selection type, self-selection ratio,
    confirmation bias factor to their respective shared lists.

    Args:
    shared_lists: Shared lists to append results to
    states: Number of states
    pct: Percent missing data
    rmse: Root mean squared error 
    N: Number of agents
    T: Number of observations
    ss: Self-selection type
    ss_ratio: Self-selection ratio
    cbf: Confirmation bias factor
    
    Returns:
    Tuple of updated shared lists
    """
    shared_confirmation_states_list.append(states)
    shared_confirmation_missing_list.append(pct)
    shared_confirmation_emTime_list.append(chi)
    shared_confirmation_norm_list.append(norm)
    shared_confirmation_p_list.append(p)
    shared_confirmation_agents_list.append(N)
    shared_confirmation_obs_list.append(T)
    shared_confirmation_ss_ratio_list.append(ss_ratio)
    shared_bias_factor_list.append(cbf)
    return shared_confirmation_states_list, shared_confirmation_missing_list, shared_confirmation_emTime_list, shared_confirmation_norm_list, \
        shared_confirmation_p_list, shared_confirmation_agents_list, shared_confirmation_obs_list, shared_confirmation_ss_ratio_list, \
            shared_bias_factor_list

def process_confirmation(other_args, states, N, T):
    """Runs confirmation bias simulations and appends results.

    The function introduces confirmation bias, missing data, 
    and self-selection, calculates RMSE, and appends results 
    to shared lists after each run.

    Args:
    shared_lists: Lists to append outputs to
    states: Number of states
    N: Number of agents 
    T: Number of time steps
    cbf: Confirmation bias factor
    
    Returns:
    Tuple of updated shared lists
    """
    shared_confirmation_states_list, shared_confirmation_missing_list, shared_confirmation_emTime_list, shared_confirmation_norm_list, \
        shared_confirmation_p_list, shared_confirmation_agents_list, shared_confirmation_obs_list, shared_confirmation_ss_ratio_list, \
        shared_bias_factor_list = other_args
    for r, pct, cbf in list(itertools.product(np.linspace(0, 0.8, 5), np.linspace(0, 0.8, 5), np.linspace(0.1, 0.9, 5))):
        if not (pct == 0 and r > 0):
            observed, data = introduce_confirmation_bias(states, N, cbf, T)
            ss = apply_self_selection_bias(pd.DataFrame(data), r, N, T, pct)
            result = pd.DataFrame(introduce_mcar_missing_data(pd.DataFrame(ss), pct))
                    
            start = time.time()
            estimated = extract_transition_matrix(pd.DataFrame(result), states)
            #pi_init, transition_matrix = initialize_pi_P(result,range(states))
            #final = forward_algorithm(result, transition_matrix, T, states)
            #estimated = extract_transition_matrix(final, states)
            #estimated = em_algorithm(result, T, states, 1000)
            end = time.time()
            
            shared_confirmation_states_list, shared_confirmation_missing_list, shared_confirmation_emTime_list, shared_confirmation_norm_list,\
            shared_confirmation_p_list,shared_confirmation_agents_list, shared_confirmation_obs_list, shared_confirmation_ss_ratio_list, \
            shared_bias_factor_list = cAppend(
                shared_confirmation_states_list, 
                states,
                shared_confirmation_missing_list, 
                pct,
                shared_confirmation_emTime_list, 
                end-start,
                shared_confirmation_norm_list,
                np.sum(np.abs(estimated - observed)) / 2,
                shared_confirmation_p_list,
                np.linalg.norm(estimated- observed)/np.linalg.norm(observed),
                shared_confirmation_agents_list, 
                N,
                shared_confirmation_obs_list, 
                T,
                shared_confirmation_ss_ratio_list, 
                r,
                shared_bias_factor_list,
                cbf)

def popAppend(shared_popularity_states_list, states, shared_popularity_missing_list, pct, shared_popularity_emTime_list, chi,
        shared_popularity_norm_list, norm, shared_popularity_p_list, p, shared_popularity_agents_list, N, shared_popularity_obs_list, 
        T, shared_popularity_asc_list, asc):
    """Appends popularity scenario results to shared lists.

    The function appends states, missing %, RMSE, agents,
    observations, and ascending order to their respective
    shared lists after each run.

    Args:
    shared_lists: Shared lists to append results to
    states: Number of states  
    pct: Percent missing data
    rmse: Root mean squared error
    N: Number of agents
    T: Number of observations
    asc: Ascending popularity order
    
    Returns:
    Tuple of updated shared lists
    """
    shared_popularity_states_list.append(states)
    shared_popularity_missing_list.append(pct) 
    shared_popularity_emTime_list.append(chi)
    shared_popularity_norm_list.append(norm)
    shared_popularity_p_list.append(p)
    shared_popularity_agents_list.append(N) 
    shared_popularity_obs_list.append(T)
    shared_popularity_asc_list.append(asc)
    return shared_popularity_states_list, shared_popularity_missing_list, shared_popularity_emTime_list, shared_popularity_norm_list, \
        shared_popularity_p_list, shared_popularity_agents_list, shared_popularity_obs_list, shared_popularity_asc_list

def process_popularity(other_args, states, N, T):
    """Runs popularity scenario simulations and appends results.

    The function generates Markov chains, introduces popularity bias, 
    calculates RMSE, and appends results to shared lists after each run.

    Args:
    shared_lists: Lists to append outputs to  
    states: Number of states
    N: Number of agents
    T: Number of time steps
    
    Returns:
    Tuple of updated shared lists  
    """
    shared_popularity_states_list, shared_popularity_missing_list, shared_popularity_emTime_list, shared_popularity_norm_list, \
    shared_popularity_p_list, shared_popularity_agents_list, shared_popularity_obs_list, shared_popularity_asc_list = other_args
    observed = generate_standard_transition_matrix(states)
    initial_states = generate_initial_states(observed, N)
    d1 = pd.DataFrame(generate_markov_chains(observed, initial_states, T, N))
    probs = np.array([np.sum(d1.values == state) / d1.values.size for state in range(states)])
    state_probabilities = pd.DataFrame({'State': np.arange(0, states), 'Probability': probs})
    for asc, pct in list(itertools.product([True, False], np.linspace(0,0.8,5))):
            result = pd.DataFrame(introduce_popularity_bias(d1, state_probabilities, pct, N, T, asc))
            start = time.time()
            estimated = extract_transition_matrix(result, states)
            #pi_init, transition_matrix = initialize_pi_P(result,range(states))
            #final = forward_algorithm(result, transition_matrix, T, states)
            #estimated = extract_transition_matrix(final, states)
            #estimated = em_algorithm(result, T, states, 1000)
            end = time.time()
            
            shared_popularity_states_list, shared_popularity_missing_list, shared_popularity_emTime_list, shared_popularity_norm_list,\
            shared_popularity_p_list, shared_popularity_agents_list, shared_popularity_obs_list, shared_popularity_asc_list = popAppend(
                shared_popularity_states_list, 
                states, 
                shared_popularity_missing_list, 
                pct, 
                shared_popularity_emTime_list, 
                end-start, 
                shared_popularity_norm_list,
                np.sum(np.abs(estimated - observed)) / 2,
                shared_popularity_p_list,
                np.linalg.norm(estimated - observed)/np.linalg.norm(observed),
                shared_popularity_agents_list, 
                N, 
                shared_popularity_obs_list, 
                T, 
                shared_popularity_asc_list, 
                asc)
   
def outAppend(shared_outlier_states_list, states, shared_outlier_missing_list, pct, shared_outlier_emTime_list, chi, shared_outlier_norm_list, norm,
                shared_outlier_p_list, p, shared_outlier_agents_list, N, shared_outlier_obs_list, T, shared_outlier_loud_list, loud,
                shared_outlier_outliers_list, r):
    """Appends outlier scenario results to shared lists.

    The function appends states, missing %, RMSE, agents, 
    observations, and outlier group influence to their respective 
    shared lists after each run.

    Args:
    shared_lists: Shared lists to append results to
    states: Number of states
    pct: Percent missing data
    rmse: Root mean squared error
    N: Number of agents
    T: Number of observations
    loud: Outlier group influence type
    
    Returns:
    Tuple of updated shared lists
    """
    shared_outlier_states_list.append(states)
    shared_outlier_missing_list.append(pct)
    shared_outlier_emTime_list.append(chi) 
    shared_outlier_norm_list.append(norm)
    shared_outlier_p_list.append(p) 
    shared_outlier_agents_list.append(N)
    shared_outlier_obs_list.append(T)
    shared_outlier_loud_list.append(loud)
    shared_outlier_outliers_list.append(r)
    return shared_outlier_states_list, shared_outlier_missing_list, shared_outlier_emTime_list, shared_outlier_norm_list, \
            shared_outlier_p_list, shared_outlier_agents_list, shared_outlier_obs_list, shared_outlier_loud_list, shared_outlier_outliers_list
    
def process_outlier(other_args, states, N, T):
    r, shared_outlier_states_list, shared_outlier_missing_list, shared_outlier_emTime_list, shared_outlier_norm_list, \
    shared_outlier_p_list, shared_outlier_agents_list, shared_outlier_obs_list, shared_outlier_loud_list, \
    shared_outlier_outliers_list= other_args
    """Runs outlier scenario simulations and appends results.

    The function assigns agents to groups, generates Markov chains, 
    introduces outlier bias, calculates RMSE, and appends results to 
    shared lists after each run.

    Args:
    shared_lists: Lists to append outputs to
    states: Number of states
    N: Number of agents
    T: Number of time steps
    
    Returns:
    Tuple of updated shared lists
    """

    agents = list(range(N))
    outliers = np.random.choice(agents, int(N*r), replace = False)
    group2 = np.random.choice(outliers, len(outliers) // 2, replace = False)
    group3 = np.random.choice([x for x in outliers if x not in group2], len(outliers) - len(group2), replace = False)
    group4 = [x for x in agents if x not in outliers]

    outlier_state1, outlier_state2 = np.random.choice(range(states), 2, replace=False)

    observed = generate_outlier_matrix(states, outlier_state1, outlier_state2)
    initial_states = []
    for i in range(N):
        if i in group2:
            initial_states.append(outlier_state1)
        elif i in group3:
            initial_states.append(outlier_state2)
        else:
            initial_states.append(random.choice([j for j in range(states) if j not in [outlier_state1, outlier_state2]]))

    chains = pd.DataFrame(generate_markov_chains(observed, initial_states, T, N))
    chains.loc[group2] = outlier_state1
    chains.loc[group3] = outlier_state2
    for row_index in range(N):
        indices1 = np.random.choice(list(range(T)), int(T*0.03), replace = False)
        chains.iloc[row_index, indices1] = np.random.choice([x for x in list(range(states)) if x not in [outlier_state1, outlier_state2]], 1)
        indices2 = np.random.choice(list(range(T)), int(T*0.03), replace = False)
        chains.iloc[row_index, indices2] = np.random.choice([x for x in list(range(states)) if x not in [outlier_state1, outlier_state2]], 1)

    for loud, pct in list(itertools.product(["min", 'maj'], np.linspace(0,0.8,5))):
        result = pd.DataFrame(introduce_outlier_bias(chains, group2, group3, group4, loud, pct))
        start = time.time()
        estimated = extract_transition_matrix(result, states)
        #pi_init, transition_matrix = initialize_pi_P(result,range(states))
        #final = forward_algorithm(result, transition_matrix, T, states)
        #estimated = extract_transition_matrix(final, states)
        #estimated = em_algorithm(result, T, states, 1000)
        end = time.time()

        shared_outlier_states_list, shared_outlier_missing_list, shared_outlier_emTime_list, shared_outlier_norm_list, \
        shared_outlier_p_list, shared_outlier_agents_list, shared_outlier_obs_list, shared_outlier_loud_list,\
        shared_outlier_outliers_list = outAppend(
            shared_outlier_states_list, 
            states, 
            shared_outlier_missing_list, 
            pct,
            shared_outlier_emTime_list, 
            end-start,
            shared_outlier_norm_list,
            np.sum(np.abs(estimated - observed)) / 2,
            shared_outlier_p_list,
            np.linalg.norm(estimated - observed)/np.linalg.norm(observed),
            shared_outlier_agents_list, 
            N,
            shared_outlier_obs_list, 
            T,
            shared_outlier_loud_list,
            loud,
            shared_outlier_outliers_list,
            r)

def polarAppend(shared_polarization_states_list, states, shared_polarization_missing_list, pct,
                shared_polarization_emTime_list, chi, shared_polarization_norm_list, norm, shared_polarization_p_list, p, shared_polarization_agents_list, N, 
                shared_polarization_obs_list, T, shared_polarization_clusters_list, clusters,
                shared_polarization_unaff_list, unaff_pct, shared_polarization_inter_list, inter_prob,
                shared_polarization_ss_ratio_list, ss_ratio):
    """Appends polarization scenario results to shared lists.

    The function appends states, missing %, RMSE, agents, observations, 
    clusters, unaffiliated %, interaction probability, self-selection noise,
    and self-selection ratio to their respective shared lists after each run.

    Args:
    shared_lists: Shared lists to append results to
    states: Number of states  
    N: Number of agents
    T: Number of observations
    clusters: Number of clusters
    unaff_pct: Percent unaffiliated agents
    inter_prob: Interaction probability  
    noise: Self-selection noise type
    r: Self-selection ratio
    pct: Percent missing data
    
    Returns:
    Tuple of updated shared lists  
    """

    shared_polarization_states_list.append(states)
    shared_polarization_missing_list.append(pct)
    shared_polarization_emTime_list.append(chi)
    shared_polarization_norm_list.append(norm)
    shared_polarization_p_list.append(p)
    shared_polarization_agents_list.append(N) 
    shared_polarization_obs_list.append(T)
    shared_polarization_clusters_list.append(clusters)
    shared_polarization_unaff_list.append(unaff_pct)
    shared_polarization_inter_list.append(inter_prob)
    
    shared_polarization_ss_ratio_list.append(ss_ratio)
    return shared_polarization_states_list, shared_polarization_missing_list, shared_polarization_emTime_list, \
        shared_polarization_norm_list, shared_polarization_p_list, shared_polarization_agents_list, shared_polarization_obs_list, \
            shared_polarization_clusters_list, shared_polarization_unaff_list, shared_polarization_inter_list, \
            shared_polarization_ss_ratio_list
    
def process_polar(other_args, k, N, T):
    """Runs polarization scenario simulations and appends results.

    The function generates a base transition matrix, introduces polarization 
    among agents, adds missing data, applies self-selection, and calculates RMSE.

    It appends the results of each run to the shared output lists.

    Args:
    shared_lists: Lists to append outputs to
    states: Number of states
    N: Number of agents  
    T: Number of time steps

    Returns:
    Tuple of updated shared output lists
    """
    
    shared_polarization_states_list, shared_polarization_missing_list, shared_polarization_emTime_list, shared_polarization_norm_list, \
    shared_polarization_p_list, shared_polarization_agents_list, shared_polarization_obs_list, shared_polarization_clusters_list, \
    shared_polarization_unaff_list, shared_polarization_inter_list, \
    shared_polarization_ss_ratio_list, = other_args
    states = k
    observed = generate_standard_transition_matrix(states)
    agents = list(range(N))
    clusters = np.append([2], range(4, states//2 +1,4))
    for cluster, unaff_pct in list(itertools.product(clusters, [0.1, 0.25, 0.5, 0.75])):
        chains = introduce_polarization(observed, agents, states, int(cluster), unaff_pct, 0.1, T)
        for r, pct in list(itertools.product(np.linspace(0,0.8,5), np.linspace(0, 0.8, 5))):
            if not (pct == 0 and r > 0):
                ss = apply_self_selection_bias(pd.DataFrame(chains), r, N, T, pct)
                result = pd.DataFrame(introduce_mcar_missing_data(pd.DataFrame(ss), pct))
                start = time.time()
                estimated = extract_transition_matrix(result, states)        
                #pi_init, transition_matrix = initialize_pi_P(result, range(states))
                #final = forward_algorithm(result, transition_matrix, T, states)
                #estimated = extract_transition_matrix(final, states)
                #estimated = em_algorithm(result, T, states, 1000)
                end = time.time()
                    
                
                shared_polarization_states_list, shared_polarization_missing_list, shared_polarization_emTime_list, shared_polarization_norm_list,\
                shared_polarization_p_list, shared_polarization_agents_list, shared_polarization_obs_list, shared_polarization_clusters_list, \
                shared_polarization_unaff_list, shared_polarization_inter_list, \
                shared_polarization_ss_ratio_list = polarAppend(
                    shared_polarization_states_list, 
                    states,
                    shared_polarization_missing_list, 
                    pct,
                    shared_polarization_emTime_list, 
                    end-start,
                    shared_polarization_norm_list,
                    np.sum(np.abs(estimated - observed)) / 2,
                    shared_polarization_p_list,
                    np.linalg.norm(estimated - observed)/np.linalg.norm(observed),
                    shared_polarization_agents_list, 
                    N,
                    shared_polarization_obs_list, 
                    T,
                    shared_polarization_clusters_list, 
                    cluster,
                    shared_polarization_unaff_list, 
                    unaff_pct,
                    shared_polarization_inter_list, 
                    0.05,
                    shared_polarization_ss_ratio_list, 
                    r)

def baseAppend(shared_base_states_list, states, shared_base_missing_list, pct, shared_base_emTime_list, chi, 
               shared_base_obs_norm_list, norm, shared_base_p_list, p, shared_base_ss_ratio_list, r,
               shared_base_agents_list, N, shared_base_obs_list, T):
    """Appends base scenario simulation results to shared lists.

    The function appends states, missing data %, RMSE, self-selection noise, 
    self-selection ratio, number of agents, and number of observations to their 
    respective shared list after each run.

    Args:
    shared_lists: Shared lists to append results to
    states: Number of states
    pct: Percent of missing data
    rmse: Root mean squared error
    noise: Self-selection noise type
    r: Self-selection ratio
    N: Number of agents
    T: Number of observations
    
    Returns: 
    Tuple of updated shared lists
    """
    shared_base_states_list.append(states)
    shared_base_missing_list.append(pct)
    shared_base_emTime_list.append(chi)
    shared_base_obs_norm_list.append(norm)
    shared_base_p_list.append(p)
    shared_base_ss_ratio_list.append(r)
    shared_base_agents_list.append(N)
    shared_base_obs_list.append(T)
    
    return shared_base_states_list, shared_base_missing_list, shared_base_emTime_list, shared_base_obs_norm_list, shared_base_p_list, \
            shared_base_ss_ratio_list, shared_base_agents_list, shared_base_obs_list

def process_base(other_args, states,  N, T):  
    """
    Processes base scenario runs for agent-based simulations.

    Generates transition matrices, Markov chains, and introduces missing data. 
    Calculates RMSE and appends results to shared lists after each run.

    Args:
    other_args (list): Shared output lists to append results to 
    states (int): Number of states
    N (int): Number of agents
    T (int): Number of time steps
    
    Returns:
    Tuple of updated shared output lists 
    """

    shared_base_states_list, shared_base_missing_list, shared_base_emTime_list, shared_base_obs_norm_list, shared_base_p_list, \
    shared_base_ss_ratio_list, shared_base_agents_list, shared_base_obs_list = other_args

    observed = generate_standard_transition_matrix(states)
    initial_states = generate_initial_states(observed, N)
    chains = generate_markov_chains(observed, initial_states, T, N)
    
    for r, pct in list(itertools.product(np.linspace(0,0.8,5), np.linspace(0, 0.8, 5))):
        if not (pct == 0 and r > 0):
            ss = apply_self_selection_bias(pd.DataFrame(chains), r, N, T, pct)
            result = pd.DataFrame(introduce_mcar_missing_data(pd.DataFrame(ss), pct))
            start = time.time()     
            estimated = extract_transition_matrix(result, states)
            #pi_init, transition_matrix = initialize_pi_P(result,range(states))
            #final = forward_algorithm(result, initial, T, states)
            #estimated = extract_transition_matrix(final, states)
            #estimated = em_algorithm(result, T, states, 5000)
            end = time.time()
            shared_base_states_list, shared_base_missing_list, shared_base_emTime_list, shared_base_obs_norm_list, \
            shared_base_p_list, shared_base_ss_ratio_list, shared_base_agents_list, shared_base_obs_list = baseAppend(
                shared_base_states_list, 
                states,
                shared_base_missing_list, 
                pct,
                shared_base_emTime_list, 
                end - start,
                shared_base_obs_norm_list,
                np.sum(np.abs(estimated - observed)) / 2,
                shared_base_p_list,
                np.linalg.norm(estimated - observed)/np.linalg.norm(observed),
                shared_base_ss_ratio_list, 
                r, 
                shared_base_agents_list, 
                N,
                shared_base_obs_list,
                T)
    
def process_scenario(args):
    """
    Processes agent-based simulation scenarios.

    Dispatches scenario runs to type-specific handlers. Prints status messages.

    Args:
    args (list): Arguments including scenario type, states, agents, time steps, etc.

    Returns:
    None
    
    """

    scenario_type, k, N, T, other_args = args[0], args[1], args[2], args[3], args[4:]
    
    if scenario_type == 'Confirmation':
        print(f"Scenario: {str(scenario_type)} - States: {str(k)} - Agents: {str(N)} - Obs: {str(T)} - started at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
        process_confirmation(other_args, k, N, T)
        print(f"Scenario: {str(scenario_type)} - States: {str(k)} - Agents: {str(N)} - Obs: {str(T)} - complete at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    elif scenario_type == 'Polarization':
        print(f"Scenario: {str(scenario_type)} - States: {str(k)} - Agents: {str(N)} - Obs: {str(T)} - has begun at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
        process_polar(other_args, k, N, T)
        print(f"Scenario: {str(scenario_type)} - States: {str(k)} - Agents: {str(N)} - Obs: {str(T)} - complete at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    elif scenario_type == 'Popularity':
        print(f"Scenario: {str(scenario_type)} - States: {str(k)} - Agents: {str(N)} - Obs: {str(T)} - started at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
        process_popularity(other_args, k, N, T)
        print(f"Scenario: {str(scenario_type)} - States: {str(k)} - Agents: {str(N)} - Obs: {str(T)} - complete at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    elif scenario_type == "Outlier":
        print(f"Scenario: {str(scenario_type)} - States: {str(k)} - Agents: {str(N)} - Obs: {str(T)} - Outliers: {str(other_args[0])} - started at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
        process_outlier(other_args, k, N, T)
        print(f"Scenario: {str(scenario_type)} - States: {str(k)} - Agents: {str(N)} - Obs: {str(T)} - Outliers: {str(other_args[0])} - complete at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    elif scenario_type == "Base":
        print(f"Scenario: {str(scenario_type)} - States: {str(k)} - Agents: {str(N)} - Obs: {str(T)} - started at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
        process_base(other_args, k, N, T)
        print(f"Scenario: {str(scenario_type)} - States: {str(k)} - Agents: {str(N)} - Obs: {str(T)} - complete at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    
if __name__ == '__main__':
    folder = ''
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    for run in range(5):
        # Initialize parameter lists
        confirmation_states_list = []
        confirmation_missing_list = []
        confirmation_emTime_list = []
        confirmation_norm_list = []
        confirmation_p_list = []
        confirmation_agents_list = []
        confirmation_obs_list = []
        confirmation_ss_ratio_list = []
        bias_factor_list = []
        popularity_states_list = []
        popularity_missing_list = []
        popularity_emTime_list = []
        popularity_norm_list = []
        popularity_p_list = []
        popularity_agents_list = []
        popularity_obs_list = []
        popularity_asc_list = []
        outlier_states_list = []
        outlier_missing_list = []
        outlier_emTime_list = []
        outlier_norm_list = []
        outlier_p_list = []
        outlier_agents_list = []
        outlier_obs_list = []
        outlier_loud_list = []
        outlier_outliers_list = []
        base_states_list = []
        base_missing_list = []
        base_emTime_list = []
        base_obs_norm_list = []
        base_p_list = []
        base_ss_ratio_list = []
        base_agents_list = []
        base_obs_list = []
        polarization_states_list = []
        polarization_missing_list = []
        polarization_emTime_list = []
        polarization_norm_list = []
        polarization_p_list = []
        polarization_agents_list = []
        polarization_obs_list = []
        polarization_clusters_list = []
        polarization_unaff_list = []
        polarization_inter_list = []
        polarization_ss_ratio_list = []
        
        with Manager() as manager:
            # Create shared lists for collecting results
            shared_confirmation_states_list = manager.list()
            shared_confirmation_missing_list = manager.list()
            shared_confirmation_emTime_list = manager.list()
            shared_confirmation_norm_list = manager.list()
            shared_confirmation_p_list = manager.list()
            shared_confirmation_agents_list = manager.list()
            shared_confirmation_obs_list = manager.list()
            shared_confirmation_ratio_list = manager.list()
            shared_bias_factor_list = manager.list()
            shared_popularity_states_list = manager.list()
            shared_popularity_missing_list = manager.list()
            shared_popularity_emTime_list = manager.list()
            shared_popularity_norm_list = manager.list()
            shared_popularity_p_list = manager.list()
            shared_popularity_agents_list = manager.list()
            shared_popularity_obs_list = manager.list()
            shared_popularity_asc_list = manager.list()
            shared_outlier_states_list = manager.list()
            shared_outlier_missing_list = manager.list()
            shared_outlier_emTime_list = manager.list()
            shared_outlier_norm_list = manager.list()
            shared_outlier_p_list = manager.list()
            shared_outlier_agents_list = manager.list()
            shared_outlier_obs_list = manager.list()
            shared_outlier_loud_list = manager.list()
            shared_outlier_outliers_list = manager.list()
            shared_polarization_states_list = manager.list()
            shared_polarization_missing_list = manager.list()
            shared_polarization_emTime_list = manager.list()
            shared_polarization_norm_list = manager.list()
            shared_polarization_p_list = manager.list()
            shared_polarization_agents_list = manager.list()
            shared_polarization_obs_list = manager.list()
            shared_polarization_clusters_list = manager.list()
            shared_polarization_unaff_list = manager.list()
            shared_polarization_inter_list = manager.list()
            shared_polarization_ss_ratio_list = manager.list()
            shared_base_states_list = manager.list()
            shared_base_missing_list = manager.list()
            shared_base_emTime_list = manager.list()
            shared_base_obs_norm_list = manager.list()
            shared_base_p_list = manager.list()
            shared_base_ss_ratio_list = manager.list()
            shared_base_agents_list = manager.list()
            shared_base_obs_list = manager.list()

            # Parameter values for each scenario    
            states = [4,8,16,32,64]
            agents = [50,150,300]
            observations = [150,300,500]
            args_list = [
                            ('Base', k, N, T, shared_base_states_list, shared_base_missing_list, shared_base_emTime_list, shared_base_obs_norm_list, 
                                shared_base_p_list, shared_base_ss_ratio_list, shared_base_agents_list, shared_base_obs_list)
                                for k, N, T in list(itertools.product(states, agents, observations))
                        ] + [
                            ('Polarization', k, N, T, shared_polarization_states_list, shared_polarization_missing_list, shared_polarization_emTime_list, \
                                    shared_polarization_norm_list, shared_polarization_p_list, shared_polarization_agents_list, shared_polarization_obs_list, 
                                    shared_polarization_clusters_list, shared_polarization_unaff_list, shared_polarization_inter_list, \
                                    shared_polarization_ss_ratio_list)
                                for k, N, T in list(itertools.product(states, agents, observations))
                        ] + [
                            ('Outlier', k, N, T, r, shared_outlier_states_list, shared_outlier_missing_list, shared_outlier_emTime_list, 
                                    shared_outlier_norm_list, shared_outlier_p_list, shared_outlier_agents_list, shared_outlier_obs_list, \
                                        shared_outlier_loud_list, shared_outlier_outliers_list)
                                for k, N, T, r in list(itertools.product(states, agents, observations, np.linspace(0.1, 0.5, 5)))
                        ] + [
                            ('Popularity', k, N, T, shared_popularity_states_list, shared_popularity_missing_list, shared_popularity_emTime_list, 
                                    shared_popularity_norm_list, shared_popularity_p_list, shared_popularity_agents_list, shared_popularity_obs_list, shared_popularity_asc_list)
                                for k, N, T in list(itertools.product(states, agents, observations))
                        ] + [
                            ('Confirmation', k, N, T, shared_confirmation_states_list, shared_confirmation_missing_list,
                                    shared_confirmation_emTime_list, shared_confirmation_norm_list, shared_confirmation_p_list, shared_confirmation_agents_list, \
                                    shared_confirmation_obs_list, shared_confirmation_ratio_list, shared_bias_factor_list)
                                for k, N, T in list(itertools.product(states, agents, observations))
                        ]
                        

            # Use multi-threading to each scenario 
            results = pool.map(process_scenario, args_list)

            # Collect all simulation results into appropriate lists
            confirmation_states_list = list(shared_confirmation_states_list)
            confirmation_missing_list = list(shared_confirmation_missing_list)
            confirmation_emTime_list = list(shared_confirmation_emTime_list)
            confirmation_norm_list = list(shared_confirmation_norm_list)
            confirmation_p_list = list(shared_confirmation_p_list)
            confirmation_agents_list = list(shared_confirmation_agents_list)
            confirmation_obs_list = list(shared_confirmation_obs_list)
            confirmation_ss_ratio_list = list(shared_confirmation_ratio_list)
            bias_factor_list = list(shared_bias_factor_list)
            popularity_states_list = list(shared_popularity_states_list)
            popularity_missing_list = list(shared_popularity_missing_list)
            popularity_emTime_list = list(shared_popularity_emTime_list)
            popularity_norm_list = list(shared_popularity_norm_list)
            popularity_p_list = list(shared_popularity_p_list)
            popularity_agents_list = list(shared_popularity_agents_list)
            popularity_obs_list = list(shared_popularity_obs_list)
            popularity_asc_list = list(shared_popularity_asc_list)
            outlier_states_list = list(shared_outlier_states_list)
            outlier_missing_list = list(shared_outlier_missing_list)
            outlier_emTime_list = list(shared_outlier_emTime_list)
            outlier_norm_list = list(shared_outlier_norm_list)
            outlier_p_list = list(shared_outlier_p_list)
            outlier_agents_list = list(shared_outlier_agents_list)
            outlier_obs_list = list(shared_outlier_obs_list)
            outlier_loud_list = list(shared_outlier_loud_list)
            outlier_outliers_list = list(shared_outlier_outliers_list)
            polarization_states_list = list(shared_polarization_states_list)
            polarization_missing_list = list(shared_polarization_missing_list)
            polarization_emTime_list = list(shared_polarization_emTime_list)
            polarization_norm_list = list(shared_polarization_norm_list)
            polarization_p_list = list(shared_polarization_p_list)
            polarization_agents_list = list(shared_polarization_agents_list)
            polarization_obs_list = list(shared_polarization_obs_list)
            polarization_clusters_list = list(shared_polarization_clusters_list)
            polarization_unaff_list = list(shared_polarization_unaff_list)
            polarization_inter_list = list(shared_polarization_inter_list)
            polarization_ss_ratio_list = list(shared_polarization_ss_ratio_list)
            base_states_list = list(shared_base_states_list)
            base_missing_list = list(shared_base_missing_list)
            base_emTime_list = list(shared_base_emTime_list)
            base_obs_norm_list = list(shared_base_obs_norm_list)
            base_p_list = list(shared_base_p_list)
            base_ss_ratio_list = list(shared_base_ss_ratio_list)
            base_agents_list = list(shared_base_agents_list)
            base_obs_list = list(shared_base_obs_list)

        # Use generated lists to form different dataframes and save to csv
        #polar_file = f'{folder}Polarization/polar imputed.csv'
           
        polarization_dict = {
            'run': [run] * len(polarization_agents_list),
            'agents': polarization_agents_list,
            'observations': polarization_obs_list,
            'missing': polarization_missing_list,
            'clusters': polarization_clusters_list,
            'Time': polarization_emTime_list,
            'TVD': polarization_norm_list,
            'Norm': polarization_p_list,
            'SS ratio': polarization_ss_ratio_list,
            'states': polarization_states_list,
            'unaff': polarization_unaff_list
        }
        
        confirmation_dict = {
            'run': [run] * len(confirmation_agents_list),
            'agents': confirmation_agents_list,
            'observations': confirmation_obs_list,
            'states': confirmation_states_list,
            'Time': confirmation_emTime_list,
            'TVD': confirmation_norm_list,
            'Norm': confirmation_p_list,
            'missing': confirmation_missing_list,
            'SS ratio': confirmation_ss_ratio_list,
            'bias factor': bias_factor_list
        }
        
        popularity_dict = {
            'run': [run] * len(popularity_agents_list),
            'agents': popularity_agents_list,
            'observations': popularity_obs_list,
            'states': popularity_states_list,
            'Time': popularity_emTime_list,
            'TVD': popularity_norm_list,
            'Norm': popularity_p_list,
            'missing': popularity_missing_list,
            'low': popularity_asc_list
        }
        
        outlier_dict = {
            'run': [run] * len(outlier_agents_list),
            'agents': outlier_agents_list,
            'observations': outlier_obs_list,
            'states': outlier_states_list,
            'Time': outlier_emTime_list,
            'TVD': outlier_norm_list,
            'Norm': outlier_p_list,
            'loud': outlier_loud_list,
            'outliers': outlier_outliers_list,
            'missing': outlier_missing_list
        }
        
        base_dict = {
            'run': [run] * len(base_states_list),
            'state': base_states_list,
            'missing': base_missing_list,
            'Time': base_emTime_list,
            'TVD': base_obs_norm_list,
            'Norm': base_p_list,
            'SS ratio': base_ss_ratio_list,
            'agents': base_agents_list,
            'observations': base_obs_list 
        }
        
        base_file = folder + 'base.csv' 
        polar_file = folder + 'polar.csv'
        confirmation_file = folder + 'confirmation.csv'
        popularity_file = folder + 'popular.csv'
        outlier_file = folder + 'outlier.csv' 
        print(f"Results saving for run {run}")
        if run == 0:
            pd.DataFrame(polarization_dict).to_csv(polar_file, index = False)
            pd.DataFrame(confirmation_dict).to_csv(confirmation_file, index = False)
            pd.DataFrame(popularity_dict).to_csv(popularity_file, index = False)
            pd.DataFrame(outlier_dict).to_csv(outlier_file, index = False)
            pd.DataFrame(base_dict).to_csv(base_file, index = False)
        else:
            existing_polar = pd.read_csv(polar_file)  
            updated_polar = pd.concat([existing_polar, pd.DataFrame(polarization_dict)], ignore_index = True)
            updated_polar.to_csv(polar_file, index = False)
            existing_confirmation = pd.read_csv(confirmation_file)
            updated_confirmation = pd.concat([existing_confirmation, pd.DataFrame(confirmation_dict)], ignore_index = True)
            updated_confirmation.to_csv(confirmation_file, index = False)
            existing_popularity = pd.read_csv(popularity_file)
            updated_popularity = pd.concat([existing_popularity, pd.DataFrame(popularity_dict)], ignore_index = True)
            updated_popularity.to_csv(popularity_file, index = False)
            existing_outlier = pd.read_csv(outlier_file)
            updated_outlier = pd.concat([existing_outlier, pd.DataFrame(outlier_dict)], ignore_index = True)
            updated_outlier.to_csv(outlier_file, index = False)
            existing_base = pd.read_csv(base_file)
            updated_base = pd.concat([existing_base, pd.DataFrame(base_dict)], ignore_index = True)
            updated_base.to_csv(base_file, index = False)




