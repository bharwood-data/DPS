import os
import glob
import pandas as pd
import numpy as np
import time
import itertools
from itertools import chain
from scipy.stats import chi2
import warnings
import random 
import multiprocessing
from multiprocessing import Manager, Value

warnings.filterwarnings('ignore')
# Function for calculating RMSE
def rmse(matrix, truth):
    """
    Calculates the root mean squared error (RMSE) between a matrix and a ground truth.

    Args:
        matrix (numpy.ndarray): The matrix to compare.
        truth (numpy.ndarray): The ground truth matrix.

    Returns:
        float: The RMSE between the matrix and the ground truth.

    Example:
        ```python
        import numpy as np

        # Creating matrices
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        truth = np.array([[2, 3, 4], [5, 6, 7]])

        # Calculating the RMSE
        result = rmse(matrix, truth)
        print(result)  # Output: <calculated value>
        ```
    """
    return np.sqrt(np.sum((matrix - truth) ** 2)) / truth.shape[0]

def forward_algorithm(pi, P, data, num_states):
    """
    Performs the forward algorithm for initial data imputation.

    Args:
        pi (numpy.ndarray): The initial probability vector.
        P (numpy.ndarray): The transition matrix.
        data (numpy.ndarray): The input data.

    Returns:
        numpy.ndarray: The alpha values.

    Example:
        ```python
        pi = np.array([0.2, 0.3, 0.5])
        P = np.array([[0.4, 0.3, 0.3], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        data = np.array([[1, 2, 3], [2, 1, np.nan], [3, 2, 1]])
        alpha = forward_algorithm(pi, P, data)
        print("Alpha values:")
        print(alpha)
        ```
    """

    N, T = data.shape
    forward_probabilities = np.zeros((N, T, num_states))
    
    # Initialize the first time step with the initial state probabilities
    forward_probabilities[:, 0, :] = pi
    
    # Forward Algorithm
    for t in range(1, T):
        observed_data = np.isnan(data.iloc[:, t])
        missing_data = ~observed_data

        # Handle observed data
        observed_states = data.loc[missing_data, t].astype(int)
        forward_probabilities[missing_data, t, observed_states] = 1.0

        # Calculate forward probabilities for missing data
        transition_probabilities = P.T  # Transpose the transition matrix
        forward_probabilities[missing_data, t] = np.dot(
            forward_probabilities[missing_data, t - 1], transition_probabilities
        )

        # Normalize the forward probabilities for missing data
        row_sums = np.sum(forward_probabilities[missing_data, t], axis=1)
        forward_probabilities[missing_data, t] /= row_sums[:, np.newaxis]
                    
    return forward_probabilities

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
    
    pi_init = generate_pi(Y, states)
    P_init = extract_transition_matrix(Y, k)
    
    return pi_init, P_init

def impute_missing_values(data, forward_probabilities):
    """
    Imputes missing values in the observed data using the alpha values.

    Args:
        observed_data (numpy.ndarray): The observed data.
        alpha (numpy.ndarray): The alpha values.
        max_states (int): The maximum state value.

    Returns:
        numpy.ndarray: The imputed data.

    Example:
        ```python
        observed_data = np.array([[1, 2, np.nan], [2, np.nan, 3], [3, 2, 1]])
        alpha = np.array([[0.4, 0.3, 0.3], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]])
        max_states = 3
        imputed_data = impute_missing_values(observed_data, alpha, max_states)
        print("Imputed data:")
        print(imputed_data)
        ```
    """
    # Create a mask of missing values in your data
    mask = np.isnan(data.values)

    # Calculate the index of the maximum forward probability for each missing value
    argmax_indices = np.argmax(forward_probabilities[:, :, :], axis=2)

    # Replace missing values in imputed_data using the argmax_indices
    imputed_data = data.copy()
    imputed_data.values[mask] = argmax_indices[mask]
    return imputed_data

def em_algorithm(data, T, num_states, num_iterations, tol=1e-4):
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
    
    forward_probabilities = forward_algorithm(pi_init, transition_matrix, data, num_states)
    
    for _ in range(num_iterations):
        # E-step: Impute missing data using the Forward Algorithm
        imputed_data = impute_missing_values(data, forward_probabilities)

        # M-step: Update the transition matrix
        transition_matrix = extract_transition_matrix(imputed_data, num_states)

        # Calculate log-likelihood for convergence check
        log_likelihood = calculate_log_likelihood(imputed_data, T, transition_matrix, pi_init)

        # Check for convergence based on log-likelihood change
        if abs(log_likelihood - prev_log_likelihood) < tol:
            break

        prev_log_likelihood = log_likelihood
        pi_new, transition_matrix = initialize_pi_P(imputed_data, list(range(num_states)))
        forward_probabilities = forward_algorithm(pi_new, transition_matrix, data, num_states)
        
    return transition_matrix[:num_states, :num_states]

def apply_self_selection_bias(df, ratio, N, T):
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
        posting_probs = np.random.uniform(0, 1, size = ssAgentCount)
            
        for t in range(T):
            masks = [np.random.rand(1) < p for p in posting_probs]
            
            # Update the DataFrame using masks
            for agent in ssAgents:
                if not masks[[x for x in range(len(ssAgents)) if ssAgents[x] == agent][0]][0]:
                    df.loc[agent, t] = np.nan
        
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
    for i, j in itertools.product(range(N), range(T)):
        if np.random.rand() < missing_prob:
            missing_data[i, j] = np.nan
    return missing_data

def extract_transition_matrix(Y, num_states):
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
    Y_array = Y.values
    P = np.zeros((num_states, num_states), dtype=float)

    for i, j in itertools.product(range(num_states), range(num_states)):
        # Create a mask for valid transitions
        valid_transitions = np.logical_and(~np.isnan(Y_array[:-1, :]), Y_array[:-1, :] == i)
        valid_transitions_next = Y_array[1:, :] == j

        # Count the valid transitions
        count = np.count_nonzero(np.logical_and(valid_transitions, valid_transitions_next))
        total = np.count_nonzero(valid_transitions)

        if total > 0:
            P[i, j] = count / total

    return P

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

def generate_agent_chain(transition_matrix, initial_state, num_steps):
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

        for _ in range(num_steps):
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
        vals = np.random.dirichlet([1] * (num_states - 1))
        scaled_vals = vals * 0.05
        fullRowVals = np.random.dirichlet([1] * num_states)
        for j in range(num_states):
            if (
                (i == outlier_state1
                and j == outlier_state1)
                or (i == outlier_state2
                and j == outlier_state2)
            ):
                p[i, j] = 0.95
            elif i in [outlier_state1, outlier_state2]:
                p[i, j] = np.random.choice(scaled_vals, 1, replace = False)
            elif j in [outlier_state1, outlier_state2]:
                p[i, j] = 0.1 + random.uniform(-0.05, 0.05)
            else:
                p[i, j] = np.random.choice(fullRowVals, 1, replace = False)
    
    return p / np.sum(p, axis = 1, keepdims= True)

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
    extra = int(len(input_list) % clusters)
    sublist_size = int(len(input_list) // clusters)
    return [input_list[i:i + sublist_size + e] for i in range(0, len(input_list), sublist_size) for e in range(extra +1)]
    
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
    
    return polar_matrix / np.sum(polar_matrix, axis=1, keepdims=True)

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
    polar_count = int(len(agents) * (1 - unaff_pct))
    
    polar_agents, unaff_agents = divide_agents(agents, unaff_pct)
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
    num_agents, num_steps = data.shape

    for agent in less_vocal_group:
        for t in range(num_steps):
            if np.random.rand() < missing_prob:
                data.at[agent, t] = np.nan

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
    d2 = d1.copy()
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
        
        state = sorted_states['State'].values[i]
        # Calculate the number of observations needed for the current state
        observations_needed = int(min(len(state_indices_list[state][0]), remaining_missing_count) * 0.9 / (i if i > 0 else 1))

        # Randomly generate observations for the current state
        if observations_needed > 0 and len(state_indices_list[state]) > 0:
            # Randomly select indices for the current state
            rows_to_select = np.random.choice(state_indices_list[state][0], observations_needed, replace=False)
            cols_to_select = np.random.choice(state_indices_list[state][1], observations_needed, replace=False)
            for index in list(zip(rows_to_select, cols_to_select)):
                d2.iloc[index[0], index[1]] = np.nan

            # Update the remaining missing count
            remaining_missing_count -= observations_needed
        if remaining_missing_count <= 0:
            break

    return d2   

def condition1(x):
    # Subgrouping condition for outliers
    return x % 5 == 0

def condition2(x):
    # Subgrouping condition for outliers
    return x % 2 == 0

def introduce_confirmation_bias(num_states, num_agents, confirmation_bias_factor, obs):
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
    transition_matrix = np.full((num_states, num_states), 1 / num_states)

    # Apply confirmation bias within the transition matrix
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                # Higher probability of staying in the same state (confirmation bias)
                transition_matrix[i, j] *= random.uniform(confirmation_bias_factor, 1)
            else:
                # Lower probability of moving to different states (disconfirmation bias)
                transition_matrix[i, j] *= random.uniform(0, confirmation_bias_factor / num_agents)

    # Normalize the transition matrix
    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
    
    initial_states = generate_initial_states(transition_matrix, num_agents)
        
    return transition_matrix, generate_markov_chains(transition_matrix, initial_states, obs, num_agents)

def cAppend(shared_confirmation_states_list, states, shared_confirmation_missing_list, pct, shared_confirmation_emTime_list, chi, 
        shared_confirmation_p_list, p , shared_confirmation_agents_list, N, shared_confirmation_obs_list, T, shared_confirmation_ss_ratio_list, 
        ss_ratio, shared_bias_factor_list, cbf):
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
    shared_confirmation_p_list.append(p)
    shared_confirmation_agents_list.append(N)
    shared_confirmation_obs_list.append(T)
    shared_confirmation_ss_ratio_list.append(ss_ratio)
    shared_bias_factor_list.append(cbf)
    return shared_confirmation_states_list, shared_confirmation_missing_list, shared_confirmation_emTime_list, shared_confirmation_p_list, \
        shared_confirmation_agents_list, shared_confirmation_obs_list, shared_confirmation_ss_ratio_list, shared_bias_factor_list

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
    shared_confirmation_states_list, shared_confirmation_missing_list, shared_confirmation_emTime_list, shared_confirmation_p_list,\
        shared_confirmation_agents_list, shared_confirmation_obs_list, shared_confirmation_ss_ratio_list, \
        shared_bias_factor_list = other_args
    for cbf in [0.5, 0.6, 0.7, 0.8, 0.9]:
        observed, data = introduce_confirmation_bias(states, N, cbf, T)
        for r, pct in list(itertools.product([0, 0.1, 0.25, 0.5, 0.8], np.linspace(0, 0.9, 10))):
            missing_data = pd.DataFrame(introduce_mcar_missing_data(pd.DataFrame(data), pct))
            result = apply_self_selection_bias(missing_data, r, N, T)
            start = time.time()
            #estimated = extract_transition_matrix(result, states)
            #pi_init, transition_matrix = initialize_pi_P(result,range(states))
            #forward_probabilities = forward_algorithm(pi_init, transition_matrix, result, states)
            #final = impute_missing_values(result, forward_probabilities)
            #estimated = extract_transition_matrix(final, states)
            estimated = em_algorithm(result, T, states, 1000)
            end = time.time()
            shared_confirmation_states_list, shared_confirmation_missing_list, shared_confirmation_emTime_list, shared_confirmation_p_list,\
            shared_confirmation_agents_list, shared_confirmation_obs_list, shared_confirmation_ss_ratio_list, \
            shared_bias_factor_list = cAppend(
                shared_confirmation_states_list, 
                states,
                shared_confirmation_missing_list, 
                pct,
                shared_confirmation_emTime_list, 
                end-start,
                shared_confirmation_p_list,
                np.linalg.norm(estimated- observed),\
                shared_confirmation_agents_list, 
                N,
                shared_confirmation_obs_list, 
                T,
                shared_confirmation_ss_ratio_list, \
                r,
                shared_bias_factor_list,
                cbf)

def popAppend(shared_popularity_states_list, states, shared_popularity_missing_list, pct, shared_popularity_emTime_list, chi,
        shared_popularity_p_list, p, shared_popularity_agents_list, N, shared_popularity_obs_list, T, shared_popularity_asc_list, asc):
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
    shared_popularity_p_list.append(p)
    shared_popularity_agents_list.append(N) 
    shared_popularity_obs_list.append(T)
    shared_popularity_asc_list.append(asc)
    return shared_popularity_states_list, shared_popularity_missing_list, shared_popularity_emTime_list, \
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
    shared_popularity_states_list, shared_popularity_missing_list, shared_popularity_emTime_list, \
    shared_popularity_p_list, shared_popularity_agents_list, shared_popularity_obs_list, shared_popularity_asc_list = other_args
    observed = generate_standard_transition_matrix(states)
    initial_states = generate_initial_states(observed, N)
    d1 = pd.DataFrame(generate_markov_chains(observed, initial_states, T, N))
    probs = np.array([np.sum(d1.values == state) / d1.values.size for state in range(states)])
    state_probabilities = pd.DataFrame({'State': np.arange(0, states), 'Probability': probs})
    for asc, pct in list(itertools.product([True, False], np.linspace(0,0.9,10))):
            result = introduce_popularity_bias(d1, state_probabilities, pct, N, T, asc)
            start = time.time()
            #estimated = extract_transition_matrix(result, states)
            #pi_init, transition_matrix = initialize_pi_P(result,range(states))
            #forward_probabilities = forward_algorithm(pi_init, transition_matrix, result, states)
            #final = impute_missing_values(result, forward_probabilities)
            #estimated = extract_transition_matrix(final, states)
            estimated = em_algorithm(result, T, states, 1000)
            end = time.time()
            
            shared_popularity_states_list, shared_popularity_missing_list, shared_popularity_emTime_list, \
            shared_popularity_p_list, shared_popularity_agents_list, shared_popularity_obs_list, shared_popularity_asc_list = popAppend(
                shared_popularity_states_list, 
                states, 
                shared_popularity_missing_list, 
                pct, 
                shared_popularity_emTime_list, 
                end-start, 
                shared_popularity_p_list,
                np.linalg.norm(estimated - observed),
                shared_popularity_agents_list, 
                N, 
                shared_popularity_obs_list, 
                T, 
                shared_popularity_asc_list, 
                asc)
   
def outAppend(shared_outlier_states_list, states, shared_outlier_missing_list, pct, shared_outlier_emTime_list, chi,
                shared_outlier_p_list, p, shared_outlier_agents_list, N, shared_outlier_obs_list, T, shared_outlier_loud_list, loud):
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
    shared_outlier_p_list.append(p) 
    shared_outlier_agents_list.append(N)
    shared_outlier_obs_list.append(T)
    shared_outlier_loud_list.append(loud)
    return shared_outlier_states_list, shared_outlier_missing_list, shared_outlier_emTime_list, \
            shared_outlier_p_list, shared_outlier_agents_list, shared_outlier_obs_list, shared_outlier_loud_list
    
def process_outlier(other_args, states, N, T):
    shared_outlier_states_list, shared_outlier_missing_list, shared_outlier_emTime_list, \
    shared_outlier_p_list, shared_outlier_agents_list, shared_outlier_obs_list, \
    shared_outlier_loud_list, condition1, condition2 = other_args
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

    agent_temp = list(range(N))
    group1 = [x for x in agent_temp if condition1(x)]
    group2 = [x for x in group1 if condition2(x)]
    group3 = [x for x in group1 if not condition2(x)]
    group4 = [x for x in agent_temp if x not in group1]
    
    outlier_state1, outlier_state2 = np.random.choice(range(states), 2, replace=False)
    
    observed = generate_standard_transition_matrix(states)
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
        chains.iloc[row_index, indices1] = np.random.choice([x for x in list(range(states)) if x!=outlier_state1], 1)
        indices2 = np.random.choice(list(range(T)), int(T*0.03), replace = False)
        chains.iloc[row_index, indices2] = np.random.choice([x for x in list(range(states)) if x!=outlier_state2], 1)
    
    for loud, pct in list(itertools.product(["min", 'maj'], np.linspace(0,0.9,10))):
        result = introduce_outlier_bias(chains, group2, group3, group4, loud, pct)
        start = time.time()
        #estimated = extract_transition_matrix(result, states)
        #pi_init, transition_matrix = initialize_pi_P(result,range(states))
        #forward_probabilities = forward_algorithm(pi_init, transition_matrix, result, states)
        #final = impute_missing_values(result, forward_probabilities)
        #estimated = extract_transition_matrix(final, states)
        estimated = em_algorithm(result, T, states, 1000)
        end = time.time()
                
        shared_outlier_states_list, shared_outlier_missing_list, shared_outlier_emTime_list, \
        shared_outlier_p_list, shared_outlier_agents_list, shared_outlier_obs_list, shared_outlier_loud_list = outAppend(
            shared_outlier_states_list, 
            states, 
            shared_outlier_missing_list, 
            pct,
            shared_outlier_emTime_list, 
            end-start,
            shared_outlier_p_list,
            np.linalg.norm(estimated - observed),
            shared_outlier_agents_list, 
            N,
            shared_outlier_obs_list, 
            T,
            shared_outlier_loud_list,
            loud)

def polarAppend(shared_polarization_states_list, states, shared_polarization_missing_list, pct,
                shared_polarization_emTime_list, chi, shared_polarization_p_list, p, shared_polarization_agents_list, N, 
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
    shared_polarization_p_list.append(p)
    shared_polarization_agents_list.append(N) 
    shared_polarization_obs_list.append(T)
    shared_polarization_clusters_list.append(clusters)
    shared_polarization_unaff_list.append(unaff_pct)
    shared_polarization_inter_list.append(inter_prob)
    
    shared_polarization_ss_ratio_list.append(ss_ratio)
    return shared_polarization_states_list, shared_polarization_missing_list, \
            shared_polarization_emTime_list, shared_polarization_p_list, shared_polarization_agents_list, shared_polarization_obs_list, shared_polarization_clusters_list, \
            shared_polarization_unaff_list, shared_polarization_inter_list, \
            shared_polarization_ss_ratio_list
    
def process_polar(other_args, states, N, T):
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
    
    shared_polarization_states_list, shared_polarization_missing_list, shared_polarization_emTime_list, \
    shared_polarization_p_list, shared_polarization_agents_list, shared_polarization_obs_list, shared_polarization_clusters_list, \
    shared_polarization_unaff_list, shared_polarization_inter_list, \
    shared_polarization_ss_ratio_list, = other_args

    observed = generate_standard_transition_matrix(states)
    agents = list(range(N))
    for unaff_pct, r, pct in list(itertools.product([0.1, 0.25], [0, 0.1, 0.25, 0.5, 0.8], np.linspace(0, 0.9, 10))):
        clusters = 2 if states == 4 else states/4
        result = introduce_polarization(observed, agents, states, int(clusters), unaff_pct, 0.05, T)
        missing_data = introduce_mcar_missing_data(result, pct)
        result = apply_self_selection_bias(pd.DataFrame(missing_data), r, N, T)
        start = time.time()
        #estimated = extract_transition_matrix(result, states)
        #pi_init, transition_matrix = initialize_pi_P(result,range(states))
        #forward_probabilities = forward_algorithm(pi_init, transition_matrix, result, states)
        #final = impute_missing_values(result, forward_probabilities)
        #estimated = extract_transition_matrix(final, states)
        estimated = em_algorithm(result, T, states, 1000)
        end = time.time()
        
        shared_polarization_states_list, shared_polarization_missing_list, shared_polarization_emTime_list, \
        shared_polarization_p_list, shared_polarization_agents_list, shared_polarization_obs_list, shared_polarization_clusters_list, \
        shared_polarization_unaff_list, shared_polarization_inter_list, \
        shared_polarization_ss_ratio_list = polarAppend(
            shared_polarization_states_list, 
            states,
            shared_polarization_missing_list, 
            pct,
            shared_polarization_emTime_list, 
            end-start,
            shared_polarization_p_list,
            np.linalg.norm(estimated - observed),
            shared_polarization_agents_list, 
            N,
            shared_polarization_obs_list, 
            T,
            shared_polarization_clusters_list, 
            clusters,
            shared_polarization_unaff_list, 
            unaff_pct,
            shared_polarization_inter_list, 
            0.05,
            shared_polarization_ss_ratio_list, 
            r)

def baseAppend(shared_base_states_list, states, shared_base_missing_list, pct, shared_base_emTime_list, chi, 
               shared_base_p_list, p, shared_base_ss_ratio_list, r,
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
    shared_base_p_list.append(p)
    shared_base_ss_ratio_list.append(r)
    shared_base_agents_list.append(N)
    shared_base_obs_list.append(T)
    
    return shared_base_states_list, shared_base_missing_list, shared_base_emTime_list, shared_base_p_list, shared_base_ss_ratio_list, \
                shared_base_agents_list, shared_base_obs_list
        
def process_base(other_args, states, N, T):
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

    shared_base_states_list, shared_base_missing_list, shared_base_emTime_list, shared_base_p_list, \
    shared_base_ss_ratio_list, shared_base_agents_list, shared_base_obs_list = other_args

    observed = generate_standard_transition_matrix(states)
    initial_states = generate_initial_states(observed, N)
    chains = generate_markov_chains(observed, initial_states, T, N)
    
    for r, pct in list(itertools.product([0, 0.1, 0.25, 0.5, 0.8], np.linspace(0,0.9,10))):
        missing_data = introduce_mcar_missing_data(pd.DataFrame(chains), pct)
        result = apply_self_selection_bias(pd.DataFrame(missing_data), r, N, T)
        start = time.time()
        #estimated = extract_transition_matrix(result, states)
        #pi_init, transition_matrix = initialize_pi_P(result,range(states))
        #forward_probabilities = forward_algorithm(pi_init, transition_matrix, result, states)
        #final = impute_missing_values(result, forward_probabilities)
        #estimated = extract_transition_matrix(final, states)
        estimated = em_algorithm(result, T, states, 1000)
        end = time.time()
                
        shared_base_states_list, shared_base_missing_list, shared_base_emTime_list, shared_base_p_list, \
        shared_base_ss_ratio_list, shared_base_agents_list, shared_base_obs_list = baseAppend(
            shared_base_states_list, 
            states,
            shared_base_missing_list, 
            pct,
            shared_base_emTime_list, 
            end - start,
            shared_base_p_list,
            np.linalg.norm(estimated - observed),
            shared_base_ss_ratio_list, 
            r, 
            shared_base_agents_list, 
            N,
            shared_base_obs_list,
            T)
    
def save_dict_to_csv(data_dict, file_path):
    pd.DataFrame(data_dict).to_csv(file_path, index=False)
    
def process_scenario(args):
    """
    Processes agent-based simulation scenarios.

    Dispatches scenario runs to type-specific handlers. Prints status messages.

    Args:
    args (list): Arguments including scenario type, states, agents, time steps, etc.

    Returns:
    None
    
    """

    scenario_type, states, N, T, other_args = args[0], args[1], args[2], args[3], args[4:]
    
    print(f"Scenario: {str(scenario_type)} - States: {str(states)} - Agents: {str(N)} - Obs: {str(T)}")
    
    if scenario_type == 'Confirmation':
        process_confirmation(other_args, states, N, T)
    elif scenario_type == 'Popularity':
        process_popularity(other_args, states, N, T)
    elif scenario_type == "Outlier":
        process_outlier(other_args, states, N, T)
    elif scenario_type == "Polarization":
        process_polar(other_args, states, N, T)
    else:
        process_base(other_args, states, N, T)
 
        
    print(f"Scenario: {str(scenario_type)} - States: {str(states)} - Agents: {str(N)} - Obs: {str(T)} - complete at {time.strftime('%H:%M:%S', time.gmtime(time.time()))}")
    
if __name__ == '__main__':
    folder = 'C:/Users/kymag/OneDrive/Documents/DPS/Data/My Matrices/Results/'
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)

    # Initialize parameter lists
    confirmation_states_list = []
    confirmation_missing_list = []
    confirmation_emTime_list = []
    confirmation_p_list = []
    confirmation_agents_list = []
    confirmation_obs_list = []
    confirmation_ss_ratio_list = []
    bias_factor_list = []
    popularity_states_list = []
    popularity_missing_list = []
    popularity_emTime_list = []
    popularity_p_list = []
    popularity_agents_list = []
    popularity_obs_list = []
    popularity_asc_list = []
    outlier_states_list = []
    outlier_missing_list = []
    outlier_emTime_list = []
    outlier_p_list = []
    outlier_agents_list = []
    outlier_obs_list = []
    outlier_loud_list = []
    base_states_list = []
    base_missing_list = []
    base_emTime_list = []
    base_p_list = []
    base_ss_ratio_list = []
    base_agents_list = []
    base_obs_list = []
    polarization_states_list = []
    polarization_missing_list = []
    polarization_emTime_list = []
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
        shared_confirmation_p_list = manager.list()
        shared_confirmation_agents_list = manager.list()
        shared_confirmation_obs_list = manager.list()
        shared_confirmation_ratio_list = manager.list()
        shared_bias_factor_list = manager.list()
        shared_popularity_states_list = manager.list()
        shared_popularity_missing_list = manager.list()
        shared_popularity_emTime_list = manager.list()
        shared_popularity_p_list = manager.list()
        shared_popularity_agents_list = manager.list()
        shared_popularity_obs_list = manager.list()
        shared_popularity_asc_list = manager.list()
        shared_outlier_states_list = manager.list()
        shared_outlier_missing_list = manager.list()
        shared_outlier_emTime_list = manager.list()
        shared_outlier_p_list = manager.list()
        shared_outlier_agents_list = manager.list()
        shared_outlier_obs_list = manager.list()
        shared_outlier_loud_list = manager.list()
        shared_polarization_states_list = manager.list()
        shared_polarization_missing_list = manager.list()
        shared_polarization_emTime_list = manager.list()
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
        shared_base_p_list = manager.list()
        shared_base_ss_ratio_list = manager.list()
        shared_base_agents_list = manager.list()
        shared_base_obs_list = manager.list()

        # Parameter values for each scenario    
        args_list = [
                        ('Base', states, N, T, shared_base_states_list, shared_base_missing_list, shared_base_emTime_list, shared_base_p_list, 
                                shared_base_ss_ratio_list, shared_base_agents_list, shared_base_obs_list)
                            for states, N, T in list(itertools.product([4,8,16,32,64], [50, 150, 300], [150, 300, 500]))
                    ] + [
                        ('Polarization', states, N, T, shared_polarization_states_list, shared_polarization_missing_list, shared_polarization_emTime_list, \
                                shared_polarization_p_list, shared_polarization_agents_list, shared_polarization_obs_list, shared_polarization_clusters_list, \
                                shared_polarization_unaff_list, shared_polarization_inter_list, \
                                shared_polarization_ss_ratio_list)
                            for states, N, T in list(itertools.product([4,8,16,32,64], [50, 150, 300], [150, 300, 500]))
                    ] + [
                        ('Outlier', states, N, T, shared_outlier_states_list, shared_outlier_missing_list, shared_outlier_emTime_list, 
                                shared_outlier_p_list, shared_outlier_agents_list, shared_outlier_obs_list, shared_outlier_loud_list, condition1, condition2)
                            for states, N, T in list(itertools.product([4,8,16,32,64], [50, 150, 300], [150, 300, 500]))
                    ] + [
                        ('Popularity', states, N, T, shared_popularity_states_list, shared_popularity_missing_list, shared_popularity_emTime_list, 
                                shared_popularity_p_list, shared_popularity_agents_list, shared_popularity_obs_list, shared_popularity_asc_list)
                            for states, N, T in list(itertools.product([4,8,16,32,64], [50, 150, 300], [150, 300, 500]))
                    ] + [
                        ('Confirmation', states, N, T, shared_confirmation_states_list, shared_confirmation_missing_list,
                                shared_confirmation_emTime_list, shared_confirmation_p_list, shared_confirmation_agents_list, shared_confirmation_obs_list, 
                                shared_confirmation_ratio_list, shared_bias_factor_list)
                            for states, N, T in list(itertools.product([4,8,16,32,64], [50, 150, 300], [150, 300, 500]))
                    ]
                    

        # Use multi-threading to each scenario 
        results = pool.map(process_scenario, args_list)

        # Collect all simulation results into appropriate lists
        confirmation_states_list = list(shared_confirmation_states_list)
        confirmation_missing_list = list(shared_confirmation_missing_list)
        confirmation_emTime_list = list(shared_confirmation_emTime_list)
        confirmation_p_list = list(shared_confirmation_p_list)
        confirmation_agents_list = list(shared_confirmation_agents_list)
        confirmation_obs_list = list(shared_confirmation_obs_list)
        confirmation_ss_ratio_list = list(shared_confirmation_ratio_list)
        bias_factor_list = list(shared_bias_factor_list)
        popularity_states_list = list(shared_popularity_states_list)
        popularity_missing_list = list(shared_popularity_missing_list)
        popularity_emTime_list = list(shared_popularity_emTime_list)
        popularity_p_list = list(shared_popularity_p_list)
        popularity_agents_list = list(shared_popularity_agents_list)
        popularity_obs_list = list(shared_popularity_obs_list)
        popularity_asc_list = list(shared_popularity_asc_list)
        outlier_states_list = list(shared_outlier_states_list)
        outlier_missing_list = list(shared_outlier_missing_list)
        outlier_emTime_list = list(shared_outlier_emTime_list)
        outlier_p_list = list(shared_outlier_p_list)
        outlier_agents_list = list(shared_outlier_agents_list)
        outlier_obs_list = list(shared_outlier_obs_list)
        outlier_loud_list = list(shared_outlier_loud_list)
        polarization_states_list = list(shared_polarization_states_list)
        polarization_missing_list = list(shared_polarization_missing_list)
        polarization_emTime_list = list(shared_polarization_emTime_list)
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
        base_p_list = list(shared_base_p_list)
        base_ss_ratio_list = list(shared_base_ss_ratio_list)
        base_agents_list = list(shared_base_agents_list)
        base_obs_list = list(shared_base_obs_list)

    # Use generated lists to form different dataframes and save to csv
    #polar_file = f'{folder}Polarization/polar imputed.csv'
    base_file = folder + 'Base/base - EM.csv'
    try:
        existing_base = pd.read_csv(base_file)
        run = max(existing_base['run']) + 1
    except Exception:
        run = 0
        
    polarization_dict = {
        'run': [run] * len(polarization_agents_list),
        'agents': polarization_agents_list,
        'observations': polarization_obs_list,
        'missing': polarization_missing_list,
        'clusters': polarization_clusters_list,
        'Time': polarization_emTime_list,
        'Norm': polarization_p_list,
        'SS ratio': polarization_ss_ratio_list,
        'states': polarization_states_list,
        'unaff': polarization_unaff_list
    }
    #confirmation_file = folder + 'Confirmation/confirmation imputed.csv'
    #try:
    #    existing_confirmation = pd.read_csv(confirmation_file)
    #    run = max(existing_confirmation['run']) + 1
    #except:
    #    run = 0
    confirmation_dict = {
        'run': [run] * len(confirmation_agents_list),
        'agents': confirmation_agents_list,
        'observations': confirmation_obs_list,
        'states': confirmation_states_list,
        'Time': confirmation_emTime_list,
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
        'Norm': popularity_p_list,
        'missing': popularity_missing_list,
        'low': popularity_asc_list
    }
    #popularity_file = folder + 'Popularity/popular imputed.csv'

    
    outlier_dict = {
        'run': [run] * len(outlier_agents_list),
        'agents': outlier_agents_list,
        'observations': outlier_obs_list,
        'states': outlier_states_list,
        'Time': outlier_emTime_list,
        'Norm': outlier_p_list,
        'loud': outlier_loud_list,
        'missing': outlier_missing_list
    }
    #outlier_file = folder + 'Outlier/outlier imputed.csv' '''
    
    base_dict = {
        'run': [run] * len(base_states_list),
        'state': base_states_list,
        'missing': base_missing_list,
        'Time': base_emTime_list,
        'Norm': base_p_list,
        'SS ratio': base_ss_ratio_list,
        'agents': base_agents_list,
        'observations': base_obs_list 
    }
    
    
    polar_file = f'{folder}Polarization/polar - EM.csv'
    confirmation_file = folder + 'Confirmation/confirmation - EM.csv'
    popularity_file = folder + 'Popularity/popular - EM.csv'
    outlier_file = folder + 'Outlier/outlier - EM.csv' 
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
       
    

    '''save_dict_to_csv(polarization_dict, polar_file)
    save_dict_to_csv(confirmation_dict, confirmation_file)
    save_dict_to_csv(popularity_dict, popularity_file)
    save_dict_to_csv(outlier_dict, outlier_file)
    save_dict_to_csv(base_dict, base_file)'''