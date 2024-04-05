"""
bmcUtils.py

This module provides utility functions forbiased Markov Chain sumlation.

This module provides utility functions for Bayesian Markov Chain analysis.

Author: Ben Harwood
Contact: bharwood@syr.edu
Website: https://github.com/bharwood-data/Markov-Chains

Functions:
- kl_divergence(): Calculates the Kullback-Leibler divergence between two probability distributions.
- is_valid_transition_matrix(): Checks if a matrix is a valid transition matrix.
- calculate_steady_state(): Calculates the steady-state distribution of a Markov chain.
- state_counts(): Calculates expected state counts and frequencies from chains data.
- calculate_log_likelihood(): Calculates the log likelihood for a hidden Markov model given observed data.
- generate_pi(): Generates the initial probability vector from observed data.
- generate_standard_transition_matrix(): Generates a standard transition matrix with random values.
- introduce_mcar_missing_data(): Introduces missing completely at random (MCAR) values into a dataset.
- extract_transition_matrix(): Extracts the transition matrix from a dataset.
- generate_initial_states(): Generates initial states for multiple agents using a transition matrix.
- check_negative_steady_state(): Checks for negative values in the steady-state vector.
- generate_markov_chains(): Generates Markov chains for multiple agents given a transition matrix.
"""

import pandas as pd
import numpy as np
import itertools
from scipy.stats import entropy

def kl_divergence(estimated, observed, states):
    """
    Calculates the Kullback-Leibler divergence between two probability distributions.

    Args:
        estimated (array-like): The estimated probability distribution.
        observed (array-like): The observed probability distribution.
        states (int): The number of states in the distributions.

    Returns:
        float: The Kullback-Leibler divergence between the estimated and observed distributions.

    Raises:
        ValueError: If estimated or observed contains zero values or if their lengths do not match states.
        TypeError: If estimated or observed are not array-like objects or if states is not an integer.
    """
    # Check if estimated and observed are array-like objects
    if not isinstance(estimated, (list, np.ndarray)) or not isinstance(observed, (list, np.ndarray)):
        raise TypeError("estimated and observed must be array-like objects (e.g., lists, NumPy arrays)")

    # Check if states is an integer
    if not isinstance(states, int):
        raise TypeError("states must be an integer")

    # Check if estimated and observed have the same length as states
    if len(estimated) != states or len(observed) != states:
        raise ValueError("Lengths of estimated and observed must match the number of states")

    if 0 in observed:
        observed += 0.0000000001
    if 0 in estimated:
        estimated += 0.0000000001
    kl_divergences = np.zeros(states)
    for i in range(states):
        kl_divergences[i] = entropy(observed[i], estimated[i])
    return sum(kl_divergences)

def is_valid_transition_matrix(transition_matrix):
    """
    Check if the given matrix is a valid transition matrix.

    Args:
        transition_matrix (array-like): Transition matrix.

    Returns:
        bool: True if the matrix is a valid transition matrix, False otherwise.

    Raises:
        ValueError: If transition_matrix is not array-like.
    """
    # Type checking
    if not isinstance(transition_matrix, (list, np.ndarray)):
        raise ValueError("Transition_matrix must be an array-like object.")

    # Convert the transition matrix to a NumPy array if it's not already
    transition_matrix = np.array(transition_matrix)

    # Check if all elements are non-negative
    if (transition_matrix < 0).any():
        return False

    # Check if each row sums to approximately 1
    row_sums = np.sum(transition_matrix, axis=1)
    return bool(np.allclose(row_sums, 1))

def calculate_steady_state(P):
    """
    Calculate the steady-state distribution of a Markov chain.

    Parameters:
        P (numpy.ndarray): The transition matrix of the Markov chain.

    Returns:
        numpy.ndarray: The steady-state distribution.

    Raises:
        ValueError: If P is not a square 2D numpy array,
                    or if any unsupported operations are encountered.
    """
    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2D numpy array")

    try:
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        stationary_vector = np.real(eigenvectors[:, np.argmax(eigenvalues)])

        return stationary_vector / np.sum(stationary_vector)
    except Exception as e:
        raise ValueError(f"An error occurred during steady-state distribution calculation: {e}")

def state_counts(transition_matrix, chains, N, T, pct):
    """
    Calculate expected state counts and frequencies from chains data.

    Args:
        transition_matrix (array-like): Transition matrix.
        chains (DataFrame): DataFrame containing chains data.
        N (int): Number of simulations.
        T (int): Number of time steps per simulation.
        pct (float): Percentage of time steps to disregard.

    Returns:
        tuple: A tuple containing expected state counts and state frequencies.

    Raises:
        ValueError: If the transition_matrix is not a valid transition matrix,
                    if chains is not a DataFrame with the correct structure,
                    if N, T, or pct are not numeric values,
                    if there are missing values in the chains DataFrame,
                    or if any unsupported operations are encountered.
    """
    # Validate transition_matrix
    if not is_valid_transition_matrix(transition_matrix):
        raise ValueError("Invalid transition matrix")

    # Check if chains is a DataFrame
    if not isinstance(chains, pd.DataFrame):
        raise ValueError("chains must be a DataFrame")

    # Check if N, T, and pct are numeric values
    if not all(isinstance(val, (int, float)) for val in [N, T, pct]):
        raise ValueError("N, T, and pct must be numeric values")

    # Check for zero division
    if N == 0 or T == 0 or pct == 0:
        raise ValueError("N, T, and pct must be non-zero")

    # Check for missing values in chains DataFrame
    if chains.isnull().values.any():
        raise ValueError("chains DataFrame contains missing values")

    # Calculate expected state counts
    steady_state = calculate_steady_state(transition_matrix)
    expected = steady_state * N * T * (1 - pct)

    # Calculate state frequencies
    flat = chains.values.flatten()
    frequencies = pd.Series(flat).value_counts()

    return expected, frequencies

def calculate_log_likelihood(data, N, T, transition_matrix, pi):
    """
    Calculates the log likelihood for a hidden Markov model given observed data and a transition matrix.

    Args:
        data (DataFrame): Observed sequence data with states as row index and time as column index.
        N (int): Number of observed sequences.
        T (int): Length of each observed sequence.
        transition_matrix (array-like): Square transition matrix where element i,j is p(j|i).
        pi (array-like): Initial state distribution.

    Returns:
        log_likelihood (float): The log likelihood of the observed data given the transition matrix.

    Raises:
        ValueError: If dimensions of data, transition_matrix, or pi are inconsistent,
                    if any element of pi or transition_matrix results in division by zero,
                    or if any unsupported operations are encountered.
    """
    # Validate dimensions of data, transition_matrix, and pi
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a DataFrame")

    if data.shape != (N, T):
        raise ValueError("Dimensions of data do not match N and T")

    if transition_matrix.shape[0] != transition_matrix.shape[1]:
        raise ValueError("Transition matrix must be square")

    if len(pi) != transition_matrix.shape[0]:
        raise ValueError("Dimensions of pi do not match transition_matrix")

    # Check for zero division
    if 0 in pi or 0 in transition_matrix:
        raise ValueError("Elements of pi or transition_matrix result in division by zero")

    # Initialize log likelihood
    log_likelihood = 0.0

    try:
        # Iterate over observed sequences
        for t in range(N):
            sequence = data.iloc[t, :]
            sequence_log_likelihood = 0.0

            # Add contribution from initial state distribution
            initial_state = sequence[0]
            sequence_log_likelihood += np.log(pi[initial_state])

            # Add contribution from transition matrix
            for i in range(1, T):
                from_state = sequence[i - 1]
                to_state = sequence[i]
                sequence_log_likelihood += np.log(transition_matrix[from_state, to_state])

            # Add sequence log-likelihood to total log-likelihood
            log_likelihood += sequence_log_likelihood
    except Exception as e:
        raise ValueError(f"An error occurred during calculation: {e}")

    return log_likelihood

def generate_pi(Y, states):
    """
    Calculates the initial probability vector from observed data.

    Generates the initial probability vector by calculating the proportion 
    of observations in each state.

    Args:
        Y (DataFrame): Observed sequence data with states as row index and time as column index.
        states (list): List of possible state values.

    Returns:
        pi (array-like): Initial probability vector where pi[i] is probability of starting in state i.

    Raises:
        ValueError: If dimensions of Y are inconsistent or if states contains non-unique values,
                    if the denominator in the probability calculation becomes zero,
                    or if any unsupported operations are encountered.
    """
    # Validate dimensions of Y
    if not isinstance(Y, pd.DataFrame):
        raise ValueError("Y must be a DataFrame")

    # Validate uniqueness of states
    if len(set(states)) != len(states):
        raise ValueError("states must contain unique values")

    # Calculate the initial probability vector
    try:
        pi = [
            np.sum(Y.values == state).sum() / (Y.size - np.isnan(Y).sum())
            for state in states
        ]
    except ZeroDivisionError:
        raise ValueError("Denominator in the probability calculation is zero")

    return pi
  
def generate_standard_transition_matrix(num_states):
    """
    Generates a standard transition matrix with random values.

    Args:
        num_states (int): The number of states.

    Returns:
        numpy.ndarray: The generated standard transition matrix.

    Raises:
        ValueError: If num_states is not a positive integer,
                    or if any unsupported operations are encountered.
    """
    # Validate num_states
    if not isinstance(num_states, int) or num_states <= 0:
        raise ValueError("num_states must be a positive integer")

    try:
        return np.random.dirichlet(np.ones(num_states), size=num_states)
    except Exception as e:
        raise ValueError(f"An error occurred during standard transition matrix generation: {e}")

def introduce_mcar_missing_data(data, missing_prob):
    """
    Introduces missing completely at random (MCAR) values into a dataset.

    Randomly replaces values with NaN unbiasedd on a missing probability, without any dependency on the observed values.
    The number of missing values introduced is determined by the missing probability and the size of the dataset.

    Args:
        data (pd.DataFrame): Input dataset.
        missing_prob (float): Probability of replacing a value with NaN.

    Returns:
        pd.DataFrame: Dataset with MCAR missing values introduced.

    Raises:
        ValueError: If missing_prob is not in the valid range [0, 1],
                    if data is not a Pandas DataFrame,
                    or if any unsupported operations are encountered.
    """
    # Validate data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a Pandas DataFrame")

    # Validate missing_prob
    if not 0 <= missing_prob <= 1:
        raise ValueError("missing_prob must be in the range [0, 1]")

    try:
        N, T = data.shape

        #  Calculate the total number of missing values needed unbiasedd on the specified missing probability
        needed_missing = int(missing_prob * N * T)

        # Calculate the remaining missing values needed after considering existing NaN values in the data
        remaining = max(0, needed_missing - np.isnan(data).sum().sum())

        # Flatten the data into a 1D array and convert to float
        flat = data.T.values.flatten().astype(float)

        # Identify existing NaN values in the flattened array
        existing_nan = np.isnan(flat[N:])

        # Get the indices of non-NaN values in the flattened array
        valid = np.where(~existing_nan)[0]

        # Calculate the remaining missing values to be filled
        remaining_missing = min(remaining, len(valid))

        # Randomly select indices from the valid indices to mark as NaN
        indices = np.random.choice(valid[valid >= N], size=remaining_missing, replace=False)

        # Mark the selected indices as NaN in the flattened array
        np.put(flat, indices, np.nan)

        return pd.DataFrame(flat.reshape(-1, N).T, columns=data.columns)
    except Exception as e:
        raise ValueError(f"An error occurred during MCAR missing data introduction: {e}")
    
def extract_transition_matrix(Y, states):
    """
    Extracts the transition matrix from a dataset.

    Calculates the transition probabilities between states unbiasedd on the occurrences of transitions in the dataset.
    The transition matrix represents the likelihood of transitioning from one state to another.

    Args:
        Y (pd.DataFrame): Input dataset.
        states (int): Number of possible states.

    Returns:
        np.ndarray: Transition matrix.

    Raises:
        ValueError: If Y is not a pandas DataFrame or if states is not a positive integer,
                    or if any unsupported operations are encountered.
    """
    # Validate Y
    if not isinstance(Y, pd.DataFrame):
        raise ValueError("Y must be a pandas DataFrame")

    # Validate states
    if not isinstance(states, int) or states <= 0:
        raise ValueError("states must be a positive integer")

    try:
        result = Y.values
        N, T = result.shape
        
        # Create an array to count transitions between states
        transition_counts = np.zeros((states, states), dtype=int)

        # Loop through each agent and each time step to count occurrences and transitions
        for n, t in itertools.product(range(N), range(T)):
            if t < T - 1 and not np.isnan(result[n, t]) and not np.isnan(result[n, t + 1]):
                transition_counts[int(result[n, t]), int(result[n, t + 1])] += 1
            
        valid_start_counts = np.empty((states, T - 1))   
        for t in range(T - 1):
            # Calculate the denominator for each state
            valid_start_counts[:, t] = [np.sum((result[:, t] == state) & ~np.isnan(result[:, t + 1])) for state in range(states)]

        # Sum the columns to get the total counts
        total_valid_starts = np.sum(valid_start_counts, axis=1)

        # Handle cases where the denominator is 0 to avoid division by zero
        non_zero_total = np.maximum(total_valid_starts, 1)       
                
        # Create the transition probability matrix
        P = np.zeros((states, states), dtype=float)
        P = (transition_counts.T / non_zero_total).T
        row_sums = np.sum(P, axis=1, keepdims=True)
        
        return np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums != 0)
    except Exception as e:
        raise ValueError(f"An error occurred during transition matrix extraction: {e}")

def generate_initial_states(P, num_agents, initial_state_distribution=None):
    """
    Generate initial states for multiple agents using a transition matrix.

    Parameters:
        P (numpy.ndarray): The transition matrix.
        num_agents (int): The number of agents.
        initial_state_distribution (list or numpy.ndarray, optional): Custom initial state distribution. 
            If None, the stationary distribution is used.

    Returns:
        list: A list of initial states for each agent.

    Raises:
        ValueError: If P is not a 2D numpy array,
                    or if num_agents is not a positive integer,
                    or if initial_state_distribution is provided but its length doesn't match the number of states in P,
                    or if any unsupported operations are encountered.
    """
    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        raise ValueError("P must be a 2D numpy array")

    # Validate num_agents
    if not isinstance(num_agents, int) or num_agents <= 0:
        raise ValueError("num_agents must be a positive integer")

    # Validate initial_state_distribution if provided
    if initial_state_distribution is not None:
        if len(initial_state_distribution) != P.shape[0]:
            raise ValueError("The length of initial_state_distribution must match the number of states in P")

    try:
        num_states = P.shape[0]

        if initial_state_distribution is None:
            # Calculate the steady-state distribution if not provided
            stationary_distribution = calculate_steady_state(P)
        else:
            # Use the custom initial state distribution
            stationary_distribution = initial_state_distribution

        state_list = list(range(num_states))

        return np.random.choice(state_list, num_agents, p=stationary_distribution)
    except Exception as e:
        raise ValueError(f"An error occurred during initial state generation: {e}")

def calculate_steady_state(P):
    """
    Calculate the steady-state distribution of a Markov chain.

    Parameters:
        P (numpy.ndarray): The transition matrix of the Markov chain.

    Returns:
        numpy.ndarray: The steady-state distribution.

    Raises:
        ValueError: If P is not a square 2D numpy array,
                    or if any unsupported operations are encountered.
    """
    # Validate P
    if not isinstance(P, np.ndarray) or P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2D numpy array")

    try:
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        stationary_vector = np.real(eigenvectors[:, np.argmax(eigenvalues)])

        return stationary_vector / np.sum(stationary_vector)
    except Exception as e:
        raise ValueError(f"An error occurred during steady-state distribution calculation: {e}")

def check_negative_steady_state(steady_state_vector):
    """
    Check for negative values in the steady-state vector.

    Args:
        steady_state_vector (numpy.ndarray): The steady-state vector of the Markov chain.

    Returns:
        bool: True if any negative values are found, False otherwise.

    Raises:
        ValueError: If steady_state_vector is not a 1D numpy array.
    """
    # Validate steady_state_vector
    if not isinstance(steady_state_vector, np.ndarray) or steady_state_vector.ndim != 1:
        raise ValueError("steady_state_vector must be a 1D numpy array")

    try:
        return any(steady_state_vector < 0)
    except Exception as e:
        raise ValueError(f"An error occurred during negative steady-state check: {e}")

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
    # Check if transition matrix is valid
    if not is_valid_transition_matrix(transition_matrix):
        raise ValueError("transition_matrix is not a valid transition matrix")

    if not isinstance(transition_matrix, np.ndarray) or transition_matrix.ndim != 2:
        raise ValueError("transition_matrix must be a 2D numpy array")

    if not isinstance(initial_states, (list, np.ndarray)):
        raise ValueError("initial_states must be a list or numpy array")

    if len(initial_states) != num_agents:
        raise ValueError("The length of initial_states must be equal to the number of agents")

    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")

    if not isinstance(num_agents, int) or num_agents <= 0:
        raise ValueError("num_agents must be a positive integer")

    try:
        markov_chains = [np.array(initial_states)]
        for t in range(num_steps - 1):
            current = markov_chains[t]
            probs = []
            for i in range(len(transition_matrix)):
                counts = pd.Series(current).value_counts() / num_agents
                try:
                    probs.append(counts[i])
                except:
                    probs.append(0)

            # Multiply initial probabilities by transition matrix
            next_probabilities = np.dot(probs, transition_matrix)

            # Generate next set of observations
            markov_chains.append(np.random.choice(range(len(next_probabilities)), size=num_agents, p=next_probabilities/sum(next_probabilities)))

        return pd.DataFrame(np.array(markov_chains, dtype='int').T)
    except Exception as e:
        raise ValueError(f"An error occurred during Markov chain generation: {e}")
        