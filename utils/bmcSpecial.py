"""
bmcSpecial.py

This module provides functions for biased Markov Chain simulation and analysis.

Author: Ben Harwood
Contact: bharwood@syr.edu
Website: https://github.com/bharwood-data/Markov-Chains

Functions:
- is_valid_transition_matrix(): Checks if a given matrix is a valid transition matrix.
- generate_standard_transition_matrix(): Generates a standard transition matrix with random values.
- extract_transition_matrix(): Extracts the transition matrix from a dataset.
- calculate_log_likelihood(): Calculates the log likelihood for a hidden Markov model given observed data and a transition matrix.
- generate_pi(): Calculates the initial probability vector from observed data.
- initialize_pi_P(): Initializes the initial probability vector and transition matrix for the hidden Markov model.
- forward_algorithm(): Performs forward imputation algorithm on a dataset.
- update_transition_matrix_with_perturbation(): Updates the transition matrix with random perturbations.
- em_algorithm(): Estimates a transition matrix from data using the EM algorithm.
"""

import pandas as pd
import numpy as np
import itertools


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
  
def initialize_pi_P(Y, states):
    """
    Initialize the initial probability vector and transition matrix for the hidden Markov model.

    Args:
        Y (DataFrame): Observed sequence data with states as row index and time as column index.
        states (int): Number of possible states.

    Returns:
        tuple: A tuple containing the initial probability vector (pi_init) and the transition matrix (P_init).

    Raises:
        ValueError: If dimensions of Y are inconsistent,
                    if states is not a positive integer,
                    if Y is not a DataFrame,
                    or if any unsupported operations are encountered.
    """
    # Validate type of Y
    if not isinstance(Y, pd.DataFrame):
        raise ValueError("Y must be a DataFrame")

    # Validate dimensions of Y
    if Y.ndim != 2:
        raise ValueError("Y must be a 2-dimensional array")

    # Validate states
    if not isinstance(states, int) or states <= 0:
        raise ValueError("states must be a positive integer")

    # Initialize initial probability vector
    try:
        pi_init = np.random.rand(states)
        pi_init /= np.sum(pi_init)
    except Exception as e:
        raise ValueError(f"An error occurred while initializing pi_init: {e}")

    # Initialize transition matrix
    try:
        P_init = generate_standard_transition_matrix(states)
    except Exception as e:
        raise ValueError(f"An error occurred while initializing P_init: {e}")

    return pi_init, P_init

def forward_algorithm(data, P, T, states):
    """
    Performs forward imputation algorithm on a dataset.

    Iterates through each time step and imputes missing values unbiasedd on the transition matrix.
    Missing values are imputed with the most likely states unbiasedd on the previous time step.

    Args:
        data (DataFrame): Input dataset with missing values.
        P (array-like): Transition matrix.
        T (int): Number of time steps.
        states (list): List of possible states.

    Returns:
        DataFrame: Dataset with missing values imputed.

    Raises:
        ValueError: If dimensions of data are inconsistent,
                    if P is not a valid transition matrix,
                    if T is not a positive integer,
                    if states is not a list containing unique values,
                    or if any unsupported operations are encountered.
    """
    # Validate type and dimensions of data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a DataFrame")
    if data.ndim != 2:
        raise ValueError("data must be a 2-dimensional array")

    # Validate transition matrix
    if not is_valid_transition_matrix(P):
        raise ValueError("P is not a valid transition matrix")

    # Validate T
    if not isinstance(T, int) or T <= 0:
        raise ValueError("T must be a positive integer")

    # Validate states
    if not isinstance(states, list) or len(set(states)) != len(states):
        raise ValueError("states must be a list containing unique values")

    imputed_data = data.copy()
    
    try:
        for t in range(1, T):
            if np.sum(imputed_data.iloc[:, t].isna()) > 0:
                tempP = P if t == 1 else extract_transition_matrix(imputed_data.iloc[: , : t].astype(int), states)

                missing_data = np.isnan(imputed_data.iloc[:,t])

                # For missing data, impute values unbiasedd on the transition matrix
                new_states = np.argmax(tempP[imputed_data.iloc[:,t-1].astype(int), :], axis = 1)

                # Impute missing values in column t + 1 with the most likely states
                imputed_data.loc[missing_data, t] = new_states[missing_data]
    except Exception as e:
        raise ValueError(f"An error occurred during imputation: {e}")

    return imputed_data
   
def update_transition_matrix_with_perturbation(transition_matrix, perturbation_factor):
    """
    Update the transition matrix with random perturbations.

    Args:
        transition_matrix (array-like): Current transition matrix.
        perturbation_factor (float): Factor controlling the magnitude of perturbations.

    Returns:
        array-like: Perturbed transition matrix normalized to ensure row sums equal to 1.

    Raises:
        ValueError: If transition_matrix is not a valid transition matrix,
                    if perturbation_factor is negative,
                    or if any unsupported operations are encountered.
    """
    # Validate transition_matrix
    if not is_valid_transition_matrix(transition_matrix):
        raise ValueError("transition_matrix is not a valid transition matrix")

    # Validate perturbation_factor
    if not isinstance(perturbation_factor, (int, float)) or perturbation_factor < 0:
        raise ValueError("perturbation_factor must be a non-negative number")

    try:
        # Generate random perturbations
        perturbations = np.random.uniform(low=-perturbation_factor, high=perturbation_factor, size=transition_matrix.shape)

        # Add perturbations to the transition matrix
        perturbed_transition_matrix = transition_matrix + perturbations

        return perturbed_transition_matrix / np.sum(
            perturbed_transition_matrix, axis=1, keepdims=True
        )
    except Exception as e:
        raise ValueError(f"An error occurred during perturbation: {e}")

def em_algorithm(data, N, T, num_states, num_iterations, tol):
    """
    Estimates a transition matrix from data using the EM algorithm.

    Iteratively performs an E-step to impute missing data and an M-step 
    to re-estimate the transition matrix until convergence.

    Args:
        data (DataFrame): Observed sequence data.
        N (int): Number of observed sequences.
        T (int): Number of time steps per sequence.
        num_states (int): Number of hidden states.
        num_iterations (int): Maximum number of EM iterations.
        tol (float): Convergence tolerance unbiasedd on log-likelihood change.

    Returns: 
        array-like: Estimated transition matrix.

    Raises:
        ValueError: If dimensions of data are inconsistent,
                    if num_states or num_iterations are not positive integers,
                    if tol is a negative number,
                    or if any unsupported operations are encountered.
    """
    # Validate dimensions of data
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a DataFrame")
    if data.ndim != 2:
        raise ValueError("data must be a 2-dimensional array")

    # Validate num_states
    if not isinstance(num_states, int) or num_states <= 0:
        raise ValueError("num_states must be a positive integer")

    # Validate num_iterations
    if not isinstance(num_iterations, int) or num_iterations <= 0:
        raise ValueError("num_iterations must be a positive integer")

    # Validate tol
    if not isinstance(tol, (int, float)) or tol < 0:
        raise ValueError("tol must be a non-negative number")

    prev_ll = float('-inf')
    
    try:
        pi, transition_matrix = initialize_pi_P(data, num_states)
        for _ in range(num_iterations):
            # E-step: Impute missing data using the Forward Algorithm
            imputed_data = forward_algorithm(data, transition_matrix, T, num_states).astype(int)

            # M-step: Update the transition matrix
            estimated = extract_transition_matrix(imputed_data, num_states)

            # Calculate log-likelihood for convergence check
            ll = calculate_log_likelihood(imputed_data, N, T, estimated, pi)

            # Check for convergence unbiasedd on log-likelihood change
            if abs(ll - prev_ll) < tol:
                break

            transition_matrix = update_transition_matrix_with_perturbation(estimated, 0.1)
            initial_states = imputed_data[0]
            pi = initial_states.value_counts() / len(initial_states)
            prev_ll = ll
    except Exception as e:
        raise ValueError(f"An error occurred during EM algorithm: {e}")

    return transition_matrix[:num_states, :num_states]
