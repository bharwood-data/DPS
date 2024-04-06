# Bias Simulation Results

This folder contains the results of bias simulation experiments conducted using the provided scripts. The simulations aimed to study the effects of various biases on Markov chain transition probability estimation and included scenarios such as outlier bias, polarization bias, popularity bias, status quo bias, and unbiased simulations with self-selection bias.

## Overview

The bias simulation results provide insights into how different biases impact the estimation of transition matrices in Markov chains. Each simulation script generates multiple sets of simulations based on user-defined parameters, such as the number of agents, observations, states, and levels of missing data. The results include metrics such as Kullback-Leibler divergence, Frobenius norm-based inaccuracy, and runtime performance.

## Contents

The results are organized into subfolders corresponding to each bias simulation script:

- **outlier_results**: Results from outlier bias simulation experiments.
- **polarization_results**: Results from polarization bias simulation experiments.
- **popularity_results**: Results from popularity bias simulation experiments.
- **status_quo_results**: Results from status quo bias simulation experiments.
- **unbiased_results**: Results from unbiased simulation experiments with self-selection bias.

Each subfolder contains CSV files, plots, and summary statistics generated from the simulations.

## Usage

To interpret the bias simulation results, follow these steps:

1. **Navigate to the Desired Subfolder**: Open the subfolder corresponding to the bias simulation of interest.
2. **Review Results**: Examine the CSV files containing raw data, summary statistics, and any generated plots.
3. **Analyze Metrics**: Analyze the performance metrics, such as KL divergence and inaccuracy, to understand the impact of biases on transition matrix estimation.
4. **Compare Results**: Compare the results across different simulations to draw conclusions about the effects of biases under varying conditions.

## Requirements

No additional requirements beyond a CSV file viewer and a tool for viewing plots (e.g., matplotlib).

## Contributing

If you would like to contribute to the analysis or interpretation of the bias simulation results, please follow these guidelines:

- Fork the repository.
- Create a new branch for your analysis or feature.
- Make your changes and ensure that they are well-documented.
- Test your changes thoroughly.
- Submit a pull request with a clear description of your changes.

## License

TBA

## Contact

For questions or inquiries about the bias simulation results, please contact bharwood@syr.edu.
