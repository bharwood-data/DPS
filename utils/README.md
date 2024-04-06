# BMC Utils

This folder contains utility modules used in the Bias Simulation Files project. The utilities are designed to assist in various tasks related to Markov chain simulations and analysis.

## Overview

The BMC Utils folder includes the following modules:

- **bmcUtils.py**: This module provides utility functions for generating standard transition matrices, generating initial states, generating Markov chains, applying self-selection bias, introducing missing data, extracting transition matrices, calculating Kullback-Leibler divergence, and checking the validity of transition matrices.

- **bmcSpecial.py**: This module includes special functions tailored for specific tasks related to Markov chain simulations and analysis. It contains functions for generating custom initial distributions, constructing weighted transition matrices, and implementing the forward algorithm and EM algorithm for optimization.

## Contents

- **bmcUtils.py**: Contains utility functions for Markov chain simulations.
- **bmcSpecial.py**: Contains special functions for advanced Markov chain analysis with the forward-backward algorithm and EM-algorithm.

## Usage

To use the BMC Utils modules in your project, follow these steps:

1. **Clone the Repository**: Clone or download the Bias Simulation Files repository to your local machine.
2. **Navigate to the Utils Folder**: Open the folder containing the BMC Utils modules.
3. **Import the Modules**: Import the required modules (`bmcUtils` and/or `bmcSpecial`) into your Python scripts or notebooks.
4. **Use the Functions**: Utilize the functions provided by the modules according to your project requirements.

## Requirements

- Python
- Pandas
- Numpy
- Itertools
- Scipy.stats

## Contributing

If you would like to contribute to the development of the BMC Utils modules, please follow these guidelines:

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Make your changes and ensure that the code adheres to the project's coding standards.
- Test your changes thoroughly.
- Submit a pull request with a clear description of your changes.

## License

TBA

## Contact

For questions or inquiries about the BMC Utils modules, please contact bharwood@syr.edu.
