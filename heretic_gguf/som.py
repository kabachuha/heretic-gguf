# SPDX-License-Identifier: AGPL-3.0-or-later
# Self-Organizing Map direction collection — faithful port of
# p-e-w/heretic PR #196 (src/heretic/som.py), implementing the method of
# Piras et al., "SOM Directions are Better than One: Multi-Directional
# Refusal Suppression in Language Models" (arXiv:2511.08379, AAAI 2026):
# a hexagonal Kohonen map trained on the harmful-prompt residuals of one
# layer; the top-k most representative neuron weights become the
# "bad means" for that layer.
#
# Requires the 'minisom' package (pip install minisom).
#
# Notes on the port:
#   * The MiniSom construction, training call and the top-k neuron
#     selection (winner-count deduplication of collapsed neurons) are
#     identical to the PR.
#   * random_seed=0 is fixed by the PR for reproducibility and kept.

from collections import defaultdict

import numpy as np
from minisom import MiniSom


class SOMCalculator:
    """Trains a Self-Organizing Map and extracts neuron weights."""

    def __init__(self, som_x: int, som_y: int, iterations: int,
                 lr: float, sigma: float) -> None:
        """
        Args:
            som_x (int): Number of neurons in the x-axis of the SOM grid.
            som_y (int): Number of neurons in the y-axis of the SOM grid.
            iterations (int): Number of training iterations.
            lr (float): Learning rate.
            sigma (float): Radius of the neighborhood function.
        """
        self.som_x = som_x
        self.som_y = som_y
        self.iterations = iterations
        self.lr = lr
        self.sigma = sigma
        self.som: MiniSom | None = None

    def fit(self, data: np.ndarray) -> None:
        """
        Trains the SOM on the provided 2D data.

        Args:
            data (np.ndarray): A 2D NumPy array of shape (n_samples, n_features).
        """
        # Ensure data is 2D
        if len(data.shape) != 2:
            raise ValueError(f"Data must be a 2D array, but got shape {data.shape}")

        n_samples, n_features = data.shape

        # Initialize and train the SOM using MiniSom
        self.som = MiniSom(
            self.som_x,
            self.som_y,
            n_features,
            sigma=self.sigma,
            learning_rate=self.lr,
            random_seed=0,  # For reproducibility
            activation_distance='euclidean',
            topology='hexagonal'
        )
        self.som.random_weights_init(data)
        self.som.train_random(data, self.iterations)

    def get_top_k_neuron_weights(self, k: int) -> np.ndarray:
        """
        Gets the weights of the top-k neurons based on their frequency of
        being winners.

        Args:
            k (int): The number of top neurons to return.

        Returns:
            np.ndarray: A 2D array of shape (k, n_features) containing the
            weights of the top-k neurons.
        """
        if self.som is None:
            raise RuntimeError("SOM has not been trained yet. Call `fit()` first.")

        winners = np.array([self.som.winner(x) for x in self.som._weights.reshape(-1, self.som._weights.shape[2])])
        counts = defaultdict(int)
        for w in winners:
            counts[tuple(w)] += 1

        # Sort neurons by their count (descending) and get the top-k
        sorted_neurons = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:k]

        # Get the coordinates of the top-k neurons
        top_k_coords = [coord for coord, _ in sorted_neurons]

        # Fetch the weights for these top-k neurons
        # self.som.get_weights() has shape (som_x, som_y, n_features)
        top_k_weights = np.array([self.som.get_weights()[i, j] for i, j in top_k_coords])

        return top_k_weights
