# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Cory Shane Davis. All rights reserved.
# CosmoChain: Helper functions for the Cosmic Synapse Theory (CST) blockchain by Cory Shane Davis.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from web3 import Web3

# Constants for CST calculations
D_0 = 100  # Characteristic latency (ms) for influence decay
T_MAX = 1_000_000  # Max transaction volume (USD) for normalization
KAPPA = 0.01  # Informational scaling factor for token minting
R_0 = 1e6  # Characteristic distance (meters, ~1 Mpc) for decay
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
C = 2.99792458e8  # Speed of light (m/s)
K_B = 1.380649e-23  # Boltzmann constant (J/K)
T_CMB = 2.725  # CMB temperature (K)
HBAR = 1.054571817e-34  # Reduced Planck constant (J s)
LN_2 = 0.693147  # Natural logarithm of 2

def compute_psi_i(masses, positions, velocities, entropies, distances, lyapunovs, phi=0.1):
    """
    Compute informational energy density (psi_i) for a node using the Cosmic Synapse Theory (CST) formula.

    Args:
        masses (np.ndarray): Array of node masses (kg).
        positions (np.ndarray): Array of 12D positions (m).
        velocities (np.ndarray): Array of 12D velocities (m/s).
        entropies (np.ndarray): Array of node entropies (bits).
        distances (np.ndarray): Matrix of 3D distances (m).
        lyapunovs (np.ndarray): Array of Lyapunov exponents (1/s).
        phi (float): Chaotic coupling constant (default: 0.1).

    Returns:
        np.ndarray: Array of psi_i values (J/m^12) representing informational energy density.
    """
    n = len(masses)
    psi_i = np.zeros(n)
    v_12d = 1e-36  # Arbitrary 12D volume (m^12) for normalization

    for i in range(n):
        # Kinetic term: Accounts for node velocity and chaotic dynamics
        v_squared = np.sum(velocities[i] ** 2)
        e_chaos = masses[i] * C ** 2 * lyapunovs[i]
        kinetic = 0.5 * masses[i] * v_squared
        weight = phi * (masses[i] * C ** 2 + e_chaos) + C * lyapunovs[i] + 1
        kinetic_term = (weight * kinetic) / (masses[i] * C ** 2 + e_chaos)

        # Synaptic term: Models interactions with exponential decay based on distance
        synaptic = 0
        for j in range(n):
            if i != j:
                decay = np.exp(-distances[i, j] / R_0)
                synaptic += decay * (G * masses[i] * masses[j]) / (distances[i, j] * C ** 2)
        synaptic *= (masses[i] * C ** 2 + e_chaos)

        # Gravitational term: Accounts for gravitational interactions between nodes
        grav = 0
        for j in range(n):
            if i != j:
                grav += (G * masses[i] * masses[j]) / distances[i, j]
        grav *= -1

        # Informational term: Incorporates entropy and CMB temperature effects
        info = 0
        s_i = (K_B * C ** 3 * entropies[i]) / (4 * HBAR * G * LN_2)
        for j in range(n):
            if i != j:
                s_j = (K_B * C ** 3 * entropies[j]) / (4 * HBAR * G * LN_2)
                info += (K_B * T_CMB / C) * (s_i * s_j) / distances[i, j]
        info *= -1

        # Total psi_i: Sum of kinetic, synaptic, gravitational, and informational terms
        psi_i[i] = (kinetic_term + synaptic + grav + info) / v_12d

    return psi_i

def simulate_interactions(n_nodes, timesteps=10):
    """
    Simulate blockchain interactions for testing, generating mock data for CST calculations.

    Args:
        n_nodes (int): Number of nodes to simulate.
        timesteps (int): Number of simulation timesteps (default: 10).

    Returns:
        tuple: Arrays of masses, positions, velocities, entropies, distances, and Lyapunov exponents.
    """
    masses = np.random.uniform(1e30, 1e32, n_nodes)  # Masses in kg
    positions = np.random.uniform(-1e6, 1e6, (n_nodes, 12))  # 12D positions in m
    velocities = np.random.uniform(-1e3, 1e3, (n_nodes, 12))  # 12D velocities in m/s
    entropies = np.random.uniform(1e50, 1e52, n_nodes)  # Entropies in bits
    distances = np.random.uniform(1e5, 1e7, (n_nodes, n_nodes))  # 3D distances in m
    np.fill_diagonal(distances, 0)  # Zero distance for self-interactions
    lyapunovs = np.random.uniform(1e-10, 1e-8, n_nodes)  # Lyapunov exponents in 1/s
    return masses, positions, velocities, entropies, distances, lyapunovs

def get_contract_data(contract_address, node_address, web3_provider):
    """
    Fetch influence score and SynapCoin balance from the CosmoChain contract.

    Args:
        contract_address (str): Deployed contract address (hex).
        node_address (str): Wallet address of the node (hex).
        web3_provider (str): Web3 provider URL (e.g., Infura Sepolia endpoint).

    Returns:
        tuple: Influence score and SynapCoin balance for the node.
    """
    w3 = Web3(Web3.HTTPProvider(web3_provider))
    abi = [...]  # Replace with CosmoChain.sol ABI after compilation
    contract = w3.eth.contract(address=contract_address, abi=abi)
    influence = contract.functions.influenceScores(node_address).call()
    balance = contract.functions.getBalance(node_address).call()
    return influence, balance