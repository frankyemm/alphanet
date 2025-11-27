"""AlphaNet Package.

This package implements the AlphaNet architecture, a hybrid neural network combining:
1. Quantum Computing (Quantum Layers)
2. Binary Neural Networks (BitNet)
3. Spiking Neural Networks (SNN)

It includes modules for data loading, model definition, custom layers, and training utilities.
"""
from .layers import QuantumLayer, BitDense, SpikingLIF, MultiHeadAttention
from .model import QBitSNN
from .data import WebcamLoader, YouTubeLoader, TextLoader, OrcaMathLoader, VideoLoader, UCF101VideoLoader
from .train import create_train_state, train_epoch
