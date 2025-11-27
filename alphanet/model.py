import jax
import jax.numpy as jnp
from flax import linen as nn
from .layers import QuantumLayer, BitDense, SpikingLIF

class QBitSNN(nn.Module):
    """Quantum-BitNet-Spiking Neural Network (QBitSNN) architecture.

    This model integrates three paradigms:
    1. Quantum Layer: Encodes input features using parameterized quantum circuits.
    2. BitNet Layer: Processes encoded features with 1-bit quantized weights.
    3. Spiking Neural Network (SNN): Integrates temporal information using Leaky Integrate-and-Fire neurons.

    Attributes:
        hidden_dim (int): Dimension of the hidden layer (number of neurons).
        output_dim (int): Dimension of the output layer (number of classes).
        time_steps (int): Number of time steps to simulate in the SNN loop. Defaults to 10.
    """
    hidden_dim: int
    output_dim: int
    time_steps: int = 10
    
    @nn.compact
    def __call__(self, x):
        """Forward pass of the QBitSNN.

        Args:
            x (jax.numpy.ndarray): Input tensor of shape (Batch, Time, InputFeatures) or (Batch, InputFeatures).
                If the input is 2D (Batch, InputFeatures), it is treated as static input for each time step
                unless sliced inside the loop.

        Returns:
            tuple: A tuple containing:
                - logits (jax.numpy.ndarray): Output logits of shape (Batch, OutputDim).
                - spikes (jax.numpy.ndarray): Spike history of shape (Batch, Time, HiddenDim).
        """
        # x shape: (Batch, Time, InputFeatures)
        batch_size = x.shape[0]
        
        # Initialize SNN State (Voltage)
        voltage_state = jnp.zeros((batch_size, self.hidden_dim))
        
        # Layers (ORIGINAL ARCHITECTURE RESTORED)
        quantum = QuantumLayer(num_qubits=x.shape[-1], name='QuantumCore')
        bitnet = BitDense(features=self.hidden_dim, name='BitNetCore')
        lif = SpikingLIF(name='SpikingNeuron')
        readout = nn.Dense(self.output_dim, name='Readout')
        
        spike_history = []
        
        # Temporal Loop (SNN)
        for t in range(self.time_steps):
            if x.ndim == 3:
                xt = x[:, t, :]
            else:
                xt = x
            
            # A. Quantum Encoding
            q_out = quantum(xt)
            
            # B. BitNet Processing (Synaptic Current)
            current = bitnet(q_out)
            
            # C. Spiking Neuron Dynamics
            spike, voltage_state = lif(current, voltage_state)
            spike_history.append(spike)
            
        # Stack spikes: (Batch, Time, Hidden)
        spikes = jnp.stack(spike_history, axis=1)
        
        # Readout: Average firing rate
        firing_rate = jnp.mean(spikes, axis=1)
        
        # Final prediction
        logits = readout(firing_rate)
        
        return logits, spikes
