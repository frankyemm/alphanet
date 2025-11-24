import jax
import jax.numpy as jnp
from flax import linen as nn
from .layers import QuantumLayer, BitDense, SpikingLIF

class QBitSNN(nn.Module):
    hidden_dim: int
    output_dim: int
    time_steps: int = 10
    
    @nn.compact
    def __call__(self, x):
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
