import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import custom_vjp

# --- 1. SURROGATE GRADIENT (For SNN) ---
@custom_vjp
def spike_fn(x):
    """Computes the Heaviside step function with a custom gradient for SNNs.

    This function acts as the activation for spiking neurons. In the forward pass,
    it returns 1.0 if x > 0, else 0.0. In the backward pass, it uses a surrogate
    gradient (derivative of a sigmoid) to allow learning through the non-differentiable
    step function.

    Args:
        x (jax.numpy.ndarray): Input tensor (membrane potential minus threshold).

    Returns:
        jax.numpy.ndarray: Binary output tensor (spikes), same shape as input.
    """
    return (x > 0.0).astype(jnp.float32)

def spike_fn_fwd(x):
    """Forward pass for the spike function.

    Args:
        x (jax.numpy.ndarray): Input tensor.

    Returns:
        tuple: A tuple containing:
            - jax.numpy.ndarray: The binary spike output.
            - jax.numpy.ndarray: The input x (saved for backward pass).
    """
    return spike_fn(x), x

def spike_fn_bwd(res, g):
    """Backward pass for the spike function using a surrogate gradient.

    The surrogate gradient approximates the derivative of the Heaviside step function
    using the derivative of a fast sigmoid function.

    Args:
        res (jax.numpy.ndarray): Residuals from the forward pass (the input x).
        g (jax.numpy.ndarray): Gradient of the loss with respect to the output.

    Returns:
        tuple: A tuple containing the gradient with respect to the input x.
    """
    x = res
    beta = 10.0
    grad_x = g / (1 + jnp.abs(beta * x))**2
    return (grad_x,)

spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)

# --- 2. QUANTUM LAYER ---
class QuantumLayer(nn.Module):
    """Simulates a parameterized quantum circuit layer.

    This layer implements a simple quantum circuit where input features modulate
    rotation angles. It computes the probability of measuring state |1> for each qubit.

    Attributes:
        num_qubits (int): The number of qubits in the layer, corresponding to the input dimension.
    """
    num_qubits: int
    
    @nn.compact
    def __call__(self, x):
        """Applies the quantum layer transformation.

        Args:
            x (jax.numpy.ndarray): Input tensor of shape (..., num_qubits).
                Each feature corresponds to a rotation on a qubit.

        Returns:
            jax.numpy.ndarray: Output tensor of shape (..., num_qubits), representing
                the probability of measuring |1> for each qubit.
        """
        theta = self.param('theta', nn.initializers.normal(stddev=0.1), (self.num_qubits,))
        effective_theta = theta * x
        prob_one = jnp.sin(effective_theta / 2.0) ** 2
        return prob_one

# --- 3. BITNET DENSE LAYER ---
def quantize_weights(w):
    """Quantizes weights to -1, 0, or +1.

    This function implements a quantization scheme where weights are scaled and
    then rounded to the nearest integer in {-1, 0, 1}.

    Args:
        w (jax.numpy.ndarray): The weight matrix to quantize.

    Returns:
        jax.numpy.ndarray: Quantized weight matrix.
    """
    scale = jnp.mean(jnp.abs(w))
    w_scaled = w / (scale + 1e-6)
    w_quant = jnp.round(jnp.clip(w_scaled, -1.0, 1.0))
    return w_quant

class BitDense(nn.Module):
    """Dense layer with 1-bit quantized weights (BitNet).

    This layer implements a dense transformation using weights quantized to {-1, 0, 1}
    during the forward pass, while maintaining full-precision weights for updates.
    It uses the Straight-Through Estimator (STE) for gradient calculation.

    Attributes:
        features (int): Number of output features.
    """
    features: int
    
    @nn.compact
    def __call__(self, x):
        """Applies the BitDense layer to the input.

        Args:
            x (jax.numpy.ndarray): Input tensor of shape (..., input_features).

        Returns:
            jax.numpy.ndarray: Output tensor of shape (..., features).
        """
        w = self.param('kernel', nn.initializers.lecun_normal(), (x.shape[-1], self.features))
        w_quant = quantize_weights(w)
        # Straight-Through Estimator: use w_quant for forward, w for backward
        w_ste = w + jax.lax.stop_gradient(w_quant - w)
        return jnp.dot(x, w_ste)

# --- 4. SPIKING LAYER ---
class SpikingLIF(nn.Module):
    """Leaky Integrate-and-Fire (LIF) spiking neuron layer.

    This layer models the dynamics of biological neurons. It integrates input current
    into a membrane potential, which decays over time. When the potential exceeds
    a threshold, a spike is generated and the potential is reset.

    Attributes:
        decay (float): Decay factor for the membrane potential (0 < decay < 1).
        threshold (float): Threshold voltage for spike generation.
    """
    decay: float = 0.9
    threshold: float = 1.0
    
    @nn.compact
    def __call__(self, x, state):
        """Updates the LIF neuron state and generates spikes.

        Args:
            x (jax.numpy.ndarray): Input current injection.
            state (jax.numpy.ndarray): Previous membrane potential state.

        Returns:
            tuple: A tuple containing:
                - jax.numpy.ndarray: Spike output (1.0 for spike, 0.0 otherwise).
                - jax.numpy.ndarray: Updated membrane potential state.
        """
        new_voltage = state * self.decay + x
        spike = spike_fn(new_voltage - self.threshold)
        final_voltage = new_voltage - (self.threshold * spike)
        return spike, final_voltage


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention module for temporal context.

    This module applies multi-head self-attention to capture temporal dependencies
    in the input sequence. It includes a causal mask to ensure predictions depend
    only on past information.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
    """
    num_heads: int = 8
    head_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        """Applies multi-head self-attention to the input sequence.

        Args:
            x (jax.numpy.ndarray): Input tensor of shape (Batch, Time, Features).

        Returns:
            jax.numpy.ndarray: Output tensor of shape (Batch, Time, Features) after
                attention and projection.
        """
        # x shape: (Batch, Time, Features)
        batch_size, seq_len, features = x.shape
        
        # Project to Q, K, V
        qkv_dim = self.num_heads * self.head_dim
        
        query = nn.Dense(qkv_dim, name='query')(x)
        key = nn.Dense(qkv_dim, name='key')(x)
        value = nn.Dense(qkv_dim, name='value')(x)
        
        # Reshape for multi-head attention
        # (Batch, Time, num_heads, head_dim)
        query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (Batch, num_heads, Time, head_dim)
        query = jnp.transpose(query, (0, 2, 1, 3))
        key = jnp.transpose(key, (0, 2, 1, 3))
        value = jnp.transpose(value, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        # (Batch, num_heads, Time, Time)
        scores = jnp.matmul(query, jnp.transpose(key, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        
        # Causal mask (prevent attending to future tokens)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(mask[None, None, :, :] == 0, -1e9, scores)
        
        # Softmax attention weights
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        # (Batch, num_heads, Time, head_dim)
        attn_output = jnp.matmul(attn_weights, value)
        
        # Transpose back and reshape
        # (Batch, Time, num_heads, head_dim) -> (Batch, Time, num_heads * head_dim)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, qkv_dim)
        
        # Final projection
        output = nn.Dense(features, name='output')(attn_output)
        
        return output
