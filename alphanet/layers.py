import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import custom_vjp

# --- 1. SURROGATE GRADIENT (For SNN) ---
@custom_vjp
def spike_fn(x):
    return (x > 0.0).astype(jnp.float32)

def spike_fn_fwd(x):
    return spike_fn(x), x

def spike_fn_bwd(res, g):
    x = res
    beta = 10.0
    grad_x = g / (1 + jnp.abs(beta * x))**2
    return (grad_x,)

spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)

# --- 2. QUANTUM LAYER ---
class QuantumLayer(nn.Module):
    num_qubits: int
    
    @nn.compact
    def __call__(self, x):
        theta = self.param('theta', nn.initializers.normal(stddev=0.1), (self.num_qubits,))
        effective_theta = theta * x
        prob_one = jnp.sin(effective_theta / 2.0) ** 2
        return prob_one

# --- 3. BITNET DENSE LAYER ---
def quantize_weights(w):
    scale = jnp.mean(jnp.abs(w))
    w_scaled = w / (scale + 1e-6)
    w_quant = jnp.round(jnp.clip(w_scaled, -1.0, 1.0))
    return w_quant

class BitDense(nn.Module):
    features: int
    
    @nn.compact
    def __call__(self, x):
        w = self.param('kernel', nn.initializers.lecun_normal(), (x.shape[-1], self.features))
        w_quant = quantize_weights(w)
        w_ste = w + jax.lax.stop_gradient(w_quant - w)
        return jnp.dot(x, w_ste)

# --- 4. SPIKING LAYER ---
class SpikingLIF(nn.Module):
    decay: float = 0.9
    threshold: float = 1.0
    
    @nn.compact
    def __call__(self, x, state):
        new_voltage = state * self.decay + x
        spike = spike_fn(new_voltage - self.threshold)
        final_voltage = new_voltage - (self.threshold * spike)
        return spike, final_voltage


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention for temporal context.
    Allows the model to attend to different positions in the sequence.
    """
    num_heads: int = 8
    head_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
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
