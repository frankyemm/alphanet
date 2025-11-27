import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from .model import QBitSNN

class TrainState(train_state.TrainState):
    """Custom TrainState that includes a gradient mask for freezing layers.

    Attributes:
        grad_mask (any): A pytree matching the structure of `params`, containing
            masks (0 or 1) to be applied to gradients.
    """
    grad_mask: any = None

def create_train_state(rng, model, input_shape, learning_rate=0.0005, total_steps=10000):
    """Creates and initializes the training state.

    This function initializes the model parameters, sets up the optimizer with
    an exponential decay schedule, and initializes the gradient mask (all ones).

    Args:
        rng (jax.random.PRNGKey): Random number generator key.
        model (flax.linen.Module): The model instance.
        input_shape (tuple): Shape of the input tensor (excluding batch dimension).
        learning_rate (float): Initial learning rate. Defaults to 0.0005.
        total_steps (int): Total number of training steps (used for documentation,
            though the schedule defined here is step-based). Defaults to 10000.

    Returns:
        TrainState: The initialized training state.
    """
    params = model.init(rng, jnp.ones(input_shape))['params']
    
    # Exponential decay schedule (proven to work well)
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=1000,  # Decay every 1000 steps (~20 epochs)
        decay_rate=0.95,        # Reduce by 5% each time
        staircase=True          # Discrete steps
    )
    
    # AdamW optimizer with weight decay + gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients to prevent explosion
        optax.adamw(learning_rate=schedule, weight_decay=0.01)
    )
    
    # Initially, no mask (all 1.0)
    grad_mask = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, grad_mask=grad_mask)

def freeze_layers(state, layer_names=['QuantumCore', 'BitNetCore']):
    """Creates a mask to zero out gradients for specified layers.

    Args:
        state (TrainState): The current training state.
        layer_names (list): List of layer names (substrings) to freeze.
            Defaults to ['QuantumCore', 'BitNetCore'].

    Returns:
        TrainState: A new TrainState with the updated `grad_mask`.
    """
    def map_fn(path, param):
        # path is a tuple of keys, e.g., ('params', 'QuantumCore', 'theta')
        for name in layer_names:
            if name in path:
                return jnp.zeros_like(param) # Freeze
        return jnp.ones_like(param) # Train
    
    # Use tree_map_with_path to generate the mask based on structure
    new_mask = jax.tree_util.tree_map_with_path(map_fn, state.params)
    return state.replace(grad_mask=new_mask)

@jax.jit
def train_step(state, batch_x, batch_y):
    """Performs a single training step.

    Computes loss and gradients, applies the gradient mask (if any), and updates
    the model parameters.

    Args:
        state (TrainState): The current training state.
        batch_x (jax.numpy.ndarray): Input batch.
        batch_y (jax.numpy.ndarray): Target batch.

    Returns:
        tuple: A tuple containing:
            - state (TrainState): The updated training state.
            - loss (jax.numpy.ndarray): The loss value for the step.
            - logits (jax.numpy.ndarray): The model predictions.
    """
    def loss_fn(params):
        logits, spikes = state.apply_fn({'params': params}, batch_x)
        
        if batch_y.ndim > 1:
            batch_y_flat = batch_y[:, -1]
        else:
            batch_y_flat = batch_y
            
        batch_y_flat = jnp.squeeze(batch_y_flat)
        
        if batch_y_flat.dtype == jnp.int32 or batch_y_flat.dtype == jnp.int64:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y_flat).mean()
        else:
            loss = jnp.mean((logits - batch_y_flat) ** 2)
           # Regularization: SNN Sparsity penalty (Activity regularization)
        # We want neurons to spike rarely (energy efficiency)
        spike_loss = jnp.mean(spikes) * 0.001  # Reduced from 0.01 to allow more capacity
        return loss + spike_loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # --- APPLY FREEZING MASK ---
    # Multiply gradients by the mask (0 for frozen, 1 for trainable)
    grads = jax.tree_util.tree_map(lambda g, m: g * m, grads, state.grad_mask)
    
    state = state.apply_gradients(grads=grads)
    return state, loss, logits

def train_epoch(state, train_loader, num_batches=10):
    """Runs training for a single epoch (multiple batches).

    Args:
        state (TrainState): The current training state.
        train_loader (object): Data loader object with a `get_batch()` method.
        num_batches (int): Number of batches to process in this epoch. Defaults to 10.

    Returns:
        tuple: A tuple containing:
            - state (TrainState): The updated training state.
            - mean_loss (jax.numpy.ndarray): The average loss over the epoch.
    """
    epoch_loss = []
    for _ in range(num_batches):
        batch_x, batch_y = train_loader.get_batch()
        batch_x = jnp.array(batch_x)
        batch_y = jnp.array(batch_y)
        state, loss, _ = train_step(state, batch_x, batch_y)
        epoch_loss.append(loss)
    return state, jnp.mean(jnp.array(epoch_loss))
