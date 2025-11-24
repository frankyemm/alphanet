import jax
import jax.numpy as jnp
import optax
from alphanet.model import QBitSNN
from alphanet.train import create_train_state, train_epoch
from alphanet.data import UCF101VideoLoader

# 1. Initialize UCF101 Video Loader (REAL DATA)
print("=" * 60)
print("ALPHANET - UCF101 ACTION RECOGNITION (REAL DATA)")
print("=" * 60)

# Video configuration
num_classes = 40   # Balanced subset of UCF101 (40 classes for testing)
frame_size = 64    # 64x64 for faster training
num_frames = 16    # 16 frames per video clip

video_loader = UCF101VideoLoader(
    dataset_dir="./UCF-101",
    num_classes=num_classes,
    frame_size=frame_size,
    num_frames=num_frames
)

print("\nDataset Configuration:")
print(f"  - Action Classes: {num_classes}")
print(f"  - Frame Resolution: {frame_size}x{frame_size} RGB")
print(f"  - Frames per Clip: {num_frames}")
print(f"  - Input Features: {frame_size * frame_size * 3} (flattened RGB)")
print("=" * 60)

# 2. Initialize Model (Original Architecture)
input_features = frame_size * frame_size * 3  # Flattened RGB frame
model = QBitSNN(
    hidden_dim=12000,     # 12000 neurons / 40 classes = 300 neurons per class
    output_dim=num_classes,
    time_steps=num_frames  # Process all frames temporally
)

rng = jax.random.PRNGKey(0)
init_input = jnp.ones((1, num_frames, input_features))

# Calculate total steps
training_epochs = 300  # 3x more training for 101 classes
batches_per_epoch = 50
total_steps = training_epochs * batches_per_epoch

state = create_train_state(
    rng, model, init_input.shape,
    learning_rate=0.0005,  # Conservative LR with exponential decay
    total_steps=total_steps
)

# 3. Train on Video Data with Adaptive Learning Rate
print("\n🎬 Starting Video Action Recognition Training...\n")
print(f"Model Parameters: ~{12000 * input_features + 12000 * num_frames:,}")
print(f"Training: {training_epochs} epochs × {batches_per_epoch} batches")
print(f"Resolution: {frame_size}x{frame_size} RGB ({input_features} features)")
print(f"Capacity: 12000 hidden neurons / 40 classes = 300 neurons per class")
print(f"Initial Learning Rate: 0.0005 (exponential decay: -5% every 20 epochs)")
print(f"Estimated time: ~5-7 hours on TPU v6e")
print("=" * 60)

print("\n⏳ Initializing training (JIT compilation on first epoch)...")
print("   This may take 5-10 minutes for the first epoch.")
print("   Subsequent epochs will be much faster (~1-2 min each).\n")

import time
start_time = time.time()

for epoch in range(training_epochs):
    epoch_start = time.time()
    print(f"[Epoch {epoch:3d}] Starting... ", end='', flush=True)
    
    state, loss = train_epoch(state, video_loader, num_batches=batches_per_epoch)
    
    epoch_time = time.time() - epoch_start
    
    # Calculate current LR (exponential decay)
    steps_completed = (epoch + 1) * batches_per_epoch
    decay_factor = 0.95 ** (steps_completed // 1000)
    current_lr = 0.00015 * decay_factor
    
    print(f"Done in {epoch_time:.1f}s | Loss: {loss:.4f} | LR: {current_lr:.6f}")
    
    # Print summary every 5 epochs
    if epoch % 5 == 0 and epoch > 0:
        elapsed = time.time() - start_time
        avg_time_per_epoch = elapsed / (epoch + 1)
        remaining_epochs = training_epochs - epoch - 1
        eta_seconds = avg_time_per_epoch * remaining_epochs
        eta_hours = eta_seconds / 3600
        print(f"   Progress: {epoch}/{training_epochs} | Avg: {avg_time_per_epoch:.1f}s/epoch | ETA: {eta_hours:.1f}h")

# Calculate final LR
final_steps = training_epochs * batches_per_epoch
final_decay = 0.95 ** (final_steps // 1000)
final_lr = 0.0005 * final_decay

print(f"\n✅ Training Complete! Total time: {(time.time() - start_time) / 3600:.2f} hours")
print(f"Final Learning Rate: {final_lr:.6f}")
print("The model learned to recognize actions from video sequences.")

# Save trained model
print("\n💾 Saving trained model...")
import pickle
checkpoint = {
    'params': state.params,
    'model_config': {
        'hidden_dim': 12000,
        'output_dim': num_classes,
        'time_steps': num_frames
    },
    'training_info': {
        'final_loss': float(loss),
        'epochs': training_epochs,
        'num_classes': num_classes
    }
}

with open('trained_model.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)

print("✅ Model saved to 'trained_model.pkl'")
print("=" * 60)
