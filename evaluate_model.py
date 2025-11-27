"""Evaluation script for AlphaNet on UCF101 dataset.

This script loads a trained AlphaNet model and evaluates its performance on the
UCF101 test set. It computes various metrics including Top-1/Top-5 accuracy,
Precision, Recall, F1-Score, and generates a confusion matrix.
"""

import jax
import jax.numpy as jnp
import numpy as np
from alphanet.model import QBitSNN
from alphanet.train import create_train_state
from alphanet.data import UCF101VideoLoader
import pickle

def evaluate_model(state, test_loader, num_batches=50):
    """Evaluates the model on the test set.

    Runs the model on batches of test data and aggregates predictions.

    Args:
        state (TrainState): The trained model state.
        test_loader (UCF101VideoLoader): Data loader for the test set.
        num_batches (int): Number of batches to evaluate. Defaults to 50.

    Returns:
        tuple: A tuple containing:
            - metrics (dict): Dictionary of computed metrics (accuracy, etc.).
            - all_predictions (numpy.ndarray): Array of all predicted class indices.
            - all_labels (numpy.ndarray): Array of all true class indices.
            - all_logits (numpy.ndarray): Array of all output logits.
    """
    all_predictions = []
    all_labels = []
    all_logits = []
    
    print("Running evaluation...")
    for i in range(num_batches):
        # Get test batch
        videos, labels = test_loader.get_batch(batch_size=8)
        videos = jnp.array(videos)
        labels = jnp.array(labels)
        
        # Forward pass (no gradients needed)
        logits, _ = state.apply_fn({'params': state.params}, videos)
        
        # Get predictions
        predictions = jnp.argmax(logits, axis=-1)
        
        all_predictions.extend(np.array(predictions))
        all_labels.extend(np.array(labels))
        all_logits.append(np.array(logits))
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_batches} batches...")
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.concatenate(all_logits, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels, all_logits, test_loader.num_classes)
    
    return metrics, all_predictions, all_labels, all_logits


def compute_metrics(predictions, labels, logits, num_classes):
    """Computes comprehensive evaluation metrics.

    Args:
        predictions (numpy.ndarray): Array of predicted class indices.
        labels (numpy.ndarray): Array of true class indices.
        logits (numpy.ndarray): Array of raw output logits.
        num_classes (int): Total number of classes.

    Returns:
        dict: A dictionary containing:
            - top1_accuracy (float): Top-1 accuracy percentage.
            - top5_accuracy (float): Top-5 accuracy percentage.
            - mean_precision (float): Macro-averaged precision.
            - mean_recall (float): Macro-averaged recall.
            - mean_f1 (float): Macro-averaged F1-score.
            - per_class_accuracy (list): Accuracy per class.
            - per_class_precision (list): Precision per class.
            - per_class_recall (list): Recall per class.
            - per_class_f1 (list): F1-score per class.
            - confusion_matrix (numpy.ndarray): Confusion matrix (True x Pred).
    """
    
    # 1. Top-1 Accuracy (standard metric)
    top1_accuracy = np.mean(predictions == labels) * 100
    
    # 2. Top-5 Accuracy (for comparison with papers)
    top5_predictions = np.argsort(logits, axis=-1)[:, -5:]  # Top 5 predictions
    top5_accuracy = np.mean([labels[i] in top5_predictions[i] for i in range(len(labels))]) * 100
    
    # 3. Per-class metrics
    per_class_accuracy = []
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    
    for class_idx in range(num_classes):
        # Samples for this class
        class_mask = (labels == class_idx)
        
        if np.sum(class_mask) == 0:
            continue
            
        # True Positives, False Positives, False Negatives
        tp = np.sum((predictions == class_idx) & (labels == class_idx))
        fp = np.sum((predictions == class_idx) & (labels != class_idx))
        fn = np.sum((predictions != class_idx) & (labels == class_idx))
        
        # Metrics
        accuracy = np.mean(predictions[class_mask] == labels[class_mask]) * 100
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_accuracy.append(accuracy)
        per_class_precision.append(precision * 100)
        per_class_recall.append(recall * 100)
        per_class_f1.append(f1 * 100)
    
    # 4. Confusion Matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(labels, predictions):
        confusion_matrix[true_label, pred_label] += 1
    
    # 5. Mean Average Precision (mAP)
    mean_precision = np.mean(per_class_precision)
    mean_recall = np.mean(per_class_recall)
    mean_f1 = np.mean(per_class_f1)
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'per_class_accuracy': per_class_accuracy,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'per_class_f1': per_class_f1,
        'confusion_matrix': confusion_matrix
    }


def print_evaluation_report(metrics, class_names):
    """Prints a detailed evaluation report to the console.

    Args:
        metrics (dict): Dictionary of metrics returned by `compute_metrics`.
        class_names (list): List of class names corresponding to indices.
    """
    
    print("\n" + "=" * 70)
    print("ALPHANET - UCF101 EVALUATION REPORT")
    print("=" * 70)
    
    # Overall metrics
    print("\n📊 OVERALL METRICS:")
    print(f"  Top-1 Accuracy:     {metrics['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy:     {metrics['top5_accuracy']:.2f}%")
    print(f"  Mean Precision:     {metrics['mean_precision']:.2f}%")
    print(f"  Mean Recall:        {metrics['mean_recall']:.2f}%")
    print(f"  Mean F1-Score:      {metrics['mean_f1']:.2f}%")
    
    # Per-class breakdown
    print("\n📋 PER-CLASS PERFORMANCE:")
    print(f"{'Class':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    
    for i, class_name in enumerate(class_names):
        if i < len(metrics['per_class_accuracy']):
            print(f"{class_name:<25} "
                  f"{metrics['per_class_accuracy'][i]:>10.2f}%  "
                  f"{metrics['per_class_precision'][i]:>10.2f}%  "
                  f"{metrics['per_class_recall'][i]:>10.2f}%  "
                  f"{metrics['per_class_f1'][i]:>10.2f}%")
    
    # Confusion matrix summary
    print("\n🔢 CONFUSION MATRIX:")
    cm = metrics['confusion_matrix']
    
    # Print header
    print(f"{'True\\Pred':<15}", end="")
    for i in range(len(class_names)):
        print(f"{i:>6}", end="")
    print()
    
    # Print matrix
    for i in range(len(class_names)):
        print(f"{i:<3} {class_names[i][:10]:<11}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>6}", end="")
        print()
    
    # Best and worst classes
    print("\n⭐ BEST PERFORMING CLASSES:")
    best_indices = np.argsort(metrics['per_class_f1'])[-3:][::-1]
    for idx in best_indices:
        if idx < len(class_names):
            print(f"  {class_names[idx]}: {metrics['per_class_f1'][idx]:.2f}% F1")
    
    print("\n⚠️  WORST PERFORMING CLASSES:")
    worst_indices = np.argsort(metrics['per_class_f1'])[:3]
    for idx in worst_indices:
        if idx < len(class_names):
            print(f"  {class_names[idx]}: {metrics['per_class_f1'][idx]:.2f}% F1")
    
    print("\n" + "=" * 70)


def save_results(metrics, predictions, labels, logits, filename="evaluation_results.pkl"):
    """Saves evaluation results to a pickle file.

    Args:
        metrics (dict): computed metrics.
        predictions (numpy.ndarray): predicted labels.
        labels (numpy.ndarray): true labels.
        logits (numpy.ndarray): raw logits.
        filename (str): Output filename. Defaults to "evaluation_results.pkl".
    """
    results = {
        'metrics': metrics,
        'predictions': predictions,
        'labels': labels,
        'logits': logits
    }
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Results saved to {filename}")


# Main evaluation script
if __name__ == "__main__":
    print("=" * 70)
    print("ALPHANET - UCF101 MODEL EVALUATION")
    print("=" * 70)
    
    # 1. Load test data
    print("\n📂 Loading UCF101 test set...")
    test_loader = UCF101VideoLoader(
        dataset_dir="./UCF-101",
        num_classes=101,
        frame_size=64,   # Match existing trained model (64x64)
        num_frames=16
    )
    
    # 2. Load checkpoint first to get model config
    print("\n🧠 Loading model checkpoint...")
    try:
        with open('trained_model.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
        
        print(f"✅ Loaded checkpoint:")
        print(f"   - Final training loss: {checkpoint['training_info']['final_loss']:.4f}")
        print(f"   - Trained epochs: {checkpoint['training_info']['epochs']}")
        print(f"   - Classes: {checkpoint['training_info']['num_classes']}")
        print(f"   - Hidden dimensions: {checkpoint['model_config']['hidden_dim']}")
        
        # Use config from checkpoint
        hidden_dim = checkpoint['model_config']['hidden_dim']
        output_dim = checkpoint['model_config']['output_dim']
        time_steps = checkpoint['model_config']['time_steps']
        
    except FileNotFoundError:
        print("⚠️  Warning: No checkpoint found (trained_model.pkl)")
        print("   Using default configuration.")
        hidden_dim = 2048
        output_dim = 101
        time_steps = 16
        checkpoint = None
    
    # 3. Initialize model with correct dimensions
    print("\n🧠 Initializing model...")
    input_features = 64 * 64 * 3  # Match existing model (64x64 RGB)
    model = QBitSNN(
        hidden_dim=hidden_dim,  # Use dimension from checkpoint
        output_dim=output_dim,
        time_steps=time_steps
    )
    
    rng = jax.random.PRNGKey(0)
    init_input = jnp.ones((1, 16, input_features))
    
    # Create state (will load trained weights)
    state = create_train_state(
        rng, model, init_input.shape,
        learning_rate=0.001,
        total_steps=5000
    )
    
    # Load trained weights (already loaded above)
    if checkpoint is not None:
        state = state.replace(params=checkpoint['params'])
    else:
        print("⚠️  Warning: Using randomly initialized weights.")
        print("   Run 'python3 run_video_tpu.py' first to train the model.")
    
    # 3. Run evaluation on FULL dataset
    print("\n🧪 Running evaluation on FULL test set...")
    print("   This will take ~10-15 minutes...")
    
    # Calculate batches needed for full dataset
    total_videos = len(test_loader.video_paths)
    batch_size = 8
    num_batches = (total_videos + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"   Total videos: {total_videos}")
    print(f"   Batch size: {batch_size}")
    print(f"   Total batches: {num_batches}\n")
    
    metrics, predictions, labels, logits = evaluate_model(state, test_loader, num_batches=num_batches)
    
    # 4. Print report
    print_evaluation_report(metrics, test_loader.classes)
    
    # 5. Save results
    save_results(metrics, predictions, labels, logits)
    
    print("\n✅ Evaluation complete!")
