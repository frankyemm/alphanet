# AlphaNet: Quantum-BitNet-SNN for Video Action Recognition

A novel neural network architecture combining **Quantum Computing**, **Binary Neural Networks**, and **Spiking Neural Networks** for efficient video action recognition.

## 🎯 Architecture

AlphaNet integrates three cutting-edge paradigms:

1. **Quantum Layer**: Parameterized quantum circuits for feature transformation.
2. **BitNet Layer**: 1-bit quantized dense layers for memory efficiency.
3. **Spiking LIF Neurons**: Leaky Integrate-and-Fire neurons for temporal processing.

```
Input Video (T×H×W×C) 
    ↓
Quantum Layer (Rotation gates)
    ↓
BitNet Layer (1-bit weights)
    ↓
Spiking LIF Layer (Temporal integration)
    ↓
Output (Action classes)
```

## 📊 Results on UCF101

| Configuration | Classes | Resolution | Hidden Dim | Top-1 Acc | Top-5 Acc | Parameters |
|---------------|---------|------------|------------|-----------|-----------|------------|
| **AlphaNet-101** | 101 | 64×64 | 1024 | **57.25%** | **98.25%** | ~12.5M |
| AlphaNet-10 | 10 | 64×64 | 2048 | ~90% | ~99% | ~25M |

### Key Findings

- ✅ **57% Top-1 accuracy** on full UCF101 (101 classes) demonstrates the architecture's viability.
- ✅ **98% Top-5 accuracy** shows the model understands action semantics.
- ✅ **Efficient**: 12.5M parameters vs 100M+ for standard 3D CNNs.
- ✅ **Scalable**: Performance improves with increased capacity per class.

## 🚀 Quick Start

### Installation

Ensure you have Python 3.8+ installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/frankyemm/alphanet.git
   cd alphanet
   ```

2. Install dependencies:
   ```bash
   pip install jax[tpu] flax optax
   pip install opencv-python numpy datasets
   ```
   *Note: Adjust the JAX installation command based on your hardware (CPU/GPU/TPU). See [JAX installation guide](https://github.com/google/jax#installation).*

### Training

To train the model on the UCF101 dataset:

1. **Download UCF101 dataset**:
   ```bash
   wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
   unrar x UCF101.rar
   ```

2. **Run training**:
   ```bash
   # Train on TPU (or GPU/CPU depending on JAX installation)
   python run_video_tpu.py
   ```
   This script initializes the model, loads the data, and starts the training loop. It saves the trained model to `trained_model.pkl`.

### Evaluation

To evaluate a trained model:

```bash
python evaluate_model.py
```
This script loads `trained_model.pkl` and computes metrics like Top-1/Top-5 accuracy, precision, recall, and F1-score on the test set.

## 📁 Project Structure

```
alphanet/
├── alphanet/
│   ├── __init__.py       # Package initialization
│   ├── model.py          # QBitSNN architecture definition
│   ├── layers.py         # Custom layers (Quantum, BitNet, Spiking)
│   ├── train.py          # Training state and step functions
│   └── data.py           # Data loaders (UCF101, Synthetic, etc.)
├── run_video_tpu.py      # Main training script
├── evaluate_model.py     # Evaluation and metrics script
└── README.md             # Project documentation
```

## 🔬 Architecture Details

### Quantum Layer (`alphanet.layers.QuantumLayer`)
- **Mechanism**: Parameterized rotation gates (RY, RZ) acting on input features.
- **Function**: Encodes classical data into quantum-inspired representations.
- **Parameters**: Learnable parameters $\theta \in \mathbb{R}^d$.

### BitNet Layer (`alphanet.layers.BitDense`)
- **Mechanism**: Dense layer with 1-bit quantized weights $W \in \{-1, 0, +1\}$.
- **Function**: Performs matrix multiplication with extreme memory efficiency (up to 32x reduction).
- **Training**: Uses Straight-Through Estimator (STE) for gradient calculation.

### Spiking LIF Layer (`alphanet.layers.SpikingLIF`)
- **Mechanism**: Leaky Integrate-and-Fire neurons with surrogate gradients.
- **Function**: Captures temporal dynamics: $V(t+1) = \beta V(t) + I(t)$.
- **Spike Generation**: Emits a spike (1.0) when $V > \theta$.

## 📈 Training Configuration

Default configuration in `run_video_tpu.py`:
- **Classes**: 40 (Subset) / 101 (Full)
- **Frame Size**: 64x64
- **Hidden Dimension**: 12000 (Scales with class count)
- **Learning Rate**: 0.0005 with exponential decay
- **Epochs**: 300

## 🎓 Citation

If you use AlphaNet in your research, please cite:

```bibtex
@software{alphanet2024,
  title={AlphaNet: Quantum-BitNet-SNN for Video Action Recognition},
  author={Your Name},
  year={2024},
  url={https://github.com/frankyemm/alphanet}
}
```

## 📚 References & Citations

This work builds upon and is inspired by the following research:

### Core Technologies

**BitNet (1-bit Neural Networks)**
- Wang, H., et al. (2023). "BitNet: Scaling 1-bit Transformers for Large Language Models." Microsoft Research.
- [Paper](https://arxiv.org/abs/2310.11453)

**Spiking Neural Networks**
- Maass, W. (1997). "Networks of spiking neurons: The third generation of neural network models." Neural Networks.
- Tavanaei, A., et al. (2019). "Deep learning in spiking neural networks." Neural Networks.

**Quantum Machine Learning**
- Schuld, M., & Petruccione, F. (2018). "Supervised Learning with Quantum Computers." Springer.
- Biamonte, J., et al. (2017). "Quantum machine learning." Nature.

### Dataset

**UCF101**
- Soomro, K., Zamir, A. R., & Shah, M. (2012). "UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild." CRCV-TR-12-01.
- [Dataset](https://www.crcv.ucf.edu/data/UCF101.php)

### Infrastructure

**Google Cloud TPU**
- This research was conducted using Google Cloud TPU v6e resources.
- Framework: JAX/Flax by Google Research
- [JAX Documentation](https://github.com/google/jax)

## 📄 License

MIT License

## 🙏 Acknowledgments

- **UCF101 Dataset**: Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah (University of Central Florida)
- **BitNet**: Microsoft Research for 1-bit quantization techniques
- **Spiking Neural Networks**: Wolfgang Maass and the neuromorphic computing community
- **Quantum ML**: Maria Schuld, Francesco Petruccione, and quantum computing researchers
- **JAX/Flax Framework**: Google Research
- **Infrastructure**: Google Cloud TPU Research Credits Program
- **Community**: Open-source contributors to JAX, Flax, and Optax

## 📧 Contact

- GitHub: [@frankyemm](https://github.com/frankyemm)
- Email: frankycardona1927@gmail.com

---

**Status**: Research prototype - Achieves 57% Top-1 accuracy on UCF101, demonstrating the potential of hybrid Quantum-BitNet-SNN architectures for video understanding.
