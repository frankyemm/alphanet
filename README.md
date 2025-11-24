# AlphaNet: Quantum-BitNet-SNN for Video Action Recognition

A novel neural network architecture combining **Quantum Computing**, **Binary Neural Networks**, and **Spiking Neural Networks** for efficient video action recognition.

## 🎯 Architecture

AlphaNet integrates three cutting-edge paradigms:

1. **Quantum Layer**: Parameterized quantum circuits for feature transformation
2. **BitNet Layer**: 1-bit quantized dense layers for memory efficiency
3. **Spiking LIF Neurons**: Leaky Integrate-and-Fire neurons for temporal processing

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

- ✅ **57% Top-1 accuracy** on full UCF101 (101 classes) demonstrates the architecture's viability
- ✅ **98% Top-5 accuracy** shows the model understands action semantics
- ✅ **Efficient**: 12.5M parameters vs 100M+ for standard 3D CNNs
- ✅ **Scalable**: Performance improves with increased capacity per class

## 🚀 Quick Start

### Installation

```bash
pip install jax[tpu] flax optax
pip install opencv-python numpy
```

### Training

```bash
# Download UCF101 dataset
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar

# Train on TPU
export PJRT_DEVICE=TPU
python run_video_tpu.py
```

### Evaluation

```bash
python evaluate_model.py
```

## 📁 Project Structure

```
alphanet/
├── alphanet/
│   ├── __init__.py
│   ├── model.py          # QBitSNN architecture
│   ├── layers.py         # Quantum, BitNet, Spiking layers
│   ├── train.py          # Training utilities
│   └── data.py           # UCF101 data loader
├── run_video_tpu.py      # Training script
├── evaluate_model.py     # Evaluation script
└── README.md
```

## 🔬 Architecture Details

### Quantum Layer
- Parameterized rotation gates (RY, RZ)
- Quantum state measurement
- Learnable parameters: θ ∈ ℝ^d

### BitNet Layer
- 1-bit quantized weights: W ∈ {-1, +1}
- Sign activation function
- Memory efficient: 32× reduction

### Spiking LIF Layer
- Leaky Integrate-and-Fire neurons
- Temporal dynamics: V(t+1) = βV(t) + I(t)
- Spike threshold: V > θ

## 📈 Training Configuration

```python
# Optimal configuration (101 classes)
num_classes = 101
frame_size = 64
hidden_dim = 1024
learning_rate = 0.0005 (exponential decay)
epochs = 300
```

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

## � References & Citations

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

## �📄 License

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
