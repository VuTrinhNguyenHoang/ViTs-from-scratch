# Vision Transformers from Scratch

A clean, educational implementation of Vision Transformers (ViTs) built from scratch using PyTorch. This project demonstrates the core concepts of Vision Transformers through a well-structured, modular codebase designed for learning and research.

## 🎯 Overview

This implementation provides a complete Vision Transformer architecture including:
- **Patch Embedding**: Converts images into sequence of patch embeddings
- **Multi-Head Self-Attention**: Core attention mechanism with multiple heads
- **Transformer Encoder**: Stack of transformer blocks with layer normalization
- **Position Embeddings**: Learnable positional encoding for spatial information
- **Classification Head**: Final linear layer for image classification

The model is trained and evaluated on CIFAR-10 dataset, achieving competitvie results with a lightweight architecture.

## 🏗️ Architecture

### Model Components

```
Input Image (32x32x3)
    ↓
Patch Embedding (4x4 patches → 192-dim vectors)
    ↓ 
Add Position Embeddings + CLS Token
    ↓
Transformer Encoder (8 layers, 6 heads)
    ├── Multi-Head Self-Attention
    ├── Layer Normalization
    ├── Feed-Forward Network (MLP)
    └── Residual Connections + DropPath
    ↓
Classification Head (192 → 10 classes)
```

### Key Features

- **Modular Design**: Clean separation between patch embedding, transformer blocks, and classification head
- **Configurable Architecture**: Easy to adjust model depth, attention heads, embedding dimensions
- **Regularization**: Supports dropout, drop path, and weight decay for better generalization
- **Educational Code**: Well-commented implementation for understanding transformer mechanics

## 📁 Project Structure

```
ViTs-from-scratch/
├── src/
│   ├── models/
│   │   ├── vit.py             # Vision Transformer implementation
│   │   └── transformer.py     # Transformer encoder blocks
│   ├── datasets/
│   │   └── cifar10.py         # CIFAR-10 data loading
│   ├── train.py               # Training loop and evaluation
│   └── utils.py               # Utilities (seeding, metrics, logging)
├── configs/
│   └── vit_cifar10.yaml       # Model and training configuration
├── experiments/
│   ├── checkpoints/           # Model checkpoints (best.pt, last.pt)
│   └── logs/                  # Training metrics and logs
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/VuTrinhNguyenHoang/ViTs-from-scratch.git
cd ViTs-from-scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Train the Vision Transformer on CIFAR-10:

```bash
python src/train.py --config configs/vit_cifar10.yaml --out experiments/
```

### Configuration

Modify `configs/vit_cifar10.yaml` to experiment with different architectures:

```yaml
# Training parameters
seed: 42
epochs: 100
batch_size: 128
lr: 0.0001
weight_decay: 0.0

# Model architecture
model:
  img_size: 32          # Input image size
  patch_size: 4         # Patch size (4x4 patches)
  num_classes: 10       # Number of output classes
  embed_dim: 192        # Embedding dimension
  depth: 8              # Number of transformer layers
  num_heads: 6          # Number of attention heads
  mlp_ratio: 4.0        # MLP hidden dimension ratio
  attn_drop: 0.0        # Attention dropout
  proj_drop: 0.0        # Projection dropout
  mlp_drop: 0.1         # MLP dropout
  drop_path: 0.1        # Drop path (stochastic depth)
```

## 📊 Results

The model achieves competitive performance on CIFAR-10:
- **Architecture**: 8 layers, 6 attention heads, 192 embedding dimensions
- **Parameters**: ~2.5M parameters (lightweight compared to standard ViTs)
- **Training**: 100 epochs with Adam optimizer and cosine annealing
- **Regularization**: Dropout (0.1) and DropPath (0.1) for better generalization

Training progress is logged to `experiments/logs/` with both CSV and JSONL formats for easy analysis.

## 🔍 Key Implementation Details

### Patch Embedding
- Converts 32×32 images into 8×8 = 64 patches of size 4×4
- Uses Conv2D with stride=patch_size for efficient implementation
- Adds learnable position embeddings and CLS token

### Multi-Head Self-Attention
- Scaled dot-product attention with multiple heads
- Supports attention masking (though not used in standard ViT)
- Includes dropout for regularization

### Transformer Encoder
- Pre-normalization (LayerNorm before attention/MLP)
- Residual connections around attention and MLP blocks
- DropPath (stochastic depth) for improved training

### Training Features
- Automatic checkpointing (best and last models)
- Comprehensive logging (loss, accuracy, learning rate)
- Cosine annealing learning rate schedule
- Mixed precision support ready (can be easily added)

## 🛠️ Customization

### Adding New Datasets
Create a new dataset loader in `src/datasets/`:

```python
def get_your_dataset(batch_size=128, num_workers=2):
    # Implement data loading logic
    return trainloader, testloader
```

### Modifying Architecture
Key parameters to experiment with:
- `patch_size`: Smaller patches = more tokens, higher resolution
- `embed_dim`: Model width (usually 192, 384, 768, etc.)
- `depth`: Number of transformer layers
- `num_heads`: Number of attention heads (should divide embed_dim)
- `mlp_ratio`: Hidden dimension ratio in feed-forward network

### Advanced Features
The codebase is designed to easily add:
- Different attention patterns (sparse, local, etc.)
- Alternative position encodings (sinusoidal, relative, etc.)
- Multi-scale patch embeddings
- Knowledge distillation
- Mixed precision training

## 📚 Educational Value

This implementation is designed for learning and includes:
- **Clear Comments**: Every major component is well-documented
- **Modular Structure**: Easy to understand and modify individual components
- **Step-by-Step**: Build understanding from patches → attention → full model
- **Visualization Ready**: Easy to add attention visualizations and feature maps
- **Research Friendly**: Clean base for experimenting with ViT variants

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional datasets (ImageNet, custom datasets)
- Attention visualization tools
- Model analysis utilities
- Performance optimizations
- Documentation improvements

## 📖 References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [DeiT: Data-efficient Image Transformers](https://arxiv.org/abs/2012.12877)

## 📄 License

This project is open source and available under the MIT License.

---

Built with ❤️ for learning and research in computer vision and transformers.