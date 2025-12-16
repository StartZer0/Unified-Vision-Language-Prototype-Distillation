# Unified Vision Language Prototype Distillation (UVLP)

A novel framework for dataset distillation that extends Vision-Language Category Prototype (VLCP)(Zou et al., 2025) work approach of using image and text descriptions for prototype selection and synthesis.

## Overview

This repository, Unified Vision-Language Prototype Distillation, contains the code implementation of the master’s research project titled “Dataset Distillation via Fused Vision-Language Prototypes,” which proposes a framework to fuse image and text descriptions to discover prototypes for dataset distillation:

1. **Joint Vision-Language Clustering (JVL-C)**: Fuses CLIP image and text embeddings in a shared spherical space for more semantically coherent clustering
2. **Adaptive Alpha Optimization**: Per-cluster optimization of the vision-language fusion weight using Fisher-style objectives
4. **Dynamic Weight Scheduling**: Gap-based urgency weighting for balancing intra-cluster cohesion, text alignment, and inter-class margin

## Method

The UVLP framework operates in the following stages:

1. **Feature Extraction**: Extract CLIP image/text embeddings and VAE latents from training images
2. **Fused Embedding Creation**: Combine image and text embeddings with adaptive alpha weighting
3. **Spherical K-Means Clustering**: Cluster fused embeddings using cosine geometry
4. **Prototype Selection**: Select image and text prototypes using hubness-based scoring
5. **Image Synthesis**: Generate distilled images using Stable Diffusion Latents2Img pipeline

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Git

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Unified-Vision-Language-Prototype-Distillation.git
cd Unified-Vision-Language-Prototype-Distillation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Clone the VLCP baseline repository (required for diffusers patching and evaluation):
```bash
git clone https://github.com/StartZero0/Dataset-Distillation-via-Vision-Language-Category-Prototype.git
```

4. Set up the environment (patches diffusers with custom pipelines):
```bash
python scripts/setup_environment.py --baseline-repo ./Dataset-Distillation-via-Vision-Language-Category-Prototype
```

## Usage

### Basic Distillation

Run distillation on ImageWoof with 10 images per class:

```bash
python scripts/run_distillation.py --dataset imagewoof --ipc 10 --seed 0
```

### Command Line Options

```
--dataset       Dataset to distill (imagewoof, imagenette, imageidc)
--ipc           Images per class (default: 10)
--seed          Random seed (default: 0)
--alpha         Default fusion weight, 0=text, 1=image (default: 0.5)
--project-root  Project root directory
--finetuned-model  Path to fine-tuned Stable Diffusion model
--run-eval      Run Minimax evaluation after distillation
--eval-repeat   Number of evaluation repeats (default: 3)
```

### Running Evaluation

To run the Minimax evaluation protocol after distillation:

```bash
python scripts/run_distillation.py --dataset imagewoof --ipc 10 --run-eval
```

## Project Structure

```
Unified-Vision-Language-Prototype-Distillation/
├── README.md
├── LICENSE
├── requirements.txt
├── uvlp/                          # Main package
│   ├── configs/                   # Configuration management
│   │   └── base_config.py
│   ├── clustering/                # Clustering algorithms
│   │   ├── jvl_clustering.py      # JVL-C implementation
│   │   ├── spherical_kmeans.py    # Spherical K-Means
│   │   └── hubness_selection.py   # Hubness-based selection
│   ├── data/                      # Data loading utilities
│   │   ├── dataset_loader.py
│   │   └── clip_embeddings.py
│   ├── distillation/              # Distillation pipeline
│   │   ├── pipeline.py            # Main distillation loop
│   │   ├── latents2img.py         # Latents2Img pipeline
│   │   └── image_synthesis.py
│   ├── evaluation/                # Evaluation utilities
│   │   ├── minimax_eval.py
│   │   └── clip_diagnostics.py
│   ├── optimization/              # Optimization algorithms
│   │   ├── golden_section.py
│   │   ├── dynamic_weights.py
│   │   └── fisher_alpha.py
│   └── utils/                     # Utility functions
│       ├── environment.py
│       ├── memory.py
│       └── visualization.py
├── scripts/                       # Entry point scripts
│   ├── run_distillation.py
│   └── setup_environment.py
├── notebooks/                     # Jupyter notebooks
└── samples/                       # Sample outputs
```

## Datasets

This framework supports the following datasets: ImageWoof, ImageNette, ImageIDC


### Data Preparation

1. Download the dataset (e.g., ImageWoof):

2. Ensure the text descriptions are available from the VLCP repository:
   - Text descriptions: [VLCP text_description](https://github.com/StartZero0/Dataset-Distillation-via-Vision-Language-Category-Prototype/tree/main/03_distiilation/text_description)

## Fine-tuning Stable Diffusion

Before running distillation, you need a fine-tuned Stable Diffusion model. Follow the VLCP baseline instructions:

1. Prepare training data with text descriptions
2. Run the fine-tuning script from the VLCP repository
3. Point to the fine-tuned model using `--finetuned-model`

See the [VLCP repository](https://github.com/StartZero0/Dataset-Distillation-via-Vision-Language-Category-Prototype) for detailed fine-tuning instructions.

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | Adaptive (0.2-0.8) | Vision-language fusion weight optimized via Fisher objective (0=text only, 1=image only) |
| `ipc` | 10 | Images per class |
| `contamination` | 0.1 | LOF outlier detection threshold |
| `top_k_visual` | 6 (IPC 10), 3 (IPC 20), 1 (IPC 50) | Number of top hubness-scored samples to average for visual prototype |
| `strength` | 0.75 | Diffusion denoising strength (SDEdit) |
| `guidance_scale` | 10.5 | Classifier-free guidance scale |

## Evaluation

The framework uses the Minimax evaluation protocol from the VLCP baseline. This trains multiple architectures on the distilled dataset and evaluates on the real validation set:

- ResNet-18
- ResNet-AP (depth 10)
- ConvNet (depth 6)

Results are reported as mean accuracy across multiple runs.

## Benchmark Results

### ImageNette and ImageIDC Results

Benchmark results on ImageNette and ImageIDC datasets. The best results are **bolded** and the second-best are *underlined*.

| Dataset | IPC | Random | DiT | DM | Minimax | D⁴M | VLCP | OURS |
|---------|-----|--------|-----|-----|---------|-----|------|------|
| **ImageNette** | 10 | 54.2 ± 1.6 | 59.1 ± 0.7 | 60.8 ± 0.6 | 57.7 ± 1.2 | 60.9 ± 1.7 | *61.0 ± 0.5* | **63.0 ± 0.4** |
| | 20 | 63.5 ± 0.5 | 64.8 ± 1.2 | 66.5 ± 1.1 | 64.7 ± 0.8 | 66.3 ± 1.3 | *67.5 ± 1.0* | **67.7 ± 0.3** |
| | 50 | 76.1 ± 1.1 | 73.3 ± 0.9 | 76.2 ± 0.4 | 73.9 ± 0.3 | *77.7 ± 1.1* | **77.8 ± 0.5** | 77.5 ± 0.3 |
| **ImageIDC** | 10 | 48.1 ± 0.8 | 54.1 ± 0.4 | 52.8 ± 0.5 | 51.9 ± 1.4 | 50.3 ± 1.0 | *54.5 ± 0.6* | **55.3 ± 0.5** |
| | 20 | 52.5 ± 0.9 | 58.9 ± 0.2 | 58.5 ± 0.4 | 59.1 ± 3.7 | 55.8 ± 0.2 | *60.0 ± 0.3* | **60.3 ± 0.6** |
| | 50 | 68.1 ± 0.7 | 64.3 ± 0.6 | 69.1 ± 0.8 | 69.4 ± 1.4 | 69.1 ± 2.4 | **72.7 ± 0.4** | *72.3 ± 0.4* |

### ImageWoof Results (IPC 10)

Results on ImageWoof with IPC 10 (0.8% ratio) across different test architectures.

| Test Model | Random | K-Center | Herding | DiT | DM | GLaD | Minimax | D⁴M | VLCP | Ours | Full |
|------------|--------|----------|---------|-----|-----|------|---------|-----|------|------|------|
| ConvNet-6 | 24.3 ± 1.1 | 19.4 ± 0.9 | 26.7 ± 0.5 | **34.2 ± 1.1** | 26.9 ± 1.2 | *33.8 ± 0.9* | 33.3 ± 1.7 | 29.4 ± 0.9 | 32.5 ± 0.4 | 32.2 ± 0.6 | 86.4 ± 0.2 |
| ResNetAP-10 | 29.4 ± 0.8 | 22.1 ± 0.1 | 32.0 ± 0.3 | 34.7 ± 0.5 | 30.3 ± 1.2 | 32.9 ± 0.9 | 36.2 ± 3.2 | 33.2 ± 2.1 | *36.6 ± 0.2* | **37.7 ± 0.3** | 87.5 ± 0.5 |
| ResNet-18 | 27.7 ± 0.9 | 21.1 ± 0.4 | 30.2 ± 1.2 | 34.7 ± 0.4 | 33.4 ± 0.7 | 31.7 ± 0.8 | 35.7 ± 1.6 | 32.3 ± 1.2 | *35.9 ± 0.3* | **36.0 ± 0.4** | 89.3 ± 1.2 |


## Acknowledgments

This work builds upon:
- [VLCP: Vision-Language Category Prototype](https://zou-yawen.github.io/DD_via_vision-language/)
- [CLIP](https://github.com/openai/CLIP)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Diffusers](https://github.com/huggingface/diffusers)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

