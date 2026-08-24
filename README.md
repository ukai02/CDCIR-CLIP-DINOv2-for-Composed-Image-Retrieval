# CDCIR — CLIP-DINOv2 for Composed Image Retrieval

This repository contains the training notebooks, source code, and demo application for **CDCIR** (Group 5). This project addresses spatial feature deficiencies in standard CLIP backbones for Composed Image Retrieval (CIR), where global text-image embeddings consistently fail to capture fine-grained, highly localized image modifications requested by textual feedback.

## Key Architectural Innovations
* **TQDKV Cross-Attention:** Architected a custom Text-Query DINO-Key-Value (TQDKV) cross-attention layer to compose text token embeddings dynamically.
* **Sigmoid-Gated Fusion:** Integrates highly localized DINOv2 spatial patch features to significantly enhance structural retention.
* **Dual-Objective Contrastive Optimization:** Formulated a pipeline combining a Symmetric InfoNCE loss with an auxiliary regularizer to stabilize cross-attention gradient flows and enhance bidirectional query-target alignment.

## Performance
Evaluated on the **FashionIQ** benchmark, the model outperforms standard baseline CLIP4CIR and complex diffusion-based retrieval architectures utilizing only **7.22M trainable parameters**:
* **Average R@10:** 40.22%
* **Average R@50:** 61.83%

---

## Repository Structure
```text
├── group5_report.pdf             # Comprehensive project report detailing methodology
├── group5_training.ipynb         # Main training notebook and result visualization
└── cdcir/                        # Demo tool folder
    ├── app.py                    # Gradio demo application
    ├── models.py                 # Core CDCIR architecture
    ├── feature_extraction.py     # CLIP & DINOv2 feature extraction pipelines
    ├── retrieval.py              # Gallery construction and cosine-similarity retrieval
    ├── requirements.txt          # Python dependencies
    ├── eval_bundle.pt            # Evaluation bundle (config + best model weights)
    ├── val_clip_cls.pt           # Precomputed CLIP CLS features for gallery images
    └── val_images/               # Gallery images (FashionIQ validation set)
```

---

## 1. Training & Visualisation

All training and evaluation code is contained entirely within `group5_training.ipynb`. 

### Precomputed Features
To optimize training, we precompute and cache all image and text features from the FashionIQ dataset. The following files are hosted publicly on Google Drive:
> **Download Link:** [FashionIQ Features Google Drive](https://drive.google.com/drive/folders/1N9Bt86Z2KxjtvSqOynnNUENadDx_6n9x)

| File | Description |
| :--- | :--- |
| `best_cdcir.pt` | Best CDCIR model weights saved during the training phase |
| `clip_cls_fashioniq.pt` | Precomputed CLIP CLS features for all FashionIQ images |
| `dino_patches_fashioniq.pt` | Precomputed DINOv2 patch tokens for all FashionIQ images |
| `text_cls_fashioniq.pt` | Precomputed CLIP text CLS features for all FashionIQ captions |
| `text_tokens_fashioniq.pt` | Precomputed CLIP token‑level text features for all FashionIQ captions |

*Note: These files are automatically downloaded from the Google Drive link during execution. The Jupyter notebook can be run end-to-end without manual intervention.*

---

## 2. Interactive Demo Tool

The repository includes a Gradio web application that allows you to upload a reference image, input a textual modification query, and dynamically retrieve the most semantically and structurally similar images from the FashionIQ validation gallery.

### Hosted Demo
Try the live model directly on Hugging Face Spaces:
> **Live Demo:** [CDCIR on Hugging Face Spaces](https://huggingface.co/spaces/chadhurbala/CDCIR)

### Running Locally
To launch the Gradio app on your local machine:

1. Navigate to the demo directory:
   ```bash
   cd cdcir
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify that the required `.pt` files and `val_images/` gallery exist in the `cdcir/` folder.
4. Launch the application:
   ```bash
   python app.py
   ```
5. Open the generated local URL (e.g., `http://127.0.0.1:7860`) in your web browser.

---

## Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.1.0
* torchvision ≥ 0.16.0
* open_clip_torch ≥ 2.24.0
* Pillow ≥ 10.0.0
* tqdm ≥ 4.66.0
* Gradio
