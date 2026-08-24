# Group 5 - CDCIR — CLIP-DINOv2 for Composed Image Retrieval

---

## Submitted File Structure

```
├── group5_report.pdf
├── group5_training.ipynb         # Training notebook and result visualisation
└── cdcir/                        # Demo tool folder
    ├── app.py                    # Gradio demo application
    ├── models.py                 # CDCIR architecture
    ├── feature_extraction.py     # CLIP & DINOv2 feature extraction pipelines
    ├── retrieval.py              # Gallery construction and cosine-similarity retrieval
    ├── requirements.txt          # Python dependencies
    ├── eval_bundle.pt            # Evaluation bundle (config + best model weights)
    ├── val_clip_cls.pt           # Precomputed CLIP CLS features for gallery images
    └── val_images/               # Gallery images (FashionIQ validation set)
```

---

## 1. Training & Visualisation

All training and evaluation code is in **`group5_training.ipynb`**.

### Precomputed Features

We precompute and cache all image and text features from the **FashionIQ** dataset. The following files are available on a public Google Drive:

> Download link: https://drive.google.com/drive/folders/1N9Bt86Z2KxjtvSqOynnNUENadDx_6n9x

| File                          | Description                                                        |
| ----------------------------- | ------------------------------------------------------------------ |
| `best_cdcir.pt`              | Best cdcir model weights (saved during training)                   |
| `clip_cls_fashioniq.pt`      | Precomputed CLIP CLS features for all FashionIQ images             |
| `dino_patches_fashioniq.pt`  | Precomputed DINOv2 patch tokens for all FashionIQ images           |
| `text_cls_fashioniq.pt`      | Precomputed CLIP text CLS features for all FashionIQ captions      |
| `text_tokens_fashioniq.pt`   | Precomputed CLIP token‑level text features for all FashionIQ captions |

> These files are downloaded from the Google Drive link during training and download code is available as part of the notebook. The notebook can be ran end to end.

## 2. Demo Tool

The demo is a Gradio web application that lets you upload a reference image, type a text modification, and retrieve the most similar images from the FashionIQ validation gallery.

### Running Locally

1. Navigate to the demo folder:

   ```
   cd cdcir
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

   This installs PyTorch, torchvision, open_clip_torch, Pillow, tqdm, and Gradio.

3. Ensure required files are present in the `cdcir/` folder.

4. Launch the app:

   ```
   python app.py
   ```

5. A local URL (e.g., `http://127.0.0.1:7860`) will appear in the terminal. Open it in your web browser to use the demo.

### Hosted Demo

The demo is also hosted on Hugging Face Spaces:

> https://huggingface.co/spaces/chadhurbala/CDCIR

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.1.0
- torchvision ≥ 0.16.0
- open_clip_torch ≥ 2.24.0
- Pillow ≥ 10.0.0
- tqdm ≥ 4.66.0
- Gradio
