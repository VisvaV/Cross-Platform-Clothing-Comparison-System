# Cross-Platform Clothing Comparison System

A Python-based system that lets users find visually similar clothing products across multiple e-commerce platforms (Myntra, Ajio, H&M, Zara, Amazon Fashion, Flipkart Fashion, ASOS, Uniqlo) by uploading an image, entering a text description, or both.

## How It Works

The system runs in two phases:

1. **Offline Pipeline** — scrapes products, downloads images, generates fused embeddings (CNN + color + texture), builds a FAISS index, and fits a TF-IDF vectorizer. Run once before starting the app.
2. **Online App** — a Streamlit UI that loads the pre-built index and serves search queries in real time.

### Feature Extraction

- **CNN (2048-dim)**: ResNet50 pretrained on ImageNet (optionally fine-tuned on DeepFashion)
- **Color (512-dim)**: HSV histogram via PIL + NumPy
- **Texture (256-dim)**: LBP histogram via scikit-image
- **Fused embedding (2816-dim)**: L2-normalized concatenation of all three

---

## Project Structure

```
├── app/                    # Streamlit UI
├── embeddings/             # Offline pipeline orchestrator
├── features/               # CNN, color, texture, fusion extractors
├── models/                 # CNN encoder, text encoder, triplet model
├── ranking/                # Ranking and scoring utilities
├── scraper/                # Per-platform Playwright scrapers
├── training/               # DeepFashion dataset loader + triplet training script
├── utils/                  # Image and text utilities, SQLite helpers
├── vector_search/          # FAISS index manager + similarity search
├── data/                   # Generated artifacts (DB, index, images)
├── DeepFashion/            # Locally available DeepFashion dataset
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.9+
- `torch` and `torchvision` must already be installed in your environment (they are not listed in `requirements.txt`)
- CUDA-capable GPU recommended (CPU fallback is supported)

### Install Dependencies

```bash
pip install -r requirements.txt
```

> If OpenCV fails to import due missing system libraries (for example `libGL.so.1`),
> this project still runs because color feature extraction now uses PIL + NumPy only.

### Install Playwright Browsers

```bash
playwright install chromium
```

---

## Running the Offline Pipeline

The pipeline scrapes products, downloads images, generates embeddings, and builds the FAISS index. Run this once before launching the app.

```bash
python -m embeddings.generate_embeddings --categories "hoodies" "dresses" "t-shirts" --limit 50
```

**Arguments:**
- `--categories` — space-separated list of clothing categories to scrape
- `--limit` — max products to scrape per platform per category (default: 50)

**Artifacts produced:**
- `data/products.db` — SQLite product metadata
- `data/images/` — downloaded product images
- `data/faiss_index.bin` — FAISS index
- `data/faiss_id_map.pkl` — FAISS position → product ID mapping
- `data/tfidf_vectorizer.pkl` — fitted TF-IDF vectorizer
- `data/text_embeddings.npy` — text embedding matrix

---

## Launching the App

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser. You can:
- Upload a clothing image to search by visual similarity
- Enter a text description to search by keyword
- Use both for a hybrid search (weighted: 65% image + 25% text + 10% price score)
- Switch between "Best Match" and "Lowest Price" sort modes

> The app requires the offline pipeline to have been run first. If the FAISS index is missing, the app will display instructions to run the pipeline.

### Quick Run Checklist

1. Install Python dependencies.
2. Install Playwright browser once (`playwright install chromium`).
3. Build data/index artifacts:
   ```bash
   python -m embeddings.generate_embeddings --categories "hoodies" "dresses" "t-shirts" --limit 50
   ```
4. Start app:
   ```bash
   streamlit run app/streamlit_app.py
   ```
5. Open the local Streamlit URL shown in terminal (usually `http://localhost:8501`).

---

## Fine-Tuning on DeepFashion

To fine-tune the ResNet50 CNN on the locally available DeepFashion dataset using triplet loss:

```bash
python training/train_triplet_network.py --epochs 10 --batch-size 32 --lr 0.0001
```

**Arguments:**
- `--epochs` — number of training epochs (default: 10)
- `--batch-size` — batch size (default: 32)
- `--lr` — learning rate (default: 0.0001)

Fine-tuned weights are saved to `models/resnet50_finetuned.pth`. The CNN feature extractor automatically loads these weights when the file exists, falling back to ImageNet pretrained weights otherwise.

> Requires the DeepFashion dataset to be present at `DeepFashion/` with `Eval/list_eval_partition.txt` and `img/` subdirectories.

---

## Environment Notes

- `torch` and `torchvision` are assumed to be pre-installed and are **not** included in `requirements.txt`. Install them separately if needed: https://pytorch.org/get-started/locally/
- FAISS GPU (`faiss-gpu`) is listed in `requirements.txt`. If you don't have a GPU, replace it with `faiss-cpu`.
- The system automatically falls back to CPU if CUDA is unavailable — no code changes needed.
