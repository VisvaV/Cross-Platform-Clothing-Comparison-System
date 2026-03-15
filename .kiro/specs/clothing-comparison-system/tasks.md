# Implementation Plan

- [x] 1. Project scaffold and shared utilities




  - Create the full directory structure: `features/`, `scraper/`, `models/`, `training/`, `embeddings/`, `vector_search/`, `ranking/`, `app/`, `utils/`, `data/images/`
  - Add `__init__.py` to each package directory
  - Write `utils/image_utils.py`: `download_image(url, save_path)`, `load_image(path)`, `preprocess_for_cnn(image)` with ImageNet normalization
  - Write `utils/text_utils.py`: `clean_text(text)`, `extract_price(raw_str)` with ₹/$/£ parsing
  - Write `requirements.txt` with all dependencies (no torch/torchvision — already installed)
  - _Requirements: 1.2, 7.1, 8.9_

- [x] 2. CNN feature extractor





  - [x] 2.1 Implement `features/cnn_features.py` with `CNNFeatureExtractor` class


    - Load ResNet50 with `ResNet50_Weights.DEFAULT`, replace `model.fc` with `nn.Identity()`
    - Move model to CUDA device when available, call `model.eval()`
    - Load fine-tuned weights from `models/resnet50_finetuned.pth` if file exists, else use ImageNet weights
    - `extract(image: PIL.Image) -> np.ndarray` — runs inference, returns L2-normalized 2048-dim vector
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 13.6_
  - [x] 2.2 Implement `models/cnn_encoder.py` as a re-export of `CNNFeatureExtractor`


    - _Requirements: 1.1_

- [x] 3. Color and texture feature extractors






  - [x] 3.1 Implement `features/color_features.py` with `extract_color_features(image: PIL.Image) -> np.ndarray`

    - Convert PIL image to BGR numpy array, then to HSV via `cv2.cvtColor`
    - Compute 3D histogram with `cv2.calcHist`, bins=[8,8,8], flatten to 512-dim, L2-normalize
    - Return zero vector of shape (512,) on any exception and log the error
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.2 Implement `features/texture_features.py` with `extract_texture_features(image: PIL.Image) -> np.ndarray`

    - Convert PIL image to grayscale numpy array
    - Compute LBP via `skimage.feature.local_binary_pattern(gray, P=24, R=3, method='uniform')`
    - Build 256-bin histogram, L2-normalize, return shape (256,)
    - Return zero vector of shape (256,) on any exception and log the error
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 4. Feature fusion module





  - Implement `features/feature_fusion.py` with `fuse_features(cnn, color, texture) -> np.ndarray`
    - Concatenate three vectors: `np.concatenate([cnn, color, texture])` → shape (2816,)
    - L2-normalize the concatenated vector before returning
  - Implement `extract_fused_embedding(image_path: str, extractor: CNNFeatureExtractor) -> np.ndarray`
    - Load image, call all three extractors, call `fuse_features`, return normalized 2816-dim vector
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 5. Text encoder





  - Implement `models/text_encoder.py` with `TextEncoder` class
    - `fit(texts: list[str])`: trains `TfidfVectorizer(max_features=10000)` on cleaned product titles
    - `transform(text: str) -> np.ndarray`: transforms a single string to a dense vector
    - `save(path)` / `load(path)` using `joblib.dump` / `joblib.load`
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 6. SQLite database layer





  - Implement database initialization and CRUD in `utils/db_utils.py`
    - `init_db(db_path)`: creates `products` table with columns: id, product_title, brand, price, platform, category, product_url (UNIQUE), image_url, image_path
    - `insert_product(conn, record: dict)`: inserts with `INSERT OR IGNORE` to skip duplicates on product_url
    - `get_product_by_id(conn, product_id: int) -> dict`
    - `get_all_products(conn) -> list[dict]`
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 7. Platform scrapers





  - [x] 7.1 Implement `scraper/hm_scraper.py` — `async scrape_products(category, limit) -> list[dict]`


    - Chromium headless, Chrome user-agent, `networkidle` wait, auto-scroll loop, 1–3s random delay
    - CSS selector fallbacks for title, price, brand, image, product URL
    - Full try/except — return `[]` on failure
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  - [x] 7.2 Implement `scraper/zara_scraper.py` with same contract


    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  - [x] 7.3 Implement `scraper/myntra_scraper.py` with same contract


    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  - [x] 7.4 Implement `scraper/ajio_scraper.py` with same contract


    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_






  - [x] 7.5 Implement `scraper/amazon_scraper.py` with same contract
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  - [x] 7.6 Implement `scraper/flipkart_scraper.py` with same contract
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  - [x] 7.7 Implement fallback scrapers: `scraper/asos_scraper.py` and `scraper/uniqlo_scraper.py`
    - _Requirements: 8.8_

- [x] 8. FAISS index manager





  - Implement `vector_search/faiss_index.py` with `FaissIndex` class
    - `build(embeddings: np.ndarray, id_map: list[int])`: creates `faiss.IndexFlatIP(2816)`, adds all normalized embeddings
    - Wrap with `faiss.index_cpu_to_gpu` when GPU resource is available, fall back to CPU silently
    - `save(index_path, map_path)`: writes index binary and pickles id_map
    - `load(index_path, map_path)`: loads both artifacts
    - `search(query: np.ndarray, k: int) -> list[tuple[int, float]]`: returns (product_id, score) pairs
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 9. Offline embedding generation pipeline





  - Implement `embeddings/generate_embeddings.py` with `run_pipeline(categories, limit_per_platform)`
    - Step 1: For each category × platform, call `scrape_products`, insert records into SQLite
    - Step 2: Download images to `data/images/` using `download_image`, update `image_path` in DB
    - Step 3: For each product without a stored embedding, call `extract_fused_embedding`, accumulate embeddings and id_map
    - Step 4: Fit `TextEncoder` on all product titles, save to `data/tfidf_vectorizer.pkl`; compute and save text embedding matrix to `data/text_embeddings.npy`
    - Step 5: Build `FaissIndex` from accumulated embeddings, save to `data/faiss_index.bin` and `data/faiss_id_map.pkl`
    - Step 6: Log total products indexed
    - Add `if __name__ == "__main__"` CLI entry point with `--categories` and `--limit` args
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 10. Similarity search engine





  - Implement `vector_search/similarity_search.py` with `SimilaritySearch` class
    - Constructor loads `FaissIndex`, `TextEncoder`, text embedding matrix, and SQLite connection
    - `search_by_image(image_path, k=20)`: extract fused embedding → FAISS search → SQLite lookup → return result dicts with `similarity_score`
    - `search_by_text(query, k=20)`: TF-IDF transform → cosine similarity against `text_embeddings.npy` → top-K → SQLite lookup
    - `search_hybrid(image_path, query, k=20)`: compute both scores, normalize prices, apply `compute_hybrid_score`, sort descending
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 11. Ranking engine





  - Implement `ranking/ranking_engine.py`
    - `rank_by_score(results: list[dict]) -> list[dict]`: sort by `similarity_score` descending
    - `rank_by_price(results: list[dict]) -> list[dict]`: sort by `price` ascending
    - `compute_hybrid_score(image_sim, text_sim, normalized_price) -> float`: `0.65*image_sim + 0.25*text_sim + 0.10*(1/(normalized_price+1e-6))`
    - `normalize_prices(results: list[dict]) -> list[dict]`: min-max normalize `price` field across result set
  - _Requirements: 11.1, 11.2, 11.3_

- [x] 12. DeepFashion dataset loader





  - Implement `training/dataset_loader.py` with `DeepFashionTripletDataset(Dataset)`
    - Parse `DeepFashion/Eval/list_eval_partition.txt`: build dict of `item_id → [image_paths]` for the requested split
    - Filter to items with at least 2 images (needed for anchor/positive pairs)
    - `__getitem__`: pick anchor and positive from same item_id (different indices), sample negative from a different item_id
    - Apply transforms: Resize(224), ToTensor, Normalize(ImageNet mean/std)
    - Prepend `datasets/DeepFashion/` to relative image paths from the annotation file
  - _Requirements: 13.1, 13.2_

- [x] 13. Triplet model and training script





  - [x] 13.1 Implement `models/triplet_model.py` with `TripletNet(nn.Module)`


    - Shared-weight forward pass: `forward(anchor, positive, negative)` returns three embedding tensors
    - Base encoder is a `CNNFeatureExtractor`-style ResNet50 with `fc` replaced by `nn.Identity()`
    - _Requirements: 13.3_
  - [x] 13.2 Implement `training/train_triplet_network.py`


    - Instantiate `DeepFashionTripletDataset`, `DataLoader`, `TripletNet`, `nn.TripletMarginLoss(margin=0.3)`
    - Move model to CUDA device when available; move each batch of tensors to device
    - Train loop: forward → loss → backward → optimizer step; log loss per epoch
    - Save best model weights to `models/resnet50_finetuned.pth`
    - Add `if __name__ == "__main__"` CLI with `--epochs`, `--batch-size`, `--lr` args
    - _Requirements: 13.3, 13.4, 13.5_

- [x] 14. Streamlit application





  - Implement `app/streamlit_app.py`
    - Page config: wide layout, title "Clothing Comparison System"
    - Input section: `st.file_uploader` for image, `st.text_input` for description, Search button
    - Sort control: `st.radio` with "Best Match" / "Lowest Price" options
    - On search: instantiate `SimilaritySearch`, dispatch to correct search method based on inputs
    - Display results in 3-column grid using `st.columns`; each card: `st.image`, title, platform, price, score, `st.link_button`
    - Show `st.spinner("Searching...")` during search
    - Show `st.warning` with pipeline instructions when FAISS index file is missing
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 15. Write README.md





  - Write `README.md` with: project overview, setup steps, how to run the offline pipeline (`python -m embeddings.generate_embeddings`), how to launch the Streamlit app (`streamlit run app/streamlit_app.py`), how to run fine-tuning (`python training/train_triplet_network.py`), environment constraints note (torch/torchvision pre-installed)
  - Note: `requirements.txt` is already complete
  - _Requirements: all_
