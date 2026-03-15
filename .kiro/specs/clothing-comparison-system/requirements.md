# Requirements Document

## Introduction

The Cross-Platform Clothing Comparison System is an end-to-end MVP that enables users to find visually similar clothing products across multiple e-commerce platforms (Myntra, Ajio, H&M, Zara, Amazon Fashion, Flipkart Fashion, and fallbacks). Users can search by uploading a clothing image, entering a text description, or combining both. The system uses a ResNet50-based CNN for image feature extraction, TF-IDF for text embeddings, and FAISS for fast vector similarity search. A multi-feature visual embedding pipeline fuses CNN, color (HSV histogram), and texture (LBP) features for improved retrieval accuracy. An offline pipeline handles scraping, embedding generation, and FAISS index building. A Streamlit UI provides the search interface. CNN fine-tuning on the locally available DeepFashion dataset is supported via triplet loss training.

---

## Glossary

- **System**: The Cross-Platform Clothing Comparison System
- **CNN**: Convolutional Neural Network — ResNet50 pretrained on ImageNet, used as a feature extractor
- **Embedding**: A fixed-size numerical vector representing an image or text
- **FAISS**: Facebook AI Similarity Search — GPU-accelerated vector similarity search library
- **TF-IDF**: Term Frequency-Inverse Document Frequency — scikit-learn text vectorizer
- **Fused Embedding**: A concatenated vector of CNN (2048) + Color (512) + Texture (256) = 2816 dimensions
- **LBP**: Local Binary Pattern — texture descriptor computed on grayscale images
- **HSV Histogram**: Color histogram computed in Hue-Saturation-Value color space
- **Triplet Loss**: A metric learning loss using anchor, positive, and negative image samples
- **DeepFashion**: A locally available fashion dataset with 52,712 images across MEN/WOMEN categories
- **Product Record**: A dictionary with fields: product_title, brand, price, product_url, image_url, category, platform
- **FAISS Index**: A binary file storing all fused embeddings for fast cosine similarity search
- **Scraper**: A Playwright-based async module that extracts product records from a single e-commerce platform
- **Offline Pipeline**: A script that runs scraping, image download, embedding generation, and FAISS index building — executed once before the app starts
- **Hybrid Score**: A weighted combination: 0.65 × image_similarity + 0.25 × text_similarity + 0.10 × price_score

---

## Requirements

### Requirement 1 — CNN Image Feature Extraction

**User Story:** As a developer, I want a ResNet50-based CNN feature extractor, so that clothing images are represented as normalized 2048-dimensional embeddings.

#### Acceptance Criteria

1. THE System SHALL load a ResNet50 model pretrained on ImageNet using torchvision and remove its final fully connected layer to produce a 2048-dimensional global average pooling output.
2. THE System SHALL preprocess input images by resizing to 224×224 pixels, converting to a tensor, and normalizing using ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
3. THE System SHALL normalize each CNN embedding vector to unit length (L2 norm) before storing or returning it.
4. WHEN CUDA is available, THE System SHALL move the CNN model and all input tensors to the CUDA device for GPU-accelerated inference.
5. THE System SHALL run the CNN model in evaluation mode (model.eval()) during feature extraction to disable dropout and batch normalization updates.

---

### Requirement 2 — Color Feature Extraction

**User Story:** As a developer, I want HSV color histogram features extracted from clothing images, so that color similarity is captured in the fused embedding.

#### Acceptance Criteria

1. THE System SHALL convert each input image to HSV color space using OpenCV before computing the color histogram.
2. THE System SHALL compute a 3D color histogram with 8 bins per channel (H, S, V), producing a flattened vector of 512 dimensions.
3. THE System SHALL normalize the color histogram vector to unit length before returning it.
4. IF an image cannot be loaded or converted, THEN THE System SHALL return a zero vector of dimension 512 and log the error.

---

### Requirement 3 — Texture Feature Extraction

**User Story:** As a developer, I want LBP texture features extracted from clothing images, so that clothing patterns (stripes, checks, floral) are captured in the fused embedding.

#### Acceptance Criteria

1. THE System SHALL convert each input image to grayscale before computing LBP features.
2. THE System SHALL compute a Local Binary Pattern histogram using skimage with radius=3 and n_points=24, producing a normalized histogram of 256 dimensions.
3. THE System SHALL normalize the texture histogram vector to unit length before returning it.
4. IF an image cannot be processed, THEN THE System SHALL return a zero vector of dimension 256 and log the error.

---

### Requirement 4 — Feature Fusion

**User Story:** As a developer, I want CNN, color, and texture features fused into a single embedding, so that the FAISS index captures shape, color, and texture simultaneously.

#### Acceptance Criteria

1. THE System SHALL concatenate CNN features (2048), color features (512), and texture features (256) into a single fused embedding vector of 2816 dimensions.
2. THE System SHALL normalize the fused embedding vector to unit length before storing it in the FAISS index.
3. THE System SHALL expose a single function that accepts an image path or PIL image and returns the normalized 2816-dimensional fused embedding.

---

### Requirement 5 — Text Feature Extraction

**User Story:** As a developer, I want TF-IDF text embeddings for product titles and descriptions, so that text-based search is supported.

#### Acceptance Criteria

1. THE System SHALL train a TF-IDF vectorizer on all scraped product titles stored in the SQLite database.
2. THE System SHALL persist the fitted TF-IDF vectorizer to disk so it does not need to be retrained on every application start.
3. WHEN a user submits a text query, THE System SHALL transform the query using the fitted TF-IDF vectorizer and return a sparse vector.
4. THE System SHALL compute cosine similarity between the query TF-IDF vector and all stored product text embeddings to rank results.

---

### Requirement 6 — FAISS Vector Index

**User Story:** As a developer, I want a FAISS index storing fused image embeddings, so that top-K similar products can be retrieved in sub-second time.

#### Acceptance Criteria

1. THE System SHALL build a FAISS IndexFlatIP (inner product) index over normalized fused embeddings, which is equivalent to cosine similarity search.
2. WHEN FAISS-GPU is available, THE System SHALL use a GPU-backed FAISS index; otherwise THE System SHALL fall back to a CPU index.
3. THE System SHALL persist the FAISS index to disk and reload it on application start without recomputing embeddings.
4. THE System SHALL store a mapping from FAISS index position to product_id so retrieved results can be looked up in SQLite.
5. WHEN a query embedding is provided, THE System SHALL return the top-K product IDs and their similarity scores.

---

### Requirement 7 — SQLite Metadata Storage

**User Story:** As a developer, I want all product metadata stored in SQLite, so that product details can be retrieved after a FAISS search.

#### Acceptance Criteria

1. THE System SHALL create a SQLite database at `data/products.db` with a `products` table containing columns: id, product_title, brand, price, platform, category, product_url, image_url, image_path.
2. THE System SHALL insert each scraped product record into the database, skipping duplicates based on product_url.
3. WHEN a product ID is provided, THE System SHALL return the full product metadata record from SQLite.

---

### Requirement 8 — Web Scraping Module

**User Story:** As a developer, I want Playwright-based scrapers for each e-commerce platform, so that product data is collected automatically.

#### Acceptance Criteria

1. THE System SHALL implement a separate scraper module for each platform: Myntra, Ajio, H&M, Zara, Amazon Fashion, Flipkart Fashion.
2. EACH scraper SHALL expose a function `scrape_products(category: str, limit: int) -> list[dict]` returning product records with fields: product_title, brand, price, product_url, image_url, category, platform.
3. THE System SHALL use Playwright's async API with Chromium in headless mode and a Chrome desktop user-agent string.
4. EACH scraper SHALL wait for `networkidle` load state before parsing page content.
5. EACH scraper SHALL implement auto-scrolling until page height stops increasing to handle lazy-loaded content.
6. EACH scraper SHALL add a randomized delay of 1–3 seconds between page requests to reduce bot detection.
7. IF a platform returns an error or blocks the scraper, THEN THE System SHALL log the error, skip that platform, and continue scraping remaining platforms.
8. THE System SHALL attempt fallback platforms (ASOS, Uniqlo, Urbanic, Koovs) when primary platforms fail.
9. EACH scraper SHALL download product images to `data/images/` and store the local path in the product record.

---

### Requirement 9 — Offline Embedding Pipeline

**User Story:** As a developer, I want an offline pipeline that scrapes, embeds, and indexes all products, so that the app starts instantly without recomputing embeddings.

#### Acceptance Criteria

1. THE System SHALL execute the pipeline in order: scrape products → download images → generate fused embeddings → generate text embeddings → store in SQLite → build FAISS index.
2. THE System SHALL skip embedding generation for products that already have embeddings stored, to avoid redundant computation.
3. THE System SHALL save the FAISS index to `data/faiss_index.bin` and the TF-IDF vectorizer to `data/tfidf_vectorizer.pkl`.
4. WHEN the pipeline completes, THE System SHALL log the total number of products indexed.

---

### Requirement 10 — Search Logic

**User Story:** As a user, I want to search for clothing by image, text, or both, so that I can find similar products across platforms.

#### Acceptance Criteria

1. WHEN a user provides only an image, THE System SHALL extract the fused embedding, query the FAISS index, and return the top-K matching products ranked by cosine similarity.
2. WHEN a user provides only a text description, THE System SHALL compute TF-IDF similarity against all stored product text embeddings and return the top-K results ranked by cosine similarity.
3. WHEN a user provides both an image and a text description, THE System SHALL compute a hybrid score: 0.65 × image_similarity + 0.25 × text_similarity + 0.10 × price_score, where price_score = 1 / normalized_price.
4. THE System SHALL return results sorted by the computed score in descending order.
5. THE System SHALL support configurable K (default K=20) for the number of results returned.

---

### Requirement 11 — Ranking Engine

**User Story:** As a user, I want results ranked by best match or lowest price, so that I can find the most relevant or affordable products.

#### Acceptance Criteria

1. THE System SHALL support sorting results by similarity score (descending) as the default ranking.
2. THE System SHALL support sorting results by price (ascending) as an alternative ranking mode.
3. THE System SHALL normalize prices to the [0, 1] range across the result set before computing price_score in hybrid search.

---

### Requirement 12 — Streamlit User Interface

**User Story:** As a user, I want a Streamlit web interface, so that I can upload images, enter text, and view results without using a command line.

#### Acceptance Criteria

1. THE System SHALL render an input section at the top of the page with: a file uploader for clothing images, a text input for clothing description, and a Search button.
2. WHEN search results are available, THE System SHALL display each result as a product card showing: product image, product title, platform, price, similarity score, and a clickable product link.
3. THE System SHALL provide a sort control allowing the user to switch between "Best Match" and "Lowest Price" ordering.
4. WHEN no products are indexed, THE System SHALL display a clear message instructing the user to run the offline embedding pipeline first.
5. THE System SHALL display a loading spinner while search is in progress.

---

### Requirement 13 — CNN Fine-Tuning on DeepFashion

**User Story:** As a developer, I want to fine-tune the ResNet50 CNN on the DeepFashion dataset using triplet loss, so that the model produces fashion-specific embeddings.

#### Acceptance Criteria

1. THE System SHALL load training samples from `DeepFashion/Eval/list_eval_partition.txt`, using images with `train` status, grouped by item_id to form triplet sets.
2. THE System SHALL construct triplets where the anchor and positive share the same item_id and the negative is sampled from a different item_id.
3. THE System SHALL train using PyTorch's TripletMarginLoss with a configurable margin (default 0.3).
4. WHEN CUDA is available, THE System SHALL move the model and all triplet tensors to the CUDA device during training.
5. THE System SHALL save the fine-tuned model weights to `models/resnet50_finetuned.pth` after training completes.
6. THE System SHALL load fine-tuned weights when available, falling back to ImageNet pretrained weights if the fine-tuned file does not exist.
