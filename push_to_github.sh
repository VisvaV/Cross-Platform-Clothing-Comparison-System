#!/bin/bash

REMOTE_URL="https://github.com/VisvaV/Cross-Platform-Clothing-Comparison-System.git"

echo "Setting up remote..."
git remote remove origin 2>/dev/null
git remote add origin "$REMOTE_URL"

echo "Setting branch to main..."
git branch -M main

# ── Helper: stage and commit a single file if it has any changes ──────────────
commit_file() {
    local file="$1"
    local message="$2"

    # Check if file exists at all
    if [ ! -f "$file" ]; then
        echo "  [skip] $file — file not found"
        return
    fi

    # Stage the file
    git add "$file"

    # Check if anything is staged using git status --porcelain
    local staged
    staged=$(git status --porcelain "$file" 2>/dev/null | grep -E "^[AMDRC]" | head -1)

    if [ -z "$staged" ]; then
        git restore --staged "$file" 2>/dev/null
        echo "  [skip] $file — no changes"
        return
    fi

    git commit -m "$message"
    echo "  [ok]   committed: $file"
}

echo ""
echo "Committing files individually..."

# ── utils ─────────────────────────────────────────────────────────────────────
commit_file "utils/text_utils.py"        "fix(utils): return None from extract_price on failure instead of 0.0 to fix price ranking bias"
commit_file "utils/db_utils.py"          "chore(utils): db utility helpers"
commit_file "utils/image_utils.py"       "chore(utils): image download helpers"

# ── scraper base ──────────────────────────────────────────────────────────────
commit_file "scraper/_base.py"           "refactor(scraper): migrate to camoufox; fix auto_scroll infinite loop with max_scrolls cap; bump Chrome UA to 124; expand dismiss_popups selectors"

# ── scrapers ──────────────────────────────────────────────────────────────────
commit_file "scraper/ajio_scraper.py"    "feat(scraper/ajio): rewrite with camoufox for Cloudflare bypass; URL-encode category; lazy-load img fallback via data-src"
commit_file "scraper/amazon_scraper.py"  "feat(scraper/amazon): migrate to camoufox; URL-encode category with quote_plus"
commit_file "scraper/asos_scraper.py"    "feat(scraper/asos): migrate to camoufox; URL-encode category with quote_plus"
commit_file "scraper/flipkart_scraper.py" "feat(scraper/flipkart): migrate to camoufox; URL-encode category; stronger card wait; expand popup dismissal"
commit_file "scraper/hm_scraper.py"      "feat(scraper/hm): migrate to camoufox; URL-encode category with quote_plus"
commit_file "scraper/myntra_scraper.py"  "feat(scraper/myntra): migrate to camoufox; add slug map for correct path URLs; search fallback for multi-word queries"
commit_file "scraper/uniqlo_scraper.py"  "feat(scraper/uniqlo): migrate to camoufox; URL-encode category with quote_plus"
commit_file "scraper/zara_scraper.py"    "feat(scraper/zara): migrate to camoufox; URL-encode category with quote_plus"
commit_file "scraper/__init__.py"        "chore(scraper): package init"

# ── ranking ───────────────────────────────────────────────────────────────────
commit_file "ranking/ranking_engine.py"  "fix(ranking): push None-price products to end in rank_by_price; guard normalize_prices against None with neutral 0.5 fallback"
commit_file "ranking/__init__.py"        "chore(ranking): package init"

# ── embeddings ────────────────────────────────────────────────────────────────
commit_file "embeddings/generate_embeddings.py" "feat(embeddings): save text_id_map.pkl alongside text_embeddings.npy so text search maps to correct product IDs"
commit_file "embeddings/__init__.py"     "chore(embeddings): package init"

# ── vector_search ─────────────────────────────────────────────────────────────
commit_file "vector_search/faiss_index.py"       "chore(vector_search): faiss index wrapper"
commit_file "vector_search/similarity_search.py" "fix(vector_search): use text_id_map instead of faiss.id_map for text search; fix None-price crash in hybrid scoring; guard out-of-range text indices; image_url fallback"
commit_file "vector_search/__init__.py"  "chore(vector_search): package init"

# ── features ──────────────────────────────────────────────────────────────────
commit_file "features/cnn_features.py"    "chore(features): CNN feature extractor"
commit_file "features/color_features.py"  "chore(features): color feature extractor"
commit_file "features/texture_features.py" "chore(features): texture feature extractor"
commit_file "features/feature_fusion.py"  "chore(features): fused embedding builder"
commit_file "features/__init__.py"        "chore(features): package init"

# ── models ────────────────────────────────────────────────────────────────────
commit_file "models/cnn_encoder.py"    "chore(models): CNN encoder"
commit_file "models/text_encoder.py"   "chore(models): TF-IDF text encoder"
commit_file "models/triplet_model.py"  "chore(models): triplet network model"
commit_file "models/__init__.py"       "chore(models): package init"

# ── training ──────────────────────────────────────────────────────────────────
commit_file "training/dataset_loader.py"         "chore(training): dataset loader"
commit_file "training/train_triplet_network.py"  "chore(training): triplet training script"
commit_file "training/__init__.py"               "chore(training): package init"

# ── app ───────────────────────────────────────────────────────────────────────
commit_file "app/streamlit_app.py"  "fix(app): pass text_id_map_path to SimilaritySearch; seek(0) before reading upload; show upload preview; fallback to image_url when local image missing"
commit_file "app/__init__.py"       "chore(app): package init"

# ── tests ─────────────────────────────────────────────────────────────────────
commit_file "tests/seed_and_test.py" "test: save text_id_map in seed pipeline; add green-dress text search test; fix rank_by_price assertion for None prices; stdout utf-8 for Windows"

# ── root files ────────────────────────────────────────────────────────────────
commit_file "requirements.txt"  "chore: drop version pins for wider compatibility"
commit_file "migrate.py"        "chore: db migration script"
commit_file ".gitignore"        "chore: update .gitignore"
commit_file "README.md"         "docs: update README"
commit_file "push_to_github.sh" "chore: per-file commit script with unique messages"

echo ""
echo "Pushing all commits to GitHub..."
git push -u origin main

echo ""
echo "Done! All commits pushed to $REMOTE_URL"
