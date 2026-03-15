#!/bin/bash

REMOTE_URL="https://github.com/VisvaV/Cross-Platform-Clothing-Comparison-System.git"

echo "Setting up remote..."
git remote remove origin 2>/dev/null
git remote add origin "$REMOTE_URL"

echo "Staging all files..."
git add .

echo "Committing..."
git commit -m "Initial commit: Cross-Platform Clothing Comparison System"

echo "Setting branch to main..."
git branch -M main

echo "Pushing to GitHub..."
git push -u origin main

echo "Done! Files pushed to $REMOTE_URL"
