"""Triplet network training script for DeepFashion fine-tuning."""

import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.triplet_model import TripletNet
from training.dataset_loader import DeepFashionTripletDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_ROOT = "DeepFashion"
PARTITION_FILE = os.path.join("DeepFashion", "Eval", "list_eval_partition.txt")
SAVE_PATH = os.path.join("models", "resnet50_finetuned.pth")


def train(epochs: int, batch_size: int, lr: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dataset = DeepFashionTripletDataset(
        dataset_root=DATASET_ROOT,
        partition_file=PARTITION_FILE,
        split="train",
    )
    logger.info("Dataset size: %d triplet items", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    model = TripletNet().to(device)
    criterion = nn.TripletMarginLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (anchor, positive, negative) in enumerate(loader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
            loss = criterion(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info("Epoch [%d/%d]  loss: %.4f", epoch, epochs, avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.encoder.state_dict(), SAVE_PATH)
            logger.info("Saved best weights to %s (loss=%.4f)", SAVE_PATH, best_loss)

    logger.info("Training complete. Best loss: %.4f", best_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 on DeepFashion with triplet loss")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
