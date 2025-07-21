# model_selection.py

import os
import re
import pickle
import logging
from typing import Any, Dict, Tuple, List
from shutil import copy2, move
import json

import pandas as pd
from src.evaluate import evaluate_model

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logs/model_selection.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="a",
)


def load_model(model_path: str) -> Any:
    """Load a pickled model from disk."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded model from {model_path}")
    return model


def archive_current_champion(model_dir: str = "model") -> None:
    """
    Move existing champion artifacts into old_champion_versions/,
    appending a version suffix _v{n}.

    Parameters
    ----------
    model_dir : str
        Base model directory containing champion_*.*
    """
    archive_dir = os.path.join(model_dir, "old_champion_versions")
    os.makedirs(archive_dir, exist_ok=True)

    # Find next version number
    existing = os.listdir(archive_dir)
    versions = [
        int(m.group(1))
        for fn in existing
        for m in [re.match(r".+_v(\d+)\.", fn)]
        if m
    ]
    next_v = max(versions, default=0) + 1

    # Files to archive
    artifacts = ["model.pkl", "model_metrics.json", "model_feature_importance.png", "model_features.json",]
    for name in artifacts:
        src = os.path.join(model_dir, f"champion_{name}")
        if os.path.exists(src):
            base, ext = os.path.splitext(name)
            dst_name = f"champion_{base}_v{next_v}{ext}"
            dst = os.path.join(archive_dir, dst_name)
            move(src, dst)
            logger.info(f"Archived {src} → {dst}")
        else:
            logger.debug(f"No champion artifact to archive: {src}")


def promote_challenger(model_dir: str = "model") -> None:
    """
    Archive current champion, then overwrite champion_* with challenger_*.
    """
    # Archive first if champion exists
    champ_model = os.path.join(model_dir, "champion_model.pkl")
    if os.path.exists(champ_model):
        archive_current_champion(model_dir)

    # Copy challenger_ files over champion_
    artifacts = ["model.pkl", "model_metrics.json", "model_feature_importance.png", "model_features.json",]
    for name in artifacts:
        src = os.path.join(model_dir, f"challenger_{name}")
        dst = os.path.join(model_dir, f"champion_{name}")
        if os.path.exists(src):
            copy2(src, dst)
            logger.info(f"Promoted {src} → {dst}")
        else:
            logger.warning(f"Challenger artifact not found, skipping: {src}")


def select_and_promote(
    val_data: pd.DataFrame,
    model_dir: str,
    full_df: pd.DataFrame,
    demographics: pd.DataFrame
) -> Tuple[bool, Dict[str, float], Dict[str, float]]:
    """
    Compare champion and challenger on validation data and promote if challenger is better.

    Always passes full_df, demographics, and model_features to evaluate_model
    so that KNN pipelines are handled correctly.
    """
    champ_path = os.path.join(model_dir, "champion_model.pkl")
    chall_path = os.path.join(model_dir, "challenger_model.pkl")

    champ_features = json.load(open(os.path.join(model_dir, "champion_model_features.json"), 'r'))
    chall_features = json.load(open(os.path.join(model_dir, "challenger_model_features.json"), 'r'))

    # Auto‑promote if no champion exists
    if not os.path.exists(champ_path):
        logger.info("No champion found; auto‑promoting challenger.")
        promote_challenger(model_dir)
        champ_m = evaluate_model(
            load_model(os.path.join(model_dir, "champion_model.pkl")),
            val_data,
            full_df=full_df,
            demographics=demographics,
            model_features=champ_features
        )["MAPE"]
        return True, {"mape": champ_m}, {"mape": champ_m}

    # Evaluate champion
    champ_model = load_model(champ_path)
    champ_m = evaluate_model(
        champ_model,
        val_data,
        full_df=full_df,
        demographics=demographics,
        model_features=champ_features
    )["MAPE"]

    # Evaluate challenger
    chall_model = load_model(chall_path)
    chall_m = evaluate_model(
        chall_model,
        val_data,
        full_df=full_df,
        demographics=demographics,
        model_features=chall_features
    )["MAPE"]

    logger.info(f"Champion MAPE: {champ_m:.5f}; Challenger MAPE: {chall_m:.5f}")
    champion_metrics = {"mape": champ_m}
    challenger_metrics = {"mape": chall_m}

    # Promote if challenger is strictly better
    if chall_m < champ_m:
        logger.info("Challenger outperforms champion; promoting.")
        promote_challenger(model_dir)
        promoted = True
    else:
        logger.info("Champion retains status.")
        promoted = False

    return promoted, champion_metrics, challenger_metrics


def launch():
    """
    Orchestrate selection:
      1. Load val set and demographics
      2. Determine feature columns
      3. Compare & promote
    """
    val_data_path: str = "data/golden/val_data.csv"
    demographics_path: str = "data/zipcode_demographics.csv"
    target_col: str = "price"
    model_dir: str = "model"

    logger.info(f"Loading validation data from {val_data_path}")
    df_val = pd.read_csv(val_data_path)
    df = pd.read_csv("data/kc_house_data.csv")

    logger.info(f"Loading demographics data from {demographics_path}")
    demog = pd.read_csv(demographics_path)

    logger.info("Comparing champion vs challenger...")
    promoted, champ_metrics, chall_metrics = select_and_promote(
        df_val,
        model_dir=model_dir,
        full_df=df,
        demographics=demog
    )

    if promoted:
        logger.info("Challenger promoted to champion.")
    else:
        logger.info("Champion remains unchanged.")
