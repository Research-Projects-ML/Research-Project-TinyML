from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa

# Import your loader (adjust import path to match your project)
# from src.data.loaders.speech_commands_loader import build_speech_commands_index, SpeechCommandAudio
from src.data.loaders.speech_commands_loader import build_speech_commands_index  # noqa


@dataclass(frozen=True)
class MFCCConfig:
    sample_rate: int = 16000
    clip_seconds: float = 1.0
    n_mfcc: int = 10
    n_fft: int = 400           # 25ms at 16kHz
    hop_length: int = 160      # 10ms at 16kHz
    n_mels: int = 40
    fmin: int = 20
    fmax: int = 7600           # below Nyquist(8000)
    center: bool = False       # good for streaming-style features


def _ensure_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or trim waveform to exactly target_len samples."""
    if x.shape[0] == target_len:
        return x
    if x.shape[0] > target_len:
        return x[:target_len]
    pad = target_len - x.shape[0]
    return np.pad(x, (0, pad), mode="constant")


def _load_wav_16k_mono(path: Path, sr: int) -> np.ndarray:
    """Load wav as mono float32 at sr."""
    # librosa will resample if needed
    x, _ = librosa.load(str(path), sr=sr, mono=True)
    return x.astype(np.float32)


def _wav_to_mfcc(x: np.ndarray, cfg: MFCCConfig) -> np.ndarray:
    """Convert waveform -> MFCC (T, F)."""
    mfcc = librosa.feature.mfcc(
        y=x,
        sr=cfg.sample_rate,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        center=cfg.center,
    )
    # librosa returns (n_mfcc, T) -> transpose to (T, n_mfcc)
    return mfcc.T.astype(np.float32)


def _speaker_disjoint_split(
    items: List[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    items: list of dict with keys: path, label, speaker_id (speaker_id empty for noise)
    Split ONLY keyword items by speaker_id so speakers don't leak across splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # keyword items: speaker_id != "" and label not _noise_file
    keyword_items = [it for it in items if it["speaker_id"] and it["label"] != "_noise_file"]

    speakers = sorted(list({it["speaker_id"] for it in keyword_items}))
    rng = random.Random(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # remaining goes to test
    train_s = set(speakers[:n_train])
    val_s = set(speakers[n_train:n_train + n_val])
    test_s = set(speakers[n_train + n_val:])

    train = [it for it in keyword_items if it["speaker_id"] in train_s]
    val = [it for it in keyword_items if it["speaker_id"] in val_s]
    test = [it for it in keyword_items if it["speaker_id"] in test_s]

    return train, val, test


def _sample_silence_waveform(
    noise_paths: List[Path],
    cfg: MFCCConfig,
    rng: random.Random,
) -> np.ndarray:
    """
    Create a 1-second "silence" example by slicing random chunk from a random noise file.
    """
    target_len = int(cfg.sample_rate * cfg.clip_seconds)
    noise_path = rng.choice(noise_paths)
    noise, _ = librosa.load(str(noise_path), sr=cfg.sample_rate, mono=True)
    noise = noise.astype(np.float32)

    if noise.shape[0] <= target_len:
        x = _ensure_length(noise, target_len)
    else:
        start = rng.randint(0, noise.shape[0] - target_len)
        x = noise[start:start + target_len]

    # optional: random gain (helps robustness)
    gain = rng.uniform(0.3, 1.0)
    x = np.clip(x * gain, -1.0, 1.0)
    return x


def prepare_speech_commands_mfcc(
    raw_root: str | Path,                  # e.g. data/google-speech-commands
    out_dir: str | Path,                   # e.g. data/features/speech_commands_mfcc
    commands: Optional[List[str]] = None,  # 10 keywords
    mfcc_cfg: MFCCConfig = MFCCConfig(),
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    silence_per_split: Optional[int] = None,  # if None -> match avg keyword count per class in that split
) -> None:
    """
    Produces:
      out_dir/
        train.npz, val.npz, test.npz
        label_map.json
        feature_config.json
        meta_train.jsonl, meta_val.jsonl, meta_test.jsonl
    """
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build index using loader
    idx = build_speech_commands_index(str(raw_root), commands=commands, include_background_noise=True)

    # Convert loader items into simple dicts (keeps processor independent)
    # Adjust depending on your loader object type:
    items = []
    for it in idx.items:
        # if your loader uses dataclass: it.path, it.label, it.speaker_id
        items.append({"path": Path(it.path), "label": it.label, "speaker_id": it.speaker_id})

    # 2) Identify noise files (sources for silence)
    noise_paths = [it["path"] for it in items if it["label"] == "_noise_file"]
    if not noise_paths:
        raise RuntimeError("No _background_noise_ wavs found. Expected folder: _background_noise_")

    # 3) Split keyword items by speaker
    train_items, val_items, test_items = _speaker_disjoint_split(
        items=items,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # 4) Build label map (10 commands + silence)
    # Start from loader mapping but remove technical _noise_file and add silence
    label_to_id: Dict[str, int] = {k: v for k, v in idx.label_to_id.items() if k != "_noise_file"}
    if "silence" not in label_to_id:
        label_to_id["silence"] = len(label_to_id)
    id_to_label = {v: k for k, v in label_to_id.items()}

    # 5) Decide silence count per split
    # Default: match average count per keyword class in the split (roughly balanced)
    def avg_per_class(split_items: List[dict]) -> int:
        counts = {}
        for it in split_items:
            counts[it["label"]] = counts.get(it["label"], 0) + 1
        # average over command classes
        if not counts:
            return 0
        return int(sum(counts.values()) / max(1, len(counts)))

    rng = random.Random(seed)

    def build_split(split_name: str, split_items: List[dict]) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        # compute silence count
        n_sil = silence_per_split if silence_per_split is not None else avg_per_class(split_items)
        target_len = int(mfcc_cfg.sample_rate * mfcc_cfg.clip_seconds)

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        meta: List[dict] = []

        # keyword items
        for it in split_items:
            x = _load_wav_16k_mono(it["path"], sr=mfcc_cfg.sample_rate)
            x = _ensure_length(x, target_len)
            feat = _wav_to_mfcc(x, mfcc_cfg)

            X_list.append(feat)
            y_list.append(label_to_id[it["label"]])
            meta.append({"path": str(it["path"]), "label": it["label"], "speaker_id": it["speaker_id"]})

        # silence items
        for i in range(n_sil):
            x = _sample_silence_waveform(noise_paths, mfcc_cfg, rng)
            x = _ensure_length(x, target_len)
            feat = _wav_to_mfcc(x, mfcc_cfg)

            X_list.append(feat)
            y_list.append(label_to_id["silence"])
            meta.append({"path": "<generated_from_noise>", "label": "silence", "speaker_id": ""})

        X = np.stack(X_list, axis=0).astype(np.float32)  # (N, T, F)
        y = np.array(y_list, dtype=np.int64)

        # Shuffle split consistently
        perm = np.arange(X.shape[0])
        rng.shuffle(perm.tolist())
        X = X[perm]
        y = y[perm]
        meta = [meta[i] for i in perm]

        print(f"[{split_name}] X: {X.shape}, y: {y.shape}, silence: {n_sil}")
        return X, y, meta

    # 6) Build and save splits
    X_train, y_train, meta_train = build_split("train", train_items)
    X_val, y_val, meta_val = build_split("val", val_items)
    X_test, y_test, meta_test = build_split("test", test_items)

    np.savez_compressed(out_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(out_dir / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(out_dir / "test.npz", X=X_test, y=y_test)

    # 7) Save metadata / configs
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)

    with open(out_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(mfcc_cfg.__dict__, f, indent=2)

    def write_jsonl(path: Path, rows: List[dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    write_jsonl(out_dir / "meta_train.jsonl", meta_train)
    write_jsonl(out_dir / "meta_val.jsonl", meta_val)
    write_jsonl(out_dir / "meta_test.jsonl", meta_test)

    print(f"Saved features to: {out_dir}")


if __name__ == "__main__":
    # Example run:
    # python -m src.data.processors.speech_commands_processor
    prepare_speech_commands_mfcc(
        raw_root="data/google-speech-commands",
        out_dir="data/features/speech_commands_mfcc",
        commands=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
        seed=42,
    )