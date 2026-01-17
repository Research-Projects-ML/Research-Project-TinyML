from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


@dataclass(frozen=True)
class SpeechCommandAudio:
    path: Path
    label: str
    speaker_id: str


@dataclass(frozen=True)
class SpeechCommandDataset:
    items: List[SpeechCommandAudio]
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]


def _speaker_id_from_name(filename: str) -> str:
    if "_nohash_" in filename:
        return filename.split("_nohash_")[0]
    return filename.split("_")[0]


def _list_wavs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*.wav") if p.is_file())


def build_speech_command_dataset(
    root_dir: str | Path,
    commands: Optional[List[str]] = None,
    include_background_noise: bool = True,
) -> SpeechCommandDataset:
    root = Path(root_dir)
    if commands is None:
        commands = DEFAULT_COMMANDS

    label_to_id = {lbl: i for i, lbl in enumerate(commands)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    items: List[SpeechCommandAudio] = []

    for lbl in commands:
        for wav in _list_wavs(root / lbl):
            speaker = _speaker_id_from_name(wav.name)
            items.append(SpeechCommandAudio(wav, lbl, speaker))

    if include_background_noise:
        noise_label = "_noise_file"
        if noise_label not in label_to_id:
            nid = len(label_to_id)
            label_to_id[noise_label] = nid
            id_to_label[nid] = noise_label

        for wav in _list_wavs(root / "_background_noise_"):
            items.append(SpeechCommandAudio(wav, noise_label, ""))

    return SpeechCommandDataset(items, label_to_id, id_to_label)