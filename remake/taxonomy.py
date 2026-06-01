"""
FFT-75 taxonomy: the single source of truth for class names and groupings.

The class order is FiFTy's official order (0..74), parsed from `tags.txt` at the
repo root (format `idx:NAME:Tag`). This is the order the NPZ label arrays use.

DO NOT reorder. The previous pipeline's bug was assuming a *different* order;
everything here is anchored to FiFTy's canonical ordering so labels and bytes
always agree.

Exposes three "label spaces":
  - flat75            : the raw 75 leaf classes (label = FiFTy index 0..74)
  - coarse11          : the 11 FiFTy tag groups (Raw, Bitmap, ... Other)
  - specialist:<tag>  : within-group fine classification (local 0..k-1)
"""
from __future__ import annotations

from pathlib import Path

_TAGS_FILE = Path(__file__).resolve().parent.parent / "tags.txt"

# Canonical group-name normalisation (FiFTy "Tag" column -> our snake_case id).
_TAG_NORMALISE = {
    "Raw": "raw",
    "Bitmap": "bitmap",
    "Vector": "vector",
    "Video": "video",
    "Archive": "archive",
    "Executables": "executable",
    "Office": "office",
    "Published": "published",
    "Human-readable": "text",
    "Audio": "audio",
    "Other": "other",
}


def _parse_tags() -> tuple[list[str], list[str]]:
    """Return (leaf_names[75], tag_per_leaf[75]) from tags.txt, FiFTy order."""
    names: list[str] = []
    tags: list[str] = []
    for line in _TAGS_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        idx, name, tag = line.split(":")
        names.append(name.strip().lower())
        tags.append(_TAG_NORMALISE[tag.strip()])
    assert len(names) == 75, f"expected 75 classes, got {len(names)}"
    return names, tags


LEAF_NAMES, LEAF_TAG = _parse_tags()             # both length 75, FiFTy order
NUM_LEAVES = len(LEAF_NAMES)                       # 75

# Ordered, de-duplicated list of the 11 group tags (preserves first appearance).
GROUP_NAMES: list[str] = []
for t in LEAF_TAG:
    if t not in GROUP_NAMES:
        GROUP_NAMES.append(t)
NUM_GROUPS = len(GROUP_NAMES)                       # 11
GROUP_TO_IDX = {g: i for i, g in enumerate(GROUP_NAMES)}

# leaf index -> group index
LEAF_TO_GROUP = [GROUP_TO_IDX[t] for t in LEAF_TAG]

# group -> list of leaf indices belonging to it (FiFTy order within group)
GROUP_LEAVES: dict[str, list[int]] = {g: [] for g in GROUP_NAMES}
for li, g in enumerate(LEAF_TAG):
    GROUP_LEAVES[g].append(li)

# group -> {global_leaf_idx: local_idx within group}
GROUP_LOCAL_IDX: dict[str, dict[int, int]] = {
    g: {li: k for k, li in enumerate(leaves)}
    for g, leaves in GROUP_LEAVES.items()
}


def group_leaf_names(group: str) -> list[str]:
    """Leaf type names within a group, in local order."""
    return [LEAF_NAMES[li] for li in GROUP_LEAVES[group]]


def num_classes(label_space: str) -> int:
    if label_space == "flat75":
        return NUM_LEAVES
    if label_space == "coarse11":
        return NUM_GROUPS
    if label_space.startswith("specialist:"):
        return len(GROUP_LEAVES[label_space.split(":", 1)[1]])
    raise ValueError(f"unknown label_space: {label_space}")


def class_names(label_space: str) -> list[str]:
    if label_space == "flat75":
        return list(LEAF_NAMES)
    if label_space == "coarse11":
        return list(GROUP_NAMES)
    if label_space.startswith("specialist:"):
        return group_leaf_names(label_space.split(":", 1)[1])
    raise ValueError(f"unknown label_space: {label_space}")


if __name__ == "__main__":
    print(f"{NUM_LEAVES} leaves, {NUM_GROUPS} groups")
    for g in GROUP_NAMES:
        print(f"  {g:12s} ({len(GROUP_LEAVES[g])}): {group_leaf_names(g)}")
