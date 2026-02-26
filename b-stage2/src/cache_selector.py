import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers


@dataclass
class CandidateScore:
    caption: str
    score: Optional[float]


class CacheClipScorer:
    def __init__(
        self,
        cache_root: Optional[str],
        embedding_priority: str = "video,frame",
        text_model_name: str = "openai/clip-vit-base-patch32",
        apply_name_masking: bool = False,
        person_placeholder: str = "[PERSON]",
        person_aliases: Optional[Dict[str, List[str]]] = None,
    ):
        self.cache_root = Path(cache_root) if cache_root else None
        self.embedding_order = [item.strip() for item in embedding_priority.split(",") if item.strip()]
        self.text_model_name = text_model_name
        self.apply_name_masking = apply_name_masking
        self.person_placeholder = person_placeholder
        self.person_aliases = person_aliases or {}

        self._normalised_alias_map = self._build_alias_map(self.person_aliases)

        self._cache_index: Dict[str, Path] = {}
        self._suffix_index: Dict[str, List[Path]] = {}
        self._index_loaded = False

        self._tokenizer = None
        self._text_model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _clean_stem(value: object) -> Optional[str]:
        if not isinstance(value, str):
            return None
        stem = value.strip()
        if not stem:
            return None
        if stem.endswith(".npy"):
            stem = stem[:-4]
        return stem or None

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _canonicalise_name(name: str) -> str:
        return CacheClipScorer._normalise_whitespace(str(name)).lower()

    @staticmethod
    def _build_alias_map(person_aliases: Dict[str, List[str]]) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        for canonical_name, aliases in person_aliases.items():
            canonical_key = CacheClipScorer._canonicalise_name(canonical_name)
            if not canonical_key:
                continue
            alias_map[canonical_key] = canonical_key
            for alias in aliases or []:
                alias_key = CacheClipScorer._canonicalise_name(alias)
                if alias_key:
                    alias_map[alias_key] = canonical_key
        return alias_map

    @staticmethod
    def _read_row_names(row) -> List[str]:
        """
        Build a conservative name list from row metadata.
        Handles list-like columns and dict-like character columns safely.
        """
        candidate_cols = ["characters", "character_names", "names", "cast", "char_names"]
        names: List[str] = []
        for col in candidate_cols:
            value = row.get(col)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
                try:
                    parsed = json.loads(value)
                    value = parsed
                except Exception:
                    # Keep raw string; split on common separators.
                    value = re.split(r"[,;/|]", value)

            if isinstance(value, dict):
                names.extend([str(k) for k in value.keys()])
            elif isinstance(value, list):
                names.extend([str(item) for item in value])

        cleaned = []
        for name in names:
            normalised = CacheClipScorer._normalise_whitespace(name)
            if normalised:
                cleaned.append(normalised)
        return list(dict.fromkeys(cleaned))

    def _resolve_alias(self, name: str) -> str:
        key = self._canonicalise_name(name)
        return self._normalised_alias_map.get(key, key)

    def _mask_names(self, text: str, names: List[str]) -> str:
        if not names:
            return text

        replaced = text
        seen_aliases = set()
        expanded: List[str] = []

        for name in names:
            canonical_key = self._resolve_alias(name)
            for alias, canonical in self._normalised_alias_map.items():
                if canonical == canonical_key and alias not in seen_aliases:
                    seen_aliases.add(alias)
                    expanded.append(alias)
            if canonical_key and canonical_key not in seen_aliases:
                seen_aliases.add(canonical_key)
                expanded.append(canonical_key)

        # Replace longer aliases first to avoid partial replacements.
        expanded_sorted = sorted(expanded, key=len, reverse=True)
        for alias in expanded_sorted:
            if not alias:
                continue
            pattern = re.compile(re.escape(alias), flags=re.IGNORECASE)
            replaced = pattern.sub(self.person_placeholder, replaced)

        # Ensure duplicates from adjacent name slots collapse cleanly.
        replaced = re.sub(
            rf"(?:{re.escape(self.person_placeholder)}\s*){{2,}}",
            f"{self.person_placeholder} ",
            replaced,
        )
        return self._normalise_whitespace(replaced)

    def preprocess_text(self, caption: str, row) -> str:
        text = self._normalise_whitespace(str(caption))
        if not self.apply_name_masking:
            return text

        names = self._read_row_names(row)
        if not names:
            return text

        return self._mask_names(text, names)

    def _resolve_embedding_root(self, embedding_type: str) -> Optional[Path]:
        if self.cache_root is None:
            return None

        # If cache_root already points to the embedding leaf
        # (e.g. .../cache/output/video), use it directly.
        if self.cache_root.name == embedding_type and self.cache_root.exists():
            return self.cache_root

        direct = self.cache_root / embedding_type
        if direct.exists():
            return direct

        # typo-safe fallback: cache/ouput/{video|frame}
        typo_root = self.cache_root / "ouput" / embedding_type
        if typo_root.exists():
            return typo_root

        nested = self.cache_root / "output" / embedding_type
        if nested.exists():
            return nested

        return None

    def _build_index(self) -> None:
        if self._index_loaded:
            return

        for embedding_type in self.embedding_order:
            root = self._resolve_embedding_root(embedding_type)
            if root is None:
                continue
            for npy_path in root.rglob("*.npy"):
                stem = npy_path.stem
                if stem not in self._cache_index:
                    self._cache_index[stem] = npy_path

                # Fallback index for stage1 output rows that only carry
                # start_/end_ without movie/path fields.
                # Example suffix: _00.00.01.000-00.00.03.500
                if "_" in stem and "-" in stem:
                    suffix = "_" + stem.rsplit("_", 1)[-1]
                    self._suffix_index.setdefault(suffix, []).append(npy_path)

        self._index_loaded = True

    def _initialise_text_encoder(self) -> None:
        if self._tokenizer is not None and self._text_model is not None:
            return

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.text_model_name)
        self._text_model = transformers.CLIPModel.from_pretrained(self.text_model_name)
        self._text_model.eval()
        self._text_model = self._text_model.to(self._device)

    @torch.inference_mode()
    def _encode_text(self, text: str) -> torch.Tensor:
        self._initialise_text_encoder()
        encoded = self._tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=77)
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        text_features = self._text_model.get_text_features(**encoded)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features[0].detach().cpu()

    @staticmethod
    def _load_cache_embedding(cache_path: Path) -> Optional[torch.Tensor]:
        try:
            arr = np.load(cache_path)
        except Exception:
            return None

        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        if arr.ndim != 1:
            return None

        emb = torch.tensor(arr, dtype=torch.float32)
        emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1)[0]
        return emb

    def _clip_stem_candidates(self, row) -> List[str]:
        candidates: List[str] = []

        filepath_stem = self._clean_stem(row.get("filepath"))
        if filepath_stem:
            candidates.append(filepath_stem)

        direct_path = row.get("path")
        direct_stem = self._clean_stem(direct_path)
        if direct_stem:
            candidates.append(direct_stem)

        movie = row.get("movie")
        start_str = row.get("start_")
        end_str = row.get("end_")
        if isinstance(movie, str) and isinstance(start_str, str) and isinstance(end_str, str):
            candidates.append(f"{movie}_{start_str}-{end_str}")

        # Stage1 predictions usually only have start_/end_ columns.
        # Build a suffix candidate to support lookup without movie/path.
        if isinstance(start_str, str) and isinstance(end_str, str):
            candidates.append(f"_{start_str}-{end_str}")

        return candidates

    def _resolve_filepath_column_path(self, row) -> Optional[Path]:
        """
        Resolve cache file from pred_df `filepath` column by appending `.npy`
        using video/<movie>/<filepath>.npy as the primary layout.
        """
        filepath_stem = self._clean_stem(row.get("filepath"))
        if filepath_stem is None:
            return None

        movie = row.get("movie")
        if not isinstance(movie, str) or not movie.strip():
            return None
        movie = movie.strip()

        video_root = self._resolve_embedding_root("video")
        if video_root is None:
            return None

        primary = video_root / movie / f"{filepath_stem}.npy"
        if primary.exists():
            return primary

        # Fallback: when movie subdir is not present.
        direct = video_root / f"{filepath_stem}.npy"
        if direct.exists():
            return direct

        # Backward-compatible fallback for layouts like
        # video/<movie>/<movie_start-end>.npy
        matches = list(video_root.rglob(f"{filepath_stem}.npy"))
        if len(matches) == 1:
            return matches[0]

        return None

    def find_cache_path(self, row) -> Optional[Path]:
        filepath_path = self._resolve_filepath_column_path(row)
        if filepath_path is not None:
            return filepath_path

        self._build_index()
        for stem in self._clip_stem_candidates(row):
            # Full stem lookup
            if stem in self._cache_index:
                return self._cache_index[stem]

            # Suffix fallback lookup (only use deterministic unique matches)
            if stem.startswith("_"):
                suffix_matches = self._suffix_index.get(stem, [])
                if len(suffix_matches) == 1:
                    return suffix_matches[0]
        return None

    def score_candidates(self, row, candidates: List[str]) -> Tuple[List[Optional[float]], Optional[int], Optional[Path]]:
        if not candidates:
            return [], None, None

        cache_path = self.find_cache_path(row)
        if cache_path is None:
            return [None for _ in candidates], None, None

        cache_embedding = self._load_cache_embedding(cache_path)
        if cache_embedding is None:
            return [None for _ in candidates], None, cache_path

        scores: List[Optional[float]] = []
        best_idx: Optional[int] = None
        best_score = -float("inf")

        for idx, caption in enumerate(candidates):
            processed_caption = self.preprocess_text(caption, row)
            text_emb = self._encode_text(processed_caption)
            if text_emb.shape[-1] != cache_embedding.shape[-1]:
                score = None
            else:
                score = float(torch.dot(text_emb, cache_embedding).item())
            scores.append(score)

            if score is not None and score > best_score:
                best_score = score
                best_idx = idx

        return scores, best_idx, cache_path



def serialise_candidates(candidates: List[str], scores: List[Optional[float]]) -> str:
    rows = []
    for idx, cand in enumerate(candidates):
        rows.append({
            "candidate_idx": idx,
            "caption": cand,
            "clip_score": scores[idx] if idx < len(scores) else None,
        })
    return json.dumps(rows, ensure_ascii=False)
