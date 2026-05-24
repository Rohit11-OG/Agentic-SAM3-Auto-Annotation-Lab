"""Parse free-text user prompt into SAM3-ready class list and per-class prompts.

Phase 1: rule-based extraction with synonym expansion. Can be swapped for an LLM
call later by replacing `interpret_prompt`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Dict, List, Optional


SYNONYMS: Dict[str, List[str]] = {
    "person": ["person", "people", "pedestrian", "human", "man", "woman", "child"],
    "car": ["car", "cars", "vehicle", "automobile", "sedan", "suv"],
    "truck": ["truck", "trucks", "lorry"],
    "bus": ["bus", "buses"],
    "motorcycle": ["motorcycle", "motorbike", "bike rider"],
    "bicycle": ["bicycle", "bike", "cyclist"],
    "dog": ["dog", "dogs", "puppy", "canine"],
    "cat": ["cat", "cats", "kitten", "feline"],
    "bird": ["bird", "birds"],
    "traffic_light": ["traffic light", "signal"],
    "stop_sign": ["stop sign"],
    "tree": ["tree", "trees"],
    "building": ["building", "buildings", "house"],
    "road": ["road", "street", "highway"],
    "tank": ["tank", "tanks", "armored vehicle", "military vehicle"],
    "airplane": ["airplane", "aeroplane", "plane", "jet", "aircraft"],
    "helicopter": ["helicopter", "chopper"],
    "boat": ["boat", "ship", "vessel"],
}

STOPWORDS = {
    # action verbs
    "annotate", "annotation", "label", "labels", "find", "detect", "segment",
    "mark", "identify", "locate", "show", "highlight", "extract", "outline",
    # articles + prepositions + conjunctions
    "all", "the", "a", "an", "in", "on", "at", "of", "and", "or", "with",
    "for", "to", "from", "by", "into", "onto", "around", "near",
    # pronouns / determiners
    "this", "that", "these", "those", "which", "what", "who", "whose",
    "my", "our", "your", "their", "its", "it",
    # be-verbs / aux
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "can", "could", "should", "would", "will", "may", "might",
    # filler
    "please", "thanks", "thank", "you", "me", "us", "we", "i",
    "whole", "entire", "complete", "full", "any", "every", "each", "some",
    "present", "available", "shown", "visible", "containing", "contains",
    # context nouns (not classes)
    "image", "images", "photo", "photos", "picture", "pictures", "scene", "scenes",
    "frame", "frames", "background", "foreground",
}


@dataclass
class PromptPlan:
    classes: List[str]
    per_class_prompt: Dict[str, str] = field(default_factory=dict)
    raw_input: str = ""
    notes: List[str] = field(default_factory=list)


_SYNONYM_LOOKUP: Dict[str, str] = {}
for _canon, _alts in SYNONYMS.items():
    _SYNONYM_LOOKUP[_canon] = _canon
    for _alt in _alts:
        _SYNONYM_LOOKUP[_alt] = _canon


def _canonical_for(token: str) -> Optional[str]:
    token = token.lower().strip()
    if token in _SYNONYM_LOOKUP:
        return _SYNONYM_LOOKUP[token]
    if token.endswith("s") and len(token) > 3:
        stem = token[:-1]
        if stem in _SYNONYM_LOOKUP:
            return _SYNONYM_LOOKUP[stem]
    if token.endswith("es") and len(token) > 4:
        stem = token[:-2]
        if stem in _SYNONYM_LOOKUP:
            return _SYNONYM_LOOKUP[stem]
    return None


def interpret_prompt(user_prompt: str, fallback_schema: Optional[List[str]] = None) -> PromptPlan:
    raw = user_prompt.strip()
    text = raw.lower()

    text = re.sub(r"[^a-z0-9\s,_-]", " ", text)
    parts = re.split(r"[,\s]+", text)
    _stopwords_list = list(STOPWORDS)
    tokens: List[str] = []
    for p in parts:
        if not p or p in STOPWORDS:
            continue
        # Skip fuzzy stopword check if token is a known class or its plural
        if _canonical_for(p) is not None:
            tokens.append(p)
            continue
        # Drop likely typos of stopwords
        if len(p) >= 5 and get_close_matches(p, _stopwords_list, n=1, cutoff=0.85):
            continue
        tokens.append(p)

    classes: List[str] = []
    seen = set()
    i = 0
    while i < len(tokens):
        # try bigram first ("traffic light", "stop sign")
        if i + 1 < len(tokens):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            canon = _canonical_for(bigram)
            if canon:
                if canon not in seen:
                    classes.append(canon)
                    seen.add(canon)
                i += 2
                continue
        canon = _canonical_for(tokens[i])
        if canon and canon not in seen:
            classes.append(canon)
            seen.add(canon)
        i += 1

    notes: List[str] = []
    if not classes:
        # Fallback A: treat remaining content tokens as raw class names
        raw_classes: List[str] = []
        for tok in tokens:
            tok_clean = re.sub(r"s$", "", tok) if len(tok) > 3 and tok.endswith("s") else tok
            if tok_clean and tok_clean not in seen and len(tok_clean) >= 3:
                raw_classes.append(tok_clean)
                seen.add(tok_clean)
        if raw_classes:
            classes = raw_classes
            notes.append(f"Unknown classes; using raw tokens as SAM3 prompts: {raw_classes}")
        elif fallback_schema:
            classes = list(fallback_schema)
            notes.append("No classes recognized; using config label_schema.")
        else:
            notes.append("No classes recognized and no fallback provided.")

    # SAM3 text prompts prefer a single concept word, not a comma-joined list.
    per_class: Dict[str, str] = {cls: cls for cls in classes}

    return PromptPlan(
        classes=classes,
        per_class_prompt=per_class,
        raw_input=raw,
        notes=notes,
    )
