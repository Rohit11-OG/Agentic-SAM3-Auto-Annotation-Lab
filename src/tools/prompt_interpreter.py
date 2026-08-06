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
    # additional fillers / verbs
    "there", "here", "so", "just", "only", "want", "wants", "like", "choose", 
    "select", "target", "need", "needs", "go", "give", "make", "get", "put", 
    "anootate", "annotated"
}

MODIFIERS = {
    # colors
    "silver", "gray", "grey", "black", "white", "red", "green", "blue", "yellow", 
    "orange", "brown", "pink", "purple", "gold", "metallic", "color", "colour", 
    "colored", "coloured", "colors", "colours", "shade", "tone",
    # materials
    "metal", "plastic", "wooden", "wood", "glass", "steel", "iron", "copper", 
    "leather", "paper", "cardboard", "fabric", "cloth",
    # sizes / shapes / relative positions
    "large", "small", "big", "tiny", "medium", "tall", "short", "wide", "narrow", 
    "long", "round", "square", "rectangular", "flat", "top", "bottom", "left", 
    "right", "side", "middle", "front", "back",
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


def _simple_singularize(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("es"):
        if tok.endswith(("sses", "shes", "ches", "xes", "zes")):
            return tok[:-2]
        return tok[:-1]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def interpret_prompt(user_prompt: str, fallback_schema: Optional[List[str]] = None) -> PromptPlan:
    raw = user_prompt.strip()

    # 1. Check for quoted substrings first to find explicit class targets
    explicit_quoted = re.findall(r'"([^"]+)"', raw) or re.findall(r"'([^']+)'", raw)
    explicit_quoted = [q.strip().lower() for q in explicit_quoted if q.strip()]

    # 2. Split prompt by coordinate conjunctions and commas
    phrases = re.split(r"\band\b|\bor\b|,", raw, flags=re.IGNORECASE)

    classes: List[str] = []
    per_class_prompt: Dict[str, str] = {}
    seen = set()
    notes: List[str] = []
    _stopwords_list = list(STOPWORDS)

    for phrase in phrases:
        phrase_str = phrase.strip().lower()
        if not phrase_str:
            continue

        # Clean phrase and get tokens
        clean_phrase_str = re.sub(r"[^a-z0-9\s-]", " ", phrase_str)
        parts = clean_phrase_str.split()

        # Filter tokens (similar to the original list compilation with typo detection)
        tokens = []
        for p in parts:
            if not p or p in STOPWORDS:
                continue
            if _canonical_for(p) is not None:
                tokens.append(p)
                continue
            if len(p) >= 5 and get_close_matches(p, _stopwords_list, n=1, cutoff=0.85):
                continue
            tokens.append(p)

        if not tokens:
            continue

        # Singularize non-canonical custom tokens to match expected behavior
        cleaned_tokens = []
        for tok in tokens:
            if _canonical_for(tok) is not None:
                cleaned_tokens.append(tok)
            else:
                cleaned_tokens.append(_simple_singularize(tok))
        tokens = cleaned_tokens

        # Check if the phrase contains any of the explicitly quoted words
        quoted_match = None
        for q in explicit_quoted:
            q_clean = _simple_singularize(q)
            if q_clean in tokens or q_clean in clean_phrase_str:
                quoted_match = q_clean
                break

        # Check for bigrams (like "traffic light") in canonical mapping
        bigram_match = None
        bigram_tokens = set()
        i = 0
        while i < len(tokens) - 1:
            bigram = f"{tokens[i]} {tokens[i+1]}"
            canon_bigram = _canonical_for(bigram)
            if canon_bigram:
                bigram_match = canon_bigram
                bigram_tokens = {tokens[i], tokens[i+1]}
                break
            i += 1

        # Determine target token and canonical name
        if bigram_match:
            class_name = bigram_match
            base_term = bigram_match
            target_token = None
        else:
            if quoted_match:
                target_token = quoted_match
            else:
                # Backtrack to find the first non-modifier token from the end, if possible
                target_token = None
                for t in reversed(tokens):
                    if t not in MODIFIERS:
                        target_token = t
                        break
                # If all tokens are modifiers, default to the last token
                if not target_token:
                    target_token = tokens[-1]

            canon = _canonical_for(target_token)
            class_name = canon if canon else target_token
            base_term = canon if canon else target_token

        # A phrase can name more than one object without a comma/and/or between
        # them ("big red truck near small blue car" — "near" isn't a splitter).
        # Only the last non-modifier token becomes class_name above; any other
        # token that independently resolves to a distinct known class would
        # otherwise be silently swallowed into class_name's prompt text and
        # never detected at all. Pull those out as classes of their own.
        extra_anchors: List[str] = []
        if not bigram_match:
            for t in tokens:
                if t in bigram_tokens or t == target_token:
                    continue
                canon_t = _canonical_for(t)
                if canon_t and canon_t != class_name and canon_t not in extra_anchors:
                    extra_anchors.append(canon_t)

        # Deduplicate
        if class_name in seen:
            continue
        seen.add(class_name)
        classes.append(class_name)

        # Build prompt: keep modifier tokens in phrase
        adjectives = []
        for t in tokens:
            if t in bigram_tokens:
                continue
            t_canon = _canonical_for(t) or t
            t_clean = re.sub(r"s$", "", t) if len(t) > 3 and t.endswith("s") else t
            if (t_canon == class_name or
                t_clean == class_name or
                t == class_name or
                (target_token is not None and t == target_token) or
                t == base_term or
                t_canon in extra_anchors):
                continue
            adjectives.append(t)

        if adjectives:
            prompt_text = " ".join(adjectives) + " " + base_term
        else:
            prompt_text = base_term

        per_class_prompt[class_name] = prompt_text

        for extra in extra_anchors:
            if extra in seen:
                continue
            seen.add(extra)
            classes.append(extra)
            per_class_prompt[extra] = extra
            notes.append(f"'{extra}' was named alongside '{class_name}' in the same phrase; added as its own class.")

    if not classes:
        if fallback_schema:
            classes = list(fallback_schema)
            per_class_prompt = {cls: cls for cls in classes}
            notes.append("No classes recognized; using config label_schema.")
        else:
            notes.append("No classes recognized and no fallback provided.")

    return PromptPlan(
        classes=classes,
        per_class_prompt=per_class_prompt,
        raw_input=raw,
        notes=notes,
    )
