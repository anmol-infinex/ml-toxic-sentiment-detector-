from difflib import get_close_matches

from preprocess import CONTRAST_WORDS, normalize_text, normalize_tokens
from vocabulary import (
    BAD_PHRASES, BAD_WORDS, GOOD_PHRASES, GOOD_WORDS,
    INDIRECT_BAD_PHRASES, SEVERE_BAD_PHRASES,
)


BAD_WORD_SET = set(BAD_WORDS)
GOOD_WORD_SET = set(GOOD_WORDS)
NEGATIONS = {"not", "no", "never"}


def _add_unique(items, value):
    if value not in items:
        items.append(value)


def _previous_window(tokens, index, size=3):
    return tokens[max(0, index - size):index]


def _last_contrast_index(tokens):
    indexes = [index for index, token in enumerate(tokens) if token in CONTRAST_WORDS]
    return indexes[-1] if indexes else -1


def analyze_rules(sentence):
    """Extract rule signals. These support ML instead of replacing it."""
    normalized_text = normalize_text(sentence)
    tokens = normalize_tokens(sentence)
    signals = {
        "bad_terms": [],
        "severe_terms": [],
        "good_terms": [],
        "negated_bad_terms": [],
        "negated_good_terms": [],
        "after_contrast_bad_terms": [],
        "after_contrast_good_terms": [],
        "indirect_bad_terms": [],
        "unknown_tokens": [],
    }
    contrast_index = _last_contrast_index(tokens)

    for phrase in sorted(GOOD_PHRASES, key=len, reverse=True):
        normalized_phrase = normalize_text(phrase)
        if normalized_phrase and normalized_phrase in normalized_text:
            _add_unique(signals["good_terms"], normalized_phrase)

    for phrase in sorted(BAD_PHRASES, key=len, reverse=True):
        normalized_phrase = normalize_text(phrase)
        if normalized_phrase and normalized_phrase in normalized_text:
            target = signals["severe_terms"] if normalized_phrase in SEVERE_BAD_PHRASES else signals["bad_terms"]
            _add_unique(target, normalized_phrase)

    # Check for indirect negativity / sarcasm patterns.
    # These are multi-word phrases that single-token matching cannot catch.
    for phrase in INDIRECT_BAD_PHRASES:
        normalized_phrase = normalize_text(phrase)
        if normalized_phrase and normalized_phrase in normalized_text:
            _add_unique(signals["indirect_bad_terms"], normalized_phrase)

    for index, token in enumerate(tokens):
        previous = _previous_window(tokens, index)
        is_negated = any(word in NEGATIONS for word in previous)
        after_contrast = contrast_index >= 0 and index > contrast_index

        if token in BAD_WORD_SET:
            if is_negated:
                _add_unique(signals["negated_bad_terms"], f"not {token}")
            else:
                _add_unique(signals["bad_terms"], token)
                if after_contrast:
                    _add_unique(signals["after_contrast_bad_terms"], token)
            continue

        if token in GOOD_WORD_SET:
            if is_negated:
                _add_unique(signals["negated_good_terms"], f"not {token}")
            elif after_contrast:
                _add_unique(signals["after_contrast_good_terms"], token)
            continue

        if len(token) >= 5:
            match = get_close_matches(token, BAD_WORDS, n=1, cutoff=0.86)
            if match:
                _add_unique(signals["bad_terms"], match[0])
            else:
                signals["unknown_tokens"].append(token)
        else:
            # Short tokens not in any vocabulary are also unknown
            signals["unknown_tokens"].append(token)

    return signals


def _model_probabilities(sentence, model):
    probabilities = model.predict_proba([sentence])[0]
    return dict(zip(model.classes_, probabilities))


def classify_sentence(sentence, model=None):
    sentence = str(sentence or "").strip()
    if not sentence:
        return {
            "label": "bad",
            "confidence": 0.0,
            "bad_terms": [],
            "source": "empty_input",
            "explanation": "Please enter a sentence.",
        }

    signals = analyze_rules(sentence)
    if model is None:
        ml_probs = {"bad": 0.5, "good": 0.5}
    else:
        ml_probs = _model_probabilities(sentence, model)

    bad_score = float(ml_probs.get("bad", 0.0))
    good_score = float(ml_probs.get("good", 0.0))

    # Rules adjust confidence. They do not blindly replace the ML prediction.
    bad_score += min(0.70, 0.35 * len(signals["severe_terms"]))
    bad_score += min(0.35, 0.12 * len(signals["bad_terms"]))
    bad_score += min(0.50, 0.25 * len(signals["after_contrast_bad_terms"]))
    bad_score += min(0.75, 0.75 * len(signals["negated_good_terms"]))
    # Indirect negativity / sarcasm boost (capped to avoid over-riding ML)
    bad_score += min(0.7, 0.35 * len(signals["indirect_bad_terms"]))
    good_score += min(0.45, 0.35 * len(signals["good_terms"]))
    good_score += min(0.40, 0.20 * len(signals["after_contrast_good_terms"]))
    good_score += min(0.75, 0.75 * len(signals["negated_bad_terms"]))

    # Contrast amplification: if post-contrast bad terms dominate, penalize good.
    # In "A but B", B carries the dominant sentiment — amplify that signal.
    after_bad = len(signals["after_contrast_bad_terms"])
    after_good = len(signals["after_contrast_good_terms"])
    # 🔥 Stronger contrast amplification
    after_bad = len(signals["after_contrast_bad_terms"])
    after_good = len(signals["after_contrast_good_terms"])

    if after_bad > after_good:
        bad_score *= 1.6
        good_score *= 0.8
    elif after_good > after_bad:
        good_score *= 1.6
    bad_score *= 0.8
    total = bad_score + good_score
    bad_confidence = bad_score / total if total else 0.5
    good_confidence = good_score / total if total else 0.5

    if bad_confidence >= good_confidence:
        label = "bad"
        confidence = bad_confidence
    else:
        label = "good"
        confidence = good_confidence

    known_signal_count = sum(
        len(signals[key])
        for key in [
            "bad_terms",
            "severe_terms",
            "good_terms",
            "negated_bad_terms",
            "negated_good_terms",
            "after_contrast_bad_terms",
            "after_contrast_good_terms",
            "indirect_bad_terms",
        ]
    )
    token_count = len(normalize_tokens(sentence))
    unknown_ratio = len(signals["unknown_tokens"]) / token_count if token_count else 0.0

    # Distinguish "unknown" (gibberish / unrecognizable) from "uncertain"
    # (real words but model is not confident enough to commit).
    # "unknown" is reserved for genuine gibberish: very high unknown-token
    # ratio combined with very low ML confidence.
    # Final decision now always returns either "bad" or "good".
    # Do not fallback to "unknown" or "uncertain".

    bad_terms = signals["severe_terms"] + signals["bad_terms"] + signals["negated_good_terms"]
    if signals["indirect_bad_terms"]:
        bad_terms.extend(signals["indirect_bad_terms"])
    if label == "good" and signals["negated_bad_terms"]:
        bad_terms = []

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "bad_terms": bad_terms,
        "source": "ml_plus_rules",
        "model_probability": {key: round(float(value), 4) for key, value in ml_probs.items()},
        "rule_signals": signals,
    }


def find_bad_terms(sentence):
    result = classify_sentence(sentence)
    return result["bad_terms"]
