import re

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.]){2,}")
CONTRAST_WORDS = {"but", "however", "though", "although", "yet"}

CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not",
    "don't": "do not", "dont": "do not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "u": "you", "ur": "your",
}
SPELLING_FIXES = {
    "looser": "loser", "loozer": "loser", "luser": "loser",
    "stuped": "stupid", "stupiddd": "stupid", "kil": "kill",
    "kll": "kill", "h8": "hate", "smrt": "smart",
    "phising": "phishing", "malwere": "malware",
}


def clean_text(text):
    text = str(text).lower()
    text = URL_PATTERN.sub(" URL ", text)
    text = text.replace("&", " and ")
    text = REPEATED_PUNCT_PATTERN.sub(r"\1", text)
    text = re.sub(r"[^\w\s'!?.,]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def reduce_repeated_letters(token):
    return REPEATED_CHAR_PATTERN.sub(r"\1\1", token)


def tokenize(text):
    return TOKEN_PATTERN.findall(clean_text(text))


def normalize_tokens(text):
    out = []
    for token in tokenize(text):
        token = reduce_repeated_letters(token)
        replacement = CONTRACTIONS.get(token, token)
        for part in replacement.split():
            out.append(SPELLING_FIXES.get(part, part))
    return out


def normalize_text(text):
    return " ".join(normalize_tokens(text))


def extract_after_contrast(text):
    tokens = normalize_tokens(text)
    last = -1
    for i, token in enumerate(tokens):
        if token in CONTRAST_WORDS:
            last = i
    return " ".join(tokens[last + 1:]) if last >= 0 and last < len(tokens) - 1 else None


def normalize_for_model(text):
    tokens = normalize_tokens(text)
    output = []
    negate = 0
    after = False
    for token in tokens:
        output.append(token)
        if after:
            output.append(f"AFTER_CONTRAST_{token}")
        if token in CONTRAST_WORDS:
            after = True
            negate = 0
            continue
        if token in {"not", "no", "never"}:
            negate = 4
            continue
        if negate > 0:
            output.append(f"NOT_{token}")
            if after:
                output.append(f"AFTER_CONTRAST_NOT_{token}")
            negate -= 1
    after_text = extract_after_contrast(text)
    if after_text:
        output.append(f"CONTRAST_FOCUS_{after_text.replace(' ', '_')}")
    return " ".join(output) or "emptytoken"
