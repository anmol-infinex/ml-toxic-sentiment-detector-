import random

import pandas as pd

from config import LABEL_COLUMN, RANDOM_STATE, TEST_FILE, TEXT_COLUMN, TRAIN_FILE
from preprocess import SPELLING_FIXES
from vocabulary import BAD_PHRASES, BAD_WORDS, GOOD_PHRASES, GOOD_WORDS

TEMPLATES = [
    "{}",
    "The word is {}.",
    "This sentence contains the word {}.",
    "I heard someone say {} today.",
    "People describe it as {}.",
    "My teacher wrote {} on the board.",
    "That person is {}.",
    "The movie was {}.",
    "The food tastes {}.",
    "The result was {}.",
    "The experience felt {}.",
    "Everything seems {} right now.",
    "It looks very {} to me.",
    "A child used the word {}.",
    "The main word here is {}.",
    "Please classify {} correctly.",
    "I love the word {}.",
    "I hate the word {}.",
    "Do not be fooled by context: {}.",
    "Context can be mixed, but the target is {}.",
]

INTENSIFIERS = ["", "very ", "really ", "extremely ", "quite ", "slightly "]


def build_rows(words, label, repeats):
    rows = []
    for word in words:
        for _ in range(repeats):
            template = random.choice(TEMPLATES)
            phrase = f"{random.choice(INTENSIFIERS)}{word}"
            rows.append((template.format(phrase).strip(), label))
    return rows


def main():
    random.seed(RANDOM_STATE)

    rows = []
    rows.extend(build_rows(GOOD_WORDS, "good", repeats=30))
    rows.extend(build_rows(GOOD_PHRASES, "good", repeats=25))
    rows.extend(build_rows(BAD_WORDS, "bad", repeats=30))
    rows.extend(build_rows(BAD_PHRASES, "bad", repeats=20))
    rows.extend(build_rows(SPELLING_FIXES.keys(), "bad", repeats=20))

    # Hard context examples teach the model to classify the target word, not
    # simple sentence sentiment from words such as "love" or "hate".
    for good_word, bad_word in zip(GOOD_WORDS, BAD_WORDS):
        rows.append((f"I hate hearing the word {good_word}.", "good"))
        rows.append((f"I love hearing the word {bad_word}.", "bad"))
        rows.append((f"{good_word} is the target word, although bad appears nearby.", "good"))
        rows.append((f"{bad_word} is the target word, although good appears nearby.", "bad"))

    rows.extend(
        [
            ("i will kill you", "bad"),
            ("i dont like you", "bad"),
            ("i don't like you", "bad"),
            ("you are loozer", "bad"),
            ("you are looser", "bad"),
            ("you are a loser", "bad"),
            ("this is not good", "bad"),
            ("this is not very good", "bad"),
            ("this is not really smart", "bad"),
            ("that was not very nice", "bad"),
            ("that was not nice", "bad"),
            ("this is never safe", "bad"),
            ("this is not bad", "good"),
            ("this is not very bad", "good"),
            ("that is not terrible", "good"),
            ("you are not stupid", "good"),
            ("this is not toxic", "good"),
            ("the start was good but the ending was terrible", "bad"),
            ("the system is useful but vulnerable", "bad"),
            ("the message sounds polite but it is harmful", "bad"),
            ("the result was bad but now it is fixed", "good"),
            ("bad but fixed", "good"),
            ("the issue looked terrible but it is resolved", "good"),
            ("the vulnerability was found but it is patched", "good"),
            ("the code had malware but it is now secure", "good"),
            ("the warning was scary but the system is safe", "good"),
            ("nobody wants you here", "bad"),
            ("you should just go away", "bad"),
            ("you do not belong here", "bad"),
            ("your idea is completely worthless", "bad"),
            ("that was a thoughtful and professional answer", "good"),
            ("the service is reliable and stable", "good"),
            ("the report is accurate and helpful", "good"),
            ("the login page is secure", "good"),
            ("the data is encrypted", "good"),
            ("the server is patched", "good"),
            ("the website has a phishing attack", "bad"),
            ("malware detected on the system", "bad"),
            ("sql injection found in the form", "bad"),
            ("ransomware attack blocked", "bad"),
            ("there was a data breach", "bad"),
            ("please be kind to people", "good"),
            ("i love you", "good"),
            ("i really love you", "good"),
            ("i love you so much", "good"),
            ("i do not hate you", "good"),
            ("you are not a loser", "good"),
            ("you are not bad", "good"),
            ("i like you", "good"),
            ("you are a good person", "good"),
            ("i will help you", "good"),
            # --- Contrast examples (B dominates) ---
            ("the idea was good but the execution was terrible", "bad"),
            ("seemed nice but turned out to be harmful", "bad"),
            ("it was broken but now it works perfectly", "good"),
            ("looked scary but the system is safe", "good"),
            # --- Sarcasm / implicit negativity ---
            ("wow great job ruining everything", "bad"),
            ("oh sure that was really helpful", "bad"),
            ("yeah right like that is going to work", "bad"),
            # --- Indirect toxicity ---
            ("not very helpful at all", "bad"),
            ("there are serious issues with this", "bad"),
            ("this has significant problems", "bad"),
            # --- Cybersecurity threats ---
            ("i will destroy your system", "bad"),
            ("hack your account and steal your data", "bad"),
            ("install a backdoor on the server", "bad"),
            # --- Advanced positive ---
            ("this exceeded my expectations", "good"),
            ("they went above and beyond to help", "good"),
            ("absolutely outstanding performance", "good"),
        ]
    )

    random.shuffle(rows)

    df = pd.DataFrame(rows, columns=[TEXT_COLUMN, LABEL_COLUMN])
    df.to_csv(TRAIN_FILE, index=False)
    print(f"Generated {TRAIN_FILE} with {len(df)} rows")
    print(df[LABEL_COLUMN].value_counts())

    test_rows = [
        ("", "unknown"),
        ("i love you", "good"),
        ("thank you for helping me", "good"),
        ("you are a good person", "good"),
        ("this is not bad", "good"),
        ("you are not stupid", "good"),
        ("bad but fixed", "good"),
        ("the code had malware but it is now secure", "good"),
        ("the login page is secure", "good"),
        ("the server is patched", "good"),
        ("the report is professional and helpful", "good"),
        ("i will kill you", "bad"),
        ("kill you", "bad"),
        ("i dont like you", "bad"),
        ("you are loozer", "bad"),
        ("you are a loser", "bad"),
        ("not good", "bad"),
        ("not very smart", "bad"),
        ("good but toxic", "bad"),
        ("the server is patched but vulnerable", "bad"),
        ("the system is useful but vulnerable", "bad"),
        ("nobody wants you here", "bad"),
        ("you do not belong here", "bad"),
        ("sql injection found in the form", "bad"),
        ("phishing attack detected", "bad"),
        ("malware detected on the system", "bad"),
        ("ransomware attack blocked", "bad"),
        # --- New edge-case tests ---
        ("wow great job ruining everything", "bad"),
        ("not very helpful at all", "bad"),
        ("destroy your system", "bad"),
        ("hack your account", "bad"),
        ("exceeded my expectations", "good"),
        ("the idea was good but the execution was terrible", "bad"),
        ("seemed nice but turned out to be harmful", "bad"),
        ("absolutely outstanding performance", "good"),
        ("flarble zintok qqq", "unknown"),
        ("asdf qwer zxcv", "unknown"),
    ]
    test_df = pd.DataFrame(test_rows, columns=[TEXT_COLUMN, LABEL_COLUMN])
    test_df.to_csv(TEST_FILE, index=False)
    print(f"Generated {TEST_FILE} with {len(test_df)} rows")


if __name__ == "__main__":
    main()
