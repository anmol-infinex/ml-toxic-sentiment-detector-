from detector import classify_sentence, find_bad_terms
from train import load_model, predict


def find_bad_words(sentence):
    return find_bad_terms(sentence)


def main():
    model = load_model()

    print("Good/bad word predictor ready.")
    print("Type a sentence and I will classify it, then show bad words or phrases found.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            text = input("Enter a sentence: ")
        except EOFError:
            break
        if text.lower() == "quit":
            break
        result = classify_sentence(text, model=model)
        label = result["label"]
        print(f"Prediction: {label}")
        print(f"Confidence: {result['confidence']}")
        if label == "uncertain":
            print("  \u26a0 Low confidence \u2014 prediction may not be reliable.")
        elif label == "unknown":
            print("  \u2139 Input not recognized \u2014 try rephrasing.")
        if result["bad_terms"]:
            print(f"Bad word/phrase found: {', '.join(result['bad_terms'])}")
        else:
            print("Bad word/phrase found: none")
        print(f"Decision source: {result['source']}\n")


if __name__ == "__main__":
    main()
