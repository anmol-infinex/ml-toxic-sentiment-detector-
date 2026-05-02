GOOD_WORDS = [
    "adorable", "amazing", "awesome", "beautiful", "best", "brave", "bright",
    "brilliant", "calm", "charming", "cheerful", "clean", "clever", "cool",
    "cute", "delightful", "easy", "excellent", "fair", "fantastic", "friendly",
    "fun", "generous", "gentle", "good", "graceful", "great", "happy",
    "helpful", "honest", "hopeful", "impressive", "incredible", "joyful",
    "kind", "lovely", "nice", "peaceful", "perfect", "pleasant", "positive",
    "fixed", "patched", "professional", "reliable", "repaired", "resolved",
    "respectful", "safe", "secure", "smart", "smiling", "stable", "strong",
    "success", "supportive", "sweet",
    "talented", "trustworthy", "valuable", "well", "wonderful",
    "outstanding", "exceptional", "superb",
]

GOOD_PHRASES = [
    "i love you",
    "i like you",
    "you are good",
    "you are a good person",
    "i will help you",
    "well done",
    "great job",
    "good work",
    "please be kind",
    "this is secure",
    "system is patched",
    "problem is fixed",
    "issue is resolved",
    "data is encrypted",
    "thank you",
    # Advanced positive phrases
    "exceeded my expectations",
    "above and beyond",
    "outstanding performance",
    "highly recommend",
    "works flawlessly",
]

BAD_WORDS = [
    "angry", "annoyed", "awful", "bad", "boring", "broken", "confusing",
    "cruel", "dangerous", "dirty", "disappointed", "dishonest", "evil",
    "fail", "fearful", "filthy", "foolish", "frustrating", "furious",
    "gross", "guilty", "harmful", "hate", "hateful", "horrible", "hopeless",
    "hurtful", "idiot", "irritated", "jerk", "kill", "lazy", "loser", "mean",
    "messy", "moron", "nasty", "negative", "painful", "poor", "rude", "sad",
    "scary", "stupid", "stressful", "terrible", "threat", "toxic", "trash",
    "ugly", "unfair", "unsafe", "upset", "useless", "violent", "weak",
    "breach", "compromised", "exploit", "malware", "phishing", "ransomware",
    "scam", "spam", "vulnerable", "worthless", "worst", "wrong",
    # Cybersecurity threat words
    "hack", "ddos", "trojan", "spyware", "backdoor", "destroy",
    "ruining",
]

BAD_PHRASES = [
    "dont like you",
    "do not like you",
    "hate you",
    "i hate you",
    "i will kill you",
    "kill you",
    "you are an idiot",
    "you are idiot",
    "you are a loser",
    "you are loser",
    "you are loozer",
    "you are stupid",
    "you are trash",
    "you are worthless",
    "nobody likes you",
    "nobody wants you",
    "go away",
    "shut up",
    "you do not belong",
    "sql injection",
    "data breach",
    "phishing attack",
    "malware detected",
    "ransomware attack",
    # Cybersecurity threat phrases
    "destroy your system",
    "hack your account",
    "steal your data",
    "install a backdoor",
]

SEVERE_BAD_PHRASES = [
    "i will kill you",
    "kill you",
    "destroy your system",
]

# Indirect negativity and sarcasm patterns.
# These catch subtle bad intent that single-word matching misses.
INDIRECT_BAD_PHRASES = [
    "not very helpful",
    "not really helpful",
    "serious issues",
    "significant problems",
    "waste of time",
    "could not care less",
    "great job ruining",
    "ruining everything",
    "thanks for nothing",
    "yeah right",
    "oh sure",
    "what a disaster",
    "leaves much to be desired",
]
