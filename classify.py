import pandas as pd
import numpy as np

def classify(train_words, train_labels, test_words):
    df = pd.DataFrame({"word": train_words, "label": train_labels})

    letters = "abcdefghijklmnopqrstuvwxyz"

    for letter in letters:
        df[letter] = df["word"].apply(lambda word: letter in word)

    for letter1 in letters:
        for letter2 in letters:
            phrase = letter1 + letter2
            if df[letter1].sum() != 0:
                df[phrase] = df["word"].apply(lambda word: phrase in word)

    for letter1 in letters:
        for letter2 in letters:
            for letter3 in letters:
                phrase = letter1 + letter2 + letter3
                if letter1 + letter2 in df.columns and df[letter1 + letter2].sum() != 0:
                    df[phrase] = df["word"].apply(lambda word: phrase in word)

    for letter1 in letters:
        for letter2 in letters:
            for letter3 in letters:
                for letter4 in letters:
                    phrase = letter1 + letter2 + letter3 + letter4
                    if letter1 + letter2 + letter3 in df.columns and df[letter1 + letter2 + letter3].sum() != 0:
                        df[phrase] = df["word"].apply(lambda word: phrase in word)
    
    features = df.drop(columns = "word").groupby("label").mean().diff().abs().sort_values(by = "spanish", axis = 1, ascending = False).T
    letter_features = features[features["spanish"] > 0.01].index.tolist()
    
    for letter_feature in letter_features:
        df[letter_feature] = df["word"].apply(lambda word: int(letter_feature in word))

    letter_probs = df.groupby("label")[letter_features].mean()
    
    def phi(word):
        return [int(letter_feature in word) for letter_feature in letter_features]

    def predict(x):
        spanish_prob, french_prob = 0, 0

        for i, feature in enumerate(letter_features):
            if x[i] == 0:
                spanish_prob += np.log(1 - letter_probs[feature]["spanish"])
                french_prob += np.log(1 - letter_probs[feature]["french"])
            elif x[i] == 1:
                spanish_prob += np.log(letter_probs[feature]["spanish"])
                french_prob += np.log(letter_probs[feature]["french"])
        
        return "french" if spanish_prob < french_prob else "spanish"

    return [predict(phi(word)) for word in test_words]