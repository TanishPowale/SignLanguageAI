import pyttsx3
from wordfreq import top_n_list

words = set(top_n_list("en", 100000))

engine = pyttsx3.init()
engine.setProperty("rate", 150)


def get_valid_words(sentence):

    sentence_words = sentence.lower().strip().split()

    valid_words = []

    for word in sentence_words:

        if word in words:
            valid_words.append(word)

    return valid_words


text = input("Enter sentence: ")

valid_words = get_valid_words(text)

if valid_words:

    valid_sentence = " ".join(valid_words)

    print("Original:", text)
    print("Speaking:", valid_sentence)

    engine.say(valid_sentence)
    engine.runAndWait()

else:

    print("No valid words found.")
    print("TTS not started.")