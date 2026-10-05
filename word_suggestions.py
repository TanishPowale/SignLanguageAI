from wordfreq import top_n_list


# Get 10,000 common English words
words = top_n_list("en", 10000)


def get_suggestions(prefix, limit=5):

    prefix = prefix.lower().strip()

    if not prefix:
        return []

    matches = [
        word for word in words
        if word.startswith(prefix)
    ]

    return matches[:limit]


while True:

    text = input("Enter letters: ").strip()

    if text.lower() == "exit":
        break

    suggestions = get_suggestions(text)

    print("Suggestions:", suggestions)