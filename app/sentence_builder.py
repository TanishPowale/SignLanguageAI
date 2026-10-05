sentence = ""

print("Sign Language Sentence Builder")
print("Enter letters one by one.")
print("Type SPACE to add a space.")
print("Type DEL to delete the last letter.")
print("Type CLEAR to clear the sentence.")
print("Type EXIT to stop.")

while True:

    sign = input("\nEnter sign: ").strip().upper()

    if sign == "EXIT":
        break

    elif sign == "SPACE":
        sentence += " "

    elif sign == "DEL":
        sentence = sentence[:-1]

    elif sign == "CLEAR":
        sentence = ""

    elif len(sign) == 1 and sign.isalpha():
        sentence += sign

    print("Sentence:", sentence)