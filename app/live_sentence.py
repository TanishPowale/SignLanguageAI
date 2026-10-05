import cv2
import mediapipe as mp
import joblib
import pandas as pd
from collections import Counter
from wordfreq import top_n_list
import time
import pyttsx3

words = top_n_list("en", 10000)
dictionary = set(top_n_list("en", 100000))

engine = pyttsx3.init() 
engine.setProperty("rate", 150)
model = joblib.load("models/sign_model_robust.pkl")

words = top_n_list("en", 10000)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

prediction_history = []

sentence = ""

accepted_sign = ""
stable_start_time = 0
last_prediction = ""

current_confidence = 0
top_predictions = []


def get_suggestions(prefix, limit=5):

    prefix = prefix.lower().strip()

    if not prefix:
        return []

    return [
        word for word in words
        if word.startswith(prefix)
    ][:limit]

def speak_sentence(sentence):

    text_to_speak = sentence.strip()

    if text_to_speak:

        print("Speaking:", text_to_speak)

        engine.say(text_to_speak)
        engine.runAndWait()

    else:

        print("Nothing to speak.")
while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    prediction = ""
    current_confidence = 0
    top_predictions = []

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        landmarks = []

        for landmark in hand.landmark:

            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        points = pd.DataFrame(
            [landmarks]
        ).values.reshape(21, 3)

        wrist = points[0]

        points = points - wrist

        scale = abs(points).max()

        if scale != 0:
            points = points / scale

        normalized_landmarks = points.flatten()

        input_df = pd.DataFrame(
            [normalized_landmarks],
            columns=model.feature_names_in_
        )

        probabilities = model.predict_proba(
            input_df
        )[0]

        top_indices = probabilities.argsort()[-3:][::-1]

        top_predictions = [
            (
                model.classes_[i],
                probabilities[i] * 100
            )
            for i in top_indices
        ]

        current_prediction = top_predictions[0][0]

        current_confidence = top_predictions[0][1]

        prediction_history.append(
            current_prediction
        )

        if len(prediction_history) > 15:
            prediction_history.pop(0)

        prediction = Counter(
            prediction_history
        ).most_common(1)[0][0]

        stable_index = list(
            model.classes_
        ).index(prediction)

        stable_confidence = (
            probabilities[stable_index] * 100
        )

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        if prediction != last_prediction:

            stable_start_time = time.time()

            last_prediction = prediction

        stable_time = (
            time.time() - stable_start_time
        )

        if (
    stable_time >= 1.5
    and stable_confidence >= 20
):

            if prediction != accepted_sign:

                if (
                    len(prediction) == 1
                    and prediction.isalpha()
                ):

                    sentence += prediction

                    accepted_sign = prediction

                elif prediction == "space":

                    sentence += " "

                    accepted_sign = prediction

                elif prediction == "del":

                    sentence = sentence[:-1]

                    accepted_sign = prediction

    else:

        prediction_history.clear()

        accepted_sign = ""

        last_prediction = ""

        stable_start_time = time.time()

    # Current word
    current_word = sentence.split(" ")[-1]

    suggestions = get_suggestions(
        current_word
    )

    # ---------------- DISPLAY ----------------

    display_frame = cv2.flip(frame, 1)

    cv2.putText(
        display_frame,
        "Prediction: " + prediction,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        "Confidence: "
        + str(round(current_confidence, 1))
        + "%",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        "Text: " + sentence,
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        "Word Suggestions:",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    y = 195

    for i, word in enumerate(suggestions):

        cv2.putText(
            display_frame,
            str(i + 1) + ". " + word,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        y += 30

    cv2.putText(
        display_frame,
        "Press 1-5 to select word",
        (20, 355),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    # Top signs
    cv2.putText(
        display_frame,
        "Possible Signs:",
        (20, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    y = 420

    for sign, confidence in top_predictions:

        cv2.putText(
            display_frame,
            sign
            + " : "
            + str(round(confidence, 1))
            + "%",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        y += 25

        cv2.putText(
            frame,
            "B=Backspace  C=Clear  S=Speak  Q=Quit",
            (20, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    cv2.imshow(
        "Sign Language AI",
        display_frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord("q"):
        break

    # Clear
    elif key == ord("c"):

        sentence = ""
        accepted_sign = ""
        prediction_history.clear()
        last_prediction = ""
        stable_start_time = time.time()

    # Backspace
    elif key == ord("b"):

        if sentence:
            sentence = sentence[:-1]

        accepted_sign = ""
        prediction_history.clear()
        last_prediction = ""
        stable_start_time = time.time()

    # Speak valid words
    elif key == ord("s"):

        speak_sentence(sentence)

    # Select word suggestion using 1-5
    elif ord("1") <= key <= ord("5"):

        index = key - ord("1")

        if index < len(suggestions):

            selected_word = suggestions[index]

            words_in_sentence = sentence.split(" ")

            # Replace current incomplete word
            words_in_sentence[-1] = selected_word

            # Add automatic space
            sentence = " ".join(words_in_sentence) + " "

            # Reset recognition state
            accepted_sign = ""
            prediction_history.clear()
            last_prediction = ""
            stable_start_time = time.time()


cap.release()

cv2.destroyAllWindows()

hands.close()

print("Final sentence:", sentence)