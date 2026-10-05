import cv2
import mediapipe as mp
import joblib
from collections import Counter

model = joblib.load("../models/sign_model_robust.pkl")

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

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    prediction = ""

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        landmarks = []

        for landmark in hand.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        current_prediction = model.predict([landmarks])[0]

        prediction_history.append(current_prediction)

        if len(prediction_history) > 10:
            prediction_history.pop(0)

        prediction = Counter(prediction_history).most_common(1)[0][0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    else:
        prediction_history.clear()

    cv2.putText(
        frame,
        "Sign: " + prediction,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Sign Language AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()