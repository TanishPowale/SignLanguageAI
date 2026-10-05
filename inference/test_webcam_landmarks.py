import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        landmarks = []

        for landmark in hand.landmark:
            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        points = []

        for i in range(0, len(landmarks), 3):
            points.append([
                landmarks[i],
                landmarks[i + 1],
                landmarks[i + 2]
            ])

        points = __import__("numpy").array(points)

        wrist = points[0]

        points = points - wrist

        scale = abs(points).max()

        if scale != 0:
            points = points / scale

        print("Normalized wrist:", points[0])
        print("Normalized first finger point:", points[1])

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    cv2.imshow("Webcam Landmark Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()