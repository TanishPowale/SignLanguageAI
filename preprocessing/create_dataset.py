import cv2
import mediapipe as mp
import pandas as pd
import os

DATASET_PATH = "../dataset/asl_alphabet_train/asl_alphabet_train"
OUTPUT_FILE = "../dataset/landmark_dataset.csv"

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

data = []

for label in os.listdir(DATASET_PATH):

    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print("Processing:", label)

    for image_name in os.listdir(label_path):

        image_path = os.path.join(label_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            landmarks = []

            for landmark in hand.landmark:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            landmarks.append(label)

            data.append(landmarks)

df = pd.DataFrame(data)

df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset created successfully!")
print("Total samples:", len(df))
print("Saved as:", OUTPUT_FILE)

hands.close()