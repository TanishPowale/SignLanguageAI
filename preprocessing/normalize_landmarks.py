import pandas as pd

df = pd.read_csv("../dataset/landmark_dataset.csv")

features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

normalized_data = []

for row, label in zip(features.values, labels):

    landmarks = row.reshape(21, 3)

    wrist = landmarks[0]

    # Move wrist to origin
    landmarks = landmarks - wrist

    # Normalize scale
    max_distance = abs(landmarks).max()

    if max_distance != 0:
        landmarks = landmarks / max_distance

    normalized_data.append(
        list(landmarks.flatten()) + [label]
    )

normalized_df = pd.DataFrame(normalized_data)

normalized_df.to_csv(
    "../dataset/normalized_landmark_dataset.csv",
    index=False
)

print("Normalization completed!")
print("Total samples:", len(normalized_df))
print("Saved as: normalized_landmark_dataset.csv")