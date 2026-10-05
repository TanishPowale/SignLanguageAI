import pandas as pd
import numpy as np

df = pd.read_csv("../dataset/normalized_landmark_dataset.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

augmented_X = []
augmented_y = []

for landmarks, label in zip(X, y):

    points = landmarks.reshape(21, 3)

    # Original
    augmented_X.append(points.flatten())
    augmented_y.append(label)

    # Variation 1: small noise
    noise = np.random.normal(0, 0.01, points.shape)
    new_points = points + noise

    augmented_X.append(new_points.flatten())
    augmented_y.append(label)

    # Variation 2: small scale change
    scale = np.random.uniform(0.95, 1.05)
    new_points = points * scale

    augmented_X.append(new_points.flatten())
    augmented_y.append(label)

    # Variation 3: small rotation around Z axis
    angle = np.random.uniform(-10, 10) * np.pi / 180

    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])

    new_points = points @ rotation.T

    augmented_X.append(new_points.flatten())
    augmented_y.append(label)

augmented_df = pd.DataFrame(augmented_X)
augmented_df["label"] = augmented_y

augmented_df.to_csv(
    "../dataset/augmented_landmark_dataset.csv",
    index=False
)

print("Augmentation completed!")
print("Original samples:", len(df))
print("Augmented samples:", len(augmented_df))
print("Saved as: augmented_landmark_dataset.csv")