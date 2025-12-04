import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_ROOT = r"E:\school\vscode_projects\outfitGenerator\Deepfashion_dataset"

# CSV is directly inside Deepfashion_dataset now
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
IMG_ROOT = os.path.join(DATA_ROOT)

# pick a few categories so the task is manageable
SELECTED_CATEGORIES = ["Tee", "Blazer", "Cardigan", "Tank", "Hoodie"]
MAX_PER_CLASS = 750
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15


def load_metadata():
    """
    Load train_labels.csv, keep only selected categories, and
    sample up to MAX_PER_CLASS examples per category.
    """
    df = pd.read_csv(LABELS_CSV)

    # keep only the categories we’re interested in
    df = df[df["category_name"].isin(SELECTED_CATEGORIES)]

    # sample up to MAX_PER_CLASS per category
    subsets = []
    for cat, group in df.groupby("category_name"):
        n = min(MAX_PER_CLASS, len(group))
        subsets.append(group.sample(n=n, random_state=42))
    df_small = pd.concat(subsets).reset_index(drop=True)

    print("Using categories:", df_small["category_name"].unique())
    print("Total images after shrinking:", len(df_small))

    return df_small


def load_images_and_labels(df):
    """
    Load images into memory as numpy arrays and encode labels.
    """
    images = []
    labels = []

    for _, row in df.iterrows():
        rel_path = row["image_name"].replace("\\", "/")
        img_path = os.path.join(IMG_ROOT, rel_path)

        if not os.path.exists(img_path):
            # just skip missing files
            continue

        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(row["category_name"])

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    print("Loaded image tensor:", images.shape)

    # encode labels to integers 0..num_classes-1
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = list(le.classes_)
    print("Class names:", class_names)

    return images, y, class_names


def build_model(num_classes):
    """
    Very simple CNN. You can upgrade this to a pre-trained model later.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(*IMG_SIZE, 3)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def train_and_save():
    df = load_metadata()
    X, y, class_names = load_images_and_labels(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(num_classes=len(class_names))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # save model and class names so the app can load them
    model.save("shirt_model.h5")
    np.save("class_names.npy", np.array(class_names))

    print("Model and class names saved: shirt_model.h5, class_names.npy")


if __name__ == "__main__":
    train_and_save()