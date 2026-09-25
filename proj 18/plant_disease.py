import os
import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ==============================
# SETTINGS
# ==============================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/validation"

MODEL_DIR = "models"
MODEL_FILE = "plant_disease_model.keras"
CLASS_FILE = "class_names.json"

os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================
# LOAD DATASET
# ==============================

print("Loading training dataset...")

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Loading validation dataset...")

valid_data = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==============================
# GET CLASS NAMES
# ==============================

class_names = train_data.class_names

print("\nDisease Classes:")

for i, name in enumerate(class_names):
    print(i, ":", name)


# Save class names

with open(
    os.path.join(MODEL_DIR, CLASS_FILE),
    "w"
) as f:
    json.dump(class_names, f)


# ==============================
# CNN MODEL
# ==============================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(128, 128, 3)
    ),

    # Normalize image
    tf.keras.layers.Rescaling(
        1.0 / 255
    ),

    # Convolution Layer 1
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Convolution Layer 2
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Convolution Layer 3
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    # Flatten
    tf.keras.layers.Flatten(),

    # Fully Connected Layer
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout
    tf.keras.layers.Dropout(0.5),

    # Output
    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# ==============================
# COMPILE MODEL
# ==============================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


model.summary()


# ==============================
# TRAIN MODEL
# ==============================

print("\nTraining started...\n")

history = model.fit(

    train_data,

    validation_data=valid_data,

    epochs=EPOCHS
)


# ==============================
# SAVE MODEL
# ==============================

model_path = os.path.join(
    MODEL_DIR,
    MODEL_FILE
)

model.save(model_path)

print("\nModel saved:")
print(model_path)


# ==============================
# DISPLAY ACCURACY
# ==============================

train_accuracy = history.history[
    "accuracy"
]

valid_accuracy = history.history[
    "val_accuracy"
]


print("\nFinal Training Accuracy:",
      train_accuracy[-1] * 100, "%")

print("Final Validation Accuracy:",
      valid_accuracy[-1] * 100, "%")


# ==============================
# ACCURACY GRAPH
# ==============================

plt.plot(
    train_accuracy,
    label="Training Accuracy"
)

plt.plot(
    valid_accuracy,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Plant Disease Detection Accuracy"
)

plt.legend()
plt.grid()

plt.savefig(
    "accuracy.png"
)

plt.show()


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_image(image_path):

    print("\nLoading image...")

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = (
        tf.keras.utils.img_to_array(
            image
        )
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )[0]


    # Find highest probability
    index = np.argmax(
        prediction
    )

    confidence = (
        prediction[index] * 100
    )


    disease = class_names[index]


    # ==============================
    # RESULT
    # ==============================

    print("\n==============================")
    print(" AI PLANT DISEASE DETECTION")
    print("==============================")

    print(
        "Image      :",
        image_path
    )

    print(
        "Prediction :",
        disease
    )

    print(
        "Confidence : {:.2f}%".format(
            confidence
        )
    )

    print("==============================")


# ==============================
# PREDICT NEW IMAGE
# ==============================

if len(sys.argv) > 1:

    image_file = sys.argv[1]

    if os.path.exists(image_file):

        predict_image(
            image_file
        )

    else:

        print(
            "Image not found:",
            image_file
        )
