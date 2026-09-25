import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

TEST_PATH = "dataset/test"

MODEL_PATH = "models/plant_disease_model.keras"

RESULT_PATH = "simulation/results"


os.makedirs(
    RESULT_PATH,
    exist_ok=True
)


# ---------------------------------------
# Load test dataset
# ---------------------------------------

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_PATH,

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


class_names = test_dataset.class_names


# ---------------------------------------
# Load trained model
# ---------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ---------------------------------------
# Evaluate model
# ---------------------------------------

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print("\n================================")
print("        TEST RESULTS")
print("================================")

print(
    "Test Loss     :",
    test_loss
)

print(
    "Test Accuracy : {:.2f}%".format(
        test_accuracy * 100
    )
)


# ---------------------------------------
# Generate predictions
# ---------------------------------------

actual_labels = []
predicted_labels = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted = np.argmax(
        predictions,
        axis=1
    )

    actual_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predicted
    )


# ---------------------------------------
# Classification report
# ---------------------------------------

print("\n================================")
print("   CLASSIFICATION REPORT")
print("================================")

report = classification_report(

    actual_labels,

    predicted_labels,

    target_names=class_names,

    zero_division=0
)

print(report)


# ---------------------------------------
# Confusion Matrix
# ---------------------------------------

matrix = confusion_matrix(

    actual_labels,

    predicted_labels
)


display = ConfusionMatrixDisplay(

    confusion_matrix=matrix,

    display_labels=class_names
)


fig, ax = plt.subplots(
    figsize=(9, 9)
)


display.plot(

    ax=ax,

    cmap="Blues",

    xticks_rotation=45
)


plt.title(
    "Plant Disease Detection Confusion Matrix"
)

plt.tight_layout()


plt.savefig(

    os.path.join(
        RESULT_PATH,
        "confusion_matrix.png"
    )
)


plt.close()


print(
    "\nConfusion matrix saved successfully."
)
