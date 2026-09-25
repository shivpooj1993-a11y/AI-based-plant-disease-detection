import os
import json

import numpy as np
import tensorflow as tf
import pytest


MODEL_PATH = "models/plant_disease_model.keras"

CLASS_FILE = "models/class_names.json"

IMAGE_SIZE = (128, 128)


# ---------------------------------------
# Fixture: Load model
# ---------------------------------------

@pytest.fixture
def model():

    if not os.path.exists(MODEL_PATH):

        pytest.fail(
            "Trained model does not exist."
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ---------------------------------------
# Test 1: Model loading
# ---------------------------------------

def test_model_loading(model):

    assert model is not None


# ---------------------------------------
# Test 2: Input size
# ---------------------------------------

def test_input_shape(model):

    assert model.input_shape[1:] == (
        128,
        128,
        3
    )


# ---------------------------------------
# Test 3: Class file
# ---------------------------------------

def test_class_file():

    assert os.path.exists(
        CLASS_FILE
    )


# ---------------------------------------
# Test 4: Prediction output
# ---------------------------------------

def test_prediction_output(model):

    test_image = np.random.rand(

        1,
        128,
        128,
        3

    ).astype(
        np.float32
    )


    prediction = model.predict(

        test_image,

        verbose=0
    )


    assert prediction.shape[0] == 1


# ---------------------------------------
# Test 5: Probability range
# ---------------------------------------

def test_probability_range(model):

    test_image = np.random.rand(

        1,
        128,
        128,
        3

    ).astype(
        np.float32
    )


    prediction = model.predict(

        test_image,

        verbose=0
    )


    assert np.all(
        prediction >= 0
    )

    assert np.all(
        prediction <= 1
    )


# ---------------------------------------
# Test 6: Probability sum
# ---------------------------------------

def test_probability_sum(model):

    test_image = np.random.rand(

        1,
        128,
        128,
        3

    ).astype(
        np.float32
    )


    prediction = model.predict(

        test_image,

        verbose=0
    )


    total = np.sum(
        prediction[0]
    )


    assert np.isclose(
        total,
        1.0,
        atol=0.01
    )
