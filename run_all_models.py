"""Train all required acoustic models for the DNN Speech Recognizer project."""

import gc
import math
import os
import pickle

from keras import backend as K

from sample_models import (
    bidirectional_rnn_model,
    cnn_rnn_model,
    deep_rnn_model,
    final_model,
    rnn_model,
    simple_rnn_model,
)
from train_utils import train_model


EPOCHS = 20
TRAIN_JSON = "train_corpus_project.json"
VALID_JSON = "valid_corpus_project.json"
MINIBATCH_SIZE = 4
MAX_DURATION = 10.0
FIT_TRAIN_SAMPLES = 5


def has_completed_history(pickle_name):
    path = os.path.join("results", pickle_name)
    if not os.path.exists(path):
        return False
    with open(path, "rb") as history_file:
        history = pickle.load(history_file)
    losses = list(history.get("loss", [])) + list(history.get("val_loss", []))
    has_required_epochs = (
        len(history.get("loss", [])) >= EPOCHS
        and len(history.get("val_loss", [])) >= EPOCHS
    )
    return has_required_epochs and all(math.isfinite(float(value)) for value in losses)


def train_if_needed(model_name, model_factory):
    pickle_name = f"{model_name}.pickle"
    weights_name = f"{model_name}.h5"
    weights_path = os.path.join("results", weights_name)

    if has_completed_history(pickle_name) and os.path.exists(weights_path):
        print(f"Skipping {model_name}: completed artifacts already exist.")
        return

    print("=" * 80)
    print(f"Training {model_name} for {EPOCHS} epochs")
    print("=" * 80)
    model = model_factory()
    train_model(
        input_to_softmax=model,
        pickle_path=pickle_name,
        save_model_path=weights_name,
        train_json=TRAIN_JSON,
        valid_json=VALID_JSON,
        minibatch_size=MINIBATCH_SIZE,
        spectrogram=True,
        epochs=EPOCHS,
        verbose=2,
        sort_by_duration=True,
        max_duration=MAX_DURATION,
        fit_train_samples=FIT_TRAIN_SAMPLES,
    )
    K.clear_session()
    gc.collect()


def main():
    os.makedirs("results", exist_ok=True)
    train_if_needed("model_0", lambda: simple_rnn_model(input_dim=161))
    train_if_needed("model_1", lambda: rnn_model(input_dim=161, units=200, activation="relu"))
    train_if_needed(
        "model_2",
        lambda: cnn_rnn_model(
            input_dim=161,
            filters=200,
            kernel_size=11,
            conv_stride=2,
            conv_border_mode="valid",
            units=200,
            activation="tanh",
        ),
    )
    train_if_needed("model_3", lambda: deep_rnn_model(input_dim=161, units=200, recur_layers=2))
    train_if_needed("model_4", lambda: bidirectional_rnn_model(input_dim=161, units=200))
    train_if_needed("model_end", lambda: final_model())
    print("All required model artifacts are present.")


if __name__ == "__main__":
    main()
