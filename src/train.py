import pandas as pd
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint, 
    ReduceLROnPlateau
)
from src.data_loader import get_data_generators
from src.model import build_model

def save_training_history(history):
    df = pd.DataFrame(history.history)
    df["epoch"] = range(1, len(df) + 1)

    df.to_json("model/training_history.json", orient="records", indent=4)
    df.to_csv("model/training_history.csv", index=False)
    df.to_pickle("model/training_history.pkl")
    
def train_model(batch_size=64, epochs=100):
    train_gen, val_gen, _, class_weights = get_data_generators(batch_size=batch_size)

    model = build_model()

    checkpoint = ModelCheckpoint(
        filepath="model/emotion_recognition.keras",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=15,
        restore_best_weights=True,
        verbose=1,
        mode="max"
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
        mode="min"
    )

    callbacks = [checkpoint, early_stop, reduce_lr]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    best_val_acc = max(history.history["val_accuracy"])
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1
    print(f"Best Val Acc: {best_val_acc*100:.2f}% (Epoch {best_epoch})")

    save_training_history(history)

if __name__ == "__main__":
    train_model(batch_size=64, epochs=100)
