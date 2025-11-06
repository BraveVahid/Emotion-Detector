from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.data_loader import get_data_generators
from src.model import build_model


def train_model(batch_size=64, epochs=100): 
    train_gen, val_gen, _, class_weights = get_data_generators(batch_size=batch_size)
    
    model = build_model()
    model.summary()
    
    checkpoint = ModelCheckpoint(
        filepath="model/emotion_detector_best.h5",
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
        
    model.load_weights("model/emotion_detector.h5")
    
    best_val_acc = max(history.history["val_accuracy"])
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1
    print(f"Best Val Acc:   {best_val_acc*100:.2f}% (Epoch {best_epoch})")
    
    model.save("model/emotion_detector.keras")
    
    return history, model

if __name__ == "__main__":
    history, model = train_model(batch_size=64, epochs=100)
