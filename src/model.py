from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_model(input_shape=(48, 48, 1), n_classes=7):
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Conv2D(512, (3, 3), padding="same", kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation("relu"),
        Conv2D(512, (3, 3), padding="same", kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.5),

        Flatten(),
        
        Dense(1024, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.5),
        
        Dense(512, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.5),

        Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
