from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def get_data_generators(batch_size=64, img_size=(48, 48)):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.15
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.15
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        "data/fer2013/train",
        target_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = val_test_datagen.flow_from_directory(
        "data/fer2013/train",
        target_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        "data/fer2013/test",
        target_size=img_size,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=False
    )

    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))

    return train_gen, val_gen, test_gen, class_weight_dict
