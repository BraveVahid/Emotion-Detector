from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(batch_size=32, img_size=(48,48)):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.1
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        "data/fer2013/train",
        target_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="sparse",
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        "data/fer2013/train",
        target_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        "data/fer2013/test",
        target_size=img_size,
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False
    )

    return train_generator, val_generator, test_generator, train_generator.class_indices
