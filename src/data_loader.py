from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        "data/fer2013/train",
        target_size=(48,48),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical"
    )

    test_generator = test_datagen.flow_from_directory(
        "data/fer2013/test",
        target_size=(48,48),
        color_mode="grayscale",
        batch_size=batch_size,
        class_mode="categorical"
    )

    return train_generator, test_generator
