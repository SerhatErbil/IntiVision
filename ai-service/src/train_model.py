import tensorflow as tf

from config import IMAGE_WIDTH, IMAGE_HEIGHT, GESTURE_CLASSES
from preprocess import load_datasets

def build_model():
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(GESTURE_CLASSES), activation="softmax"))    
    
    return model

model = build_model()
model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]   
)
train_dataset, validation_dataset = load_datasets()
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
)
model.save("models/intivision_v1.keras")