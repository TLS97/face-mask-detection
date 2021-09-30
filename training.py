# Import libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize constants
INIT_LR = 0.001
EPOCHS = 30
BATCH_SIZE = 32

DATADIR = "Dataset/"
CLASSES = ["Masked", "No_Mask", "Incorrect_Mask"]

data = []
labels = []

print("STATUS: Loading images...")
# Loop through the directories and store path to all images in an array (image_array)
for class_ in CLASSES:
    path = os.path.join(DATADIR, class_)
    print(f"Path: {path}")
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(class_)


# Encode the labels
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Convert lists of data and labels to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split data into training and testing data
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Data Augmentation
data_aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Loading MobileNetV2
base_model = MobileNetV2(weights="imagenet", input_tensor=Input(shape=(224, 224, 3)))

# Remove last layer of MobileNetV2 and create a new output layer classifying into 3 classes
output = base_model.layers[-1].output
output = Dense(units=3, activation="softmax")(output)

model = Model(inputs=base_model.input, outputs=output)

model.summary()

# Loop over all the layers and freeze all except the last 5 layers
for layer in base_model.layers[:-15]:
    layer.trainable = False

# Compiling the model
optimizer = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)

print("STATUS: Compiling model...")
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics="accuracy",
)

# Train the model
print("STATUS: Training the model...")
history = model.fit(
    data_aug.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    validation_data=(x_test, y_test),
    validation_steps=len(x_test) // BATCH_SIZE,
    epochs=EPOCHS,
)

# Evaluate the model
print("STATUS: Evaluating the model...")
prediction_idx = model.predict(x_test, batch_size=BATCH_SIZE)

# Find index of the label with corresponding largest predicted probability
prediction_idx = np.argmax(prediction_idx, axis=1)

# Visualize classification report
print(classification_report(y_test.argmax(axis=1), prediction_idx, target_names=lb.classes_))

# Save the model
print("STATUS: Saving the model...")
model.save("classifier_model.model", save_format="h5")

# Plot training and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")