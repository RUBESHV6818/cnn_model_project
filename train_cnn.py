import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

# ========================
# 1️⃣ Data Augmentation
# ========================
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset/",
    target_size=(224, 224),  # MobileNetV2 expects 224x224 images
    batch_size=4,
    class_mode='binary',
    subset='training'
)

val_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)
val_data = val_datagen.flow_from_directory(
    "dataset/",
    target_size=(224, 224),
    batch_size=4,
    class_mode='binary',
    subset='validation'
)

# ========================
# 2️⃣ Load Pretrained MobileNetV2
# ========================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze pretrained layers

# ========================
# 3️⃣ Build the model
# ========================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ========================
# 4️⃣ Train the model
# ========================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10  # fewer epochs are enough for small dataset
)

# ========================
# 5️⃣ Save the model
# ========================
model.save("cat_dog_mobilenet.keras")
print("Model trained & saved successfully!")

# ========================
# 6️⃣ Predict & Display Test Images
# ========================
test_folder = "./test_images"
model = tf.keras.models.load_model("cat_dog_mobilenet.keras")

img_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
num_images = len(img_files)

fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
if num_images == 1:
    axes = [axes]

for ax, img_file in zip(axes, img_files):
    img_path = os.path.join(test_folder, img_file)
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    pred = model.predict(img_array, verbose=0)[0][0]
    if pred > 0.5:
        label = "Dog"
        confidence = pred * 100
    else:
        label = "Cat"
        confidence = (1 - pred) * 100

    ax.imshow(img)
    ax.set_title(f"{label} ({confidence:.2f}%)")
    ax.axis('off')

plt.tight_layout()
plt.show()
