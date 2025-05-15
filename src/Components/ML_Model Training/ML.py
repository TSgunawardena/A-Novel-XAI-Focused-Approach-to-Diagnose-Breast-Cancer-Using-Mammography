import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Paths to your dataset
train_dir = "/content/drive/MyDrive/FYP Documentation/dataset/train"
val_dir = "/content/drive/MyDrive/FYP Documentation/dataset/val"

# ✅ Data Generators
img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    rotation_range=10
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# ✅ Print label mapping
print("Class indices (label mapping):", train_generator.class_indices)

# ✅ Compute class weights
labels = train_generator.classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))
print("Computed class weights:", class_weights)

# ✅ Create CNN + ViT Model
def create_upgraded_mammo_cnn_vit_model(input_shape=(224, 224, 1)):
    inputs = layers.Input(shape=input_shape)

    # CNN Layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Patch Projection
    patch_size = 4
    projection_dim = 64
    x = layers.Conv2D(filters=projection_dim, kernel_size=(patch_size, patch_size), strides=(patch_size, patch_size))(x)
    x = layers.Reshape((-1, projection_dim))(x)
    num_patches = x.shape[1]

    # Positional Encoding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    x = x + pos_embed

    # Transformer Block
    for _ in range(5):
        x1 = layers.LayerNormalization()(x)
        attention = layers.MultiHeadAttention(num_heads=2, key_dim=projection_dim)(x1, x1)
        x2 = layers.Add()([x, attention])
        x3 = layers.LayerNormalization()(x2)
        mlp = layers.Dense(128, activation="gelu")(x3)
        mlp = layers.Dense(64, activation="gelu")(mlp)
        x = layers.Add()([x2, mlp])

    # Output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)

# ✅ Build and compile the model
model = create_upgraded_mammo_cnn_vit_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ✅ Callbacks
model_path = "/content/drive/MyDrive/best_mammo_model.keras"
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True)

# ✅ Train with class weights
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights
)

# ✅ Evaluate model
val_preds = model.predict(val_generator)
y_true = val_generator.classes
y_pred = (val_preds > 0.5).astype("int32").flatten()

# ✅ Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))

# ✅ Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ✅ ROC curve
fpr, tpr, _ = roc_curve(y_true, val_preds)
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ✅ Plot Training & Validation Accuracy and Loss
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

