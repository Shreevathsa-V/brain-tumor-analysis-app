import tensorflow as tf
import numpy as np
import os
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Augmentation Pipeline ---
# This creates a set of random transformations to apply to the training images.
# This makes the model more robust and improves accuracy.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# --- Model Definition (Must match app.py) ---
# This function defines the robust U-Net architecture.
def build_unet_model(input_shape, n_classes):
    # Add the input layer
    inputs = tf.keras.layers.Input(input_shape)
    
    # NEW: Apply data augmentation only during training
    x = data_augmentation(inputs)

    # Encoder Path
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c_class = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    
    # Classifier Head
    gap = tf.keras.layers.GlobalAveragePooling2D()(c_class)
    d1 = tf.keras.layers.Dense(128, activation='relu')(gap)
    d1 = tf.keras.layers.Dropout(0.5)(d1)
    classification_output = tf.keras.layers.Dense(n_classes, activation='softmax', name='classification_output')(d1)

    # Decoder Path (for Segmentation)
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c_class)
    u6 = tf.keras.layers.concatenate([u6, c2])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.1)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c1])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.1)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    segmentation_output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_output')(c7)

    # The model has two outputs
    model = tf.keras.Model(inputs=[inputs], outputs=[segmentation_output, classification_output])
    
    return model

# --- Main Training Function ---
def main():
    # --- Parameters ---
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 16
    # Increased epochs for better training with augmentation
    EPOCHS = 50 
    N_CLASSES = 4
    DATA_DIR = './data/Training'
    MODEL_SAVE_PATH = 'brain_tumor_model.h5'

    # --- Create Data Generators ---
    logging.info("Creating training and validation data generators...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )
    
    class_names = train_ds.class_names
    logging.info(f"Found classes: {class_names}")

    # --- Preprocessing Function ---
    def preprocess_data(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        dummy_masks = tf.zeros_like(images)[..., :1]
        return images, {'segmentation_output': dummy_masks, 'classification_output': labels}

    train_ds = train_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

    # --- Optimize Datasets for Performance ---
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # --- Build and Compile Model ---
    model = build_unet_model((*IMAGE_SIZE, 3), N_CLASSES)
    
    losses = {
        'segmentation_output': 'binary_crossentropy',
        'classification_output': 'categorical_crossentropy'
    }
    loss_weights = {
        'segmentation_output': 0.5,
        'classification_output': 0.5
    }
    
    model.compile(optimizer='adam',
                  loss=losses,
                  loss_weights=loss_weights,
                  metrics={'classification_output': 'accuracy'})

    model.summary()

    # --- Callbacks ---
    checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, 
                                                    monitor='val_classification_output_accuracy', 
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    mode='max')
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_classification_output_accuracy', 
                                                      patience=10, # Increased patience for longer training
                                                      verbose=1, 
                                                      mode='max')

    # --- Train the Model ---
    logging.info("Starting model training with data augmentation...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping]
    )

    logging.info(f"Training complete. Best model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
