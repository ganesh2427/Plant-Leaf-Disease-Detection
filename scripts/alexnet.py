import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns




training_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/gk/Desktop/data/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=64,
    image_size=(227,227),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)




validation_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/gk/Desktop/data/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=64,
    image_size=(227,227),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)




cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

cnn.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))

cnn.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))

cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(4096, activation='relu'))

cnn.add(tf.keras.layers.Dense(4096, activation='relu'))

cnn.add(tf.keras.layers.Dense(39, activation='softmax'))

# Use the standard `tf.keras.optimizers.Adam`
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn.summary()


training_history = cnn.fit(x=training_set,validation_data=validation_set,epochs=5)

#Training set Accuracy
train_loss, train_acc = cnn.evaluate(training_set)
print('Training accuracy:', train_acc)




#Validation set Accuracy
val_loss, val_acc = cnn.evaluate(validation_set)
print('Validation accuracy:', val_acc)





cnn.save('alexnet_5.keras')








import tensorflow as tf

# Load the saved model
cnn = tf.keras.models.load_model('alexnet_5.keras')

# Load the test dataset
test_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/gk/Desktop/data/test',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=64,
    image_size=(227, 227),
    shuffle=False,  # No need to shuffle for testing
)

# Evaluate the model on the test dataset
test_loss, test_acc = cnn.evaluate(test_set)
print('Test accuracy:', test_acc)























