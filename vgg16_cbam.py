import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Concatenate, Multiply

def Channel_attention_module(x, ratio=8):  
    b, _, _, channel = x.shape
    l1 = Dense(channel // ratio, activation="relu", use_bias=False)
    l2 = Dense(channel, use_bias=False)
    X1 = GlobalAveragePooling2D()(x)
    X1 = l1(X1)
    X1 = l2(X1)
    X2 = GlobalMaxPooling2D()(x)
    X2 = l1(X2)
    X2 = l2(X2)
    features = X1 + X2
    features = Activation("sigmoid")(features)
    features = Multiply()([x, features])
    return features

def spatial_attention_module(x):
    X1 = tf.reduce_mean(x, axis=-1)
    X1 = tf.expand_dims(X1, axis=-1)
    X2 = tf.reduce_max(x, axis=-1)
    X2 = tf.expand_dims(X2, axis=-1)
    features = Concatenate()([X1, X2])
    features = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(features)
    features = Multiply()([x, features])
    return features

def cbam(x):
    x = Channel_attention_module(x)
    x = spatial_attention_module(x)
    return x

def vgg16_cbam(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = cbam(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = cbam(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = cbam(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = cbam(x)

    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = cbam(x)

    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=64,
    image_size=(224,224),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)


validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=64,
    image_size=(224,224),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)


input_shape = (224, 224, 3)   
num_classes = 39  
model = vgg16_cbam(input_shape, num_classes)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(
    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

training_history = model.fit(x=training_set,validation_data=validation_set,epochs=10)

#Training set Accuracy
train_loss, train_acc = model.evaluate(training_set)
print('Training accuracy:', train_acc)

#Validation set Accuracy
val_loss, val_acc = model.evaluate(validation_set)
print('Validation accuracy:', val_acc)

model.save('vgg16_cbam.keras')


