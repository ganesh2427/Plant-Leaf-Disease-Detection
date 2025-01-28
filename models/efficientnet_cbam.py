import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tensorflow.keras.layers import Lambda,Multiply,Reshape,Multiply,Concatenate,Conv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D,Dense,GlobalMaxPooling2D

training_set = tf.keras.utils.image_dataset_from_directory(
    'train',
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
    'valid',
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



def build_efficientnet_b0_cbam(input_shape=(227, 227, 3), num_classes=38):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Stem
    x = Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Blocks
    x = efficientnet_block_cbam(x, 1, 16, 1, 1)
    x = efficientnet_block_cbam(x, 6, 24, 2, 2)  # MBConv1
    x = efficientnet_block_cbam(x, 6, 40, 2, 2)  # MBConv2
    x = efficientnet_block_cbam(x, 6, 80, 3, 2)  # MBConv3
    x = efficientnet_block_cbam(x, 6, 112, 3, 1) # MBConv4
    x = efficientnet_block_cbam(x, 6, 192, 4, 2) # MBConv5
    x = efficientnet_block_cbam(x, 6, 320, 1, 1) # MBConv6

    # Head
    x = Conv2D(1280, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # Output
    outputs = Dense(39, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def efficientnet_block_cbam(inputs, num_repeat, out_channels, expand_ratio, strides):
    x = mb_conv_block_cbam(inputs, out_channels, expand_ratio, strides)
    for _ in range(1, num_repeat):
        x = mb_conv_block_cbam(x, out_channels, expand_ratio, 1)
    return x

def mb_conv_block_cbam(inputs, out_channels, expand_ratio, strides):
    input_channels = inputs.shape[-1]

    # Expansion phase
    expanded_channels = input_channels * expand_ratio
    x = Conv2D(expanded_channels, 1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    # Depthwise convolution
    x = Conv2D(expanded_channels, 3, strides=strides, padding='same', groups=expanded_channels)(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    # CBAM module
    x = cbam_block(x)

    # Projection phase
    x = Conv2D(out_channels, 1, padding='same')(x)
    x = BatchNormalization()(x)

    # Skip connection if the input and output shapes are the same (identity)
    if strides == 1 and input_channels == out_channels:
        x = tf.keras.layers.Add()([inputs, x])
    return x

def cbam_block(inputs):
    # Channel attention module
    avg_pool = GlobalAveragePooling2D()(inputs)
    max_pool = GlobalMaxPooling2D()(inputs)
    channel_features = Concatenate()([avg_pool, max_pool])
    channel_features = Dense(inputs.shape[-1] // 2, activation='relu')(channel_features)
    channel_att = Dense(inputs.shape[-1], activation='sigmoid')(channel_features)
    channel_att = Reshape((1, 1, inputs.shape[-1]))(channel_att)
    channel_att = Multiply()([inputs, channel_att])

    # Spatial attention module
    avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(channel_att)
    max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(channel_att)
    spatial_features = Concatenate()([avg_pool, max_pool])
    spatial_att = Conv2D(1, 3, padding='same', activation='sigmoid')(spatial_features)
    spatial_att = Multiply()([channel_att, spatial_att])

    return spatial_att

# Build EfficientNetB0 with CBAM
model_cbam = build_efficientnet_b0_cbam()



model_cbam.compile(optimizer=tf.keras.optimizers.legacy.Adam(
    learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

model_cbam.summary()

training_history = model_cbam.fit(x=training_set,validation_data=validation_set,epochs=10)

#Training set Accuracy
train_loss, train_acc = model_cbam.evaluate(training_set)
print('Training accuracy:', train_acc)

#Validation set Accuracy
val_loss, val_acc = model_cbam.evaluate(validation_set)
print('Validation accuracy:', val_acc)

model_cbam.save('efficientnet_cbam.keras')



