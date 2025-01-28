import tensorflow as tf
from tensorflow.keras.layers import Reshape,Dense,Input,GlobalMaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Activation,Concatenate,Conv2D,Multiply

def Channel_attention_module(x,ratio=8):  #x is input feature map
    b,_,_,channel = x.shape
    ##shared layers
    l1 = Dense(channel//ratio,activation="relu",use_bias=False)
    l2 = Dense(channel,use_bias=False)
    
    ##Global average pooling
    X1=GlobalAveragePooling2D()(x)
    X1=l1(X1)
    X1=l2(X1)

    ##Global max pooling
    X2=GlobalAveragePooling2D()(x)
    X2=l1(X2)
    X2=l2(X2)
    
    ##add both and apply sigmoid
    features = X1+X2
    features = Activation("sigmoid")(features)
    features = Multiply()([x,features])

    return features
    

def spatial_attention_module(x):
    ##average pooling
    X1 = tf.reduce_mean(x,axis=-1)
    X1 = tf.expand_dims(X1,axis=-1)

    ##max pooling
    X2 = tf.reduce_max(x,axis=-1)
    X2 = tf.expand_dims(X2,axis=-1)
    
    ##concatenate
    features = Concatenate()([X1,X2])

    #conv layer
    features = Conv2D(1,kernel_size=7,padding='same',activation='sigmoid')(features)
    features = Multiply()([x,features])

    return features


def cbam(x):
    x = Channel_attention_module(x)
    x = spatial_attention_module(x)
    return x

inputs = Input(shape=(128,128,32))
y = cbam(inputs)
print(y.shape)


