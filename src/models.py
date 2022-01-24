import tensorflow as tf
from tensorflow.keras import layers

class Mask_G(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=3, strides=2, padding='same'):
    super(Mask_G, self).__init__()

    initializer = tf.random_normal_initializer(0., 0.02)
    self.strides = strides
    self.kernel_size = kernel_size
    self.padding = padding

    self.conv2_1 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.conv2_2 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn2 = layers.BatchNormalization()
    self.conv2_3 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn3 = layers.BatchNormalization()
    self.conv2_4 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn4 = layers.BatchNormalization()
    self.conv2_5 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn5 = layers.BatchNormalization()
    
    self.convt2_5 = layers.Conv2DTranspose(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.convt2_4 = layers.Conv2DTranspose(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.convt2_3 = layers.Conv2DTranspose(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.convt2_2 = layers.Conv2DTranspose(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.convt2_1 = layers.Conv2DTranspose(3, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)

  def call(self, inputs, training=False):
    # encoder
    enc_1 = self.conv2_1(inputs)
    enc_1 = tf.nn.leaky_relu(enc_1)
    enc_2 = self.conv2_2(enc_1)
    enc_2 = self.bn2(enc_2, training=training)
    enc_2 = tf.nn.leaky_relu(enc_2)
    enc_3 = self.conv2_3(enc_2)
    enc_3 = self.bn3(enc_3, training=training)
    enc_3 = tf.nn.leaky_relu(enc_3)
    enc_4 = self.conv2_4(enc_3)
    enc_4 = self.bn4(enc_4, training=training)
    enc_4 = tf.nn.leaky_relu(enc_4)
    enc_5 = self.conv2_5(enc_4)
    enc_5 = self.bn5(enc_5, training=training)
    enc_5 = tf.nn.leaky_relu(enc_5)
    
    # decoder
    dec_5 = self.convt2_5(enc_5)
    dec_5 = tf.nn.relu(dec_5)
    cat4 = layers.Concatenate()([dec_5, enc_4])
    dec_4 = self.convt2_4(cat4)
    dec_4 = tf.nn.relu(dec_4)
    cat3 = layers.Concatenate()([dec_4, enc_3])
    dec_3 = self.convt2_3(cat3)
    dec_3 = tf.nn.relu(dec_3)
    cat2 = layers.Concatenate()([dec_3, enc_2])
    dec_2 = self.convt2_2(cat2)
    dec_2 = tf.nn.relu(dec_2)
    cat1 = layers.Concatenate()([dec_2, enc_1])
    
    out = self.convt2_1(cat1)

    return tf.keras.activations.tanh(out)

class Face_G(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=3, strides=2, padding='same'):
    super(Face_G, self).__init__()

    initializer = tf.random_normal_initializer(0., 0.02)
    self.strides = strides
    self.kernel_size = kernel_size
    self.padding = padding

    self.conv2_1 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.conv2_2 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn2 = layers.BatchNormalization()
    self.conv2_3 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn3 = layers.BatchNormalization()
    self.conv2_4 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn4 = layers.BatchNormalization()
    self.conv2_5 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn5 = layers.BatchNormalization()
    self.conv2_6 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn6 = layers.BatchNormalization()
    
    self.convt2_6 = layers.Conv2DTranspose(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.drop2_6 = layers.Dropout(0.5)
    self.convt2_5 = layers.Conv2DTranspose(filters*8, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.drop2_5 = layers.Dropout(0.5)
    self.convt2_4 = layers.Conv2DTranspose(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.convt2_3 = layers.Conv2DTranspose(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.convt2_2 = layers.Conv2DTranspose(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.convt2_1 = layers.Conv2DTranspose(3, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)

  def call(self, inputs, training=False):
    #encoder
    enc_1 = self.conv2_1(inputs)
    enc_1 = tf.nn.leaky_relu(enc_1)
    enc_2 = self.conv2_2(enc_1)
    enc_2 = self.bn2(enc_2, training=training)
    enc_2 = tf.nn.leaky_relu(enc_2)
    enc_3 = self.conv2_3(enc_2)
    enc_3 = self.bn3(enc_3, training=training)
    enc_3 = tf.nn.leaky_relu(enc_3)
    enc_4 = self.conv2_4(enc_3)
    enc_4 = self.bn4(enc_4, training=training)
    enc_4 = tf.nn.leaky_relu(enc_4)
    enc_5 = self.conv2_5(enc_4)
    enc_5 = self.bn5(enc_5, training=training)
    enc_5 = tf.nn.leaky_relu(enc_5)
    enc_6 = self.conv2_6(enc_5)
    enc_6 = self.bn6(enc_6, training=training)
    enc_6 = tf.nn.leaky_relu(enc_6)

    #decoder
    dec_6 = self.convt2_6(enc_6)
    dec_6 = tf.nn.relu(dec_6)
    dec_6 = self.drop2_6(dec_6, training=training)
    cat5 = layers.Concatenate()([dec_6, enc_5])
    dec_5 = self.convt2_5(cat5)
    dec_5 = tf.nn.relu(dec_5)
    dec_5 = self.drop2_5(dec_5, training=training)
    cat4 = layers.Concatenate()([dec_5, enc_4])
    dec_4 = self.convt2_4(cat4)
    dec_4 = tf.nn.relu(dec_4)
    cat3 = layers.Concatenate()([dec_4, enc_3])
    dec_3 = self.convt2_3(cat3)
    dec_3 = tf.nn.relu(dec_3)
    cat2 = layers.Concatenate()([dec_3, enc_2])
    dec_2 = self.convt2_2(cat2)
    dec_2 = tf.nn.relu(dec_2)
    cat1 = layers.Concatenate()([dec_2, enc_1])
    
    out = self.convt2_1(cat1)

    return tf.keras.activations.tanh(out)

class Face_D(tf.keras.Model):
  def __init__(self, filters=64, kernel_size=4, strides=2, padding='same'):
    super(Face_D, self).__init__()

    initializer = tf.random_normal_initializer(0., 0.02)
    
    self.filters = filters
    self.strides = strides
    self.kernel_size = kernel_size
    self.padding = padding

    self.conv2_1 = layers.Conv2D(filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.conv2_2 = layers.Conv2D(filters*2, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn2 = layers.BatchNormalization()
    self.conv2_3 = layers.Conv2D(filters*4, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding, kernel_initializer=initializer, use_bias=False)
    self.bn3 = layers.BatchNormalization()
    self.zero_pad1 = layers.ZeroPadding2D()
    self.conv2_4 = layers.Conv2D(filters*8, kernel_size=self.kernel_size, strides=1, kernel_initializer=initializer, use_bias=False)
    self.bn4 = layers.BatchNormalization()
    self.zero_pad2 = layers.ZeroPadding2D()
    self.conv2_5 = layers.Conv2D(1, kernel_size=self.kernel_size, strides=1, kernel_initializer=initializer, use_bias=False)

  def call(self, inputs, training=False):
    input = inputs[0]
    target = inputs[1]
    x = layers.concatenate([input, target])
    x = self.conv2_1(x)
    x = tf.nn.leaky_relu(x)
    x = self.conv2_2(x)
    x = self.bn2(x, training=training)
    x = tf.nn.leaky_relu(x)
    x = self.conv2_3(x)
    x = self.bn3(x, training=training)
    x = tf.nn.leaky_relu(x)
    x = self.zero_pad1(x)
    x = self.conv2_4(x)
    x = self.bn4(x, training=training)
    x = tf.nn.leaky_relu(x)
    x = self.zero_pad2(x)
    x = self.conv2_5(x)

    return x
  

# mask_G = Mask_G()
# face_G = Face_G()
# face_D = Face_D()
# mask_G.build(input_shape=(None, 256, 256, 3))
# face_G.build(input_shape=(None, 256, 256, 3))
# face_D.build(input_shape=[(None, 256, 256, 3), (None, 256, 256, 3)])
