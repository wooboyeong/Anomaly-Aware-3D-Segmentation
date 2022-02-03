import tensorflow as tf
import tensorflow_addons as tfa

class Activation(tf.keras.layers.Layer):

    def __init__(self):
        super(Activation, self).__init__()
        self.norm = tfa.layers.InstanceNormalization()
        self.relu = tf.keras.layers.LeakyReLU(alpha=0.1)
    
    def call(self, x):
        x = self.norm(x)
        return self.relu(x)

class ConvBlock(tf.keras.layers.Layer):
    
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv3D(filters, kernel_size=3, padding='same')
        self.activation = Activation()
    
    def call(self, x):
        x = self.conv(x)
        return self.activation(x)

class DownsamplingBlock(tf.keras.layers.Layer):
    
    def __init__(self, filters):
        super(DownsamplingBlock, self).__init__()
        self.conv = tf.keras.layers.Conv3D(filters, kernel_size=3, strides=2, padding='same')
        self.activation = Activation()
    
    def call(self, x):
        x = self.conv(x)
        return self.activation(x)

class UpsamplingBlock(tf.keras.layers.Layer):
    
    def __init__(self, filters):
        super(UpsamplingBlock, self).__init__()
        self.upconv = tf.keras.layers.Conv3DTranspose(filters, kernel_size=3, strides=2, padding='same')
        self.activation = Activation()
    
    def call(self, x, skip_layer):
        x = self.upconv(x) + skip_layer
        return self.activation(x)

class C(tf.keras.Model):

    def __init__(self, base_filters=8):
        super(C, self).__init__()

        filters = base_filters
        self.conv1 = DownsamplingBlock(filters)

        filters //= 2
        self.conv2 = DownsamplingBlock(filters)

        filters //= 2
        self.conv3 = DownsamplingBlock(filters)

        filters //= 2
        self.conv4 = DownsamplingBlock(filters)

        self.flatten = tf.keras.layers.Flatten()
        self.z = tf.keras.layers.Dense(100)

        self.out = tf.keras.layers.Dense(10*24*24)
        self.reshape = tf.keras.layers.Reshape((10, 24, 24, 1))
        
        self.sigmoid = tf.keras.layers.Activation('sigmoid', dtype='float32')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        z = self.z(x)
        z = self.out(z)
        z = self.reshape(z)
        return self.sigmoid(z)

class G(tf.keras.Model):

    def __init__(self, base_filters=8):
        super(G, self).__init__()

        filters = base_filters
        self.conv1_1 = ConvBlock(filters)
        self.conv1_2 = ConvBlock(filters)

        filters *= 2
        self.conv2_1 = DownsamplingBlock(filters)
        self.conv2_2 = ConvBlock(filters)

        filters *= 2
        self.conv3_1 = DownsamplingBlock(filters)
        self.conv3_2 = ConvBlock(filters)

        filters *= 2
        self.conv4_1 = DownsamplingBlock(filters)
        self.conv4_2 = ConvBlock(filters)

        filters *= 2
        self.conv5_1 = DownsamplingBlock(filters)
        self.conv5_2 = ConvBlock(filters)

        self.C = C(base_filters)

        filters //= 2
        self.upconv4 = UpsamplingBlock(filters)
        self.conv4_3 = ConvBlock(filters)
        self.conv4_4 = ConvBlock(filters)

        filters //= 2
        self.upconv3 = UpsamplingBlock(filters)
        self.conv3_3 = ConvBlock(filters)
        self.conv3_4 = ConvBlock(filters)

        filters //= 2
        self.upconv2 = UpsamplingBlock(filters)
        self.conv2_3 = ConvBlock(filters)
        self.conv2_4 = ConvBlock(filters)

        filters //= 2
        self.upconv1 = UpsamplingBlock(filters)
        self.conv1_3 = ConvBlock(filters)
        self.conv1_4 = ConvBlock(filters)
        self.out = tf.keras.layers.Conv3D(1, kernel_size=1)
        
        self.sigmoid = tf.keras.layers.Activation('sigmoid', dtype='float32')

    def call(self, inputs):
        masked_images, images = inputs
        
        x1 = self.conv1_1(masked_images)
        x1 = self.conv1_2(x1)
        x2 = self.conv2_1(x1)
        x2 = self.conv2_2(x2)
        x3 = self.conv3_1(x2)
        x3 = self.conv3_2(x3)
        x4 = self.conv4_1(x3)
        x4 = self.conv4_2(x4)
        x5 = self.conv5_1(x4)
        x5 = self.conv5_2(x5)
        
        z = self.C(images)
        x5 = tf.keras.layers.concatenate([x5, z])
        
        x4 = self.upconv4(x5, x4)
        x4 = self.conv4_3(x4)
        x4 = self.conv4_4(x4)
        x3 = self.upconv3(x4, x3)
        x3 = self.conv3_3(x3)
        x3 = self.conv3_4(x3)
        x2 = self.upconv2(x3, x2)
        x2 = self.conv2_3(x2)
        x2 = self.conv2_4(x2)
        x1 = self.upconv1(x2, x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        y = self.out(x1)
        return self.sigmoid(y)
