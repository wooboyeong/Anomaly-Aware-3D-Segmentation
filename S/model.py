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

class UNET(tf.keras.Model):
    
    def __init__(self, output_channels, base_filters=16):
        super(UNET, self).__init__()

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
        self.output5 = tf.keras.layers.Conv3D(output_channels, kernel_size=1)
        self.upsample5 = tf.keras.layers.UpSampling3D()

        filters //= 2
        self.upconv4 = UpsamplingBlock(filters)
        self.conv4_3 = ConvBlock(filters)
        self.conv4_4 = ConvBlock(filters)
        self.output4 = tf.keras.layers.Conv3D(output_channels, kernel_size=1)
        self.upsample4 = tf.keras.layers.UpSampling3D()

        filters //= 2
        self.upconv3 = UpsamplingBlock(filters)
        self.conv3_3 = ConvBlock(filters)
        self.conv3_4 = ConvBlock(filters)
        self.output3 = tf.keras.layers.Conv3D(output_channels, kernel_size=1)
        self.upsample3 = tf.keras.layers.UpSampling3D()

        filters //= 2
        self.upconv2 = UpsamplingBlock(filters)
        self.conv2_3 = ConvBlock(filters)
        self.conv2_4 = ConvBlock(filters)
        self.output2 = tf.keras.layers.Conv3D(output_channels, kernel_size=1)
        self.upsample2 = tf.keras.layers.UpSampling3D()

        filters //= 2
        self.upconv1 = UpsamplingBlock(filters)
        self.conv1_3 = ConvBlock(filters)
        self.conv1_4 = ConvBlock(filters)
        self.output1 = tf.keras.layers.Conv3D(output_channels, kernel_size=1)
        
        self.softmax = tf.keras.layers.Activation('softmax', dtype='float32')
    
    def call(self, x):
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x2 = self.conv2_1(x1)
        x2 = self.conv2_2(x2)
        x3 = self.conv3_1(x2)
        x3 = self.conv3_2(x3)
        x4 = self.conv4_1(x3)
        x4 = self.conv4_2(x4)
        x5 = self.conv5_1(x4)
        x5 = self.conv5_2(x5)
        y5 = self.output5(x5)
        x4 = self.upconv4(x5, x4)
        x4 = self.conv4_3(x4)
        x4 = self.conv4_4(x4)
        y4 = self.output4(x4) + self.upsample5(y5)
        x3 = self.upconv3(x4, x3)
        x3 = self.conv3_3(x3)
        x3 = self.conv3_4(x3)
        y3 = self.output3(x3) + self.upsample4(y4)
        x2 = self.upconv2(x3, x2)
        x2 = self.conv2_3(x2)
        x2 = self.conv2_4(x2)
        y2 = self.output2(x2) + self.upsample3(y3)
        x1 = self.upconv1(x2, x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        y1 = self.output1(x1) + self.upsample2(y2)
        return self.softmax(y1)
