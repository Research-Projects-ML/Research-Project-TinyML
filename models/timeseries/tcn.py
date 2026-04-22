import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package='tinyml')
class CausalDilatedConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, name_prefix, **kwargs):
        super().__init__(**kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size
        self.dilation_rate = dilation_rate
        self.name_prefix   = name_prefix

        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            use_bias=False,
            name=f'{name_prefix}_conv'
        )
        self.bn   = layers.BatchNormalization(name=f'{name_prefix}_bn')
        self.relu = layers.ReLU(name=f'{name_prefix}_relu')

    def build(self, input_shape):
        self.conv.build(input_shape)
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.bn.build(conv_output_shape)
        self.relu.build(conv_output_shape)
        super().build(input_shape)

    def call(self, x, training=False):
        return self.relu(self.bn(self.conv(x), training=training))

    def get_config(self):
        base_config = super().get_config()
        custom_config = {
            'filters':       self.filters,
            'kernel_size':   self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'name_prefix':   self.name_prefix,
        }
        return {**base_config, **custom_config}


@tf.keras.utils.register_keras_serializable(package='tinyml')
class LastTimestep(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        return x[:, -1, :]

    def get_config(self):
        return super().get_config()


def causal_conv_block(x, filters, kernel_size, dilation, dropout_rate, block_name, training=None):
    residual = x

    x = CausalDilatedConv1D(
        filters, kernel_size, dilation,
        name_prefix=f'{block_name}_conv1'
    )(x, training=training)
    x = layers.Dropout(dropout_rate, name=f'{block_name}_drop1')(x, training=training)

    x = CausalDilatedConv1D(
        filters, kernel_size, dilation,
        name_prefix=f'{block_name}_conv2'
    )(x, training=training)
    x = layers.Dropout(dropout_rate, name=f'{block_name}_drop2')(x, training=training)

    if residual.shape[-1] != filters:
        residual = layers.Conv1D(
            filters, kernel_size=1,
            use_bias=False,
            name=f'{block_name}_proj'
        )(residual)

    x = layers.Add(name=f'{block_name}_add')([x, residual])
    x = layers.ReLU(name=f'{block_name}_relu')(x)

    return x


def build_tcn(input_channels, num_classes, num_blocks, num_channels, kernel_size, dropout, model_name):
    inputs = keras.Input(shape=(128, input_channels), name='input')

    x = inputs
    for i in range(num_blocks):
        dilation = 2 ** i
        x = causal_conv_block(
            x,
            filters=num_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            dropout_rate=dropout,
            block_name=f'tcn_block_{i}'
        )

    x = LastTimestep(name='last_timestep')(x)
    outputs = layers.Dense(num_classes, name='classifier')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=model_name)


def get_teacher(config):
    return build_tcn(
        input_channels=config['input_channels'],
        num_classes=config['num_classes'],
        num_blocks=config['teacher_num_blocks'],
        num_channels=config['teacher_num_channels'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
        model_name='tcn_teacher'
    )


def get_student(config):
    return build_tcn(
        input_channels=config['input_channels'],
        num_classes=config['num_classes'],
        num_blocks=config['student_num_blocks'],
        num_channels=config['student_num_channels'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
        model_name='tcn_student'
    )