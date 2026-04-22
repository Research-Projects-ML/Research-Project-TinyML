import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def residual_block(x, out_filters, stride, block_name):
    in_filters = x.shape[-1]  # channels-last

    # First conv
    out = layers.Conv2D(
        out_filters, kernel_size=3,
        strides=stride, padding='same',
        use_bias=False, name=f'{block_name}_conv1'
    )(x)
    out = layers.BatchNormalization(name=f'{block_name}_bn1')(out)
    out = layers.ReLU(name=f'{block_name}_relu1')(out)

    # Second conv
    out = layers.Conv2D(
        out_filters, kernel_size=3,
        strides=1, padding='same',
        use_bias=False, name=f'{block_name}_conv2'
    )(out)
    out = layers.BatchNormalization(name=f'{block_name}_bn2')(out)

    # Shortcut projection if shape changes
    if stride != 1 or in_filters != out_filters:
        x = layers.Conv2D(
            out_filters, kernel_size=1,
            strides=stride, use_bias=False,
            name=f'{block_name}_proj'
        )(x)
        x = layers.BatchNormalization(name=f'{block_name}_proj_bn')(x)

    # Residual addition
    out = layers.Add(name=f'{block_name}_add')([out, x])
    out = layers.ReLU(name=f'{block_name}_relu2')(out)

    return out


def build_resnet8(num_classes, base_filters, model_name, dropout_rate=0.0):
    inputs = keras.Input(shape=(32, 32, 3), name='input')

    # Entry conv
    x = layers.Conv2D(base_filters, kernel_size=3,padding='same', use_bias=False,name='entry_conv')(inputs)
    x = layers.BatchNormalization(name='entry_bn')(x)
    x = layers.ReLU(name='entry_relu')(x)

    # Stage 1: same channels, no downsampling
    x = residual_block(x, base_filters, stride=1, block_name='stage1')

    # Stage 2: double channels, stride-2 downsampling
    x = residual_block(x, base_filters * 2, stride=2, block_name='stage2')

    # Stage 3: double channels again, stride-2 downsampling
    x = residual_block(x, base_filters * 4, stride=2, block_name='stage3')

    x = layers.GlobalAveragePooling2D(name='gap')(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name='classifier_dropout')(x)
    outputs = layers.Dense(num_classes, name='classifier')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=model_name)


def get_teacher(config):
    return build_resnet8(
        num_classes=config['num_classes'],
        base_filters=config['teacher_base_filters'],
        model_name='resnet8_teacher',
        dropout_rate=0.3        
    )


def get_student(config):
    return build_resnet8(
        num_classes=config['num_classes'],
        base_filters=config['student_base_filters'],
        model_name='resnet8_student',
        dropout_rate=0.0
    )