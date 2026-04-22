import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from models.timeseries.tcn import CausalDilatedConv1D, LastTimestep


def compute_channel_importance(layer):
    """
    Channels with low L1 norm contribute little to the output and are candidates for removal.
    For Conv2D: weights shape is (H, W, C_in, C_out)
    For Conv1D: weights shape is (kernel, C_in, C_out)
    Result shape: (C_out,) — one score per output channel
    """
    weights = layer.get_weights()[0]

    if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.DepthwiseConv2D):
        importance = np.sum(np.abs(weights), axis=(0, 1, 2))
    elif isinstance(layer, layers.Conv1D):
        importance = np.sum(np.abs(weights), axis=(0, 1))
    else:
        raise ValueError(f"Unsupported layer type for pruning: {type(layer)}")

    return importance


def get_channels_to_keep(importance_scores, prune_ratio):
    """
    Returns indices of channels to keep after pruning.
    """
    num_channels = len(importance_scores)
    num_to_keep  = max(1, int(num_channels * (1 - prune_ratio)))
    sorted_indices = np.argsort(importance_scores)
    keep_indices   = sorted_indices[num_channels - num_to_keep:]
    return np.sort(keep_indices)


def prune_conv2d(layer, keep_indices_out, keep_indices_in=None):
    """
    Returns a new Conv2D layer with pruned input and output channels.
    Weight tensor for Conv2D: (H, W, C_in, C_out)

    The new layer has:
    - filters = len(keep_indices_out)
    - No bias if original had none
    - Same kernel size, strides, padding, activation
    """
    weights = layer.get_weights()
    kernel  = weights[0]

    kernel = kernel[:, :, :, keep_indices_out]

    if keep_indices_in is not None:
        kernel = kernel[:, :, keep_indices_in, :]

    new_layer = layers.Conv2D(
        filters=len(keep_indices_out),
        kernel_size=layer.kernel_size,
        strides=layer.strides,
        padding=layer.padding,
        use_bias=layer.use_bias,
        name=layer.name
    )

    dummy_input_channels = (
        len(keep_indices_in) if keep_indices_in is not None
        else kernel.shape[2]
    )
    new_layer.build((None, None, None, dummy_input_channels))

    if layer.use_bias:
        bias = weights[1][keep_indices_out]
        new_layer.set_weights([kernel, bias])
    else:
        new_layer.set_weights([kernel])

    return new_layer


def prune_conv1d(layer, keep_indices_out, keep_indices_in=None):
    """
    Returns a new Conv1D layer with pruned input and output channels.
    Weight tensor for Conv1D: (kernel_size, C_in, C_out)
    """
    weights = layer.get_weights()
    kernel  = weights[0]
    kernel  = kernel[:, :, keep_indices_out]

    if keep_indices_in is not None:
        kernel = kernel[:, keep_indices_in, :]

    new_layer = layers.Conv1D(
        filters=len(keep_indices_out),
        kernel_size=layer.kernel_size[0],
        strides=layer.strides[0],
        padding=layer.padding,
        dilation_rate=layer.dilation_rate[0],
        use_bias=layer.use_bias,
        name=layer.name
    )

    new_layer.build((None, None, kernel.shape[1]))
    if layer.use_bias:
        bias = weights[1][keep_indices_out]
        new_layer.set_weights([kernel, bias])
    else:
        new_layer.set_weights([kernel])

    return new_layer


def prune_batchnorm(layer, keep_indices, input_rank):
    """
    Batch Normalization has 4 parameter arrays: gamma, beta, moving_mean, moving_variance.
    All are 1D arrays of shape (C,) — one value per channel.
    When channels are pruned from the preceding conv, the corresponding
    BN parameters must also be removed to maintain alignment.
    """
    gamma, beta, moving_mean, moving_var = layer.get_weights()

    new_layer = layers.BatchNormalization(
        momentum=layer.momentum,
        epsilon=layer.epsilon,
        name=layer.name
    )

    n             = len(keep_indices)
    spatial_ones  = [1] * (input_rank - 2)
    build_shape   = tuple([None] + spatial_ones + [n])

    new_layer.build(build_shape)
    new_layer.set_weights([
        gamma[keep_indices],
        beta[keep_indices],
        moving_mean[keep_indices],
        moving_var[keep_indices]
    ])

    return new_layer


def prune_dense(layer, keep_indices_in):
    weights = layer.get_weights()
    kernel  = weights[0]
    bias    = weights[1]
    kernel  = kernel[keep_indices_in, :]

    new_layer = layers.Dense(
        units=layer.units,
        name=layer.name
    )
    new_layer.build((None, len(keep_indices_in)))
    new_layer.set_weights([kernel, bias])

    return new_layer


def apply_structured_pruning(model, prune_ratio, domain):
    if domain == 'image':
        return _prune_resnet8(model, prune_ratio)
    elif domain == 'timeseries':
        return _prune_tcn(model, prune_ratio)
    else:
        raise ValueError(f"Unknown domain: {domain}")


def _prune_resnet8(model, prune_ratio):
    """
    Structured pruning for ResNet-8.

    ResNet-8 has residual connections that constrain pruning:
    - The output of conv2 in each block must match the shortcut output
    - If the shortcut is a projection (1x1 conv), it must also be pruned
      to match the same output channels
    - If the shortcut is identity, conv2 output channels must equal
      the block's input channels — cannot be independently pruned

    Strategy:
    - Stage1 (identity shortcut): prune conv1 output channels freely,
      but conv2 output channels must be fixed (equal to block input).
      We only prune conv1's internal channels in this case.
    - Stage2, Stage3 (projection shortcut): prune conv2 and projection
      conv together — they must produce identical channel sets.
    """
    inputs = keras.Input(shape=model.input_shape[1:], name='input')

    entry_conv = model.get_layer('entry_conv')
    entry_bn   = model.get_layer('entry_bn')

    entry_importance = compute_channel_importance(entry_conv)
    entry_keep       = get_channels_to_keep(entry_importance, prune_ratio)

    x = prune_conv2d(entry_conv, entry_keep)(inputs)
    x = prune_batchnorm(entry_bn, entry_keep, input_rank=4)(x)
    x = layers.ReLU(name='entry_relu')(x)

    prev_keep = entry_keep

    for stage_name in ['stage1', 'stage2', 'stage3']:
        conv1 = model.get_layer(f'{stage_name}_conv1')
        bn1   = model.get_layer(f'{stage_name}_bn1')
        conv2 = model.get_layer(f'{stage_name}_conv2')
        bn2   = model.get_layer(f'{stage_name}_bn2')

        has_proj = True
        try:
            proj    = model.get_layer(f'{stage_name}_proj')
            proj_bn = model.get_layer(f'{stage_name}_proj_bn')
        except ValueError:
            has_proj = False

        # Prune conv1 output channels freely
        conv1_importance = compute_channel_importance(conv1)
        conv1_keep       = get_channels_to_keep(conv1_importance, prune_ratio)

        shortcut = x

        x = prune_conv2d(conv1, conv1_keep, keep_indices_in=prev_keep)(x)
        x = prune_batchnorm(bn1, conv1_keep, input_rank=4)(x)
        x = layers.ReLU(name=f'{stage_name}_relu1')(x)

        if has_proj:
            # Projection shortcut: conv2 and proj must produce same channels
            conv2_importance = compute_channel_importance(conv2)
            conv2_keep       = get_channels_to_keep(conv2_importance, prune_ratio)

            x        = prune_conv2d(conv2, conv2_keep, keep_indices_in=conv1_keep)(x)
            x        = prune_batchnorm(bn2, conv2_keep, input_rank=4)(x)
            shortcut = prune_conv2d(proj, conv2_keep, keep_indices_in=prev_keep)(shortcut)
            shortcut = prune_batchnorm(proj_bn, conv2_keep, input_rank=4)(shortcut)

            x = layers.Add(name=f'{stage_name}_add')([x, shortcut])
            x = layers.ReLU(name=f'{stage_name}_relu2')(x)

            prev_keep = conv2_keep

        else:
            x = prune_conv2d(conv2, prev_keep, keep_indices_in=conv1_keep)(x)
            x = prune_batchnorm(bn2, prev_keep, input_rank=4)(x)
            x = layers.Add(name=f'{stage_name}_add')([x, shortcut])
            x = layers.ReLU(name=f'{stage_name}_relu2')(x)

    x = layers.GlobalAveragePooling2D(name='gap')(x)

    dense   = model.get_layer('classifier')
    outputs = prune_dense(dense, prev_keep)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=model.name + '_pruned')


def _prune_tcn(model, prune_ratio):
    """
    Structured pruning for TCN.

    TCN blocks have residual connections with optional projection.
    Within each block, two causal conv layers share the same filter count.
    The residual projection (if present) must output the same channels
    as the second conv in the block.

    Since all blocks after the first use the same num_channels for both
    input and output, the projection only appears at block 0 where
    input_channels (sensor channels) != num_channels.

    Pruning strategy:
    - All blocks: prune both conv layers to the same keep_indices
      (they must match for the residual addition)
    - Projection conv (block 0 only): prune to same keep_indices as conv2
    """
    inputs = keras.Input(shape=model.input_shape[1:], name='input')
    x      = inputs

    block_idx = 0
    prev_keep = None

    causal_wrappers = [
        layer for layer in model.layers
        if isinstance(layer, CausalDilatedConv1D)
    ]

    block_pairs = [
        (causal_wrappers[i], causal_wrappers[i + 1])
        for i in range(0, len(causal_wrappers), 2)
    ]

    while block_idx < len(block_pairs):
        conv1_wrapper, conv2_wrapper = block_pairs[block_idx]

        conv1_inner = conv1_wrapper.conv
        conv2_inner = conv2_wrapper.conv
        bn1_inner   = conv1_wrapper.bn
        bn2_inner   = conv2_wrapper.bn

        has_proj = True
        try:
            proj_layer = model.get_layer(f'tcn_block_{block_idx}_proj')
        except ValueError:
            has_proj = False

        # Average importance across both convs in the block before selecting
        # channels to keep — both convs must share the same keep_indices
        # for the residual addition to remain valid
        importance = (
            compute_channel_importance(conv1_inner) +
            compute_channel_importance(conv2_inner)
        ) / 2
        block_keep = get_channels_to_keep(importance, prune_ratio)

        residual = x

        # Build conv1 wrapper, set pruned weights, then call
        c1_kernel = conv1_inner.get_weights()[0]
        if prev_keep is not None:
            c1_kernel = c1_kernel[:, prev_keep, :]
        c1_kernel    = c1_kernel[:, :, block_keep]
        g1, b1, mm1, mv1 = bn1_inner.get_weights()

        new_conv1 = CausalDilatedConv1D(
            filters=len(block_keep),
            kernel_size=conv1_inner.kernel_size[0],
            dilation_rate=conv1_inner.dilation_rate[0],
            name_prefix=f'tcn_block_{block_idx}_conv1'
        )
        new_conv1.build(x.shape)
        new_conv1.conv.set_weights([c1_kernel])
        new_conv1.bn.set_weights([
            g1[block_keep], b1[block_keep],
            mm1[block_keep], mv1[block_keep]
        ])
        x = new_conv1(x)

        drop1_rate = model.get_layer(f'tcn_block_{block_idx}_drop1').rate
        x = layers.Dropout(drop1_rate, name=f'tcn_block_{block_idx}_drop1')(x)

        # Build conv2 wrapper, set pruned weights, then call
        c2_kernel = conv2_inner.get_weights()[0]
        c2_kernel = c2_kernel[:, block_keep, :]
        c2_kernel = c2_kernel[:, :, block_keep]
        g2, b2, mm2, mv2 = bn2_inner.get_weights()

        new_conv2 = CausalDilatedConv1D(
            filters=len(block_keep),
            kernel_size=conv2_inner.kernel_size[0],
            dilation_rate=conv2_inner.dilation_rate[0],
            name_prefix=f'tcn_block_{block_idx}_conv2'
        )
        new_conv2.build(x.shape)
        new_conv2.conv.set_weights([c2_kernel])
        new_conv2.bn.set_weights([
            g2[block_keep], b2[block_keep],
            mm2[block_keep], mv2[block_keep]
        ])
        x = new_conv2(x)

        drop2_rate = model.get_layer(f'tcn_block_{block_idx}_drop2').rate
        x = layers.Dropout(drop2_rate, name=f'tcn_block_{block_idx}_drop2')(x)

        if has_proj:
            proj_kernel = proj_layer.get_weights()[0]
            if prev_keep is not None:
                proj_kernel = proj_kernel[:, prev_keep, :]
            proj_kernel = proj_kernel[:, :, block_keep]

            new_proj = layers.Conv1D(
                filters=len(block_keep),
                kernel_size=1,
                use_bias=False,
                name=f'tcn_block_{block_idx}_proj'
            )
            new_proj.build(residual.shape)
            new_proj.set_weights([proj_kernel])
            residual = new_proj(residual)

        x = layers.Add(name=f'tcn_block_{block_idx}_add')([x, residual])
        x = layers.ReLU(name=f'tcn_block_{block_idx}_relu')(x)

        prev_keep = block_keep
        block_idx += 1

    if prev_keep is None:
        raise RuntimeError(
            "[_prune_tcn] No TCN blocks found. "
            "Expected names: tcn_block_0_conv1_causal_conv. "
            "Run: [print(l.name) for l in model.layers] to check."
        )

    x       = LastTimestep(name='last_timestep')(x)
    dense   = model.get_layer('classifier')
    outputs = prune_dense(dense, prev_keep)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=model.name + '_pruned')