import numpy as np
import tensorflow as tf
import keras
from keras import layers
from models.timeseries.tcn import CausalDilatedConv1D, LastTimestep


def compute_channel_importance(layer):
    """
    L1-norm importance score per output channel.
    - For Conv2D: Height, Width, Channel_in
    - For Conv1D: Kernel_size, Channel_in
    Returns shape (C_out,) one score per output channel.
    """
    weights = layer.get_weights()[0]
    if isinstance(layer, layers.Conv2D):
        return np.sum(np.abs(weights), axis = (0, 1, 2))
    elif isinstance(layer, layers.Conv1D):
        return np.sum(np.abs(weights), axis = (0, 1))
    else:
        raise ValueError(
            f"Unsupported layer type for importance scoring: {type(layer)}"
        )


def get_channels_to_keep(importance_scores, prune_ratio):
    """
    Returns sorted indices of channels to keep.
    Keeps the (1 - prune_ratio) fraction with highest L1 importance. Always keeps at least 1 channel.
    """
    num_channels = len(importance_scores)
    num_to_keep = max(1, int(num_channels * (1 - prune_ratio)))
    sorted_indices = np.argsort(importance_scores)
    keep_indices = sorted_indices[num_channels - num_to_keep:]
    return np.sort(keep_indices)


def prune_conv2d(layer, keep_indices_out, keep_indices_in=None):
    """
    Returns a new Conv2D with pruned input and output channels.
    Weight tensor: (H, W, C_in, C_out)
    - keep_indices_out: which output channels to keep
    - keep_indices_in: which input channels to keep (None = keep all)
    """
    weights = layer.get_weights()
    kernel  = weights[0]

    kernel = kernel[:, :, :, keep_indices_out]
    if keep_indices_in is not None:
        kernel = kernel[:, :, keep_indices_in, :]

    in_channels = kernel.shape[2]

    new_layer = layers.Conv2D(filters=len(keep_indices_out),kernel_size=layer.kernel_size,strides=layer.strides,
                              padding=layer.padding,use_bias=layer.use_bias,name=layer.name)
    new_layer.build((None, None, None, in_channels))

    if layer.use_bias:
        new_layer.set_weights([kernel, weights[1][keep_indices_out]])
    else:
        new_layer.set_weights([kernel])

    return new_layer


def prune_conv1d(layer, keep_indices_out, keep_indices_in=None):
    """
    Returns a new Conv1D with pruned input and output channels.
    Weight tensor: (kernel_size, C_in, C_out)
    - keep_indices_out: which output channels to keep
    - keep_indices_in: which input channels to keep (None = keep all)
    """
    weights = layer.get_weights()
    kernel = weights[0]

    kernel = kernel[:, :, keep_indices_out]
    if keep_indices_in is not None:
        kernel = kernel[:, keep_indices_in, :]

    in_channels = kernel.shape[1]

    new_layer = layers.Conv1D(filters=len(keep_indices_out),kernel_size=layer.kernel_size[0],strides=layer.strides[0],
                              padding=layer.padding,dilation_rate=layer.dilation_rate[0],use_bias=layer.use_bias,name=layer.name)
    new_layer.build((None, None, in_channels))

    if layer.use_bias:
        new_layer.set_weights([kernel, weights[1][keep_indices_out]])
    else:
        new_layer.set_weights([kernel])

    return new_layer


def prune_batchnorm(layer, keep_indices, input_rank):
    """
    Returns a new BatchNormalization with parameters sliced to keep_indices.
    BN has 4 arrays: gamma, beta, moving_mean, moving_variance
    input_rank: 4 for Conv2D (NHWC), 3 for Conv1D (NTC).
    """
    gamma, beta, moving_mean, moving_var = layer.get_weights()

    new_layer = layers.BatchNormalization(momentum=layer.momentum,epsilon=layer.epsilon,name=layer.name)

    n = len(keep_indices)
    spatial_ones = [1] * (input_rank - 2)
    new_layer.build(tuple([None] + spatial_ones + [n]))

    new_layer.set_weights([gamma[keep_indices],beta[keep_indices],moving_mean[keep_indices],moving_var[keep_indices],])

    return new_layer


def prune_dense(layer, keep_indices_in):
    """
    Returns a new Dense with pruned input connections.
    Output units are unchanged classifier output must equal num_classes.
    Guards against use_bias=False to avoid indexing weights[1] when absent.
    """
    weights = layer.get_weights()
    kernel = weights[0][keep_indices_in, :]

    new_layer = layers.Dense(units=layer.units,use_bias=layer.use_bias,name=layer.name)
    new_layer.build((None, len(keep_indices_in)))

    if layer.use_bias:
        new_layer.set_weights([kernel, weights[1]])
    else:
        new_layer.set_weights([kernel])

    return new_layer


def apply_structured_pruning(model, prune_ratio, domain):
    if domain == 'image':
        return _prune_resnet8(model, prune_ratio)
    elif domain == 'timeseries':
        return _prune_tcn(model, prune_ratio)
    else:
        raise ValueError(f"Unknown domain: '{domain}'. 'image' or 'timeseries'.")
    

def _prune_resnet8(model, prune_ratio):
    """
    Residual connection constraints:
    - Stage1 has an identity shortcut: conv2 output channels must equal block input channels. 
    We prune conv1 freely but fix conv2 output channels to match the incoming channel count (prev_keep).
    - Stage2, Stage3 have projection shortcuts (1x1 conv): conv2 and the projection conv must produce identical output channels, 
    so they are pruned together to the same keep_indices.
    """
    inputs = keras.Input(shape=model.input_shape[1:], name='input')

    entry_conv = model.get_layer('entry_conv')
    entry_bn   = model.get_layer('entry_bn')

    entry_importance = compute_channel_importance(entry_conv)
    entry_keep = get_channels_to_keep(entry_importance, prune_ratio)

    x = prune_conv2d(entry_conv, entry_keep)(inputs)
    x = prune_batchnorm(entry_bn, entry_keep, input_rank=4)(x)
    x = layers.ReLU(name='entry_relu')(x)

    prev_keep = entry_keep

    for stage_name in ['stage1', 'stage2', 'stage3']:
        conv1 = model.get_layer(f'{stage_name}_conv1')
        bn1 = model.get_layer(f'{stage_name}_bn1')
        conv2 = model.get_layer(f'{stage_name}_conv2')
        bn2 = model.get_layer(f'{stage_name}_bn2')

        has_proj = True
        try:
            proj = model.get_layer(f'{stage_name}_proj')
            proj_bn = model.get_layer(f'{stage_name}_proj_bn')
        except ValueError:
            has_proj = False

        conv1_importance = compute_channel_importance(conv1)
        conv1_keep = get_channels_to_keep(conv1_importance, prune_ratio)

        shortcut = x

        x = prune_conv2d(conv1, conv1_keep, keep_indices_in=prev_keep)(x)
        x = prune_batchnorm(bn1, conv1_keep, input_rank=4)(x)
        x = layers.ReLU(name=f'{stage_name}_relu1')(x)

        if has_proj:
            # Projection shortcut — prune conv2 freely, project shortcut
            # to the same output channels so the residual addition is valid
            conv2_importance = compute_channel_importance(conv2)
            conv2_keep = get_channels_to_keep(conv2_importance, prune_ratio)

            x = prune_conv2d(conv2, conv2_keep, keep_indices_in=conv1_keep)(x)
            x = prune_batchnorm(bn2, conv2_keep, input_rank=4)(x)
            shortcut = prune_conv2d(proj, conv2_keep, keep_indices_in=prev_keep)(shortcut)
            shortcut = prune_batchnorm(proj_bn, conv2_keep, input_rank=4)(shortcut)

            x = layers.Add(name=f'{stage_name}_add')([x, shortcut])
            x = layers.ReLU(name=f'{stage_name}_relu2')(x)

            prev_keep = conv2_keep

        else:
            # Identity shortcut: conv2 output must match block input channels
            # so the residual addition is valid without a projection
            x = prune_conv2d(conv2, prev_keep, keep_indices_in=conv1_keep)(x)
            x = prune_batchnorm(bn2, prev_keep, input_rank=4)(x)
            x = layers.Add(name=f'{stage_name}_add')([x, shortcut])
            x = layers.ReLU(name=f'{stage_name}_relu2')(x)

    x = layers.GlobalAveragePooling2D(name='gap')(x)
    dense   = model.get_layer('classifier')
    outputs = prune_dense(dense, prev_keep)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=model.name + '_pruned')


def _build_pruned_causal_block(
    x,
    block_idx,
    conv1_kernel, bn1_weights,
    conv2_kernel, bn2_weights,
    kernel_size, dilation_rate,
    drop1_rate, drop2_rate,
    proj_kernel=None,
):
    """
    Builds one pruned TCN block via a dummy forward pass through new
    CausalDilatedConv1D wrappers, then transplants the pruned weights.

    Using a forward pass rather than manual build() calls ensures the inner conv and bn sublayers are built to the correct shapes before
    set_weights is called avoiding shape mismatches from build order ambiguity in custom layers.
    """
    n_out = conv1_kernel.shape[-1]

    # First causal conv
    wrapper1 = CausalDilatedConv1D(
        filters=n_out,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        name_prefix=f'tcn_block_{block_idx}_conv1'
    )
    # Dummy pass builds all sublayers to the right shapes
    _ = wrapper1(x, training=False)
    wrapper1.conv.set_weights([conv1_kernel])
    wrapper1.bn.set_weights(bn1_weights)

    residual = x
    x = wrapper1(x, training=False)
    x = layers.Dropout(drop1_rate, name=f'tcn_block_{block_idx}_drop1')(x)

    # Second causal conv
    n_out2 = conv2_kernel.shape[-1]

    wrapper2 = CausalDilatedConv1D(
        filters=n_out2,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        name_prefix=f'tcn_block_{block_idx}_conv2'
    )
    _ = wrapper2(x, training=False)
    wrapper2.conv.set_weights([conv2_kernel])
    wrapper2.bn.set_weights(bn2_weights)

    x = wrapper2(x, training=False)
    x = layers.Dropout(drop2_rate, name=f'tcn_block_{block_idx}_drop2')(x)

    # Residual connection
    if proj_kernel is not None:
        proj = layers.Conv1D(
            filters=n_out2,
            kernel_size=1,
            use_bias=False,
            name=f'tcn_block_{block_idx}_proj'
        )
        _ = proj(residual)
        proj.set_weights([proj_kernel])
        residual = proj(residual)

    x = layers.Add(name=f'tcn_block_{block_idx}_add')([x, residual])
    x = layers.ReLU(name=f'tcn_block_{block_idx}_relu')(x)

    return x


def _prune_tcn(model, prune_ratio):
    """
    Structured pruning, TCN block constraints:
    - Each block has two CausalDilatedConv1D layers followed by a residual add.
    - Both conv layers in a block must output the same channel count for the residual addition to be valid.
    - We average L1 importance across both convs before selecting keep_indices, then apply that same set to both.
    - The projection conv (present when block input channels differ from num_channels, typically only block 0) must output the same channels
      as the block, so it is pruned to the same keep_indices.
    """
    # Collect CausalDilatedConv1D wrappers in order
    causal_wrappers = [
        layer for layer in model.layers
        if isinstance(layer, CausalDilatedConv1D)
    ]

    if not causal_wrappers:
        raise RuntimeError(
            "[_prune_tcn] No CausalDilatedConv1D layers found in model. "
            "Check model architecture and layer registration."
        )

    if len(causal_wrappers) % 2 != 0:
        raise RuntimeError(
            f"[_prune_tcn] Expected an even number of CausalDilatedConv1D layers "
            f"(2 per block), found {len(causal_wrappers)}."
        )

    block_pairs = [
        (causal_wrappers[i], causal_wrappers[i + 1])
        for i in range(0, len(causal_wrappers), 2)
    ]

    inputs = keras.Input(shape=model.input_shape[1:], name='input')
    x      = inputs

    prev_keep = None

    for block_idx, (wrapper1, wrapper2) in enumerate(block_pairs):
        conv1_inner = wrapper1.conv
        conv2_inner = wrapper2.conv
        bn1_inner = wrapper1.bn
        bn2_inner = wrapper2.bn

        # Average importance across both convs — they share keep_indices
        importance = (
            compute_channel_importance(conv1_inner) +
            compute_channel_importance(conv2_inner)
        ) / 2.0
        block_keep = get_channels_to_keep(importance, prune_ratio)

        # Slice conv1 kernel: (kernel, C_in, C_out)
        # C_in comes from the previous block's keep_indices (or original
        # input channels for block 0)
        c1_kernel = conv1_inner.get_weights()[0]
        if prev_keep is not None:
            c1_kernel = c1_kernel[:, prev_keep, :]
        c1_kernel = c1_kernel[:, :, block_keep]

        g1, b1, mm1, mv1 = bn1_inner.get_weights()
        bn1_weights = [
            g1[block_keep], b1[block_keep],
            mm1[block_keep], mv1[block_keep]
        ]

        # Slice conv2 kernel: input comes from conv1 output (block_keep),
        # output also block_keep (residual constraint)
        c2_kernel = conv2_inner.get_weights()[0]
        c2_kernel = c2_kernel[:, block_keep, :]
        c2_kernel = c2_kernel[:, :, block_keep]

        g2, b2, mm2, mv2 = bn2_inner.get_weights()
        bn2_weights = [
            g2[block_keep], b2[block_keep],
            mm2[block_keep], mv2[block_keep]
        ]

        # Projection kernel if present
        proj_kernel = None
        try:
            proj_layer = model.get_layer(f'tcn_block_{block_idx}_proj')
            proj_kernel = proj_layer.get_weights()[0]
            if prev_keep is not None:
                proj_kernel = proj_kernel[:, prev_keep, :]
            proj_kernel = proj_kernel[:, :, block_keep]
        except ValueError:
            pass

        drop1_rate = model.get_layer(f'tcn_block_{block_idx}_drop1').rate
        drop2_rate = model.get_layer(f'tcn_block_{block_idx}_drop2').rate

        x = _build_pruned_causal_block(
            x=x,
            block_idx=block_idx,
            conv1_kernel=c1_kernel,
            bn1_weights=bn1_weights,
            conv2_kernel=c2_kernel,
            bn2_weights=bn2_weights,
            kernel_size=conv1_inner.kernel_size[0],
            dilation_rate=conv1_inner.dilation_rate[0],
            drop1_rate=drop1_rate,
            drop2_rate=drop2_rate,
            proj_kernel=proj_kernel,
        )

        prev_keep = block_keep

    if prev_keep is None:
        raise RuntimeError(
            "[_prune_tcn] prev_keep is None after iterating blocks — "
            "no blocks were processed."
        )

    x = LastTimestep(name='last_timestep')(x)
    dense   = model.get_layer('classifier')
    outputs = prune_dense(dense, prev_keep)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=model.name + '_pruned')