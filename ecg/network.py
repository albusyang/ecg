import tensorflow as tf

def _bn_relu(layer, dropout=0, **params):
    """Batch normalization and ReLU activation"""
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.Activation(params["conv_activation"])(layer)

    if dropout > 0:
        layer = tf.keras.layers.Dropout(params["conv_dropout"])(layer)

    return layer

def add_conv_weight(layer, filter_length, num_filters, subsample_length=1, **params):
    """Add convolutional layer"""
    layer = tf.keras.layers.Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        kernel_initializer=params["conv_init"])(layer)
    return layer

def add_conv_layers(layer, **params):
    """Add a series of convolutional layers"""
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
                    layer,
                    params["conv_filter_length"],
                    params["conv_num_filters_start"],
                    subsample_length=subsample_length,
                    **params)
        layer = _bn_relu(layer, **params)
    return layer

def resnet_block(layer, num_filters, subsample_length, block_index, **params):
    """Add a ResNet block"""
    def zeropad(x):
        """Zero-padding function"""
        y = tf.keras.backend.zeros_like(x)
        return tf.keras.backend.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        """Shape function for Lambda layer"""
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = tf.keras.layers.MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 and block_index > 0
    if zero_pad:
        shortcut = tf.keras.layers.Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(layer, dropout=params["conv_dropout"] if i > 0 else 0, **params)
        layer = add_conv_weight(layer, params["conv_filter_length"], num_filters, subsample_length if i == 0 else 1, **params)
    layer = tf.keras.layers.Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, **params):
    """Calculate number of filters at a specific index"""
    return 2**int(index / params["conv_increase_channels_at"]) * num_start_filters

def add_resnet_layers(layer, **params):
    """Add a series of ResNet blocks"""
    layer = add_conv_weight(layer, params["conv_filter_length"], params["conv_num_filters_start"], subsample_length=1, **params)
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(index, params["conv_num_filters_start"], **params)
        layer = resnet_block(layer, num_filters, subsample_length, index, **params)
    layer = _bn_relu(layer, **params)
    return layer

def add_output_layer(layer, **params):
    """Add output layer"""
    layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(params["num_categories"]))(layer)
    return tf.keras.layers.Activation('softmax')(layer)

def add_compile(model, **params):
    """Compile the model"""
    optimizer = tf.keras.optimizers.Adam(lr=params["learning_rate"], clipnorm=params.get("clipnorm", 1))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

def build_network(**params):
    """Build the entire network"""
    inputs = tf.keras.layers.Input(shape=params['input_shape'], dtype='float32', name='inputs')

    if params.get('is_regular_conv', False):
        layer = add_conv_layers(inputs, **params)
    else:
        layer = add_resnet_layers(inputs, **params)

    output = add_output_layer(layer, **params)
    model = tf.keras.Model(inputs=[inputs], outputs=[output])
    if params.get("compile", True):
        add_compile(model, **params)
    return model
