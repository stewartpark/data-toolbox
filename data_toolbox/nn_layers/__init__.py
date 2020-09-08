from tensorflow.keras.layers import Add, Dropout as KerasDropout, LayerNormalization, deserialize


def Residual(layer, n_outputs=None, choose=0, normalization_layer_class=LayerNormalization, normalization_layer_options={}):
    def __inner__(in_, *args, **kwargs):
        out = layer(in_, *args, **kwargs)
        norm = normalization_layer_class(
            **normalization_layer_options) if normalization_layer_class else lambda x: x
        if n_outputs is None:
            out = Add()([in_, out])
            out = norm(out)
            return out
        else:
            outs = out
            out = Add()([in_, outs[choose]])
            out = norm(out)
            return [out if i == choose else o for i, o in enumerate(outs)]
    return __inner__


def Dropout(layer, rate=0.2):
    def __inner__(in_, *args, **kwargs):
        out = layer(in_, *args, **kwargs)
        out = KerasDropout(rate)(out)
        return out
    return __inner__


def MultiHead(layer, n_heads=None):
    if type(layer) in (list, tuple):
        layers = layer
    else:
        if n_heads is None:
            raise Exception("n_heads is missing.")

        layer_info = {'class_name': layer.__class__.__name__,
                      'config': layer.get_config()}
        n_heads, original_name = int(n_heads), layer_info['config']['name']
        layers = []
        for i in range(n_heads):
            layer_info['config']['name'] = f'{original_name}_multihead_{i}'
            layers.append(deserialize(layer_info))

    def __inner__(in_, *args, **kwargs):
        return [l(in_, *args, **kwargs) for l in layers]

    return __inner__
