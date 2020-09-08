from tensorflow.keras.layers import Add, Dropout as KerasDropout, LayerNormalization


def Residual(Layer, n_outputs=None, choose=0, normalization_layer_class=LayerNormalization, normalization_layer_options={}):
    def __inner__(in_, **kwargs):
        out = Layer(in_, **kwargs)
        norm = normalization_layer_class(**normalization_layer_options) if normalization_layer_class else lambda x: x
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


def Dropout(Layer, rate=0.2):
    def __inner__(*args, **kwargs):
        out = Layer(*args, **kwargs)
        out = KerasDropout(rate)(out)
        return out
    return __inner__
