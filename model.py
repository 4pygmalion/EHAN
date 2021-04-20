import tensorflow as tf


def build_EHAN(config):
    '''

    Parameters
    ----------
    config: dict
        key: k . # of visiting
        c : Voca size
        m : output vector size of embedding layer. conventionally, 4 squared root of c
            e.g.) 4 ~ 3.5 = sqrt^{4}(100)
        n_d = number of unit of GRU layer
        n_h = number of unit in dense lyaer  (o)
        

    Returns
    -------

    '''
    x = tf.keras.layers.Input((k, c))
    v = tf.keras.layers.Lambda(lambda x: x / tf.norm(x, axis=0), name='norm')(x)
    v = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(m, use_bias=False), name='soft_embedding')(v)
    h_d_bidrection = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(n_d, return_sequences=True))(v)
    omega_d = tf.keras.layers.Dense(k, name='omega_att')(h_d_bidrection)
    omega_att = tf.keras.layers.Lambda(lambda x: tf.linalg.diag_part(x), name='diagonal')(omega_d)
    omega_att = tf.keras.layers.Softmax()(omega_att)

    c_d = tf.keras.layers.Lambda(lambda x: tf.einsum('ij,k->ijk', x, tf.ones(n_d*2)))(omega_att)
    c_d = tf.keras.layers.Multiply()([h_d_bidrection, c_d])

    c_tilde = tf.keras.layers.Flatten()(c_d)
    o = tf.keras.layers.Dense(n_h)(c_tilde)
    y = tf.keras.layers.Dense(2, activation='softmax')(o)

    model = tf.keras.models.Model(x, y)

    return model
