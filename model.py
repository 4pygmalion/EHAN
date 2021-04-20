import tensorflow as tf


def build_EHAN(config):
    '''

    Parameters
    ----------
    config: dict
        key: k . # of visiting
        c_d : Voca size of diagnosis code
        c_m : Voca size of medicatio code
        m : output vector size of embedding layer. conventionally, 4 squared root of c
            e.g.) 4 ~ 3.5 = sqrt^{4}(100)
        n_d = number of unit of GRU layer
        n_h = number of unit in dense lyaer  (o)
        

    Returns
    -------
    tf.keras.models.Sequntial
    
    
    '''
    # Diagnosis part
    
    x = tf.keras.layers.Input((k, c_m))
    v = tf.keras.layers.Lambda(lambda x: x / tf.norm(x, axis=0), name='norm')(x)
    v = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(m, use_bias=False), name='soft_embedding')(v)
    h_d_bidrection = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(n_d, return_sequences=True))(v)
    omega_d = tf.keras.layers.Dense(k, name='omega_att')(h_d_bidrection)
    omega_att = tf.keras.layers.Lambda(lambda x: tf.linalg.diag_part(x), name='diagonal')(omega_d)
    omega_att = tf.keras.layers.Softmax()(omega_att)

    c_d = tf.keras.layers.Lambda(lambda x: tf.einsum('ij,k->ijk', x, tf.ones(n_d*2)))(omega_att)
    c_d = tf.keras.layers.Multiply()([h_d_bidrection, c_d])
    c_d = tf.keras.layers.Flatten()(c_d)
    
    
    x_m = tf.keras.layers.Input((k, c_m))
    v_m = tf.keras.layers.Lambda(lambda x: x / tf.norm(x, axis=0), name='norm')(x_m)
    v_m = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(m, use_bias=False), name='soft_embedding')(v_m)
    h_m_bidrection = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(n_d, return_sequences=True))(v_m)
    omega_m = tf.keras.layers.Dense(k, name='omega_att')(h_m_bidrection)
    omega_att_m = tf.keras.layers.Lambda(lambda x: tf.linalg.diag_part(x), name='diagonal')(omega_m)
    omega_att_m = tf.keras.layers.Softmax()(omega_att_m)

    c_m = tf.keras.layers.Lambda(lambda x: tf.einsum('ij,k->ijk', x, tf.ones(n_d*2)))(omega_att_m)
    c_m = tf.keras.layers.Multiply()([h_m_bidrection, c_d])
    c_m = tf.keras.layers.Flatten()(c_m)
    
    c_tilde = tf.keras.layers.concat()[c_m, c_d]
    
    o = tf.keras.layers.Dense(n_h)(c_tilde)
    y = tf.keras.layers.Dense(2, activation='softmax')(o)
    
    model = tf.keras.models.Model([x, m_d], y)

    return model
