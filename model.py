import tensorflow as tf


class EHAN(object):

    def __init__(self, config):
        '''

        Parameters
        ----------
        config: dict
            keys:
            k . # of visiting
            c_d : Voca size of diagnosis code
            c_m : Voca size of medicatio code
            m : output vector size of embedding layer. conventionally, 4 squared root of c
                e.g.) 4 ~ 3.5 = sqrt^{4}(100)
            n_d = number of unit of GRU layer
            n_h = number of unit in dense lyaer  (o)
        '''

        self.k = config['k']
        self.c_d = config['c_d']
        self.c_m = config['c_m']
        self.m = config['m']
        self.n_d = config['n_d']
        self.n_h = config['n_h']



    def build_model(self):
        '''
        Parameters
        ----------


        Returns
        -------
        tf.keras.models.Sequntial


        '''

        # Diagnosis part
        x_d = tf.keras.layers.Input((self.k, self.c_d))
        v_d = tf.keras.layers.Lambda(lambda x: x / tf.norm(x, axis=0), name='norm_d')(x_d)
        v_d = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.m, use_bias=False), name='soft_embedding_d')(v_d)
        h_d_bidrection = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.n_d, return_sequences=True))(v_d)
        omega_d = tf.keras.layers.Dense(self.k, name='omega_att')(h_d_bidrection)
        omega_att_d = tf.keras.layers.Lambda(lambda x: tf.linalg.diag_part(x), name='diagonal')(omega_d)
        omega_att_d = tf.keras.layers.Softmax()(omega_att_d)

        c_d = tf.keras.layers.Lambda(lambda x: tf.einsum('ij,k->ijk', x, tf.ones(self.n_d * 2)))(omega_att_d)
        c_d = tf.keras.layers.Multiply()([h_d_bidrection, c_d])
        c_d = tf.keras.layers.Flatten()(c_d)

        # Medication part
        x_m = tf.keras.layers.Input((self.k, self.c_m))
        v_m = tf.keras.layers.Lambda(lambda x: x / tf.norm(x, axis=0), name='norm_m')(x_m)
        v_m = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.m, use_bias=False), name='soft_embedding_m')(v_m)
        h_m_bidrection = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.n_d, return_sequences=True))(v_m)
        omega_m = tf.keras.layers.Dense(self.k, name='omega_att_m')(h_m_bidrection)
        omega_att_m = tf.keras.layers.Lambda(lambda x: tf.linalg.diag_part(x), name='diagonal_m')(omega_m)
        omega_att_m = tf.keras.layers.Softmax()(omega_att_m)

        c_m = tf.keras.layers.Lambda(lambda x: tf.einsum('ij,k->ijk', x, tf.ones(self.n_d * 2)))(omega_att_m)
        c_m = tf.keras.layers.Multiply()([h_m_bidrection, c_m])
        c_m = tf.keras.layers.Flatten()(c_m)

        c_tilde = tf.keras.layers.Concatenate()([c_m, c_d])

        o = tf.keras.layers.Dense(self.n_h)(c_tilde)
        y = tf.keras.layers.Dense(2, activation='softmax')(o)

        model = tf.keras.models.Model([x_d, x_m], y)

        return model
