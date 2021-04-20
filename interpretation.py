import tensorflow as tf

class GradCAM(object):
    
    def __init__(self, model):
        self.model = model

    def _create_subgraph(self):
        subgraph = tf.keras.models.Model([self.model.inputs], 
                                         [self.model.get_layer('element_multiply').output, self.model.output])
        return subgraph
        
    def generated_grad_cam(self, tensor_x, class_index=0):
        
        # L_c_d (Equation 14) :
        with tf.GradientTape() as tape:
            subgraph = self._create_subgraph()
            subgraph.layers[-1].activation = tf.keras.activations.relu
            att_output, prediction = subgraph(tensor_x)
            loss = prediction[:, class_index]
        
        Lc = tape.gradient(loss, att_output)[0]
        
        # GAP (Equation 15)
        w_grad_d =  tf.reduce_mean(Lc, axis=0)


        # g_c_d (Equatio 16)
        g = tf.einsum('i,j->ij', w_grad_d, tf.ones(k))
        g = tf.linalg.diag_part(Lc @ g)
        
        return g
