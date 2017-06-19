import tensorflow as tf

def mkMask(input_tensor, maxLen):
    shape_of_input = tf.shape(input_tensor)
    shape_of_output = tf.concat(0, [shape_of_input, [maxLen]])
    
    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)

def reduce_avg(reduce_tensor, mask_tensor, lengths_tensor, dim=-2):
    """
    Args:
        reduce_tensor : which tensor to average dtype float point
        mask_tensor   : same shape as reduce_tensor
        lengths_tensor : same rank as tf.reduce_sum(reduce_tensor * mask_tensor, reduction_indices=k)
        dim : which dim to average
    """
    red_sum = tf.reduce_sum(reduce_tensor * tf.to_float(mask_tensor), reduction_indices=[dim], keep_dims=False)
    red_avg = red_sum / (tf.to_float(lengths_tensor) + 1e-20)
    return red_avg
