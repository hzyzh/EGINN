import tensorflow.compat.v1 as tf


hidden_size=3

x = tf.constant([[1.0, 2.0, 3.0], [3.0, 4.0, 4.0], [3.2, 4.2, 5.0]])
# 1 means original and 0 means replaced
y = tf.constant([1,0,1])
#y = tf.one_hot(y, 2)

print(y)


output_weights = tf.get_variable(
      name="Weight_for_RTD",
      shape=[hidden_size, 2]
    )

input_tensor = tf.layers.dense(
          x,
          units=hidden_size)

output_bias = tf.get_variable(
    "output_bias",
    shape=[2],
    initializer=tf.zeros_initializer())
logits = tf.matmul(input_tensor, output_weights)
logits = tf.nn.bias_add(logits, output_bias)
print(logits)
log_probs = tf.nn.log_softmax(logits, axis=-1)
print(log_probs)

origPredProb=tf.gather(log_probs, [1], axis=1)
print(origPredProb)
y = tf.cast(y, tf.float32)
y = tf.expand_dims(y, axis=-1)
print(y)


res = origPredProb*y + (1-y)*(1-origPredProb)
print(res)


loss = tf.math.reduce_sum(res)

print(loss)



