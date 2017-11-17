import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32, name="weight")
b = tf.Variable([-.3], dtype=tf.float32, name="bias")
# Model input and output
x = tf.placeholder(tf.float32, name="x")
linear_model = tf.add(tf.multiply(W, x), b, "output")
y = tf.placeholder(tf.float32, name="y")

# loss
loss = tf.reduce_sum(tf.square(linear_model - y), name="loss") # sum of the squares
# optimizer
grad_decent_rate = tf.constant(0.01, tf.float32, name="grad_decent_rate")
optimizer = tf.train.GradientDescentOptimizer(grad_decent_rate)
train = optimizer.minimize(loss, name="train")

init = tf.global_variables_initializer()
tf.train.write_graph(tf.get_default_graph().as_graph_def(), 'models', 'linear_train.pb', as_text=False)
