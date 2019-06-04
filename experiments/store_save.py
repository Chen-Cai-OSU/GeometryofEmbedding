import tensorflow as tf

# Create some variables.

v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

inc_v1 = v1.assign(v1 + 1)
dec_v2 = v2.assign(v2 - 1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

vars = [v for v in tf.global_variables()]
print (vars)
print(vars[0].name)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
    sess.run(init_op)
    # Do some work with the model.
    inc_v1.op.run()
    dec_v2.op.run()
    # Save the variables to disk.
    save_path = saver.save(sess, "/tmp/model.ckpt")


# restore
tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver([v1, v2])

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "/tmp/model.ckpt")
    print("Model restored.")
    # Check the values of the variables
    print("v1 : %s" % v1.eval())
    print("v2 : %s" % v2.eval())


from tensorflow.python import pywrap_tensorflow
reader = pywrap_tensorflow.NewCheckpointReader("//home/cai.507/Documents/DeepLearning/AmpliGraph-master/save_models/my-model-1")
var_to_shape_map = reader.get_variable_to_shape_map()  # 'var_to_shape_map' is a dictionary contains every tensor in the model


if 'v1' in var_to_shape_map.keys():
    print(var_to_shape_map['v1'])
