import os
import cv2
import random
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy
from tensorflow.core.protobuf import saver_pb2

xs = []
ys = []

train_batch_pointer = 0
val_batch_pointer = 0

OUTPUTDIR = './output'


with open("steering_data/data.txt") as f:
    for line in f:
        xs.append("steering_data/" + line.split()[0])
        ys.append(float(line.split()[1]) * 3.14159265 / 180)


# shuffling the images
num_images = len(xs)


c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def load_train_batch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(train_xs[(train_batch_pointer + i) % num_train_images])[-150:], (200, 66)) / 255.0)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    return x_out, y_out

def load_test_batch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(0, batch_size):
        x_out.append(cv2.resize(cv2.imread(val_xs[(val_batch_pointer + i) % num_val_images])[-150:], (200, 66)) / 255.0)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    return x_out, y_out
    
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')
  
# Creating the model architeture
x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = x

#Conv 1
W_conv1 = weight_variable([5, 5, 3, 24])
b_conv1 = bias_variable([24])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)

#Conv 2
W_conv2 = weight_variable([5, 5, 24, 36])
b_conv2 = bias_variable([36])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)

#Conv 3
W_conv3 = weight_variable([5, 5, 36, 48])
b_conv3 = bias_variable([48])
  
#Conv 4
W_conv4 = weight_variable([3, 3, 48, 64])
b_conv4 = bias_variable([64])

h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)

#Conv 5
W_conv5 = weight_variable([3, 3, 64, 64])
b_conv5 = bias_variable([64])

h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)

#FCL 1
W_fc1 = weight_variable([1152, 1164])
b_fc1 = bias_variable([1164])

h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#FCL 2
W_fc2 = weight_variable([1164, 100])
b_fc2 = bias_variable([100])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#FCL 3
W_fc3 = weight_variable([100, 50])
b_fc3 = bias_variable([50])

h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

#FCL 3
W_fc4 = weight_variable([50, 10])
b_fc4 = bias_variable([10])

h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

#Output
W_fc5 = weight_variable([10, 1])
b_fc5 = bias_variable([1])

y = tf.multiply(tf.atan(tf.matmul(h_fc4_drop, W_fc5) + b_fc5), 2)


# Training the model

def train_model():
   sess = tf.InteractiveSession()

   L2NormConst = 0.001

   train_vars = tf.trainable_variables()

   loss = tf.reduce_mean(tf.square(tf.subtract(y_, y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
   train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
   sess.run(tf.global_variables_initializer())

   tf.summary.scalar("loss", loss)
   merged_summary_op = tf.summary.merge_all()

   saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)

   logs_path = './logs'
   summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

   for epoch in range(epochs):
      for i in range(int(driving_data.num_images/batch_size)):
         xs, ys = load_train_batch(batch_size)
         train_step.run(feed_dict={x: xs, y_: ys, keep_prob: 0.8})
         if i % 10 == 0:
           xs, ys = load_test_batch(batch_size)
           loss_value = loss.eval(feed_dict={x:xs, y_: ys, keep_prob: 1.0})
           print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

         # write logs at every iteration
         summary = merged_summary_op.eval(feed_dict={x:xs, y_: ys, keep_prob: 1.0})
         summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

         if i % batch_size == 0:
           if not os.path.exists(OUTPUTDIR):
             os.makedirs(OUTPUTDIR)
           checkpoint_path = os.path.join(OUTPUTDIR, "self_driving_model.ckpt")
           filename = saver.save(sess, checkpoint_path)
       print("Model saved in file: %s" % filename)
       

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Train the self driving car model')
   parser.add_argument('--epochs', default='30', help='enter epoch number')
   parser.add_argument('--batch_size', default='100', help='enter batch size')
   args = parser.parse_args()
   
   train_model(args.epochs, args.batch_size)
