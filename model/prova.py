
import numpy as np
import tensorflow as tf
import torch

a = np.array([5.0,22.0,22.0,22.0])
b = np.array([0.0,0.9,0.9,0.9])



dot_product = tf.tensordot(a, b, axes=1)
norm_tensor1 = tf.norm(a, ord='euclidean')
norm_tensor2 = tf.norm(b, ord='euclidean')
cosine_distance = dot_product / (norm_tensor1 * norm_tensor2)

print("euclidean_distance", tf.exp(-tf.square(tf.norm(a - b))))
print( "cosine_distance",tf.exp(-(1-cosine_distance)))

print("euclid ",tf.norm(norm_tensor1 - norm_tensor2))