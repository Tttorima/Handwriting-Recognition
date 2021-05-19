# Handwriting Recognition

> Data source: MNIST data set (large data set containing handwritten digital images)
>
> It consists of 0-9 handwritten digital pictures, which are 10 categories, each of which is 28 * 28. Black and white images are single channel.

```pyt
from tensorflow.examples.tutorials.mnist import input_data
```



> Method：tensorflow
>
> Tensorflow provides the related types of handwritten numeral recognition

```python
import tensorflow as tf
```



> Details：
>
> Neural network construction using tensorflow
>
> Single layer neural network

```python
# Input
X = tf.placeholder(dtype = tf.float32, shape=[None,784])
# Output
y = tf.placeholder(dtype = tf.float32, shape=[None,10])

# Single layer neural network
# Use softmax function
W = tf.Variable(tf.zeros(shape=[784,10]))
b = tf.Variable(tf.zeros(shape=[1,10]))

# Matmul
z = tf.matmul(X,W) + b

# Softmax
a = tf.nn.softmax(z）
                  
# loss function:Cross entropy loss function
loss = -tf.reduce_sum(y*tf.log(a))
```



> Optimization：GradientDescentOptimizer
>
> Learning rate：0.01

​	The maximum accuracy of the single layer neural network is 92%



> Improvement：Add hidden layer
>
> The number of neurons was set to 256
>
> Weight initialization：Standard normal distribution

```python
# placeholder(Input & Output)
X=tf.placeholder(dtype=tf.float32,shape=[None,784])
y=tf.placeholder(dtype=tf.float32,shape=[None,10])

# Weight initialization
W=tf.Variable(tf.random_normal(shape=[784,256],stddev=0.1))
b=tf.Variable(tf.zeros(shape=[1,256]))
z=tf.matmul(X,W)+b

# neural network.relu
# If the input value < 0, it will be assigned as 0
# Else (value>=0) will remain unchanged
a=tf.nn.relu(z)

# second layer
W2=tf.Variable(tf.random_normal(shape=[256,10],stddev=0.05))
b2=tf.Variable(tf.zeros(shape=[1,10]))
z2=tf.matmul(a,W2)+b2

# softmax : convert score to probability
a2=tf.nn.softmax(z2)
```



> Result

​	The maximun accuracy is 98%, which is 6% higher than that of single layer nn.