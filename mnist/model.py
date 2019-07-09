import tensorflow as tf

# Softmax Regression Model
# rnn训练接口  还需要完善
def regression(x, keep_prob):
    def weight_variable(shape):
        # 权重的初始化
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        # 偏置项的初始化
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 第一层
    W1 = weight_variable([784, 500])
    b1 = bias_variable([500])
    preactivate1 = tf.matmul(x, W1) + b1
    L1 = tf.nn.relu(preactivate1)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    # 第二层
    W2 = weight_variable([500, 10])
    b2 = bias_variable([10])
    preactivate2 = tf.matmul(L1, W2) + b2
    y = tf.nn.softmax(preactivate2)
    return y, [W1, b1, W2, b2]


# Multilayer Convolutional Network
# cnn训练函数接口
def convolutional(x, keep_prob):
    def conv2d(x, W):
        # 实现卷积
        # x代表输入图像矩阵 W代表卷积核
        # keep_prob 代表每个元素被保留的概率
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        # stride [1, x_movement, y_movement, 1]
        # 接着定义池化pooling，为了得到更多的图片信息，padding时我们选的是一次一步，也就是strides[1]=strides[2]=1，这样得到的图片尺寸没有变化，
        # 而我们希望压缩一下图片也就是参数能少一些从而减小系统的复杂度，因此我们采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。
        # pooling 有两种，一种是最大值池化，一种是平均值池化，本例采用的是最大值池化tf.max_pool()。池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape):
        # 权重的初始化
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        # 偏置项的初始化
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    with tf.name_scope('first_layer'):
    # 隐层1
    # First Convolutional Layer
    # 我们需要处理我们的xs，把xs的形状变成[-1,28,28,1]，-1代表先不考虑输入的图片例子多少这个维度，
    # 后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1，例如如果是RGB图像，那么channel就是3。
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        # 卷积核patch的大小是5x5，因为黑白图片channel是1所以输入是1，输出是32个featuremap（经验值）
        W_conv1 = weight_variable([3, 3, 1, 32])  # 卷积核
        # variable_summaries(W_conv1)

        b_conv1 = bias_variable([32])
        # variable_summaries(b_conv1)

        # relu激励函数
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        # tf.summary.histogram('activations01', h_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        L1 = tf.nn.dropout(h_pool1, keep_prob=keep_prob)

    with tf.name_scope('second_layer'):
        # 隐层2
        # Second Convolutional Layer 加厚一层，图片大小14*14
        W_conv2 = weight_variable([3, 3, 32, 64])
        # variable_summaries(W_conv2)

        b_conv2 = bias_variable([64])
        # variable_summaries(b_conv2)

        h_conv2 = tf.nn.relu(conv2d(L1, W_conv2) + b_conv2)
        # tf.summary.histogram('activations02', h_conv2)
        # 池化层
        h_pool2 = max_pool_2x2(h_conv2)
        L2 = tf.nn.dropout(h_pool2, keep_prob=keep_prob)

    with tf.name_scope('third_layer'):
        # 隐层3
        # Third Convolutional Layer 加厚一层，图片大小14*14
        W_conv3 = weight_variable([3, 3, 64, 128])
        # variable_summaries(W_conv2)

        b_conv3 = bias_variable([128])
        # variable_summaries(b_conv2)

        h_conv3 = tf.nn.relu(conv2d(L2, W_conv3) + b_conv3)
        # tf.summary.histogram('activations02', h_conv2)
        # 池化层
        h_pool3 = max_pool_2x2(h_conv3)
        L3 = tf.nn.dropout(h_pool3, keep_prob=keep_prob)

    with tf.name_scope('densely_layer'):
        # 隐层4 全连接层
        # Densely Connected Layer 加厚一层，图片大小7*7
        W_fc1 = weight_variable([128 * 4 * 4, 625])
        # variable_summaries(W_fc1)

        # 特征经验 2的n次方，如32,64,1024
        b_fc1 = bias_variable([625])
        # variable_summaries(b_fc1)

        h_pool2_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # tf.summary.histogram('activations03', h_fc1)

        # Dropout 剪枝
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

    with tf.name_scope('output_layer'):
        # 输出层
        # Readout Layer
        W_fc2 = weight_variable([625, 10])
        # variable_summaries(W_fc2)
        b_fc2 = bias_variable([10])
        # variable_summaries(b_fc2)
        out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        # tf.summary.histogram('output', out)
        y = tf.nn.softmax(out)  # 最终返回是softmax分类器

    return y, [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
