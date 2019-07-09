import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 绘制参数变化
def variable_summaries(var):
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)

def regression(x, keep_prob):
    # 神经网络的实现
    def weight_variable(shape):
        # 权重的初始化
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        # 偏置项的初始化
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 第一层
    # summaries_1
    W1 = weight_variable([784, 500])
    variable_summaries(W1)

    b1 = bias_variable([500])
    variable_summaries(b1)

    preactivate1 = tf.matmul(x, W1) + b1
    tf.summary.histogram('linear_compute01', preactivate1)

    L1 = tf.nn.relu(preactivate1)
    tf.summary.histogram('activations01', L1)

    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

    # 第二层
    # summaries_2
    W2 = weight_variable([500, 10])
    variable_summaries(W2)

    b2 = bias_variable([10])
    variable_summaries(b2)

    preactivate2 = tf.matmul(L1, W2) + b2
    tf.summary.histogram('linear_compute02', preactivate2)

    # 输出层
    y = tf.nn.softmax(preactivate2)
    return y, [W1, b1, W2, b2]

def main():
    # 训练函数
    data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    log_dir = 'data/log'    # 输出日志保存的路径

    with tf.variable_scope("regression"):
        x = tf.placeholder(tf.float32, [None, 784])
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        # softmax函数
        y, variables = regression(x, keep_prob)

    # train
    y_ = tf.placeholder("float", [None, 10])
    #交叉熵
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    tf.summary.scalar('loss', cross_entropy)
    #梯度下降  学习率定为0.001
    learning_rate = 0.001
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    '''
    tf.argmax(input, axis=None, name=None, dimension=None)
    此函数是对矩阵按行或列计算最大值
    input：输入Tensor
    axis：0表示按列，1表示按行
    name：名称
    dimension：和axis功能一样，默认axis取值优先。新加的字段
    '''
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #计算均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(variables)
    acc = 0

    # summaries合并
    merged = tf.summary.merge_all()
    

    with tf.Session() as sess:
        # 写到指定的磁盘路径中
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test')

        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            #每次取100个实例
            batch_xs, batch_ys = data.train.next_batch(100)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob : 0.9})
            train_writer.add_summary(summary, i)

        for j in range(1000):
            batch_test = data.test.next_batch(100)
            summary, s = sess.run([merged, accuracy], feed_dict={x: batch_test[0], y_: batch_test[1], keep_prob : 1.0})
            if j % 10 == 0:
                print(s)
            test_writer.add_summary(summary, j)
            acc += s
        # total:0.948700001776
        print('total:' + str(acc / 1000))

        path = saver.save(
            sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
            write_meta_graph=False, write_state=False)
        print("Saved:", path)

        train_writer.close()
        test_writer.close()

if __name__ == '__main__':
    main()