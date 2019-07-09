from django.shortcuts import render
import numpy as np
import tensorflow as tf
from . import model
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import base64
from urllib import parse
from PIL import Image

@csrf_exempt
def get_data(request):
    print(tf.__version__)

    x = tf.placeholder("float", [None, 784])
    sess = tf.Session()

    # restore trained data
    with tf.variable_scope("regression"):
        keep_prob1 = tf.placeholder("float")
        # 调用接口函数
        y1, variables = model.regression(x, keep_prob1)
    # tf.reset_default_graph()
    saver = tf.train.Saver(variables)
    saver.restore(sess, "./regression.ckpt")

    with tf.variable_scope("convolutional"):
        keep_prob2 = tf.placeholder("float")
        # 调用接口函数
        y2, variables = model.convolutional(x, keep_prob2)
    # tf.reset_default_graph()
    saver = tf.train.Saver(variables)
    saver.restore(sess, "./convolutional.ckpt")

    results = list(request.POST.keys())
    if request.method == 'POST':
        print('-----------POST-------------')
        print(results[0][22:])
        data = request.POST.get('data')
        host = request.POST.get('host')
        print(host)
        image = request.POST.get('img')
        image = parse.unquote(image)
        image = image[22:]
        image = base64.b64decode(image)
        fh = open("imageToSave.png", "wb")
        fh.write(image)
        fh.close()
        print("save success!")
        im = Image.open("imageToSave.png")
        rgb_im = im.convert('RGB')
        rgb_im.save('colors.jpg')

        im = Image.open('colors.jpg')
        newImage = im.resize((28, 28), Image.ANTIALIAS).convert("L")
        img = np.array(im.resize((28, 28), Image.ANTIALIAS).convert("L"))
        newImage.save('finished.jpg')
        imdata = img.reshape([1, 784])
        imdata = 1 - (imdata / 255)
        # 28 * 28 = 784个元素
        # 去掉首尾元素
        data = data.strip('[]')
        # ，分割
        data_list = data.split(',')
        # 像素点预处理 转为灰度图
        print('输入元素点数量：')
        print(len(data_list))
        input = ((255 - np.array(data_list, dtype=np.uint8)) / 255.0).reshape(1, 784)
        # print(input)
        # 测试时dropout为1.0
        output1 = sess.run(y1, feed_dict={x: input, keep_prob1: 1.0}).flatten().tolist()
        output2 = sess.run(y2, feed_dict={x: imdata, keep_prob2: 1.0}).flatten().tolist()
        # output1 = [0.019258225336670876, 0.036380793899297714, 0.6032975912094116, 0.12827254831790924, 0.01951347477734089, 0.04987751692533493, 0.09156624972820282, 0.03392419591546059, 0.007857662625610828, 0.010051618330180645]
        print(output1)
        print(output2)
        print('-----------OVER-------------')
        # 最终返回前端是json数据
        return JsonResponse({
            "results":[output1, output2]
        })

