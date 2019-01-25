#!/usr/bin/env python
from __future__ import division #fix division py2 (precise)
import numpy
import scipy.special
import sys
import random
import imageio
from PIL import Image

class NN:
    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate,load=False):
        #set nodes
        self.inodes = input_layer
        self.hnodes = hidden_layer
        self.onodes = output_layer
        self.lr = learning_rate

        #set weight
        if load:
            self.who = numpy.loadtxt("./NN/who", dtype=float)
            self.wih = numpy.loadtxt("./NN/wih", dtype=float)
        else:
            self.wih = numpy.random.normal(0.0, pow(self.inodes,-0.5), (self.hnodes,self.inodes) )#set link between input layer and hidden layer
            self.who = numpy.random.normal(0.0, pow(self.hnodes,-0.5), (self.onodes,self.hnodes) )#set link between hidden layer and output layer

        #set activation function (sigmoid)
        self.activate = lambda x : scipy.special.expit(x)

    def train(self, input_nodes,target, detail=False):
        #import list into 2d array
        inputs = numpy.array(input_nodes, ndmin=2).T
        #reformat target
        target = numpy.array(target, ndmin=2).T
        #sigmoid of input layer and hidden layer
        dotih = numpy.dot(self.wih,inputs)
        input_out = self.activate(dotih)

        #sigmoid from hidden layer to output layer
        dotho = numpy.dot(self.who,input_out)
        output = self.activate(dotho)

        #calc error
        #print numpy.argmax(target), numpy.argmax(output)
        hidden_error = target - output

        #backpropagated to hidden layer
        hidden_output = numpy.dot(self.who.T, hidden_error)

        gradient = self.lr * numpy.dot((hidden_error*output*(1.0-output)),numpy.transpose(input_out))
        gradient1 = self.lr * numpy.dot((hidden_output*input_out*(1.0-input_out)),numpy.transpose(inputs))

        if detail:
            out = {}
            out['inputs'] = inputs
            out['target'] = target
            out['dotih'] = dotih
            out['input_out'] = input_out
            out['dotho'] = dotho
            out['out'] = output
            out['wih'] = self.wih
            out['who'] = self.who
            out['error'] = hidden_error
            out['gradient'] = gradient

        #update link between hidden and output
        self.who += gradient

        #update link between hidden and input layer
        self.wih += gradient1

        if detail:
            float_formatter = lambda x: "%.2f" % x
            numpy.set_printoptions(formatter={'float_kind':float_formatter},threshold=sys.maxsize)
            return out

        pass


    def test(self, input_nodes, flat=False):
        #import list into 2d array
        input_nodes = numpy.array(input_nodes, ndmin=2).T

        #sigmoid of input layer and hidden layer
        dotih = numpy.dot(self.wih,input_nodes)
        hidden_out = self.activate(dotih)

        #sigmoid from hidden layer to output layer
        dotho = numpy.dot(self.who,hidden_out)
        out = self.activate(dotho)

        if flat:
            return out

        output = {'flat': input_nodes,'wih': self.wih, 'dotih': dotih, 'hidden_out': hidden_out,'who':self.who,'dotho':dotho,'out':out}

        return output

    def save(self):
        numpy.savetxt("./NN/wih",self.wih)
        numpy.savetxt("./NN/who",self.who)
        pass

#define the brain
learning_rate = 0.1
nodes_in_input_layer = 784
nodes_in_hidden_layer = 200
nodes_in_output_layer = 10
brain = NN(nodes_in_input_layer,nodes_in_hidden_layer,nodes_in_output_layer,learning_rate,load = True)

def accuration():
    #Test
    data_file = open("./NN/mnist_test.csv", 'r')
    data_test = data_file.readlines()
    data_file.close()
    score = []

    for line in data_test:
        #split by the comma
        explode = line.split(',')

        #rescale and shifting
        #don't include the first character because it's the label
        #divide by 255 to make it in range 0-1
        #multiply it by 0.99 to make it in range 0.0 - 0.99
        #add 0.01 so the lower value is 0.01 not 0
        pixel = (numpy.asfarray(explode[1:]) / 255 * 0.99) + 0.01

        #test and get the higher
        test = brain.test(pixel,flat=True)
        result = numpy.argmax(test)
        expectation = int(explode[0])
        #print result,expectation
        if result == expectation:
            score.append(1)
        else:
            score.append(0)

    #change to percentage
    score_arr = numpy.asarray(score)
    accuration = score_arr.sum() / score_arr.size
    akurasi = accuration*100
    return akurasi


def test(filename):
    guess = {}
    ori_filename = "./static/ori/"+filename
    filename = "./static/clean/"+filename
    im = Image.open(ori_filename).convert('L')
    guess["ori_size"] = im.size
    res = im.resize((28,28), Image.ANTIALIAS)
    res.save(filename)

    #test with real image
    img = imageio.imread(filename,as_gray=True)
    guess['clean_size'] = img.shape
    guess['img_ori'] = str(img)
    img_reshape = img.reshape(784)
    guess['img_reshape'] = str(img_reshape)
    img_data = 255-img_reshape
    guess['img_range'] = str(img_data)
    pixel = (numpy.asfarray(img_data)/255 * 0.99) + 0.01
    guess['pixel'] = str(pixel)
    test = brain.test(pixel)

    float_formatter = lambda x: "%.2f" % x
    numpy.set_printoptions(formatter={'float_kind':float_formatter})

    #sys.exit()
    guess['guess'] = numpy.argmax(test['out'])
    guess['percentage'] = numpy.amax(test['out'])*100
    guess['test'] = test
    return guess
