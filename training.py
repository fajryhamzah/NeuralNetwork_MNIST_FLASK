import NN.NeuralNetwork as NN
import random
import sys
import numpy

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

print "Akurasi sekarang : "+str(NN.accuration())+"%"
#train this shit out
epoch = int(raw_input("How much epoch? "))
data_file = open("./NN/mnist_train.csv", 'r')
data_list = data_file.readlines()
data_file.close()
random.shuffle(data_list)
for a in range(epoch):
    counter = 0
    for line in data_list:
        #if counter >= 100:
        #    break
        progress(counter, len(data_list), status=' Iter '+str(a)+'/'+str(epoch-1))
        #split by the comma
        explode = line.split(',')

        #rescale and shifting
        #don't include the first character because it's the label
        #divide by 255 to make it in range 0-1
        #multiply it by 0.99 to make it in range 0.0 - 0.99
        #add 0.01 so the lower value is 0.01 not 0
        pixel = (numpy.asfarray(explode[1:]) / 255 * 0.99) + 0.01

        #get the label and make it into array
        label = numpy.zeros(NN.nodes_in_output_layer)+0.01
        label[int(explode[0])] = 0.99

        #train it
        NN.brain.train(pixel,label)
        counter += 1
pass

NN.brain.save()
print "\nAkurasi : "+str(NN.accuration())+"%"
