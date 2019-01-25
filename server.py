from flask import Flask, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from flask import render_template
import NN.NeuralNetwork as NN
import os
import sys
import numpy
from random import randint

app = Flask(__name__)
app.secret_key = 'some_secret'
allowed_file = ["png","jpeg","jpg"]
upload_folder = "./static/ori/"
app.config['UPLOAD_FOLDER'] = upload_folder

def allowed(filename):
    allow = set(allowed_file)
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allow


@app.route("/")
def hello():
    user = {'l1': NN.nodes_in_input_layer,'l2': NN.nodes_in_hidden_layer,'l3': NN.nodes_in_output_layer,'akurasi' : NN.accuration()}
    user['file'] = filter(None,[v if v.split('.').pop() in allowed_file else "" for v in os.listdir("static/clean")])
    return render_template('home.html', title='Home', info=user)


@app.route("/training")
def train():
    user = {'l1': NN.nodes_in_input_layer,'l2': NN.nodes_in_hidden_layer,'l3': NN.nodes_in_output_layer,'akurasi' : NN.accuration()}
    user['exist'] = os.path.isfile("./NN/mnist_train.csv")
    return render_template('training.html', title='Training', info=user)

@app.route("/trainingcoba")
def traintest():
    user = {}
    randomline = randint(0,25)
    with open("./NN/mnist_train.csv") as fp:
        for i, line in enumerate(fp):
            if i == randomline:
                data = line
                break

    explode = data.split(',')

    #rescale and shifting
    #don't include the first character because it's the label
    #divide by 255 to make it in range 0-1
    #multiply it by 0.99 to make it in range 0.0 - 0.99
    #add 0.01 so the lower value is 0.01 not 0
    real = explode[1:]
    bagi = numpy.asfarray(real)/255
    kali = bagi*0.99
    tambah = kali+0.01

    #get the label and make it into array
    label = numpy.zeros(NN.nodes_in_output_layer)+0.01
    label[int(explode[0])] = 0.99

    user['ori'] = data
    user['label'] = explode[:1]
    user['real'] = real
    user['bagi'] = bagi
    user['kali'] = kali
    user['tambah'] = tambah
    user['matrix_label'] = label
    user['train'] = NN.brain.train(tambah,label, detail=True)

    return render_template('trainingcoba.html', title='Training Coba', info=user)

@app.route("/test",methods=["post","get"])
def test():
    user = {}
    if request.method == "GET":
        user['clean'] = request.args.get("img")
    else:
        if 'file' not in request.files:
            flash('No file part')
            return redirect("/")
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect("/")
        if file and allowed(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            user['clean'] = filename
        else:
            flash('Format file tidak didukung(png,jpeg,jpg)')
            return redirect("/")


    user['guess'] = NN.test(user['clean'])
    #return user
    return render_template('test.html', title='Test', info=user)


if __name__ == "__main__":
    app.run()
