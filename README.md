# Description

This is a virtual environtment of Neural Network of handwriting digit classifier (MNIST) made with python and flask. This model has 97,56% accuration.


# How to use

- Install virtualenv using pip (2.7)
- Activate the virtualenv
- Install dependencies using python
- Download the training dataset and test dataset

[Training datasets](http://www.pjreddie.com/media/files/mnist_train.csv)

[Test datasets](http://www.pjreddie.com/media/files/mnist_test.csv)

- Put the downloaded file into NN folder
- Export some flask environtment variables
```
EXPORT FLASK_APP=server.py
```
- Run this command:
```
flask run
```

