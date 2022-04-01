# Venture Funding with Deep Learning
This analysis works through the process of creating a deep neural network model that predicts whether startup applicants will be successful if funded by a venture capital firm named Alphabet Soup. It uses a CSV file that contains data on more than 34,000 organizations that have received funding from Alphabet Soup over the years. The CSV file contains various features such as use case, income amount, and ask amount. It also includes a column that indicates whether or not the startup ultimately became successful. The features in this provided dataset are used to create a binary classification model that will predict whether a startup applicant will become a successful business.

---

## Technologies 
This analysis uses [Google Colab](https://colab.research.google.com/?utm_source=scs-index) and the standard Python 3.8 libraries. In addition, this project requires the following libraries and/or dependencies: 
* [pandas](https://pandas.pydata.org/) - a software library designed for open source data analysis and manipulation
* [tensorflow](https://www.tensorflow.org/) - end-to-end open source library to help you develop and train machine learning models
* [sklearn](https://scikit-learn.org/stable/) - simple and efficient tools for predictive data analysis; built on NumPy, SciPy, and matplotlib
* [Path](https://pypi.org/project/path/) - implements path objects as first-class entities, allowing common operations on files to be invoked on those path objects directly

---

## Installation Guide

Before running the application, first install the following libraries in the first code cell (or in your terminal):
```
pip install -U scikit-learn
pip install --upgrade tensorflow
```

Verify the installations:
```
conda list scikit-learn
python -c "import tensorflow as tf;print(tf.__version__)"
python -c "import tensorflow as tf;print(tf.keras.__version__)"
```

---

## Usage
To interact with the analysis of venture funding with deep neural network models:
1. Clone the repository
`git clone https://github.com/ccroft6/Venture_Funding_Deep_Learning.git`

2. Open a Google Colab notebook and upload the "GC_venture_funding_with_deep_learning.ipynb" file that you cloned 

**OR** 

Open the terminal at this repository location. Activate the environment and launch jupyter lab:
```
conda activate dev
jupyter lab 
```
*On some computers, tensorflow may not install correctly. Therefore, Google Colab is offered as an alternative way to use the tensorflow library without any issues.*

---

## Methods

1. Preprocess data for a neural network model.

2. Use the model-fit-predict pattern to compile and evaluate a binary classification model using a neural network.

3. Optimize the neural network model.

---

## Results 
An initial neural network model was created to see how it would perform. Then, three alternative neural network models were created to optimize the model and try to improve on the initial model's predictive accuracy. 

### Initial Model
**Pre-settings:**
* Number of features - 116
* Activation of hidden layers function - ```relu```
* Activation of output layer - ```sigmoid```
* Number of epochs - 50

Model: "sequential"
_________________________________________________________________
Layer (type)               Output Shape              Param #   
 dense (Dense)               (None, 58)                6786      
                                                                 
 dense_1 (Dense)             (None, 29)                1711      
                                                                 
 dense_2 (Dense)             (None, 1)                 30        
                                                                 
--------------- -------------------------------- ----------------
Total params: 8,527
Trainable params: 8,527
Non-trainable params: 0

**Original Model Results:**
268/268 - 0s - loss: 0.5587 - accuracy: 0.7307 - 240ms/epoch - 896us/step
Loss: 0.5587013959884644, Accuracy: 0.7307288646697998

### Alternative Model 1
**Pre-settings:**
* Number of features - 116
* Activation of hidden layers function - ```relu```
* Activation of output layer - ```sigmoid```
* Number of epochs - 50

Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
 dense_3 (Dense)             (None, 58)                6786      
                                                                 
 dense_4 (Dense)             (None, 29)                1711      
                                                                 
 dense_5 (Dense)             (None, 15)                450       
                                                                 
 dense_6 (Dense)             (None, 1)                 16        
                                                                 
---------- ------------- ------------- ----------- --------------
Total params: 8,963
Trainable params: 8,963
Non-trainable params: 0

**Alternative Model 1 Results:**
268/268 - 0s - loss: 8.1631 - accuracy: 0.4708 - 337ms/epoch - 1ms/step
Loss: 8.163082122802734, Accuracy: 0.4707871675491333

### Alternative Model 2
**Pre-settings:**
* Number of features - 116
* Activation of hidden layers function - ```relu```
* Activation of output layer - ```sigmoid```
* Number of epochs - 100

Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
 dense_7 (Dense)             (None, 58)                6786      
                                                                 
 dense_8 (Dense)             (None, 29)                1711      
                                                                 
 dense_9 (Dense)             (None, 1)                 30        
                                                                 
=================================================================
Total params: 8,527
Trainable params: 8,527
Non-trainable params: 0

**Alternative Model 2 Results:**
268/268 - 0s - loss: 0.5605 - accuracy: 0.7299 - 323ms/epoch - 1ms/step
Loss: 0.5605130195617676, Accuracy: 0.729912519454956

### Alternative Model 3

**Pre-settings:**
* Number of features - 116
* Activation of hidden layers function - ```relu```
* Activation of output layer - ```sigmoid```
* Number of epochs - 80

Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
 dense_10 (Dense)            (None, 80)                9360      
                                                                 
 dense_11 (Dense)            (None, 40)                3240      
                                                                 
 dense_12 (Dense)            (None, 20)                820       
                                                                 
 dense_13 (Dense)            (None, 1)                 21        
                                                                 
--------- -------------- ------------- ---------------------------
Total params: 13,441
Trainable params: 13,441
Non-trainable params: 0

**Alternative Model 3 Results:**
268/268 - 0s - loss: 0.5640 - accuracy: 0.7308 - 247ms/epoch - 922us/step
Loss: 0.5640198588371277, Accuracy: 0.7308454513549805

## Conclusions
In alternative model 1, a third hidden node layer was added. The number of epochs and the number of nodes in the first and second layers remained the same as the initial model. By adding a third layer, the loss increased greatly from 0.6 (initial model) to 8.2, and the accuracy decreased from 73% to 47%. 

In alternative model 2, everything stayed the same as the initial model, but the number of epochs was increased from 50 to 100. By using 100 epochs, the loss stayed the same as the initial model at 0.6 and the accuracy stayed similar at 73%. 

In alternative model 3, three hidden node layers were used, more nodes were added per hidden layer (started at 80 instead of 56 for the first hidden layer), and 80 epochs were run. Despite these changes, the loss stayed the same as the initial model at 0.6 and the accuracy stayed similar at 73%.

In conclusion, there would need to be more trial and error to see if dropping some feature columns, adding more neurons/nodes to a hidden layer, and/or using different activation function for the hidden layers would help increase the accuracy. It takes a lot of trial and error before obtaining a desired neural network model. 

---

## Contributors
Catherine Croft

Email: catherinecroft1014@gmail.com

LinkedIn: [catherine-croft](https://www.linkedin.com/in/catherine-croft-4715481aa/)

---

## License 
MIT