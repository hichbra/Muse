# -*- coding: latin-1 -*-
from __future__ import division
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy import genfromtxt
import csv
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

class AdaBoostHalf:

    def __init__(self, training_set):
        self.training_set = training_set
        self.N = len(self.training_set)
        self.weights = np.ones(self.N)/self.N
        self.RULES = []
        self.ALPHA = []

    def set_rule(self, func, test=False):
        print()
        errors = np.array([t[1]!=func(t[0]) for t in self.training_set]) # Evaluation de la regle de decision

        e = (errors*self.weights).sum()
        if test:
            return e
        alpha = 0.5 * np.log((1-e)/e)  # Note, this line can be deleted since we are using the 1/2 trick
        print ('e={} a={}'.format(e, alpha))
        w = np.zeros(self.N)
        sRight = np.sum(self.weights[errors])
        sWrong = np.sum(self.weights[~errors])
        for i in range(self.N):
            if errors[i] == 1:
                w[i] = self.weights[i]/(2.0*sRight)
            else:
                 w[i] = self.weights[i]/(2.0*sWrong)

        self.weights = w / w.sum()
        print(self.weights)
        self.RULES.append(func)
        self.ALPHA.append(alpha)
        print('alpha = {}'.format(alpha))

    def evaluate(self):
        test = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        NR = len(self.RULES)

        for (x,l) in self.training_set:
            hx = [alpha * rules(x) for alpha, rules in zip(self.ALPHA, self.RULES)]
            test.append(np.sign(l) == np.sign(sum(hx)))
            print (x, np.sign(l) == np.sign(sum(hx)))
        return test

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

"""""""""""""""""""""''''''""""""""""""""""""""""""
"""""""""""" MISE EN PLACE DES DONNEES """"""""""""
"""""""""""""""""""""''''''""""""""""""""""""""""""

my_data = genfromtxt('/home/hicham/Bureau/Stage/Dataset/dataset_1/datasetAssisMD.csv', delimiter=';')
input_signal = np.reshape(my_data[:,1:2].T, -1, 2) # 2eme colonne
capteur = np.reshape(my_data[:,27:28].T, -1, 2)
capteur[capteur > 30] = 100
capteur[capteur <= 30] = 0

SIZE = len(input_signal)

# Frequence d'echantillonnage fs=333.333 Hz
# Filtre passe bande [1 10] Hz et d'ordre 4
FiltreMin = 0.5
FiltreMax = 30
fs = 333.333
Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]

# b, a = signal.butter(1, 10, 'low', analog=True)
b, a = signal.butter(4,  Wn, 'bandpass')
output_signal = signal.filtfilt(b, a, input_signal)

sortie = np.zeros((input_signal.shape[0],2))

for i in range(input_signal.shape[0]):
    sortie[i][0] = output_signal[i]

    if capteur[i] == 100:
        sortie[i][1] = 1

""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return weights

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def get_data():

    #plt.plot(input_signal)
    #plt.plot(output_signal)
    #plt.plot(sortie)

    #plt.show()

    a = np.zeros((input_signal.shape[0],2))
    #print( "sortie = ",sortie[1:5])

    """ Creation des fenetres temporelles """
    TAILLE_FENETRE = 70 # 16

    dataset = np.zeros((input_signal.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire

    #cpt = 0
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset[i][j] = sortie[cpt][0]
            cpt = cpt+1

        dataset[i][TAILLE_FENETRE] = sortie[cpt][1]

    data = dataset[0:dataset.shape[0],0:TAILLE_FENETRE]
    target = dataset[0:dataset.shape[0],TAILLE_FENETRE].astype(int)

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    #all_X = data
    print("-----------------------------------------------------------------\n")
    #print("Data = ",data)
    #print("DataSize = ",len(data))
    #print("Target = ",target)
    #print("N = ",N)
    #print("all_X = ",all_X)

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    #print("all_Y = ",all_Y)

    print(all_X.shape)
    print(all_Y.shape)
    return train_test_split(all_X, all_Y, test_size=0.3, random_state=RANDOM_SEED)


global ECHELLE_PREDICTION
global ECHELLE_SORTIE
ECHELLE_PREDICTION = 200
ECHELLE_SORTIE = 100
"""def useModel(input_signal_Random, model, filtreMin, filtreMax):
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("Graph/EEG/"+model+"/"+model+".meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./Graph/EEG/"+model))

    sortie_Random = np.zeros((input_signal_Random.shape[0],2))

    TAILLE_FENETRE = 70 # 16

    """""" --- FILTRE --- """"""
    FiltreMin = filtreMin
    FiltreMax = filtreMax
    fs = 333.333
    Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]
    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_Random = signal.filtfilt(b, a, input_signal_Random)
    """""" -------------- """"""

    for i in range(input_signal_Random.shape[0]):
        sortie_Random[i][0] = output_signal_Random[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_Random[i][1] = 1


    dataset_RANDOM = np.zeros((input_signal_Random.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM[i][j] = sortie_Random[cpt][0]
            cpt = cpt+1

        dataset_RANDOM[i][TAILLE_FENETRE] = sortie_Random[cpt][1]

    data = dataset_RANDOM[0:dataset_RANDOM.shape[0],0:TAILLE_FENETRE+1]

    fenetres = []
    prediction = []
    prediction_visualisation = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(data.shape[0]):
        fenetre_a_predire = data[i].reshape((1, 71))
        fenetres.append(fenetre_a_predire)
        prediction.append(sess.run(predict, feed_dict={X: fenetre_a_predire})[0])

    print("TAILLE FENETRE",fenetres)
    print("TAILLE",len(prediction))

    for i in range(len(prediction)):
        #for j in range(TAILLE_FENETRE):
        prediction_visualisation.append(prediction[i])

    sess.close()

    #return prediction_visualisation
    return {'predict':prediction_visualisation, 'input':input_signal_Random ,'output':output_signal_Random }
"""
def useModel(fenetre_a_predire, model, signal):


    #prediction = []
    #prediction_visualisation = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    prediction = model.run(predict, feed_dict={X: fenetre_a_predire[signal]})[0]

    #print("TAILLE FENETRE",fenetres)
    #print("TAILLE",prediction)

    #for i in range(len(prediction)):
        #for j in range(TAILLE_FENETRE):
        #prediction_visualisation.append(prediction[i])

    #model.close()

    #return prediction_visualisation
    return prediction

def main():

    train_X, test_X, train_y, test_y = get_data()

    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 70                # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # Symbols
    global X
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    W1 = init_weights((x_size, h_size))
    w_1 = tf.Variable(W1, name="W1")
    W2 = init_weights((h_size, y_size))
    w_2 = tf.Variable(W2, name="W2")

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    global predict
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    # Save
    saver = tf.train.Saver()
    tf.add_to_collection('vars', w_1)
    tf.add_to_collection('vars', w_2)

    # Run SGD


    """ LANCEMENT DU TEST """
    """ -------------------------- SIGNAL 1 l_ear NOUVEAU -------------------------- """
    my_data = genfromtxt("/home/hicham/Bureau/Stage/Dataset/dataset_1/datasetAssisMD.csv", delimiter=';')
    input_signal_l_ear = np.reshape(my_data[:,1:2].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_l_ear)
    capteur = np.reshape(my_data[:,27:28].T, -1, 2)
    for i in range(len(capteur)):
        if capteur[i] > 30:
            capteur[i] = ECHELLE_SORTIE
        else:
             capteur[i] = 0

    sortie_Random = np.zeros((input_signal_l_ear.shape[0],2))

    TAILLE_FENETRE = 70 # 16

    """ --- FILTRE --- """
    FiltreMin = 1
    FiltreMax = 10
    fs = 333.333
    Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]
    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_l_ear = signal.filtfilt(b, a, input_signal_l_ear)
    """ -------------- """

    for i in range(input_signal_l_ear.shape[0]):
        sortie_Random[i][0] = output_signal_l_ear[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_Random[i][1] = 1


    dataset_RANDOM = np.zeros((input_signal_l_ear.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM[i][j] = sortie_Random[cpt][0]
            cpt = cpt+1

        dataset_RANDOM[i][TAILLE_FENETRE] = sortie_Random[cpt][1]

    data_l_ear = dataset_RANDOM[0:dataset_RANDOM.shape[0],0:TAILLE_FENETRE+1]

    #------------------------------- Sortie a predire --------------
    sortie = np.zeros((input_signal_l_ear.shape[0],2))

    for i in range(input_signal_l_ear.shape[0]):
        sortie[i][0] = output_signal_l_ear[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie[i][1] = 1

    #sortie[sortie == 1,] = ECHELLE_SORTIE
    sortie = np.delete(sortie, 0, 1)
    #----------------------------------------------------------------
    """ -------------------------- SIGNAL 2 l_forehead NOUVEAU -------------------------- """

    input_signal_l_forehead = np.reshape(my_data[:,2:3].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_l_forehead)

    sortie_l_forehead = np.zeros((input_signal_l_forehead.shape[0],2))

    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_l_forehead = signal.filtfilt(b, a, input_signal_l_forehead)

    for i in range(input_signal_l_forehead.shape[0]):
        sortie_l_forehead[i][0] = output_signal_l_forehead[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_l_forehead[i][1] = 1

    dataset_RANDOM2 = np.zeros((input_signal_l_forehead.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM2[i][j] = sortie_l_forehead[cpt][0]
            cpt = cpt+1

        dataset_RANDOM2[i][TAILLE_FENETRE] = sortie_l_forehead[cpt][1]

    data_l_forehead = dataset_RANDOM2[0:dataset_RANDOM2.shape[0],0:TAILLE_FENETRE+1]

    """ -------------------------- SIGNAL 3 r_forehead NOUVEAU -------------------------- """

    input_signal_r_forehead = np.reshape(my_data[:,3:4].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_r_forehead)

    sortie_r_forehead = np.zeros((input_signal_r_forehead.shape[0],2))

    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_r_forehead = signal.filtfilt(b, a, input_signal_r_forehead)

    for i in range(input_signal_r_forehead.shape[0]):
        sortie_r_forehead[i][0] = output_signal_r_forehead[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_r_forehead[i][1] = 1

    dataset_RANDOM3 = np.zeros((input_signal_r_forehead.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM3[i][j] = sortie_r_forehead[cpt][0]
            cpt = cpt+1

        dataset_RANDOM3[i][TAILLE_FENETRE] = sortie_r_forehead[cpt][1]

    data_r_forehead = dataset_RANDOM3[0:dataset_RANDOM3.shape[0],0:TAILLE_FENETRE+1]


    """ -------------------------- SIGNAL 4 r_ear NOUVEAU -------------------------- """

    input_signal_r_ear = np.reshape(my_data[:,4:5].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_r_ear)

    sortie_r_ear = np.zeros((input_signal_r_ear.shape[0],2))

    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_r_ear = signal.filtfilt(b, a, input_signal_r_ear)

    for i in range(input_signal_r_ear.shape[0]):
        sortie_r_ear[i][0] = output_signal_r_ear[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_r_ear[i][1] = 1

    dataset_RANDOM4 = np.zeros((input_signal_r_ear.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM4[i][j] = sortie_r_ear[cpt][0]
            cpt = cpt+1

        dataset_RANDOM4[i][TAILLE_FENETRE] = sortie_r_ear[cpt][1]

    data_r_ear = dataset_RANDOM4[0:dataset_RANDOM4.shape[0],0:TAILLE_FENETRE+1]

    """ -------------------------- SIGNAL 5 l_ear BETA NOUVEAU -------------------------- """
    input_signal_l_ear_BETA = np.reshape(my_data[:,18:19].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_l_ear_BETA)

    sortie_Random_BETA = np.zeros((input_signal_l_ear_BETA.shape[0],2))

    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_l_ear_BETA = signal.filtfilt(b, a, input_signal_l_ear_BETA)

    for i in range(input_signal_l_ear_BETA.shape[0]):
        sortie_Random_BETA[i][0] = output_signal_l_ear_BETA[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_Random_BETA[i][1] = 1


    dataset_RANDOM_BETA = np.zeros((input_signal_l_ear_BETA.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM_BETA[i][j] = sortie_Random_BETA[cpt][0]
            cpt = cpt+1

        dataset_RANDOM_BETA[i][TAILLE_FENETRE] = sortie_Random_BETA[cpt][1]

    data_l_ear_BETA = dataset_RANDOM_BETA[0:dataset_RANDOM_BETA.shape[0],0:TAILLE_FENETRE+1]

    """ -------------------------- SIGNAL 6 BETA l_forehead NOUVEAU -------------------------- """

    input_signal_l_forehead_BETA = np.reshape(my_data[:,19:20].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_l_forehead_BETA)

    sortie_l_forehead_BETA = np.zeros((input_signal_l_forehead_BETA.shape[0],2))

    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_l_forehead_BETA = signal.filtfilt(b, a, input_signal_l_forehead_BETA)

    for i in range(input_signal_l_forehead_BETA.shape[0]):
        sortie_l_forehead_BETA[i][0] = output_signal_l_forehead_BETA[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_l_forehead_BETA[i][1] = 1

    dataset_RANDOM2_BETA = np.zeros((input_signal_l_forehead_BETA.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM2_BETA[i][j] = sortie_l_forehead_BETA[cpt][0]
            cpt = cpt+1

        dataset_RANDOM2_BETA[i][TAILLE_FENETRE] = sortie_l_forehead_BETA[cpt][1]

    data_l_forehead_BETA = dataset_RANDOM2_BETA[0:dataset_RANDOM2_BETA.shape[0],0:TAILLE_FENETRE+1]

    """ -------------------------- SIGNAL 7 BETA r_forehead NOUVEAU -------------------------- """

    input_signal_r_forehead_BETA = np.reshape(my_data[:,20:21].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_r_forehead_BETA)

    sortie_r_forehead_BETA = np.zeros((input_signal_r_forehead_BETA.shape[0],2))

    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_r_forehead_BETA = signal.filtfilt(b, a, input_signal_r_forehead_BETA)

    for i in range(input_signal_r_forehead_BETA.shape[0]):
        sortie_r_forehead_BETA[i][0] = output_signal_r_forehead_BETA[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_r_forehead_BETA[i][1] = 1

    dataset_RANDOM3_BETA = np.zeros((input_signal_r_forehead_BETA.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM3_BETA[i][j] = sortie_r_forehead_BETA[cpt][0]
            cpt = cpt+1

        dataset_RANDOM3_BETA[i][TAILLE_FENETRE] = sortie_r_forehead_BETA[cpt][1]

    data_r_forehead_BETA = dataset_RANDOM3_BETA[0:dataset_RANDOM3_BETA.shape[0],0:TAILLE_FENETRE+1]


    """ -------------------------- SIGNAL 8 BETA r_ear NOUVEAU -------------------------- """

    input_signal_r_ear_BETA = np.reshape(my_data[:,21:22].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_r_ear_BETA)

    sortie_r_ear_BETA = np.zeros((input_signal_r_ear_BETA.shape[0],2))

    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_r_ear_BETA = signal.filtfilt(b, a, input_signal_r_ear_BETA)

    for i in range(input_signal_r_ear_BETA.shape[0]):
        sortie_r_ear_BETA[i][0] = output_signal_r_ear_BETA[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_r_ear_BETA[i][1] = 1

    dataset_RANDOM4_BETA = np.zeros((input_signal_r_ear_BETA.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM4_BETA[i][j] = sortie_r_ear_BETA[cpt][0]
            cpt = cpt+1

        dataset_RANDOM4_BETA[i][TAILLE_FENETRE] = sortie_r_ear_BETA[cpt][1]

    data_r_ear_BETA = dataset_RANDOM4_BETA[0:dataset_RANDOM4_BETA.shape[0],0:TAILLE_FENETRE+1]



    """ --------------------------------- BOOSTING -------------------------------------- """
    exemples = []
    for i in range(data_l_ear.shape[0]):
        fenetre_a_predire_l_ear = data_l_ear[i].reshape((1, 71))
        fenetre_a_predire_l_forehead = data_l_forehead[i].reshape((1, 71))
        fenetre_a_predire_r_forehead = data_r_forehead[i].reshape((1, 71))
        fenetre_a_predire_r_ear = data_r_ear[i].reshape((1, 71))
        fenetre_a_predire_l_ear_BETA = data_l_ear_BETA[i].reshape((1, 71))
        fenetre_a_predire_l_forehead_BETA = data_l_forehead_BETA[i].reshape((1, 71))
        fenetre_a_predire_r_forehead_BETA = data_r_forehead_BETA[i].reshape((1, 71))
        fenetre_a_predire_r_ear_BETA = data_r_ear_BETA[i].reshape((1, 71))

        exemples.append(({"l_ear":fenetre_a_predire_l_ear, "l_forehead":fenetre_a_predire_l_forehead, "r_forehead":fenetre_a_predire_r_forehead,"r_ear":fenetre_a_predire_r_ear,
        "l_ear_BETA":fenetre_a_predire_l_ear_BETA, "l_forehead_BETA":fenetre_a_predire_l_forehead_BETA, "r_forehead_BETA":fenetre_a_predire_r_forehead_BETA,"r_ear_BETA":fenetre_a_predire_r_ear_BETA}, 1)) # 2*sortie[i][0]-1
    #print(fenetre_a_predire_l_ear)
    #print(fenetre_a_predire_l_forehead)

    #print(exemples)
    m = AdaBoostHalf(exemples)
    #prediction = useModel(fenetre_a_predire, "l_ear")

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("Graph/EEG/l_ear/l_ear.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./Graph/EEG/l_ear"))
    m.set_rule(lambda x: 2*useModel(x, sess, "l_ear")-1)
    #sess.close()

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("Graph/EEG/l_forehead/l_forehead.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./Graph/EEG/l_forehead"))
    m.set_rule(lambda x: 2*useModel(x, sess, "l_forehead")-1)
    #sess.close()

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("Graph/EEG/r_forehead/r_forehead.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./Graph/EEG/r_forehead"))
    m.set_rule(lambda x: 2*useModel(x, sess, "r_forehead")-1)
    #sess.close()

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("Graph/EEG/r_ear/r_ear.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./Graph/EEG/r_ear"))
    m.set_rule(lambda x: 2*useModel(x, sess, "r_ear")-1)

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("Graph/EEG/BETA/l_ear/l_ear.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./Graph/EEG/BETA/l_ear"))
    m.set_rule(lambda x: 2*useModel(x, sess, "l_ear_BETA")-1)
    #sess.close()

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("Graph/EEG/BETA/l_forehead/l_forehead.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./Graph/EEG/BETA/l_forehead"))
    m.set_rule(lambda x: 2*useModel(x, sess, "l_forehead_BETA")-1)
    #sess.close()

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("Graph/EEG/BETA/r_forehead/r_forehead.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./Graph/EEG/BETA/r_forehead"))
    m.set_rule(lambda x: 2*useModel(x, sess, "r_forehead_BETA")-1)
    #sess.close()

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("Graph/EEG/BETA/r_ear/r_ear.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./Graph/EEG/BETA/r_ear"))
    m.set_rule(lambda x: 2*useModel(x, sess, "r_ear_BETA")-1)
    #sess.close()
    
    test = m.evaluate()
    print(test)

    """ --------------> AFFICHAGE """

    sortie = np.zeros((input_signal_l_ear.shape[0],2))

    for i in range(input_signal_l_ear.shape[0]):
        sortie[i][0] = output_signal_l_ear[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie[i][1] = 1

    sortie[sortie == 1,] = ECHELLE_SORTIE
    sortie = np.delete(sortie, 0, 1)

    plt.plot(input_signal_l_ear, label='Signal l_ear')
    plt.plot(output_signal_l_ear, label='Signal filtre l_ear')
    plt.plot(sortie, label='Sortie')

    for i in range(len(test)):
        if test[i] == True : test[i] = ECHELLE_SORTIE*2
        else : test[i] = 0

    plt.plot(test, label='test boosting')

    """ -------------------------- SIGNAL 1 l_ear -------------------------- """"""
    my_data = genfromtxt("/home/hicham/Bureau/Stage/Dataset/dataset_1/datasetAssisMD.csv", delimiter=';')
    input_signal_Random = np.reshape(my_data[:,1:2].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_Random)
    capteur = np.reshape(my_data[:,27:28].T, -1, 2)
    for i in range(len(capteur)):
        if capteur[i] > 30:
            capteur[i] = ECHELLE_SORTIE
        else:
             capteur[i] = 0

    dataload = useModel(input_signal_Random, "l_ear", 1, 10)
    input_signal_Random = dataload['input']
    output_signal_Random = dataload['output']

    plt.plot(input_signal_Random, label='Signal l_ear')
    plt.plot(output_signal_Random, label='Signal filtre l_ear')

    prediction_visualisation = dataload['predict']
    prediction_visualisation = [ECHELLE_PREDICTION if x==1 else x for x in prediction_visualisation]
    plt.plot(prediction_visualisation, label='Prediction l_ear')

    #------------------------------- Sortie a predire --------------
    sortie = np.zeros((input_signal_Random.shape[0],2))

    for i in range(input_signal_Random.shape[0]):
        sortie[i][0] = output_signal_Random[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie[i][1] = 1

    sortie[sortie == 1,] = ECHELLE_SORTIE
    sortie = np.delete(sortie, 0, 1)
    plt.plot(sortie,  label='Sortie')
    #----------------------------------------------------------------

    """""" -------------------------- SIGNAL 2 l_forehead -------------------------- """"""

    dataload = useModel("/home/hicham/Bureau/Stage/Dataset/dataset_1/datasetAssisMD.csv", 2, "l_forehead", 1, 10)
    input_signal_Random = dataload['input']
    output_signal_Random = dataload['output']

    plt.plot(input_signal_Random, label='Signal l_forehead')
    plt.plot(output_signal_Random, label='Signal filtre l_forehead')

    prediction_visualisation = dataload['predict']
    prediction_visualisation = [ECHELLE_PREDICTION if x==1 else x for x in prediction_visualisation]
    plt.plot(prediction_visualisation, label='Prediction l_forehead')

    #------------------------------- Sortie a predire --------------
    sortie = np.zeros((input_signal_Random.shape[0],2))

    for i in range(input_signal_Random.shape[0]):
        sortie[i][0] = output_signal_Random[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie[i][1] = 1

    sortie[sortie == 1,] = ECHELLE_SORTIE
    sortie = np.delete(sortie, 0, 1)
    plt.plot(sortie,  label='Sortie')

    """""" -------------------------- SIGNAL 3 r_forehead -------------------------- """"""

    dataload = useModel("/home/hicham/Bureau/Stage/Dataset/dataset_1/datasetAssisMD.csv", 3, "r_forehead", 1, 10)
    print(dataload['predict'])
    input_signal_Random = dataload['input']
    output_signal_Random = dataload['output']

    plt.plot(input_signal_Random, label='Signal r_forehead')
    plt.plot(output_signal_Random, label='Signal filtre r_forehead')

    prediction_visualisation = dataload['predict']
    prediction_visualisation = [ECHELLE_PREDICTION if x==1 else x for x in prediction_visualisation]
    plt.plot(prediction_visualisation, label='Prediction r_forehead')

    #------------------------------- Sortie a predire --------------
    sortie = np.zeros((input_signal_Random.shape[0],2))

    for i in range(input_signal_Random.shape[0]):
        sortie[i][0] = output_signal_Random[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie[i][1] = 1

    sortie[sortie == 1,] = ECHELLE_SORTIE
    sortie = np.delete(sortie, 0, 1)
    plt.plot(sortie,  label='Sortie')

    """""" -------------------------- SIGNAL 4 r_ear -------------------------- """"""

    dataload = useModel("/home/hicham/Bureau/Stage/Dataset/dataset_1/datasetAssisMD.csv", 4, "r_ear", 1, 10)
    input_signal_Random = dataload['input']
    output_signal_Random = dataload['output']

    plt.plot(input_signal_Random, label='Signal r_ear')
    plt.plot(output_signal_Random, label='Signal filtre r_ear')

    prediction_visualisation = dataload['predict']
    prediction_visualisation = [ECHELLE_PREDICTION if x==1 else x for x in prediction_visualisation]
    plt.plot(prediction_visualisation, label='Prediction r_ear')

    #------------------------------- Sortie a predire --------------
    sortie = np.zeros((input_signal_Random.shape[0],2))

    for i in range(input_signal_Random.shape[0]):
        sortie[i][0] = output_signal_Random[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie[i][1] = 1

    sortie[sortie == 1,] = ECHELLE_SORTIE
    sortie = np.delete(sortie, 0, 1)
    plt.plot(sortie,  label='Sortie')
    """
    plt.legend(bbox_to_anchor=(0.88, 0.88), loc=2, borderaxespad=0.)
    plt.show()

    #sess.close()

    #m = AdaBoostHalf(examples)
    #m.set_rule(lambda x: 2*(x[0] < 1.5)-1)

if __name__ == '__main__':
    main()
