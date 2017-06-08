from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy import genfromtxt
import csv
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

"""""""""""""""""""""''''''""""""""""""""""""""""""
"""""""""""" MISE EN PLACE DES DONNEES """"""""""""
"""""""""""""""""""""''''''""""""""""""""""""""""""
NB_DATA = 90
input_signal = []
sortie_signal = []
for i in range(NB_DATA):
    try:
        my_data = genfromtxt("/home/hicham/Bureau/Stage/Dataset/dataset_stimulus/dataset/0/"+str(i)+".csv", delimiter=';')
        sortie_signal.append(0)
    except:
        my_data = genfromtxt("/home/hicham/Bureau/Stage/Dataset/dataset_stimulus/dataset/1/"+str(i)+".csv", delimiter=';')
        sortie_signal.append(1)

    input_signal.append(np.reshape(my_data[:,12:13].T, -1, 2))
    #2:3 l_forehead 500 100 / 100
    #11:12 theta_absolutel_forehead 92.54 / 62.5
    #12:13 theta_absoluter_forehead   23000 98.51 / 75.00
    #18:19 beta_absolutel_ear  27000 77.78% / 87.5
    #23:24 gamma_absolutel_forehead 55000 85.07 / 87.5

# Frequence d'echantillonnage fs=333.333 Hz
# Filtre passe bande [1 10] Hz et d'ordre 4
# ou [0.5 30]
FiltreMin = 0.5
FiltreMax = 30
fs = 333.333
Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]

# b, a = signal.butter(1, 10, 'low', analog=True)
b, a = signal.butter(4, Wn, 'bandpass')
output_signal = []
for i in range(NB_DATA):
    output_signal.append(signal.filtfilt(b, a, input_signal[i]))



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

    a = np.zeros((NB_DATA,2))
    #print( "sortie = ",sortie[1:5])

    """ Creation des fenetres temporelles """
    global TAILLE_FENETRE
    TAILLE_FENETRE = 185
    dataset = np.zeros((NB_DATA, TAILLE_FENETRE+1)) # +1 pour la sortie desire

    for i in range(NB_DATA):
        for j in range(TAILLE_FENETRE):
            dataset[i][j] = output_signal[i][j]
        dataset[i][TAILLE_FENETRE] = sortie_signal[i]

    print("DATASET", dataset)
    data = dataset[0:dataset.shape[0],0:TAILLE_FENETRE]
    target = dataset[0:dataset.shape[0],TAILLE_FENETRE].astype(int)

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    #all_X = data
    print("-----------------------------------------------------------------\n")
    print("Data = ",data)
    print("DataSize = ",len(data))
    print("Target = ",target)
    print("N = ",N)
    print("all_X = ",all_X)

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    print("all_Y = ",all_Y)

    print(all_X.shape)
    print(all_Y.shape)
    return train_test_split(all_X, all_Y, train_size=0.9, test_size=0.1, random_state=RANDOM_SEED)

def main():

    train_X, test_X, train_y, test_y = get_data()

    # Layer's sizes
    x_size = train_X.shape[1]
    h_size = 185
    y_size = train_y.shape[1]

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

    """ LANCEMENT DU TEST """
    """ -------------------------- SIGNAL 1 l_ear -------------------------- """

    # Run SGD
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('Graph/STIMULUS/theta_absoluter_forehead/theta_absoluter_forehead.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./Graph/STIMULUS/theta_absoluter_forehead'))

    ECHELLE_PREDICTION = 200

    #my_data = genfromtxt('/home/hicham/Bureau/Stage/Dataset/dataset_1/datasetAssisMD.csv', delimiter=';')
    sortie_signal = 1
    my_data = genfromtxt("/home/hicham/Bureau/Stage/Dataset/dataset_stimulus/dataset/"+str(sortie_signal)+"/98.csv", delimiter=';')

    input_signal_Random = np.reshape(my_data[:,12:13].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_Random)

    sortie_Random = np.zeros((input_signal_Random.shape[0],2))

    """ --- FILTRE --- """
    FiltreMin = 0.5
    FiltreMax = 30
    fs = 333.333
    Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]
    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_Random = signal.filtfilt(b, a, input_signal_Random)
    """ -------------- """

    dataset_RANDOM = np.zeros((1, TAILLE_FENETRE+1)) # +1 pour la sortie desire

    for j in range(TAILLE_FENETRE):
        dataset_RANDOM[0][j] = output_signal_Random[j]
    dataset_RANDOM[0][TAILLE_FENETRE] = sortie_signal

    data = dataset_RANDOM[0:dataset_RANDOM.shape[0],0:TAILLE_FENETRE+1]

    prediction = []

    fenetre_a_predire = data[0].reshape((1, TAILLE_FENETRE+1))
    prediction.append(sess.run(predict, feed_dict={X: fenetre_a_predire})[0])

    print("PREDICTION",prediction)


    plt.plot(input_signal_Random, label='Signal l_ear')
    plt.plot(output_signal_Random, label='Signal filtre l_ear')

    sess.close()

    """ -------------------------- SIGNAL 2 l_forehead -------------------------- """"""

    sess = tf.Session()
    new_saver2 = tf.train.import_meta_graph('Graph/EEG/l_forehead/l_forehead.meta')
    new_saver2.restore(sess, tf.train.latest_checkpoint('./Graph/EEG/l_forehead'))

    input_signal_Random2 = np.reshape(my_data[:,2:3].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_Random2)

    sortie_Random2 = np.zeros((input_signal_Random2.shape[0],2))

    TAILLE_FENETRE = 70 # 16

    FiltreMin = 1
    FiltreMax = 10
    fs = 333.333
    Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]
    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_Random2 = signal.filtfilt(b, a, input_signal_Random2)

    for i in range(input_signal_Random2.shape[0]):
        sortie_Random2[i][0] = output_signal_Random2[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_Random2[i][1] = 1

    dataset_RANDOM2 = np.zeros((input_signal_Random2.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM2[i][j] = sortie_Random2[cpt][0]
            cpt = cpt+1

        dataset_RANDOM2[i][TAILLE_FENETRE] = sortie_Random2[cpt][1]

    data = dataset_RANDOM2[0:dataset_RANDOM2.shape[0],0:TAILLE_FENETRE+1]

    prediction = []
    prediction_visualisation= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(data.shape[0]):
        fenetre_a_predire2 = data[i].reshape((1, TAILLE_FENETRE+1))
        prediction.append(sess.run(predict, feed_dict={X: fenetre_a_predire2})[0])

    print("TAILLE",len(prediction))

    for i in range(len(prediction)):
        prediction_visualisation.append(prediction[i])

    #plt.plot(input_signal_Random2, label='Signal l_forehead')
    #plt.plot(output_signal_Random2, label='Signal filtre l_forehead')

    prediction_visualisation = [ECHELLE_PREDICTION if x==1 else x for x in prediction_visualisation]
    plt.plot(prediction_visualisation, label='Prediction l_forehead')
    print("TAILLE",len(prediction_visualisation))

    """""" -------------------------- SIGNAL 3 r_forehead -------------------------- """"""

    sess = tf.Session()
    new_saver3 = tf.train.import_meta_graph('Graph/EEG/r_forehead/r_forehead.meta')
    new_saver3.restore(sess, tf.train.latest_checkpoint('./Graph/EEG/r_forehead'))

    input_signal_Random3 = np.reshape(my_data[:,3:4].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_Random3)

    sortie_Random3 = np.zeros((input_signal_Random3.shape[0],2))

    TAILLE_FENETRE = 70 # 16

    FiltreMin = 1
    FiltreMax = 10
    fs = 333.333
    Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]
    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_Random3 = signal.filtfilt(b, a, input_signal_Random3)

    for i in range(input_signal_Random3.shape[0]):
        sortie_Random3[i][0] = output_signal_Random3[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_Random3[i][1] = 1

    dataset_RANDOM3 = np.zeros((input_signal_Random3.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM3[i][j] = sortie_Random3[cpt][0]
            cpt = cpt+1

        dataset_RANDOM3[i][TAILLE_FENETRE] = sortie_Random3[cpt][1]

    data = dataset_RANDOM3[0:dataset_RANDOM3.shape[0],0:TAILLE_FENETRE+1]

    prediction = []
    prediction_visualisation= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(data.shape[0]):
        fenetre_a_predire2 = data[i].reshape((1, TAILLE_FENETRE+1))
        prediction.append(sess.run(predict, feed_dict={X: fenetre_a_predire2})[0])

    print("TAILLE",len(prediction))

    for i in range(len(prediction)):
        prediction_visualisation.append(prediction[i])

    #plt.plot(input_signal_Random3, label='Signal r_forehead')
    #plt.plot(output_signal_Random3, label='Signal filtre r_forehead')

    prediction_visualisation = [ECHELLE_PREDICTION if x==1 else x for x in prediction_visualisation]
    plt.plot(prediction_visualisation, label='Prediction r_forehead')
    print("TAILLE",len(prediction_visualisation))

    """""" -------------------------- SIGNAL 4 r_ear -------------------------- """"""

    sess = tf.Session()
    new_saver4 = tf.train.import_meta_graph('Graph/EEG/r_ear/r_ear.meta')
    new_saver4.restore(sess, tf.train.latest_checkpoint('./Graph/EEG/r_ear'))

    input_signal_Random4 = np.reshape(my_data[:,4:5].T, -1, 2) # 2eme colonne
    SIZE = len(input_signal_Random4)

    sortie_Random4 = np.zeros((input_signal_Random4.shape[0],2))

    TAILLE_FENETRE = 70 # 16

    FiltreMin = 1
    FiltreMax = 10
    fs = 333.333
    Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]
    b, a = signal.butter(4,  Wn, 'bandpass')
    output_signal_Random4 = signal.filtfilt(b, a, input_signal_Random4)

    for i in range(input_signal_Random4.shape[0]):
        sortie_Random4[i][0] = output_signal_Random4[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_Random4[i][1] = 1

    dataset_RANDOM4 = np.zeros((input_signal_Random4.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire
    for i in range(SIZE-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM4[i][j] = sortie_Random4[cpt][0]
            cpt = cpt+1

        dataset_RANDOM4[i][TAILLE_FENETRE] = sortie_Random4[cpt][1]

    data = dataset_RANDOM4[0:dataset_RANDOM4.shape[0],0:TAILLE_FENETRE+1]

    prediction = []
    prediction_visualisation= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(data.shape[0]):
        fenetre_a_predire2 = data[i].reshape((1, TAILLE_FENETRE+1))
        prediction.append(sess.run(predict, feed_dict={X: fenetre_a_predire2})[0])

    print("TAILLE",len(prediction))

    for i in range(len(prediction)):
        prediction_visualisation.append(prediction[i])

    #plt.plot(input_signal_Random4, label='Signal r_ear')
    #plt.plot(output_signal_Random4, label='Signal filtre r_ear')

    prediction_visualisation = [ECHELLE_PREDICTION if x==1 else x for x in prediction_visualisation]
    plt.plot(prediction_visualisation, label='Prediction r_ear')
    print("TAILLE",len(prediction_visualisation))
    """
    plt.legend(bbox_to_anchor=(0.88, 0.88), loc=2, borderaxespad=0.)
    plt.show()

    sess.close()

if __name__ == '__main__':
    main()
