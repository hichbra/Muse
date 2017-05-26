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
# Beta = 18 19
# Gamma = 22 23
my_data = genfromtxt('/home/hicham/Bureau/Stage/Dataset/dataset_3/datasetLong1.csv', delimiter=';')
input_signal = np.reshape(my_data[:,4:5].T, -1, 2) # 2eme colonne
my_data2 = genfromtxt('/home/hicham/Bureau/Stage/Dataset/dataset_3/datasetLong2.csv', delimiter=';')
input_signal2 = np.reshape(my_data2[:,4:5].T, -1, 2) # 2eme colonne
my_data3 = genfromtxt('/home/hicham/Bureau/Stage/Dataset/dataset_3/datasetLong4.csv', delimiter=';')
input_signal3 = np.reshape(my_data3[:,4:5].T, -1, 2) # 2eme colonne

capteur1 = np.reshape(my_data[:,27:28].T, -1, 2)
capteur2 = np.concatenate((capteur1, np.reshape(my_data2[:,27:28].T, -1, 2)), axis=0)
capteur = np.concatenate((capteur2, np.reshape(my_data3[:,27:28].T, -1, 2)), axis=0)

capteur[capteur > 30] = 100
capteur[capteur <= 30] = 0
SIZE1 = len(input_signal)
SIZE2 = len(input_signal2)
SIZE3 = len(input_signal3)
#print(SIZE)
# Frequence d'echantillonnage fs=333.333 Hz
# Filtre passe bande [1 10] Hz et d'ordre 4
# ou [0.5 30]
FiltreMin = 1
FiltreMax = 10
fs = 333.333
Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]

# b, a = signal.butter(1, 10, 'low', analog=True)
b, a = signal.butter(4,  Wn, 'bandpass')
output_signal = signal.filtfilt(b, a, input_signal)
output_signal2 = signal.filtfilt(b, a, input_signal2)
output_signal3 = signal.filtfilt(b, a, input_signal3)

sortie = np.zeros((input_signal.shape[0]+input_signal2.shape[0]+input_signal3.shape[0],2))

#sortie = capteur
for i in range(input_signal.shape[0]):
    sortie[i][0] = output_signal[i]
    #print("yoooooooooooooooooooooooooooooooooooooooooooo",capteur[i])
    if capteur[i] == 100:
        sortie[i][1] = 1
j = 0
for i in range(input_signal.shape[0],input_signal2.shape[0]+input_signal.shape[0]):
    sortie[i][0] = output_signal2[j]
    #print("yoooooooooooooooooooooooooooooooooooooooooooo",capteur[i])
    if capteur[i] == 100:
        sortie[i][1] = 1
    j=j+1

j = 0
for i in range(input_signal.shape[0]+input_signal2.shape[0], input_signal3.shape[0]+input_signal.shape[0]+input_signal2.shape[0]):
    sortie[i][0] = output_signal3[j]
    #print("yoooooooooooooooooooooooooooooooooooooooooooo",capteur[i])
    if capteur[i] == 100:
        sortie[i][1] = 1
    j=j+1



print("Sortie = ",sortie)
"""
sortie = np.zeros((input_signal.shape[0],2))

for i in range(input_signal.shape[0]):
    sortie[i][0] = output_signal[i]
    if output_signal[i] > 4:
        sortie[i][1] = 1
"""
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

    dataset = np.zeros(((SIZE1-TAILLE_FENETRE)+(SIZE2-TAILLE_FENETRE)+(SIZE3-TAILLE_FENETRE), TAILLE_FENETRE+1)) # +1 pour la sortie desire

    #cpt = 0
    for i in range(SIZE1-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset[i][j] = sortie[cpt][0]
            cpt = cpt+1

        dataset[i][TAILLE_FENETRE] = sortie[cpt][1]

    for i in range((SIZE1-TAILLE_FENETRE),(SIZE1-TAILLE_FENETRE)+(SIZE2-TAILLE_FENETRE)):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset[i][j] = sortie[cpt][0]
            cpt = cpt+1

        dataset[i][TAILLE_FENETRE] = sortie[cpt][1]

    for i in range((SIZE1-TAILLE_FENETRE)+(SIZE2-TAILLE_FENETRE),(SIZE1-TAILLE_FENETRE)+(SIZE2-TAILLE_FENETRE)+(SIZE3-TAILLE_FENETRE)):
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
    return train_test_split(all_X, all_Y, train_size=0.6, test_size=0.4, random_state=RANDOM_SEED)

def main():

    """""""""""""""""""""""""""""""""""""""
    """""""""""" CLASSIFICATION """""""""""
    """""""""""""""""""'"""""""""""""""""""

    train_X, test_X, train_y, test_y = get_data()

    # Layer's sizes
    x_size = train_X.shape[1]
    h_size = 70
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

    # Run SGD
    global sess
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    print(train_X[0:1])

    yep = 0 ;
    for epoch in range(200):
        # Train with each example
        for i in range(len(train_X)):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        if (test_accuracy > 0.8):
            yep = 1

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    # Save

    #saver.save(sess, 'Graph/EEG/l_ear/l_ear')
    #saver.save(sess, 'Graph/EEG/l_forehead/l_forehead')
    #saver.save(sess, 'Graph/EEG/r_forehead/r_forehead')
    #saver.save(sess, 'Graph/EEG/r_ear/r_ear')

    #saver.save(sess, 'Graph/EEG/BETA/l_ear/l_ear')
    #saver.save(sess, 'Graph/EEG/BETA/l_forehead/l_forehead')
    #saver.save(sess, 'Graph/EEG/BETA/r_forehead/r_forehead')
    #saver.save(sess, 'Graph/EEG/BETA/r_ear/r_ear')

    #saver.save(sess, 'Graph/EEG/GAMMA/l_ear/l_ear')
    #saver.save(sess, 'Graph/EEG/GAMMA/l_forehead/l_forehead')
    #saver.save(sess, 'Graph/EEG/GAMMA/r_forehead/r_forehead')
    #saver.save(sess, 'Graph/EEG/GAMMA/r_ear/r_ear')

    #saver.save(sess, 'Graph/LONG4/l_ear/l_ear')
    #saver.save(sess, 'Graph/LONG4/l_forehead/l_forehead')
    #saver.save(sess, 'Graph/LONG4/r_forehead/r_forehead')
    saver.save(sess, 'Graph/LONG4/r_ear/r_ear')
    print(yep)

if __name__ == '__main__':
    main()

    ECHELLE_PREDICTION = 200
    ECHELLE_SORTIE = 100

    """ Creation des fenetres temporelles """
    #my_data = genfromtxt('/home/hicham/Bureau/Stage/Dataset/dataset_1/datasetAssisMD2.csv', delimiter=';')

    SIZE = len(input_signal)
    capteur1 = np.reshape(my_data[:,27:28].T, -1, 2)
    capteur2 = np.concatenate((capteur1, np.reshape(my_data2[:,27:28].T, -1, 2)), axis=0)
    capteur = np.concatenate((capteur2, np.reshape(my_data3[:,27:28].T, -1, 2)), axis=0)
    for i in range(len(capteur)):
        if capteur[i] > 30:
            capteur[i] = ECHELLE_SORTIE
        else:
             capteur[i] = 0

    sortie_Random = np.zeros((input_signal.shape[0]+input_signal2.shape[0]+input_signal3.shape[0],2))
    TAILLE_FENETRE = 70 # 16
    output_signal_Random  = signal.filtfilt(b, a, input_signal)
    output_signal2_Random = signal.filtfilt(b, a, input_signal2)
    output_signal3_Random = signal.filtfilt(b, a, input_signal3)


    for i in range(input_signal.shape[0]):
        sortie_Random[i][0] = output_signal_Random[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_Random[i][1] = 1
    j = 0
    for i in range(input_signal.shape[0],input_signal2.shape[0]+input_signal.shape[0]):
        sortie_Random[i][0] = output_signal2_Random[j]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_Random[i][1] = 1
        j=j+1

    j = 0
    for i in range(input_signal.shape[0]+input_signal2.shape[0], input_signal3.shape[0]+input_signal.shape[0]+input_signal2.shape[0]):
        sortie_Random[i][0] = output_signal3_Random[j]
        if capteur[i] == ECHELLE_SORTIE:
            sortie_Random[i][1] = 1
        j=j+1


    dataset_RANDOM = np.zeros(((SIZE1-TAILLE_FENETRE)+(SIZE2-TAILLE_FENETRE)+(SIZE3-TAILLE_FENETRE), TAILLE_FENETRE+1)) # +1 pour la sortie desire

    #cpt = 0
    for i in range(SIZE1-TAILLE_FENETRE):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM[i][j] = sortie_Random[cpt][0]
            cpt = cpt+1

        dataset_RANDOM[i][TAILLE_FENETRE] = sortie_Random[cpt][1]

    for i in range((SIZE1-TAILLE_FENETRE),(SIZE1-TAILLE_FENETRE)+(SIZE2-TAILLE_FENETRE)):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM[i][j] = sortie_Random[cpt][0]
            cpt = cpt+1

        dataset_RANDOM[i][TAILLE_FENETRE] = sortie_Random[cpt][1]

    for i in range((SIZE1-TAILLE_FENETRE)+(SIZE2-TAILLE_FENETRE),(SIZE1-TAILLE_FENETRE)+(SIZE2-TAILLE_FENETRE)+(SIZE3-TAILLE_FENETRE)):
        cpt = i
        for j in range(TAILLE_FENETRE):
            dataset_RANDOM[i][j] = sortie_Random[cpt][0]
            cpt = cpt+1

        dataset_RANDOM[i][TAILLE_FENETRE] = sortie_Random[cpt][1]



    data = dataset_RANDOM[0:dataset_RANDOM.shape[0],0:TAILLE_FENETRE+1]

    prediction_de_la_mort = []
    prediction_de_la_mort_qui_tue = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(data.shape[0]):
        fenetre_a_predire = data[i].reshape((1, TAILLE_FENETRE+1))
        prediction_de_la_mort.append(sess.run(predict, feed_dict={X: fenetre_a_predire})[0])

    print("TAILLE",len(prediction_de_la_mort))

    for i in range(len(prediction_de_la_mort)):
        #for j in range(TAILLE_FENETRE):
        prediction_de_la_mort_qui_tue.append(prediction_de_la_mort[i])

    #plt.plot(input_signal, label='Signal')
    plt.plot(output_signal_Random, label='Signal filtre')

    prediction_de_la_mort_qui_tue = [ECHELLE_PREDICTION if x==1 else x for x in prediction_de_la_mort_qui_tue]
    plt.plot(prediction_de_la_mort_qui_tue, label='Prediction')

    sortie = np.zeros((input_signal.shape[0]+input_signal2.shape[0]+input_signal3.shape[0],2))

    #sortie = capteur
    for i in range(input_signal.shape[0]):
        sortie[i][0] = output_signal_Random[i]
        #print("yoooooooooooooooooooooooooooooooooooooooooooo",capteur[i])
        if capteur[i] == ECHELLE_SORTIE:
            sortie[i][1] = 1
    j = 0
    for i in range(input_signal.shape[0],input_signal2.shape[0]+input_signal.shape[0]):
        sortie[i][0] = output_signal2_Random[j]
        #print("yoooooooooooooooooooooooooooooooooooooooooooo",capteur[i])
        if capteur[i] == ECHELLE_SORTIE:
            sortie[i][1] = 1
        j=j+1

    j = 0
    for i in range(input_signal.shape[0]+input_signal2.shape[0], input_signal3.shape[0]+input_signal.shape[0]+input_signal2.shape[0]):
        sortie[i][0] = output_signal3_Random[j]
        #print("yoooooooooooooooooooooooooooooooooooooooooooo",capteur[i])
        if capteur[i] == ECHELLE_SORTIE:
            sortie[i][1] = 1
        j=j+1

    sortie[sortie == 1,] = ECHELLE_SORTIE

    sortie = np.delete(sortie, 0, 1)
    plt.plot(sortie,  label='Sortie')
    plt.legend(bbox_to_anchor=(0.88, 0.88), loc=3, borderaxespad=0.)



    plt.show()
