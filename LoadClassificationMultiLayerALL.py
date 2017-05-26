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
input_signal_l_ear = np.reshape(my_data[:,1:2].T, -1, 2) # 2eme colonne
input_signal_l_forehead = np.reshape(my_data[:,2:3].T, -1, 2) # 2eme colonne
input_signal_r_forehead = np.reshape(my_data[:,3:4].T, -1, 2) # 2eme colonne
input_signal_r_ear = np.reshape(my_data[:,4:5].T, -1, 2) # 2eme colonne


capteur = np.reshape(my_data[:,27:28].T, -1, 2)
capteur[capteur > 30] = 100
capteur[capteur <= 30] = 0

SIZE = len(input_signal_l_ear)
print(SIZE)
print("DATA", input_signal_l_ear)

"""-----------------------------------------------"""
# Frequence d'echantillonnage fs=333.333 Hz
# Filtre passe bande [1 10] Hz et d'ordre 4
# ou [0.5 30]
FiltreMin = 1
FiltreMax = 10
fs = 333.333
Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]
"""-----------------------------------------------"""

# b, a = signal.butter(1, 10, 'low', analog=True)
b, a = signal.butter(4,  Wn, 'bandpass')
output_signal_l_ear = signal.filtfilt(b, a, input_signal_l_ear)
output_signal_l_forehead = signal.filtfilt(b, a, input_signal_l_forehead)
output_signal_r_forehead = signal.filtfilt(b, a, input_signal_r_forehead)
output_signal_r_ear = signal.filtfilt(b, a, input_signal_r_ear)

sortie = np.zeros((input_signal_l_ear.shape[0],5))


for i in range(input_signal_l_ear.shape[0]):
    sortie[i][0] = output_signal_l_ear[i]
    sortie[i][1] = output_signal_l_forehead[i]
    sortie[i][2] = output_signal_r_forehead[i]
    sortie[i][3] = output_signal_r_ear[i]

    if capteur[i] == 100:
        sortie[i][4] = 1

#print("Sortie = ",sortie)


""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""

def get_data():

    #plt.plot(input_signal)
    #plt.plot(output_signal)
    #plt.plot(sortie)

    #plt.show()

    a = np.zeros((input_signal_l_ear.shape[0],2))
    #print( "sortie = ",sortie[1:5])

    """ Creation des fenetres temporelles """
    TAILLE_FENETRE = 70 # 16

    dataset = np.zeros((input_signal_l_ear.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1)) # +1 pour la sortie desire




    data = sortie[:,0:4]

    target = []
    for i in range(len(sortie)):
        target.append(sortie[i][4].astype(int))

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    #all_X = data
    print("-----------------------------------------------------------------\n")
    #print("Data = ",data)
    print("DataSize = ",len(data))
    print("Target = ",target)
    print("N = ",N)
    #print("all_X = ",all_X)

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    print("all_Y = ",all_Y)

    print(all_X.shape)
    print(all_Y.shape)
    return train_test_split(all_X, all_Y, train_size=0.6, test_size=0.4, random_state=RANDOM_SEED)

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def main():

    """""""""""""""""""""""""""""""""""""""
    """""""""""" CLASSIFICATION """""""""""
    """""""""""""""""""'"""""""""""""""""""
    train_X, test_X, train_y, test_y = get_data()

    # Parameters
    learning_rate = 0.001
    training_epochs = 5000
    batch_size = 1
    display_step = 1

    # Network Parameters
    n_hidden_1 = 10 # 1st layer number of features
    n_hidden_2 = 10 # 2nd layer number of features
    n_input = train_X.shape[1] # MNIST data input (img shape: 28*28)
    n_classes = train_y.shape[1] # MNIST total classes (0-9 digits)
    print("n_classes", n_classes)
    # tf Graph input
    global X
    X = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    global predict
    predict = multilayer_perceptron(X, weights, biases)

    # Backward propagation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))
    updates = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Save
    saver = tf.train.Saver()
    tf.add_to_collection('vars', weights['h1'])
    tf.add_to_collection('vars', weights['h2'])
    tf.add_to_collection('vars', weights['out'])

    """ LANCEMENT DU TEST """
    """ -------------------------- SIGNAL 1 l_ear -------------------------- """

    # Run SGD
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph('Graph/MULTILAYER/LONG/LONG.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./Graph/MULTILAYER/LONG'))

    ECHELLE_PREDICTION = 200
    ECHELLE_SORTIE = 100

    #my_data = genfromtxt('/home/hicham/Bureau/Stage/Dataset/dataset_1/datasetAssisMD.csv', delimiter=';')
    my_data = genfromtxt('/home/hicham/Bureau/Stage/Dataset/dataset_3/datasetLong1.csv', delimiter=';')
    input_signal_l_ear = np.reshape(my_data[:,1:2].T, -1, 2) # 2eme colonne
    input_signal_l_forehead = np.reshape(my_data[:,2:3].T, -1, 2) # 2eme colonne
    input_signal_r_forehead = np.reshape(my_data[:,3:4].T, -1, 2) # 2eme colonne
    input_signal_r_ear = np.reshape(my_data[:,4:5].T, -1, 2) # 2eme colonne

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
    """ -------------- """

    output_signal_l_ear = signal.filtfilt(b, a, input_signal_l_ear)
    output_signal_l_forehead = signal.filtfilt(b, a, input_signal_l_forehead)
    output_signal_r_forehead = signal.filtfilt(b, a, input_signal_r_forehead)
    output_signal_r_ear = signal.filtfilt(b, a, input_signal_r_ear)

    sortie = np.zeros((input_signal_l_ear.shape[0],5))


    for i in range(input_signal_l_ear.shape[0]):
        sortie[i][0] = output_signal_l_ear[i]
        sortie[i][1] = output_signal_l_forehead[i]
        sortie[i][2] = output_signal_r_forehead[i]
        sortie[i][3] = output_signal_r_ear[i]

        if capteur[i] == 100:
            sortie[i][4] = 1


    dataset = np.zeros((input_signal_l_ear.shape[0]-TAILLE_FENETRE, TAILLE_FENETRE+1))
    data = sortie[:,0:4]

    prediction = []
    prediction_visualisation = []

    for i in range(data.shape[0]):
        fenetre_a_predire = np.array([1, output_signal_l_ear[i], output_signal_l_forehead[i], output_signal_r_forehead[i], output_signal_r_ear[i]]).reshape((1, 5))
        #print(fenetre_a_predire)
        prediction.append(sess.run(predict, feed_dict={X: fenetre_a_predire})[0])

    print("TAILLE",len(prediction))

    for i in range(len(prediction)):
        if prediction[i][0] > prediction[i][1]:
            prediction_visualisation.append(-ECHELLE_SORTIE)
        else:
            prediction_visualisation.append(ECHELLE_SORTIE)

    #plt.plot(input_signal_l_ear, label='Signal l_ear')
    plt.plot(output_signal_l_ear, label='Signal filtre l_ear')

    #prediction_visualisation = [ECHELLE_PREDICTION if x==1 else x for x in prediction_visualisation]
    plt.plot(prediction_visualisation, label='Prediction l_ear')

    #------------------------------- Sortie a predire --------------
    sortie = np.zeros((input_signal_l_ear.shape[0],2))

    for i in range(input_signal_l_ear.shape[0]):
        sortie[i][0] = output_signal_l_ear[i]
        if capteur[i] == ECHELLE_SORTIE:
            sortie[i][1] = 1

    sortie[sortie == 1,] = ECHELLE_SORTIE
    sortie = np.delete(sortie, 0, 1)
    plt.plot(sortie,  label='Sortie')
    #----------------------------------------------------------------

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
