from liblo import *

import sys
import time
import serial
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy import genfromtxt
import csv
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
"""
ser = serial.Serial("/dev/ttyACM0",timeout=1)
print (ser)

dataset = open("datasetDeboutMD.csv", "w")
# Initialisation du capteur main (vide la sortie)
start_time = time.time()
capteur = -1
while (time.time() - start_time) < 2:
    donnee=str(ser.readline())
    try:
        capteur = int(donnee)
    except ValueError:
        print("Error value ")

    ser.flush()

"""
class MuseServer(ServerThread):
    #listen for messages on port 5000
    acc_x = -1
    acc_y = -1
    acc_z = -1
    l_ear  = -1
    l_forehead = -1
    r_forehead = -1
    r_ear = -1
    r_aux = -1
    quantization = -1
    dropped_samples = -1

    delta_absolutel_ear = -1
    delta_absolutel_forehead = -1
    delta_absoluter_forehead = -1
    delta_absoluter_ear = -1

    theta_absolutel_ear = -1
    theta_absolutel_forehead = -1
    theta_absoluter_forehead = -1
    theta_absoluter_ear = -1

    alpha_absolutel_ear = -1
    alpha_absolutel_forehead = -1
    alpha_absoluter_forehead = -1
    alpha_absoluter_ear = -1

    beta_absolutel_ear = -1
    beta_absolutel_forehead = -1
    beta_absoluter_forehead = -1
    beta_absoluter_ear = -1

    gamma_absolutel_ear = -1
    gamma_absolutel_forehead = -1
    gamma_absoluter_forehead = -1
    gamma_absoluter_ear = -1



    def __init__(self):
        ServerThread.__init__(self, 5000)

    #receive accelrometer data
    @make_method('/muse/acc', 'fff')
    def acc_callback(self, path, args):
        MuseServer.acc_x, MuseServer.acc_y, MuseServer.acc_z = args
        #print "%s %f %f %f" % (path, acc_x, acc_y, acc_z)

    #receive EEG data
    @make_method('/muse/eeg', 'fffff')
    def eeg_callback(self, path, args):
        MuseServer.l_ear, MuseServer.l_forehead, MuseServer.r_forehead, MuseServer.r_ear , MuseServer.r_aux = args
        #print "%s %f %f %f %f %f" % (path, MuseServer.l_ear, MuseServer.l_forehead, MuseServer.r_forehead, MuseServer.r_ear, MuseServer.r_aux)

    @make_method('/muse/eeg/quantization', 'iiii')
    def quantiz_callback(self, path, args):
        MuseServer.quantization = args
        print (args)

    @make_method('/muse/eeg/dropped_samples', 'i')
    def drop_callback(self, path, args):
        MuseServer.dropped_samples = args
        print "%s %i" % (path, MuseServer.dropped_samples)

    @make_method('/muse/elements/alpha_absolute', 'ffff')
    def alphaabs_callback(self, path, args):
	MuseServer.alpha_absolutel_ear, MuseServer.alpha_absolutel_forehead, MuseServer.alpha_absoluter_forehead, MuseServer.alpha_absoluter_ear  =args
	#print ("alpha", args)

    @make_method('/muse/elements/beta_absolute', 'ffff')
    def betaabs_callback(self, path, args):
	MuseServer.beta_absolutel_ear, MuseServer.beta_absolutel_forehead, MuseServer.beta_absoluter_forehead, MuseServer.beta_absoluter_ear  =args
        #print ("beta", args)

    @make_method('/muse/elements/delta_absolute', 'ffff')
    def deltaabs_callback(self, path, args):
        MuseServer.delta_absolutel_ear, MuseServer.delta_absolutel_forehead, MuseServer.delta_absoluter_forehead, MuseServer.delta_absoluter_ear  =args
        #print ("delta", args)

    @make_method('/muse/elements/theta_absolute', 'ffff')
    def thetaabs_callback(self, path, args):
        MuseServer.theta_absolutel_ear, MuseServer.theta_absolutel_forehead, MuseServer.theta_absoluter_forehead, MuseServer.theta_absoluter_ear  =args
        #print ("theta", args)

    @make_method('/muse/elements/gamma_absolute', 'ffff')
    def gammaabs_callback(self, path, args):
        MuseServer.gamma_absolutel_ear, MuseServer.gamma_absolutel_forehead, MuseServer.gamma_absoluter_forehead, MuseServer.gamma_absoluter_ear  =args
        #print ("gamma", args)

    #handle unexpected messages
    @make_method(None, None)
    def fallback(self, path, args, types, src):
       """print "Unknown message \
        \n\t Source: '%s' \
        \n\t Address: '%s' \
        \n\t Types: '%s ' \
        \n\t Payload: '%s'" \
        % (src.url, path, types, args)"""

try:
    server = MuseServer()
except ServerError, err:
    print str(err)
    sys.exit()

server.start()

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
    return train_test_split(all_X, all_Y, test_size=0.3, random_state=RANDOM_SEED)

def useModel(fenetre_a_predire, model):
    prediction = model.run(predict, feed_dict={X: fenetre_a_predire})[0]
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

    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("Graph/EEG/BETA/l_ear/l_ear.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint("./Graph/EEG/BETA/l_ear"))

    plt.ion()
    plt.show()
    """ -----------TRAITEMENT DES DONNEES------------ """
    i=0
    fenetre_a_predire = []
    output_signal_l_ear = []
    """ --- FILTRE --- """
    FiltreMin = 1
    FiltreMax = 10
    fs = 333.333
    Wn = [[2*FiltreMin/fs], [2*FiltreMax/fs]]
    b, a = signal.butter(4,  Wn, 'bandpass')
    """ -------------- """

    while 1:
    	#print(server.acc_x)
        print(i)

    	time.sleep(0.05)
    	#donnee=str(ser.readline())
    	try:
            """
    		capteur = int(donnee)

    		dataset.write(str(i)+";"+str(server.l_ear)+";"+str(server.l_forehead)+";"+str(server.r_forehead)+";"+str(server.r_ear)+";"+str(server.r_aux)+";"
    			""+str(server.delta_absolutel_ear)+";"+str(server.delta_absolutel_forehead)+";"+str(server.delta_absoluter_forehead)+";"+str(server.delta_absoluter_ear)+";"
    			""+str(server.theta_absolutel_ear)+";"+str(server.theta_absolutel_forehead)+";"+str(server.theta_absoluter_forehead)+";"+str(server.theta_absoluter_ear)+";"
    			""+str(server.alpha_absolutel_ear)+";"+str(server.alpha_absolutel_forehead)+";"+str(server.alpha_absoluter_forehead)+";"+str(server.alpha_absoluter_ear)+";"
    			""+str(server.beta_absolutel_ear)+";"+str(server.beta_absolutel_forehead)+";"+str(server.beta_absoluter_forehead)+";"+str(server.beta_absoluter_ear)+";"
    			""+str(server.gamma_absolutel_ear)+";"+str(server.gamma_absolutel_forehead)+";"+str(server.gamma_absoluter_forehead)+";"+str(server.gamma_absoluter_ear)+";"
    			""+str(server.dropped_samples)+";"
    			""+str(capteur)+"\n")
            """

            """eeg.append(server.l_ear)
            #eeg.append(server.l_forehead)
            #eeg.append(server.r_forehead)
            #eeg.append(server.r_ear)"""


            TAILLE_FENETRE = 71
            if i < TAILLE_FENETRE:
                fenetre_a_predire.append(server.l_ear)
            else:
                fenetre_a_predire = np.array(fenetre_a_predire)
                fenetre_a_predire = fenetre_a_predire.reshape((1, TAILLE_FENETRE))

                output_signal_l_ear = signal.filtfilt(b, a, fenetre_a_predire)
                print("Out",output_signal_l_ear)
                prediction = useModel(output_signal_l_ear, sess)
                print("Prediction",prediction)
                #sess.close()

                fenetre_a_predire = np.delete(fenetre_a_predire, 0) # Supprime le premier enregistrement
                fenetre_a_predire = np.append(fenetre_a_predire, server.l_ear) # Ajoute un nouvel enregistrement
                #print(fenetre_a_predire)


            i=i+1
    	except ValueError as e:
    		print("Error value ", e)

    	#ser.flush()



if __name__ == '__main__':
    main()
