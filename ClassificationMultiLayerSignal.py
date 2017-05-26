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
# Alpha = 14 15
# Beta = 18 19
# Gamma = 22 23
my_data = genfromtxt('/home/hicham/Bureau/Stage/Dataset/dataset_3/datasetLong1.csv', delimiter=';')
input_signal_l_ear = np.reshape(my_data[:,1:2].T, -1, 2) # 2eme colonne
input_signal_l_forehead = np.reshape(my_data[:,2:3].T, -1, 2)
input_signal_r_forehead = np.reshape(my_data[:,3:4].T, -1, 2)
input_signal_r_ear = np.reshape(my_data[:,4:5].T, -1, 2)

input_signal_alpha_l_ear = np.reshape(my_data[:,14:15].T, -1, 2) # 2eme colonne
input_signal_alpha_l_forehead = np.reshape(my_data[:,15:16].T, -1, 2)
input_signal_alpha_r_forehead = np.reshape(my_data[:,16:17].T, -1, 2)
input_signal_alpha_r_ear = np.reshape(my_data[:,17:18].T, -1, 2)

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

output_signal_alpha_l_ear = signal.filtfilt(b, a, input_signal_alpha_l_ear)
output_signal_alpha_l_forehead = signal.filtfilt(b, a, input_signal_alpha_l_forehead)
output_signal_alpha_r_forehead = signal.filtfilt(b, a, input_signal_alpha_r_forehead)
output_signal_alpha_r_ear = signal.filtfilt(b, a, input_signal_alpha_r_ear)

sortie = np.zeros((input_signal_l_ear.shape[0],9))


for i in range(input_signal_l_ear.shape[0]):
    sortie[i][0] = output_signal_l_ear[i]
    sortie[i][1] = output_signal_l_forehead[i]
    sortie[i][2] = output_signal_r_forehead[i]
    sortie[i][3] = output_signal_r_ear[i]

    sortie[i][4] = output_signal_alpha_l_ear[i]
    sortie[i][5] = output_signal_alpha_l_forehead[i]
    sortie[i][6] = output_signal_alpha_r_forehead[i]
    sortie[i][7] = output_signal_alpha_r_ear[i]

    if capteur[i] == 100:
        sortie[i][8] = 1

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

    data = sortie[:,0:8]

    target = []
    for i in range(len(sortie)):
        target.append(sortie[i][8].astype(int))

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
    training_epochs = 2000
    batch_size = 2
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


    """
    yep = 0 ;
    for epoch in range(2):
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
        """


    # Run SGD
    # Run SGD
    global sess
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    print("train_X",train_X[0:1])
    print("train_y",train_y[0:1])


    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_X)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = train_X[i: i + 1], train_y[i: i + 1]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([updates, cost], feed_dict={X: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    #correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
    #print("correct_prediction", correct_prediction)
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #print("Accuracy:", accuracy.eval({X: train_X[i: i + 1], y: train_y[i: i + 1]}))


    """ ------------ Save ------------- """
    saver.save(sess, 'Graph/MULTILAYER/LONG/LONG')
    #saver.save(sess, 'Graph/MULTILAYER/EEG/l_forehead/l_forehead')
    #saver.save(sess, 'Graph/MULTILAYER/EEG/r_forehead/r_forehead')
    #saver.save(sess, 'Graph/MULTILAYER/EEG/r_ear/r_ear')
    #print(yep)

if __name__ == '__main__':
    main()

    ECHELLE_PREDICTION = 200
    ECHELLE_SORTIE = 100


    prediction = []
    prediction_visualisation = []#[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(output_signal_l_ear.shape[0]):
        fenetre_a_predire = np.array([1, output_signal_l_ear[i], output_signal_l_forehead[i], output_signal_r_forehead[i], output_signal_r_ear[i], output_signal_alpha_l_ear[i], output_signal_alpha_l_forehead[i], output_signal_alpha_r_forehead[i], output_signal_alpha_r_ear[i]]).reshape((1, 9))

        prediction.append(sess.run(predict, feed_dict={X: fenetre_a_predire})[0])

    print("TAILLE",len(prediction))

    for i in range(len(prediction)):
        if prediction[i][0] > prediction[i][1]:
            prediction_visualisation.append(-ECHELLE_SORTIE)
        else:
            prediction_visualisation.append(ECHELLE_SORTIE)

    #plt.plot(input_signal_l_ear, label='Signal')
    plt.plot(output_signal_l_ear, label='Signal filtre')

    #prediction_visualisation = [ECHELLE_PREDICTION if x==1 else x for x in prediction_visualisation]
    plt.plot(prediction_visualisation, label='Prediction')


    sortie = np.zeros((input_signal_l_ear.shape[0],2))

    # Sortie a predire
    for i in range(input_signal_l_ear.shape[0]):
        sortie[i][0] = output_signal_l_ear[i]

        if capteur[i] == ECHELLE_SORTIE:
            sortie[i][1] = 1
    sortie[sortie == 1,] = ECHELLE_SORTIE
    sortie = np.delete(sortie, 0, 1)
    plt.plot(sortie,  label='Sortie')
    plt.legend(bbox_to_anchor=(0.88, 0.88), loc=3, borderaxespad=0.)

    plt.show()
