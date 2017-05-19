from liblo import *

import sys
import time
import serial

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

if __name__ == "__main__":
    i=0
    while 1:
	#print(server.acc_x)
	time.sleep(0.05)
	donnee=str(ser.readline())
	try:
		capteur = int(donnee)
		
		dataset.write(str(i)+";"+str(server.l_ear)+";"+str(server.l_forehead)+";"+str(server.r_forehead)+";"+str(server.r_ear)+";"+str(server.r_aux)+";"
			""+str(server.delta_absolutel_ear)+";"+str(server.delta_absolutel_forehead)+";"+str(server.delta_absoluter_forehead)+";"+str(server.delta_absoluter_ear)+";"
			""+str(server.theta_absolutel_ear)+";"+str(server.theta_absolutel_forehead)+";"+str(server.theta_absoluter_forehead)+";"+str(server.theta_absoluter_ear)+";"
			""+str(server.alpha_absolutel_ear)+";"+str(server.alpha_absolutel_forehead)+";"+str(server.alpha_absoluter_forehead)+";"+str(server.alpha_absoluter_ear)+";"
			""+str(server.beta_absolutel_ear)+";"+str(server.beta_absolutel_forehead)+";"+str(server.beta_absoluter_forehead)+";"+str(server.beta_absoluter_ear)+";"
			""+str(server.gamma_absolutel_ear)+";"+str(server.gamma_absolutel_forehead)+";"+str(server.gamma_absoluter_forehead)+";"+str(server.gamma_absoluter_ear)+";"			
			""+str(server.dropped_samples)+";"
			""+str(capteur)+"\n")

		ser.flush()

		i+=1
	except ValueError:
		print("Error value ")

	ser.flush()
	print(capteur)
