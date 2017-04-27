import serial
import time
ser = serial.Serial("/dev/ttyACM0",timeout=1)
print (ser)

dataset = open("dataset.csv", "a")
# Initialisation (vide la sortie)
start_time = time.time()
capteur = -1
while (time.time() - start_time) < 2:
    donnee=str(ser.readline())
    try:
        capteur = int(donnee)
    except ValueError:
        print("Error value ")

    ser.flush()

# Enregistrement des valeurs
i = 0
while 1:
    donnee=str(ser.readline())
    try:
        capteur = int(donnee)

        # Capteur EEG

        dataset.write(str(i)+";"+"a;b;c"+";"+str(capteur)+"\n")
        i+=1
    except ValueError:
        print("Error value ")

    ser.flush()
    print(capteur)
