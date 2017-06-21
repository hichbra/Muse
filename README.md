# Muse

## Pré-Requis

Pour faire fonctionner le protocole il faut installer avant tout :
- Tensorflow
- Python (avec les librairies sys, time, serial, scipy, matplotlib, numpy, csv, tensorflow, sklearn, random, os)
- Osc

Et disposer d'un Muse Interaxon en guise d'Interface Cerveau-Machine

## Fonctionnement

1. Télechargez l'archive "Protocole"
2. Executez "1_MuseRecupSignaux.py" : Le protocole se lance et les données sont enregistrées dans le dossier data/
3. Executez "2_ClassificationStimulus.py" : La classification se lance, le modèle prédictif est stocké dans le dossier model/
  Par défaut réglé sur la variable gamma_absolutel_forehead (colonne 23:24) avec 55000 itérations.
4. Executez "3_MuseTraitement.py" : Utilisation du modèle afin de discriminer votre état actuel 
