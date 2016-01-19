#-*- coding:Utf-8 -*-
#Code done by : Timothee Lesort and Florian Bordes#
#Some part of the code are highly inspired by code from lab of the course IFT3395-6390-A-A15 in the University of Montreal given by M. Pascal Vincent#

import numpy as np
import gzip,cPickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import pylearn2
import theano

### Fonctions utilitaires ###
#Calcul d'un softmax numéricalement stable
# Entrée : matrice x
def softmax(x):
    w = np.array(x)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True)
    return dist

#Function relu
# Entrée : vecteur x
def relu(x):
    return np.maximum(0,x)
    
    #Dérivé de la fonction rectifieur
# Entrée : vecteur x
def rect_derive(x):
    return x > 0
    
#Function sigmoid
# Entrée : vecteur x
def sigmoid(x):
    return 1/(1+np.exp(-x))
    

#Initialisation des poids avec loi uniforme
# Entrée : nombre n
def init_weight(nc, no):
    bound = 1/np.sqrt(nc)
    return np.random.uniform(low=-bound, high=bound, size=(no,nc))



#Onehot renvoyant une matrice
def onehot(m,y):
    t = np.zeros((len(y), m))
    for i in range(len(y)):
        t[i,y[i]]= 1
    return t

### Class Multilayer Percepeptron ###
class MLP(object):
    #Initialisation du modele avec les parametres
    def __init__(self, inputs, outputs, n_in, n_neurone, n_out):
        #On separe les ensembles d'entrainement, de validation et de test
        self.x, self.xv, self.xt = inputs
        self.y, self.yv, self.yt = outputs
        #On initialise les parametres
        self.w1 = init_weight(n_in, n_neurone)
        #self.b1 = np.random.uniform(-0.1,0.1,n_neurone)
        self.b1 =np.zeros(n_neurone)
        self.w2 = init_weight(n_neurone, n_out)
        #self.b2 = np.random.uniform(-0.1,0.1,n_out)
        self.b2 = np.zeros(n_out)
        #Autre
        self.n_out = n_out
        self.n_neurone=n_neurone

    #Forward propagation
    def fprop(self, x):
        return softmax(np.tensordot(self.w2, sigmoid(np.tensordot(self.w1, x, (1,1)).transpose() + self.b1), (1,1)).transpose() + self.b2)

    #Backward propagation
    def backprop(self, x,y,o):
        grad_oa = o - onehot(self.n_out,y)
        h_a = np.tensordot(self.w1, x, (1,1)).transpose() + self.b1
        h_s = sigmoid(h_a)
        grad_b2 = grad_oa
        grad_W2 = np.dot(grad_oa.T, h_s)
        grad_hs = np.tensordot(self.w2, grad_oa, (0,1)).transpose()
        grad_ha = grad_hs * h_s*(1-h_s)
        grad_W1 = np.dot(grad_ha.T, x)
        grad_b1=grad_ha
        return grad_W1, grad_b1, grad_W2, grad_b2

    #Fonction de test comptant les erreurs
    def test(self, x, y):
        erreur = 0
        cout=0
        
        for i in range(len(x)):
            o = np.argmax(self.fprop(x[i:i+1]))
            cout=cout-np.log(np.max(self.fprop(x[i:i+1])))
            if y[i] != o:
                erreur = erreur + 1.0
        print 'erreur moyenne', '%.6f'%(erreur/(len(x)))
        print 'cout total', cout
        print 'cout moyen', cout/(len(x))
        return erreur/(len(x)), cout/(len(x))

    #Fonction d'apprentissage des parametres
    def train(self, l = 0.04, m = 0.00005, batch_size = 25, epoch_max = 20):
        print( 'start train',time.strftime('%d/%m/%y %H:%M',time.localtime()))
        nb_batch = len(self.x) / batch_size
        nbepoch = 0
        errTrain, errValid, errTest, coutTrain, coutValid, coutTest = [], [], [], [], [], []
        #Pour chaque epoch
        while(nbepoch < epoch_max):            
            print 'epoch',nbepoch
            print( time.strftime('%d/%m/%y %H:%M',time.localtime()))
            #Faire l'entrainement sur chaque batch
            for i in range(nb_batch - 1):
                #Recuperer les bounds du batch
                low = i*batch_size
                up = batch_size*(i+1)
                #Forward propagation
                o = self.fprop(self.x[low:up])
                #Backward propagation
                grad_W1, grad_b1, grad_W2, grad_b2 = self.backprop(self.x[low:up], self.y[low:up], o)
                #Weights and bias update with weight decay
                self.w1 = self.w1 - l * grad_W1 - m * 2 * np.sum(self.w1, axis=0)
                self.b1 = self.b1 - l * np.sum(grad_b1, axis=0)
                self.w2 = self.w2 - l * grad_W2 - m * 2 * np.sum(self.w2, axis=0)
                self.b2 = self.b2 - l * np.sum(grad_b2, axis=0)
            nbepoch = nbepoch + 1
            #Erreur sur l'ensemble d'entrainement
            print "------------------------"
            print "---train---"
            erreurClassTrain, coutMoyTrain= self.test(self.x,self.y)
            errTrain.append(erreurClassTrain)            
            coutTrain.append(coutMoyTrain)
            #Erreur sur l'ensemble de validation
            print "---valid---"
            erreurClassValid, coutMoyValid= self.test(self.xv,self.yv)
            errValid.append(erreurClassValid)            
            coutValid.append(coutMoyValid)
            #Erreur sur l'ensemble de test
            print "---test---"
            erreurClassTest, coutMoyTest= self.test(self.xt,self.yt)
            errTest.append(erreurClassTest)            
            coutTest.append(coutMoyTest)
            
            plt.figure(1)
            plt.plot(errTrain, 'g-', errValid, 'r-', errTest, 'b-')
            plt.savefig('result.png')
            plt.figure(2)
            plt.plot(coutTrain, 'g-', coutValid, 'r-', coutTest, 'b-')
            plt.savefig('result2.png')
            
            texte=np.transpose((errTrain,coutTrain,errValid,coutValid,errTest,coutTest))
            #enregistrement sous forme de colonne [errTrain;coutTrain;errValid;coutValid;errTest;coutTest]
            np.savetxt('save_errors', texte)


#Utilisation de moon pour tester le MLP
class moon():
    def __init__(self):
        #On charge les donnees
        moon = np.loadtxt('2moons.txt')
        X = moon[:,:-1]
        Y = moon[:,-1]
        #On creer les sets d'entrainement, de validation et de test
        self.x=X[:600]
        xv=X[600:900]
        xt=X[900:1100]
        self.y=Y[:600]
        yv=Y[600:900]
        yt=Y[900:1100]
        #On cree le modele
        self.inputs = self.x, xv, xt
        self.outputs = self.y, yv, yt
        self.model = MLP(self.inputs, self.outputs, 2, 10, 2)
    
    def train(self):
        self.model.train()

#Utilisation de mnist pour tester le MLP
def mnist():
    #On charge les donnees
    print 'début',time.strftime('%d/%m/%y %H:%M',time.localtime())
    #f=gzip.open('mnist.pkl.gz')
    #data=pickle.load(f)
    
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    #On creer les sets d'entrainement, de validation et de test
    x = train_set[0]
    y = train_set[1]
    xv =valid_set[0]
    yv = valid_set[1]
    xt = test_set[0]
    yt = test_set[1]
    #On cree le modele
    inputs = x, xv, xt
    outputs = y, yv, yt
    model = MLP(inputs, outputs, 28*28, 1000, 10)
    #On entraine le modele
    
    print 'avant train',time.strftime('%d/%m/%y %H:%M',time.localtime())
    model.train() 
    print 'apres train',time.strftime('%d/%m/%y %H:%M',time.localtime())

if __name__ == '__main__':

    mnist()

