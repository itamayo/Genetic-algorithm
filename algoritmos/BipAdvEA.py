import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from copy import deepcopy
import numpy as np
import sys
import multiprocessing
global toolbox
import time
from numpy import *

"""
   Este algoritmo es una implementacion basada en algoritmo genetico simple con generacion de poblacion a traves
   procedimineto gready, en el cual permite genera una poblacion inicial con cierto criterio con el objetivo de
   dar un resultado lo más optimo posible al paradigma de particion de un grafo en dos fracciones que maximizen
   la suma de los pesos asociados a aristas que unen vertices de diferentes conjuntos.

"""

##########################################################
#
#  Dado la modularizacion de la libreria deap, y la posible
#  paralelazacion de flujos de calculo, en muchos casos
#  es posible utilizar varios procesadores en paralelo.
#  Para calcular los resultados no se ha utilizado esta
#  opcion ya que no sería comparable con los demas algoritmos,
#  pero si se ha usado para buscar los mejores conocidos.
#
#
##########################################################


global multiProcess
multiProcess = False

##########################################################
#
#  Clase nodo, define la estructura
#  del vertice, con su identificador y coste que tiene
#  en un contexto  particular, ademas de su valor(0,1)
#
##########################################################

class nodo:
    def __init__(self,_id,coste,value):
        self.id=_id
        self.coste=coste
        self.value = value

#########################################################################
#  Funcion de cruce entre padres diferentes para generar hijos
#  con los rasgos que maximizen los pesos entre los conjuntos de vertices
#  Dado que se utiliza gready para genera la poblacion, el Path
#  crossover permite encontrar las carasteristicas que interesan
#  heredar (Mas informacion en el articulo).
#  Basicamente se basa en el algoritmo path crossover reducido,
#  ya que realizarlo es muy costos cuando la permutacion es >40
#
#
#
#  @param individuo1
#  @param indiviudo2
#  @return los mejores 2 hijos herederos
#
#########################################################################

def cruce(ind1,ind2):
    global toolbox
    hijo1 = []
    hijos = []
    id1 = random.randint(0, nvec - 3)
    dim = nvec - 1
    i = 0
    while i<(int(nvec/3)):
        if id1 == dim:
            id2=0
        else:
            id2=id1+1
        hijo1 = deepcopy(ind1)
        aux = hijo1[id1]
        hijo1[id1] = hijo1[id2]
        hijo1[id2] = aux
        hijo2 =deepcopy(ind2)
        aux = hijo2[id1]
        hijo2 [id1] = hijo2 [id2]
        hijo2 [id2] = aux
        hijo1Fitness = toolbox.evaluate(hijo1)
        hijo2Fitness = toolbox.evaluate(hijo2)
        if hijo1Fitness>hijo2Fitness:
            hijos.append(hijo1)
        else:
            hijos.append(hijo2)
        if id2 ==0:
            id1=1
        else:
            id1+=1
        i+=1
    hijos.sort(key=lambda x: x.fitness.values, reverse=True)
    return hijos[0],hijos[1]

######################################################################
#  Funcion analiza el coste de añadir un 0,1 al vector en construccion
#  @param vector en construccion
#  @param nodo a añadir
#  @param valor a añadir
#  @return matriz de coste
#
######################################################################

def getCoste(tnpSol,x,bin1):
    global pesos
    coste  = 0

    if (len(tnpSol)>1):
        for i in tnpSol:
            for j in tnpSol:
               if i!=j:
                   coste += pesos[i.id,j.id]
            coste+=pesos[i.id,x]
    else:
        coste+=pesos[tnpSol[0].id,x]
    return coste

######################################################################
#  Funcion parte del algoritmo Gready, se seleccionan log(N)*2
#  Mejores elementos segun su probabilidad inversas al coste sobre
#  elemento seleccionado.
#
#  @param lista de N nodos seleccionables con sus respectivos costes
#  sobre el elemento seleccionado
#  @param vector en construccion
#  @return el nodo seleccionado
#
#####################################################################

def seleccionar(lista,tnpVector):
      global listaPosibles
      sumFval= np.sum(c.coste for c in lista)
      listaProbs = []
      evals = []
      for i in lista:
         listaProbs.append(i.coste/sumFval)
         evals.append(i.id)

      listaProbs.sort(key=lambda x: x, reverse=True)
      selectedValue = np.random.choice(evals, size=(1,), p=listaProbs,replace=False)
      nodoId = lista[lista == int(selectedValue[0])].id
      listaPosibles = [i for i in listaPosibles if i!=nodoId]
      nodo = lista[lista == int(selectedValue[0])]
      numOfOnes =  np.sum([n.value for n in tnpVector])
      numOfZeros = len(tnpVector) - numOfOnes
      if numOfOnes>=numOfZeros:
          oneProb = 0.5 - numOfOnes/nvec
          zeroProb = 1 - oneProb
      else:
          zeroProb = 0.5 - numOfZeros/nvec
          oneProb = 1 - zeroProb
      if (oneProb<=0):
          oneProb = 0
          zeroProb =1
      if (zeroProb<=0):
          oneProb=1
          zeroProb=0

      binariValue = np.random.choice([0,1], size=(1,), p=[zeroProb,oneProb])
      nodo.value = binariValue[0]
      return nodo

######################################################################
#  Funcion parte del algoritmo Gready, calcula los costes del nodo
#  seleccionado respecto a una lista de posibles nodos.
#
#
#  @param nodo seleccionado
#  @param lista de posibles nodos
#  @return lista con costes de los posibles nodos
#
#####################################################################

def calcularCostes(constrctLista,listaP):
      costeDeRutaSel = []
      for x in listaP:
            nod1 = nodo(x,getCoste(constrctLista,x,0),0)
            nod2 = nodo(x,getCoste(constrctLista,x,1),1)
            costeDeRutaSel.append(nod1)
            costeDeRutaSel.append(nod2)
      costeDeRutaSel.sort(key=lambda x: x.coste, reverse=False)
      return costeDeRutaSel

######################################################################
#  Funcion constructor de Gready, se genera un vector
#  elegiendo con cierta aleatoriedad de una lista ordena por
#  costes del elemento seleccionado a los demas, que ademas asegura
#  las restricciones
#
#
#  @return permutacion
#
#####################################################################

def GreadyConstructor():
      global listaPosibles
      global vector
      if len(vector) ==0:
          sol = []
          selectedNum = random.randint(0,nvec-1)
          selectedNodo = nodo(selectedNum,100000,random.randint(0,2))
          sol.append (selectedNodo)
          listaPosibles = [i for i in range(nvec) if i!=selectedNodo.id]
          for i in range(nvec-1):
              lista = calcularCostes(sol,listaPosibles)
              selectedNodo = seleccionar(lista,sol)
              sol.append(selectedNodo)
          vector = [v.value for v in sol]
      vectorNum = vector[0]
      vector = vector[1:]
      return vectorNum

################################################################################
#   Funcion que calcula el Fvalor de la funcion particion
#   de un grafo de los pesos entre vertices de diferentes grafos
#   @param: vector Vector binario que define la distribucion de vert. en 2 subgr
#   @return fvalor de la particion
################################################################################

def BipEvaluator(vector1):

    ## Leemos los pesos a traves de la funcion antes definida
    ## Leemos la dimension de la matrix, como es quadratica nos quedamos con la primera dimension
    n = nvec
    balance =  np.sum(vector1) # Numero de nodos en una de las partes
    factible=(balance==n/2)
    ## Verificamos que la particiones este compuestas por el mismo numero de nodos/vertices
    f = 0
    sol_vect = copy(vector1)
    ## Creamos los grafo separados para iterar sobre estos
    ## Solo iteranos sobre los vertices que componen diferentes grafos
    ## Consegimos reducir el bucle de 45 del ejercicio propuesto a 25
    grafo1 =  np.where(sol_vect == 0)[0]
    grafo2 =  np.where(sol_vect == 1)[0]
    for i in range(len(grafo1)):
        for j in range(len(grafo2)):
            f += pesos[grafo1[i],grafo2[j]]
    if (not factible):
        #print("No esta balanceado (0s y 1s) prueba otra vez...")
        return f/2,
    return f, # devuelve suma de pesos

##################################################################
#
#  Funcion que inicializa el framework deap que nos ayuda a la
#  hora de construir un sistema evolutivo. Entre otras cosas,
#  se define que tenemos un problema de maximizacion,de como generar
#  los individuos y la poblacion, como evaluar estos y finalmente
#  como evolucionar de una generacion de otra a traves de la funcion
#  cruce, mutacion y seleccion.
#
###################################################################

def initToolbox():
    ## Se crea una clase FitnessMin para la minimizacion del fval/funcion de coste

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    ## Se crea una clase individuo asociada a la clase FitnessMin
    ## ya que nos interesa minizar
    creator.create("Individual", list, fitness=creator.FitnessMax)
    ## creacion del objeto Toolbox de DEAP
    ## permite acceder a los algoritmos internos, que facilita la implementacion
    ## de algoritmos evolutivos.
    toolbox = base.Toolbox()
    if multiProcess:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    ## Utilizaremos una representacion basada en vector binaria
    ## Registramos la funcion GreadyConstructor que se encarga mas tarde en
    ## generar el vector del individuo
    toolbox.register("digVector", GreadyConstructor)
    ## Definimos como va ser el individuo a traves de nperm y los valores que devuelve la funcion digVector
    toolbox.register("individual", tools.initRepeat, creator.Individual,
        toolbox.digVector, nvec)
    ## Establecemos la estructura de la poblacion a partir de los individuos
    ## Se inicializara mas tarde
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    ## Definimos la funcion de evaluacion del individuo
    toolbox.register("evaluate", BipEvaluator)
    ## Definimos funcion de cruce para la evolucion
    ## que tenga en cuenta el vector y sus restricciones que permite heredar las Mejores
    ## carasteristicas
    toolbox.register("mate", cruce)
    ## Mutar con probabilidad de 0.05, cambiar dos atributos entre si
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    ## Establecemos seleccion por torneo con un parametro de torneo = 3
    toolbox.register("select", tools.selTournament,tournsize=3)
    return toolbox

###################################################
# Lectura de los pesos del fichero BIPART
# @param ruta del fichero BIPART
# @return matriz de pesos
###################################################
def Read_Bipart_Instance(fname):
 hdl = open(fname, 'r')
 mylist = hdl.readlines()
 hdl.close()
 n = eval(mylist[0])
 edge_weights = np.zeros((n,n))      # Pesos de las aristas
 for i in range(n):
   for j,val in enumerate(mylist[i+1].split()): ## Simetria en la matriz, se evita rebundancia j=i+1
     edge_weights[i,j]=eval(val)
 return edge_weights


################################################################################
#  Funcion principal que implementa la busqueda basada en el algoritmo
#  genetico simple. Se obtiene los mejores en npob/ngen iteraciones
#  @param ruta del fichero
#  @param numero de la poblacion
#  @param nunmero de generaciones a tratar
#  @return mejor permutacion,los mejores evaluaciones (npob/ngen)
#
################################################################################


def BipAdvEA(fname,npob1,ngen):
    random.seed(64)
    global evals
    global pesos
    global nvec
    global npob
    global toolbox
    global vector
    vector = []
    stats = []
    pesos = Read_Bipart_Instance(fname)
    nvec = pesos.shape[0]
    tic = time.clock()
    toolbox = initToolbox()
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pob = toolbox.population(n=npob1)

    algorithms.eaSimple(pob, toolbox,cxpb=0.7, mutpb=0.2, ngen=ngen,verbose=False)
    best_sol = tools.selBest(pob, k=int(npob1/ngen))
    evals = []
    for sol in best_sol:
        evals.append(sol.fitness.values[0])
    evals.sort(reverse=True)
    toc = time.clock()
    return best_sol[0], evals

##############################################################
#
#  Llamada normal a la funcion QAPAdvLocalSearch
#    python3 QAPAdvLocalSearch.py
#  Llamada para recorge datos
#    python3 QAPAdvLocalSearch.py test
#  Llamada para buscar los mejores
#    python3 QAPAdvLocalSearch.py multiprocess
#
##############################################################

if __name__ == "__main__":
    if len(sys.argv)>2 and sys.argv[2] == 'multiprocess':
        print("Utilizando",multiprocessing.cpu_count(),"cpus")
        multiProcess = True
    if len(sys.argv)>1 and sys.argv[1]=='test':
        global pob
        print("Analizando algoritmos para obtencion de datos")
        files = ["Cebe.bip.n10.1","Cebe.bip.n10.2","Cebe.bip.n10.3","Cebe.bip.n10.4",
        "Cebe.bip.n20.1","Cebe.bip.n20.2","Cebe.bip.n20.3","Cebe.bip.n20.4","Cebe.bip.n50.1","Cebe.bip.n50.2"]
        fits=[]
        times = []
        for f in files:
            stats = []
            fits = []
            times = []
            pobs = []
            mins = []
            maxs = []
            for i in range(10):
                tic = time.clock()
                sol,values = BipAdvEA("../Instances/BIPART/"+f,200,20)
                toc = time.clock()
                times.append(toc-tic)
                fits+=values
                mins.append(values[-1])
                maxs.append(values[0])
                print("Iteracion ",i," de ",f," parametros ",200,20)
            stats.append([max(maxs),mean(mins),mean(maxs),mean(fits),var(fits),mean(times)])
            savetxt("results/lastBipEA"+f+".200.20",stats)
    else:
        sol,evals = BipAdvEA("../Instances/BIPART/Cebe.bip.n20.4",1200,60)
        print(sol,evals)
