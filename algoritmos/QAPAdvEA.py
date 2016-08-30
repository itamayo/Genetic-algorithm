import random
import sys
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
from numpy import *
import time
from copy import deepcopy
import multiprocessing


"""
   Este algoritmo es una implementacion basada en algoritmo genetico simple con generacion de poblacion a traves
   procedimineto gready, en el cual permite genera una poblacion inicial con cierto criterio con el objetivo de
   dar un resultado lo más optimo posible al paradigma QAP.

"""

##########################################################
#
#  Dado la modularizacion de la libreria deap, y la posible
#  paralelazacion de flujos de calculo, en muchos casos
#  es posible utilizar varios procesadores en paralelo.
#  Para calcular los resultados no se ha utilizado esta
#  opcion ya que no sería comparable con los demas algoritmos,
#  pero si sea usado para buscar los mejores conocidos.
#
#
##########################################################


global multiProcess
multiProcess = False

##########################################################
#
#  Clase nodo, define la estructura
#  del edificio, con su identificador y coste que tiene
#  el edifico en contexto en particular
#
##########################################################

class nodo:
    def __init__(self,_id,coste):
        self.id=_id
        self.coste=coste

##################################################################
#  Funcion que calcula el Fitness de la funcion QAP dada
#  una matrix de distancias, otra de flujos y la permutacion
#  como opcion de orden a evaluar.
#
#  @param: perm Permutacion que se quiere valorar.Array[1,2,3,4]
#  @return fval de la permutacion
#
###################################################################


def QAPEvaluator(perm):
    global nperm
    factible = True
    for i in range(nperm):
        if i not in perm:
            factible = False
    ## Inicializacion de las matrixe, todo a cero
    f = 0;
    ## f = sum_{i}^{n} \sum_{j}^{n} d_{_{i,j}}f_{\delta (i)\delta (j)
    for i in range(nperm):
     for j in range(nperm):
        f += ddistancias[i,j]*dflujos[perm[i],perm[j]];
    if (not factible): return f*2,
    return f,

##################################################################
#  Funcion de cruce entre padres diferentes para generar hijos
#  con los rasgos que minimizen los flujos entre los edificios
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
###################################################################

def cruce(ind1,ind2):
    global toolbox
    hijo1 = []
    hijos = []
    id1 = random.randint(0, nperm - 1)
    dim = nperm - 1
    i = 0
    while i<(int(nperm/3)):
        if id1 == dim:
            id2=0
        else:
            id2=id1+1
        gen1 = ind1[id1]
        gen2 = ind1[id2]
        hijo1 = deepcopy(ind1)
        aux = hijo1[id1]
        hijo1[id1] = hijo1[id2]
        hijo1[id2] = aux
        hijo2 =deepcopy(ind2)
        p2id1 = hijo2.index(gen1)
        p2id2 = hijo2.index(gen2)
        aux = hijo2[p2id1]
        hijo2 [p2id1] = hijo2 [p2id2]
        hijo2 [p2id2] = aux
        hijo1Fitness = toolbox.evaluate(hijo1)
        hijo2Fitness = toolbox.evaluate(hijo2)
        if hijo1Fitness<hijo2Fitness:
            hijos.append(hijo1)
        else:
            hijos.append(hijo2)
        if id2 ==0:
            id1=1
        else:
            id1+=1
        i+=1
    hijos.sort(key=lambda x: x.fitness.values, reverse=False)
    return hijos[0],hijos[1]

##################################################################
#
#  Funcion que inicializa el framework deap que nos ayuda a la
#  hora de construir un sistema evolutivo. Entre otras cosas,
#  se define que tenemos un problema de mimizacion,de como generar
#  los individuos y la poblacion, como evaluar estos y finalmente
#  como evolucionar de una generacion de otra a traves de la funcion
#  cruce, mutacion y seleccion.
#
###################################################################

def initToolbox():
    ## Se crea una clase FitnessMin para la minimizacion del fval/fitness de coste
    global multiProcess
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    ## Se crea una clase individuo asociada a la clase FitnessMin
    ## ya que nos interesa minizar
    creator.create("Individual", list, fitness=creator.FitnessMin)
    ## creacion del objeto Toolbox de DEAP
    ## permite acceder a los algoritmos internos, que facilita la implementacion
    ## de algoritmos evolutivos.
    toolbox = base.Toolbox()
    ## En caso de que el flag multiProcess este activo, activar multiproceso
    if multiProcess:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    ## Registramos la funcion GreadyConstruction como generador de la permutacio
    toolbox.register("digPermutacion", GreadyConstructor)
    ## Definimos como va ser el individuo a traves de nperm y los valores que devuelve la funcion GreadyConstruction
    toolbox.register("individual", tools.initRepeat, creator.Individual,
        toolbox.digPermutacion, nperm)
    ## Establecemos la estructura de la poblacion a partir de los individuos construidos por Gready
    ## Se inicializara mas tarde
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    ## Definimos la funcion de evaluacion del individuo
    toolbox.register("evaluate", QAPEvaluator)
    ## Definimos funcion de cruce para la evolucion
    ## que tenga en cuenta la permutacion que permite heredar las Mejores
    ## carasteristicas
    toolbox.register("mate", cruce)
    ## Mutar con probabilidad de 0.2, cambiar dos atributos entre si
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    ## Establecemos seleccion por torneo con un parametro de torneo = 3
    toolbox.register("select", tools.selTournament,tournsize=3)
    return toolbox

######################################################################
#  Funcion parte del algoritmo Gready, se seleccionan log(N)*2
#  Mejores elementos segun su probabilidad inversas al coste sobre
#  elemento seleccionado.
#
#  @param lista de N nodos seleccionables con sus respectivos costes
#  sobre el elemento seleccionado
#  @return el nodo seleccionado
#
#####################################################################

def seleccionar(lista):
      global listaPosibles
      global selectedValue
      listaProbs = []
      lista.sort(key=lambda x: x.coste, reverse=False)
      evals = []
      j=0
      n = len(lista)
      if (n>=3):
          n = int(math.log(n)*2)
          lista = lista[0:n]
      else:
          lista = lista[0:n]

      listaProbs = [1/n for x in lista]
      evals = [x.id for x in lista]
      selectedValue = np.random.choice(evals, size=(1,), p=listaProbs)
      listaPosibles = [i for i in listaPosibles if i!=selectedValue[0]]
      return [i for i in lista if i.id==selectedValue[0]][0]

######################################################################
#  Funcion parte del algoritmo GRASP, calcula los costes del nodo
#  seleccionado respecto a una lista de posibles nodos.
#
#
#  @param nodo seleccionado
#  @param lista de posibles nodos
#  @return lista con costes de los posibles nodos
#
#####################################################################

def calcularCostes(sel,listaP):
      global matCostes
      costeDeRutaSel = []
      for x in listaP:
         nod = nodo(x,matCostes[sel.id,x])
         costeDeRutaSel.append(nod)
      costeDeRutaSel.sort(key=lambda x: x.coste, reverse=False)
      return costeDeRutaSel

######################################################################
#  Funcion constructor de Gready, se genera una permutacion
#  elegiendo con cierta aleatoriedad de una lista ordena por
#  costes del elemento seleccionado a los demas
#
#
#  @return permutacion
#
#####################################################################

def GreadyConstructor():
      global listaPosibles
      global valorNoAperecidos
      global selectedValue
      global permutacion
      lista = []
      if len(permutacion)==0:
          valorNoAperecidos = [i for i in range(nperm)]
          sol = []
          if len(valorNoAperecidos)==0:
              selectedNum = random.randint(0,nperm-1)
          else:
              nNoaparecidos = len(valorNoAperecidos)
              probs = [1/nNoaparecidos for x in valorNoAperecidos]
              selectedNum  =  np.random.choice(valorNoAperecidos, size=(1,), p=probs)[0]
          valorNoAperecidos =  [i for i in range(nperm) if i!=selectedNum]
          selectedNodo = nodo(selectedNum,0)
          permutacion.append(selectedNodo.id)
          listaPosibles = [i for i in range(nperm) if i!=selectedNodo.id]
          for i in range(nperm-1):
              lista = calcularCostes(selectedNodo,listaPosibles)
              selectedNodo = seleccionar(lista)
              permutacion.append(selectedNodo.id)

      nodo1 = permutacion[0]
      permutacion = permutacion[1:]
      return nodo1

######################################################################
#  Funcion que inializa los valores de coste entre los diferentes
#  edificios
#
#  @return matriz de coste
#
######################################################################

def getCostes():
    global ddistancias
    global dflujos
    for i in range(nperm):
        for j in range(nperm):
            matCostes[i][j] = dflujos[i][j]
    return matCostes

######################################################################
#  Funcion que lee las matrices de flujos y de distancias dado
#  una ruta de fichero.
#
#  @param ruta del fichero
#  @return permutacion
#
#####################################################################

def leerMatrices(fname):
    global ddistancias
    global dflujos
    global nperm
    global qap
    ## Abrimos el fichero (formato lectura) que contiene las matrices
    hdl = open(fname, 'r')
    ## Leemos los datos del fichero y lo guardamos en memoria
    qap = hdl.readlines()
    nperm = eval(qap[0])
    ## Cerramos el recurso (fichero)
    hdl.close()
    ## TODO: verificar que todos los puntos aparecen en la permutacion
    ## Inicializacion de las matrixe, todo a cero
    dflujos  = np.zeros((nperm,nperm))
    ddistancias  = np.zeros((nperm,nperm))
    for i in range(nperm):
      for j,val in enumerate(qap[i+1].split()):
        dflujos[i,j]=eval(val)

    for x in range(nperm):
       for k,val2 in enumerate(qap[nperm+x+1].split()):
        ddistancias[x,k]=eval(val2)

################################################################################
#  Funcion principal que implementa la busqueda basada en el algoritmo
#  genetico simple. Se obtiene los mejores en npob/ngen iteraciones
#  @param ruta del fichero
#  @param numero de la poblacion
#  @param nunmero de generaciones a tratar
#  @return mejor permutacion,los mejores evaluaciones (npob/ngen)
#
################################################################################

def QAPAdvEA(fname,npob,ngen):
    global evals
    global qap
    global nperm
    global pob
    global permutacion
    global matCostes
    global toolbox

    random.seed(64)
    permutacion = []
    best_values = []
    stats = []
    tic = time.clock()
    leerMatrices(fname)
    matCostes = np.zeros((nperm,nperm))
    matCostes = getCostes()
    toolbox = initToolbox()
    pob = toolbox.population(n=npob)
    algorithms.eaSimple(pob, toolbox,cxpb=0.7, mutpb=0.2, ngen=ngen,verbose=False)
    best_sol = tools.selBest(pob, k=int(npob/ngen))
    best_sol.sort(key=lambda x: x.fitness.values[0], reverse=True)
    evals = []
    for sol in best_sol:
        evals.append(sol.fitness.values[0])
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
                files = ["Cebe.qap.n10.1","Cebe.qap.n10.2","Cebe.qap.n20.1","Cebe.qap.n20.2",
                "Cebe.qap.n30.1","Cebe.qap.n30.2","Cebe.qap.n40.1","Cebe.qap.n40.2","Cebe.qap.n50.1","Cebe.qap.n50.2"]
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
                        sol,values = QAPAdvEA("../Instances/QAP/"+f,200,20)
                        toc = time.clock()
                        times.append(toc-tic)
                        fits+=values
                        mins.append(values[-1])
                        maxs.append(values[0])
                        pobs.append(pob)
                        print("Iteracion ",i," de ",f," parametros ",200,20)
                    stats.append([min(mins),mean(mins),mean(maxs),mean(fits),var(fits),mean(times),mean(pob)])
                    savetxt("results/lastQapEA"+f+".200.20",stats)
    else:
        tic = time.clock()
        sol,values = QAPAdvEA("../Instances/QAP/Cebe.qap.n50.1",200,20)
        toc = time.clock()
        print(sol,values,toc-tic)
