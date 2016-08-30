#!/usr/bin/env python3
import sys
import numpy as np
from numpy import *
import copy
import random
import time
"""
   Este algoritmo es una implementacion basada en GRASP (Greedy randomized adaptive search procedure),
   en concreto se basa en la cardinalidad. El objetivo del algortimo es obtener el resultado mas optimo
   en un tiempo relativamente admisible (3') en la problematica QAP. El objetivo no es, obtener el mejor conocido
   (si se obtiene mucho mejor), esta mas orientado  a desenpeñar un funcion que pueda a ir evolucionando hacia el
   optimo global, y solo dependa los recursos que se quieran aplicar.

"""

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
 global qap
 n = eval(qap[0])
 perm = np.asarray(perm) - 1
 ## Inicializar el f valor
 f = 0;
 ## f = sum_{i}^{n} \sum_{j}^{n} d_{_{i,j}}f_{\delta (i)\delta (j)
 for i in range(n):
     for j in range(n):
        f += ddistancias[i,j]*dflujos[perm[i],perm[j]];
 return f

##############################################################################
#  Funcion que busca el optimo local de la funcion quadratica de una dado una
#  permutacion inicial;el cual calcula un fvalor de según distancias y
#  flujo entre nodos.
#
#    @param: max_eval maxima interacciones permitidas en la busqueda
#    @param: perm Permutacion que se quiere valorar.Array[1,2,3,4]
#    @return mejor fitness,mejor permutacion encontrada
#
##############################################################################

def QAPLocalSearch(permV,max_eval):
     global qap
     global numeroEvaluaciones
     ## En el formato QAP,el primer valor define la dimension de la matrix
     n = eval(qap[0])
     perm = [p.id for p in permV]
     ## Inicializamos variables
     ## Sol: solucion del vector elegido
     ## Sol_tnp: solucion del mejor vecino
     sol = 0
     sol_tnp = -1
     numeroEvaluaciones=0
     notImproved = 0
     ## Buscamos la mejor solucion hasta que alla 3 iteraciones sin mejora o se alla
     ## alcanzado la maxima de iteraciones
     ## Si encontramos una mejor permutacion en la vecindad actualizamos la permutacion
     while (notImproved<3):
        ## Calculamos el fvalor del permutacion actual
        sol_tnp = sol = QAPEvaluator(perm)
        if numeroEvaluaciones==max_eval:
            break
        numeroEvaluaciones+=1
        ## Buscamos entre la vecinda el mejor vecino
        for i in range(nperm):
            for j in range(nperm):
                ## Inicializar la permutacion para usar como nuevo vecino
                ## Intercambiar posiciones, para encontrar un nuevo vecino
                if nperm<11:
                        tnp_vector = copy.copy(perm)
                        aux = perm[i]
                        tnp_vector[i] = perm[j]
                        tnp_vector[j] = aux
                        ## Evaluar  vecino
                        fval = QAPEvaluator(tnp_vector)
                        if numeroEvaluaciones==max_eval:
                            break
                        numeroEvaluaciones+=1
                        ## Si el vecino es mejor que la anterior permutacion (vecina)
                        ## Actualizar como mejor vecino
                        if (fval < sol_tnp):
                            sol_tnp = fval
                            perm = tnp_vector
                ## Si permutacion es mayor a 10, entonces valorar vecino siempre
                ## y cuando sea mejor que el anterior
                else:
                    if j!=0 and i != j and (matCostes[i,j]<matCostes[i,j-1]):
                            tnp_vector = copy.copy(perm)
                            aux = tnp_vector[i]
                            tnp_vector[i] = perm[j]
                            tnp_vector[j] = aux
                            ## Evaluar  vecino
                            fval = QAPEvaluator(tnp_vector)
                            if numeroEvaluaciones==max_eval:
                                break
                            numeroEvaluaciones+=1

                            ## Si el vecino es mejor que la anterior permutacion (vecina)
                            ## Actualizar como mejor vecino
                            if (fval < sol_tnp):
                                sol_tnp = fval
                                perm = tnp_vector
        if (sol_tnp<sol):
            notImproved =0
        else:
            notImproved+=1
     return sol,perm

######################################################################
#  Funcion parte del algoritmo GRASP, se seleccionan log(N)*2
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
#  Funcion constructor de GRASP, se genera una permutacion
#  elegiendo con cierta aleatoriedad de una lista ordena por
#  costes del elemento seleccionado a los demas
#
#
#  @return permutacion
#
#####################################################################

def GraspConstructor():
      global listaPosibles
      global concurrencia
      global valorNoAperecidos
      global selectedValue
      lista = []
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
      sol.append(selectedNodo)
      listaPosibles = [i for i in range(nperm) if i!=selectedNodo.id]
      for i in range(nperm-1):
          lista = calcularCostes(selectedNodo,listaPosibles)
          selectedNodo = seleccionar(lista)
          sol.append(selectedNodo)
      return sol

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
    ## Leer la matriz correspondiente a las distancias
    ## Al ser una matrix de distancias, siendo simetricas se evita la sobreinformacion
    for i in range(nperm):
      for j,val in enumerate(qap[i+1].split()):
        dflujos[i,j]=eval(val)
    ## Leer la matriz correspondiente a los flujos
    for x in range(nperm):
       for k,val2 in enumerate(qap[nperm+x+1].split()):
        ddistancias[x,k]=eval(val2)


################################################################################
#  Funcion principal que implementa la busqueda basada en el algoritmo
#  GRASP. Esta compuesto por un bucle que solo termina si; se a sobrepasado
#  el maximo de evaluaciones posibles, o si no se han mejorado las ultimas 3
#  iteraciones ademas de haber recorrido %66 de las maxima iteraciones.
#  Cada max_eval/k guarda las mejores k soluciones.
#  Para realizar la busqueda utiliza la permutacion creada por el constructor de
#  de GRASP, y busca sobre esa base los vecinos de este, en caso mejorar
#  se guarda como mejor resultado de la iteraccion.
#
#  @param ruta del fichero
#  @param permutacion (no es necesario en este caso)
#  @param maximo de evaluaciones
#  @param cada cuanto guardar
#  @return mejor permutacion,los mejores evaluaciones (npob/ngen)
#
################################################################################


def QAPAdvLocalSearch(fname,perm,max_eval,k):
    global qap
    global nperm
    global numeroEvaluaciones
    global perms
    global valorNoAperecidos
    global matCostes
    valorNoAperecidos= []
    tic = time.clock()
    exists = False
    leerMatrices(fname)
    best_sols = []
    tnp_best_sols=[]
    stats = []
    perms = []
    numeroEvaluaciones = 0
    i = 0
    repeated = 0
    notImproved = 0
    matCostes = np.zeros((nperm,nperm))
    matCostes = getCostes()

    while i < max_eval and (notImproved<3 or i<(int(max_eval*0.667))):
        perm = GraspConstructor()
        i+=1
        sol = QAPLocalSearch(perm,max_eval)
        tnp_best_sols.append(sol)
        if i%(max_eval/k) == 0:
            tnp_best_sols.sort(key=lambda x: x[0], reverse=False)
            if len(best_sols)>0 and (best_sols[0][0]< tnp_best_sols[0][0]):
                notImproved+=1
            else:
                notImproved=0
            tnp_best_sols += best_sols
            tnp_best_sols.sort(key=lambda x: x[0], reverse=False)
            media = np.median([x[0] for x in tnp_best_sols])
            varianza = np.nanvar([x[0] for x in tnp_best_sols])
            stats.append([tnp_best_sols[-1][0],tnp_best_sols[0][0],media,varianza,numeroEvaluaciones])
            best_sols.append(tnp_best_sols[0])
            tnp_best_sols =[]

    best_eval = [s[0] for s in best_sols ]
    return best_sols[0][1],best_eval

##############################################################
#
#  Llamada normal a la funcion QAPAdvLocalSearch
#    python3 QAPAdvLocalSearch.py
#  Llamada para recorge datos
#    python3 QAPAdvLocalSearch.py test
#
##############################################################

if __name__ == "__main__":
        if len(sys.argv)>1 and sys.argv[1]=='test':
            print("Analizando algoritmos para obtencion de datos")
            files = ["Cebe.qap.n10.1","Cebe.qap.n10.2","Cebe.qap.n20.1","Cebe.qap.n20.2",
            "Cebe.qap.n30.1","Cebe.qap.n30.2","Cebe.qap.n40.1","Cebe.qap.n40.2","Cebe.qap.n50.1","Cebe.qap.n50.2"]
            for f in files:
                stats = []
                fits = []
                times = []
                mins = []
                maxs = []
                for i in range(10):
                    tic = time.clock()
                    sol,values = QAPAdvLocalSearch("../Instances/QAP/"+f,[],200,20)
                    toc = time.clock()
                    times.append(toc-tic)
                    fits+=values
                    mins.append(values[-1])
                    maxs.append(values[0])
                    print("Iteracion ",i," de ",f," parametros ",200,20)
                stats.append([min(mins),mean(mins),mean(maxs),mean(fits),var(fits),mean(times)])
                savetxt("results/lastQapLS"+f+".200.20",stats)
        else:
            tic = time.clock()
            sols,evals = QAPAdvLocalSearch("../Instances/QAP/Cebe.qap.n30.1",[],200,20)
            toc = time.clock()
            print(sols,evals,toc-tic)
