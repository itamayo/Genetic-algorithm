#!/usr/bin/env python3
import sys
import numpy as np
import copy
import random
import time
from numpy import *

"""
   Este algoritmo es una implementacion basada en GRASP (Greedy randomized adaptive search procedure),
   en concreto se basa en la cardinalidad. El objetivo del algortimo es obtener el resultado mas optimo
   en un tiempo relativamente admisible (3') en la problematica de la biparticion de un grafo dado sus pesos.
   El objetivo no es, obtener el mejor conocido (si se obtiene mucho mejor), esta mas orientado  a desenpeñar
   un funcion que pueda a ir evolucionando hacia el optimo global, y solo dependa los recursos que se quieran aplicar.

"""

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

####################################################################################################
#  Funcion que busca el optimo local de una vencidad maximizando el fvalor de BipEvaluator
#  @param: vector , vector binario que define la distribucion de vertice en dos subgrafos
#  @return mejor vector encontado y fvalor del mejor vector encontrados
####################################################################################################
def BipLocalSearch(vector):
    global pesos
    global max_eval
    sol = 1
    sol_result = 0
    vecinSol = 0
    num_it = 0
    notImproved = 0
    ## Iterar hasta que no exista un vector (sol_result) o supere la maximas iteraciones
    while (sol > sol_result and num_it<max_eval):
         ## Realizar copia del vector para poder operar sin problemas de referencias de memoria
         sol_vect = copy(vector)
         ## Dividir el vector entre diferentes ya que solo interesa cambiar de lugar los diferentes
         grafo1 =  np.where(sol_vect == 0)[0]
         grafo2 =  np.where(sol_vect == 1)[0]
         ## Evaluar el fvalor del actual vector
         vecinSol = sol_result = BipEvaluator(sol_vect)
         num_it+=1
         ## Crear la vecindad del vector selecionado, y buscar si alguno mejora el actual
         for i in range(len(grafo1)):
            for j in range(len(grafo2)):
                tnp_vector = copy(sol_vect)
                tnp_vector[grafo1[i]] = sol_vect[grafo2[j]]
                tnp_vector[grafo2[j]] = sol_vect[grafo1[i]]
                tnpsol = BipEvaluator(tnp_vector)
                num_it+=1
                if (tnpsol>vecinSol):
                    vecinSol = tnpsol
                    vector = copy(tnp_vector)
                    notImproved = 0
                else:
                   notImproved+= 1
                   if (notImproved>5):break
         sol = vecinSol
    return sol,sol_vect[0]

################################################################################
#  Funcion que calcula el Fvalor de la funcion particion de un grafo de los
#  pesos entre vertices de diferentes grafos
#
#  @param: Vector binario que define la distribucion de vertice en dos subgrafos
#  @return fvalor de la particion
################################################################################

def BipEvaluator(vector):
    balance =  np.sum(vector) # Numero de nodos en una de las partes
    factible=(balance==nvec/2)
    ## Verificamos que la particiones este compuestas por el mismo numero de nodos/vertices
    f = 0
    ## Creamos los grafo separados para iterar sobre estos
    ## Solo iteranos sobre los vertices que componen diferentes grafos
    ## Consegimos reducir el bucle de 45 del ejercicio propuesto a 25
    grafo1 =  np.where(vector == 0)[0]
    grafo2 =  np.where(vector == 1)[0]

    for i in range(len(grafo1)):
        for j in range(len(grafo2)):
            f += pesos[grafo1[i],grafo2[j]]

    return f # devuelve suma de pesos

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
      costeDeRutaSel.sort(key=lambda x: x.coste, reverse=True)
      return costeDeRutaSel

######################################################################
#  Funcion constructor de GRASP, se genera un vector
#  elegiendo con cierta aleatoriedad de una lista ordena por
#  costes del elemento seleccionado a los demas
#
#
#  @return permutacion
#
#####################################################################

def GraspConstructor():
      global listaPosibles
      sol = []
      selectedNum = random.randint(0,nvec-1)
      selectedNodo = nodo(selectedNum,100000,random.randint(0,2))
      sol.append (selectedNodo)
      listaPosibles = [i for i in range(nvec) if i!=selectedNodo.id]
      for i in range(nvec-1):
          lista = calcularCostes(sol,listaPosibles)
          selectedNodo = seleccionar(lista,sol)
          sol.append(selectedNodo)

      return sol

###################################################
# Lectura de los pesos del fichero BIPART
# @param ruta del fichero BIPART
# @return matriz de pesos
###################################################

def Read_Bipart_Instance(fname):
 global pesos
 global nvec
 hdl = open(fname, 'r')
 mylist = hdl.readlines()
 hdl.close()
 nvec = eval(mylist[0])
 pesos = np.zeros((nvec,nvec))      # Pesos de las aristas
 for i in range(nvec):
   for j,val in enumerate(mylist[i+1].split()): ## Simetria en la matriz, se evita rebundancia j=i+1
     pesos[i,j]=eval(val)
 return pesos

################################################################################
#  Funcion principal que implementa la busqueda basada en el algoritmo
#  GRASP. Esta compuesto por un bucle que solo termina si; se a sobrepasado
#  el maximo de evaluaciones posibles, o si no se han mejorado las ultimas 3
#  iteraciones ademas de haber recorrido %66 de las maxima iteraciones.
#  Cada max_eval/k guarda las mejores k soluciones.
#  Para realizar la busqueda utiliza el vector creado por el constructor de
#  de GRASP, y busca sobre esa base los vecinos de este, en caso mejorar
#  se guarda como mejor resultado de la iteraccion.
#
#  @param ruta del fichero
#  @param vector a analizar (no es necesario en este caso)
#  @param maximo de evaluaciones
#  @param cada cuanto guardar
#  @return mejor vector,los mejores evaluaciones (npob/ngen)
#
################################################################################

def BipAdvLocalSearch(fname,vector,_max_eval,k):
    global edge_weights
    global nvec
    global max_eval
    max_eval = _max_eval
    Read_Bipart_Instance(fname)
    best_sols = []
    tnp_best_sols = []
    notImproved = 0
    stats = []
    i = 0
    tic = time.clock()
    while i < max_eval and (notImproved<=3 or i<(int(max_eval*0.667))):
        vector = GraspConstructor()
        vector = [v.value for v in vector]
        sol = BipLocalSearch(vector)
        tnp_best_sols.append(sol)
        if i%(max_eval/k) == 0:
            tnp_best_sols.sort(key=lambda x: x[1], reverse=True)
            if len(best_sols)>0 and (best_sols[0][0]< tnp_best_sols[0][0]):
                notImproved+=1
            else:
                notImproved=0
            tnp_best_sols += best_sols
            tnp_best_sols.sort(key=lambda x: x[0], reverse=True)
            media = np.median([x[0] for x in tnp_best_sols])
            varianza = np.nanvar([x[0] for x in tnp_best_sols])
            stats.append([tnp_best_sols[-1][0],tnp_best_sols[0][0],media,varianza,i])
            best_sols.append(tnp_best_sols[0])
            tnp_best_sols =[]
        i+=1
    best_eval = [s[0] for s in best_sols ]
    return best_sols[0],best_eval

##############################################################
#
#  Llamada normal a la funcion QAPAdvLocalSearch
#    python3 BipAdvLocalSearch.py
#  Llamada para recorge datos
#    python3 BipAdvLocalSearch.py test
#
##############################################################

if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]=='test':
        print("Analizando algoritmos para obtencion de datos")
        files = ["Cebe.bip.n10.1","Cebe.bip.n10.2","Cebe.bip.n10.3","Cebe.bip.n10.4",
        "Cebe.bip.n20.1","Cebe.bip.n20.2","Cebe.bip.n20.3","Cebe.bip.n20.4","Cebe.bip.n50.1","Cebe.bip.n50.2"]
        for f in files:
            stats = []
            fits = []
            times = []
            mins = []
            maxs = []
            for i in range(10):
                tic = time.clock()
                sol,values = BipAdvLocalSearch("../Instances/BIPART/"+f,[],200,20)
                toc = time.clock()
                times.append(toc-tic)
                fits+=values
                mins.append(values[-1])
                maxs.append(values[0])
                print("Iteracion ",i," de ",f," parametros ",200,20)
            stats.append([max(maxs),mean(mins),mean(maxs),mean(fits),var(fits),mean(times)])
            savetxt("results/lastBIPLS"+f+".200.20",stats)
    else:
            tic = time.clock()
            sol,evals = BipAdvLocalSearch("../Instances/BIPART/Cebe.bip.n20.4",[],4200,20)
            toc = time.clock()
            print(sol,evals,toc-tic)
