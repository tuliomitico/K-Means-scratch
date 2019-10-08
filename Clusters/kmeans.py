import random as rd
import numpy as np
import pandas as pd
from math import sqrt
from matplotlib import pyplot as plt
from collections import defaultdict

class KMedias(object):
    def __init__(self, X, n_grupos=3, nr_init_centroides=1,itera_max=500):
        """
        Atributos que sao utilizados durante toda a classe KMedias.
        
        Parametros:
        -----------
        
        X: DataFrame,NumPy Matrix;
        Dados que serao procurados grupos de pontos.
        
        n_grupos:int;
        Numero de grupos que serao separados.
        
        n_insta:int;
        Numero de amostras obtidas.
        
        n_ctrts:int;
        Numero de features(caracteristicas).
        
        valor_inertia:list;
        Para armazenar o ultimo elemento da iteracao atual.
        
        itera_max:int;
        Numero de vezes que os centros de grupos serao movidos.
        
        toda_itera_cjdados:list;
        Todos os clusters e conjuntos de pontos.
        
        """
        self.X = X
        self.n_grupos = n_grupos
        self.n_insta,self.n_ctrts = self.X.shape
        self.nr_init_centroides = nr_init_centroides
        self.valor_inertia =[]
        self.itera_max = itera_max
        self.toda_itera_cjdados = []
    
    def checa_dados(self):
        """
        Verifica os dados entrados
        """
        if type(self.X) is pd.DataFrame:
            #cabecalho = list(X.columns) # era pra devolver as caracteristicas no final da computacao
            self.X = self.X.values
            return self.X
        elif type(self.X) is np.ndarray:
            pass
        else:
            raise ValueError("Tipo de dado inserido não é válido: {}".format(type(self.X)))
            
    def checa_dados_previstos(self,X):
        """
        Verifica os dados entradas, porem advindos de fora da classe KMedias(KMeans)
        
        Parametros
        ----------
        
        X: Recebe qualquer tipo de dados, mas só aceita
        Arrays do Numpy e DataFrames.
        """
        if type(X) is pd.DataFrame:
            #cabecalho = list(X.columns) # era pra devolver as caracteristicas no final da computacao
            X = X.values
            return X
        elif type(X) is np.ndarray:
            return X
        else:
            raise ValueError("Tipo de dado inserido não é válido: {}".format(type(X)))
	
    def __logmin_aux(self,x,esq,direi):
        if esq==direi:
            minim=x[esq]
            return minim
        if direi==esq+1:
            if x[esq]<x[direi]:
                minim=x[esq]
            else:
                minim=x[direi]
            return minim
        meio = (esq+direi)//2
        min1 = self.__logmin_aux(x,esq,meio)
        min2 = self.__logmin_aux(x,meio+1,direi)
        if min1 < min2:
            minim = min1
        else:
            minim = min2
        return minim
	
    def logmin(self,x):
        return self.__logmin_aux(x,0,len(x)-1)

    def __index(self,ent,x,comeco):
        """
        Funcao recursiva que encontra o indice de um dado valor.
        
        Parametros
        ----------
        ent : array ou lista
        A estrutura a ser percorrida.
        
        x: int,float,string
        Valor a ser procurado o indice.
        
        comeco: int
        Ponto de partida a ser varrido a busca.
        
        Retorno
        -------
        
        menorIndex: list
        Retorna uma lista de inteiros dos indices encontrados.
        """
        if comeco == len(ent):
            return []
            
        menorIndex = self.__index(ent,x,comeco + 1) 
  
    
        if (ent[comeco] == x): 
            aux = [0 for i in range(len(menorIndex) + 1)] 
            aux[0] = comeco

            for i in range(len(menorIndex)): 

                aux[i + 1] = menorIndex[i] 
  
            return aux 
        else: 
            return menorIndex
    
    def all_index(self,ent,x):
        """
        Funcao auxiliar que simpllifica a chamada de index.
        
        Parametros
        ----------
        
        ent: array ou lista
        A estrutura a ser percorrida
        
        x: int,string,float,double;
        Elemento a ser procurado
        """
        saida = self.__index(ent,x,0)
        for i in saida:
            return i
        
    @staticmethod
    def dist_euclidiana(n_c,n_X):
        """
        Funcao que faz o calculo da distancia euclidiana de dois pontos cartesianos.
        
        Parametros
        -----------
        n_c: tipo narray,list; 
        Centroides
        
        n_X: tipo narray,list; 
        Conjunto de dados inseridos
        
        ---------------
        Retorna: float; Retorna a distancia computada
        
        PS: A ordem posta não altera os produtos, pois é praticamente uma equacao modular(f(x)=|x|).
        """
        dim, soma = len(n_c), 0

        for i in range(dim):
            soma += pow(n_c[i] - n_X[i], 2)
        
    
        return np.sqrt(soma)
        
    
    def criar_centroides_aleatorios(self):
        """
        Funcao que cria os centros de massa aleatoriamente com base no conjunto de dados fornecidas
        sendo geradas pelos seus indices.
        """
        index_conj_alea = rd.sample(range(0,self.n_insta),self.n_grupos)
        centroide_inic_alea = self.X[index_conj_alea]
        return centroide_inic_alea
        
    def atrib_pontos_conj_para_centroides_prox(self,centroides):
        """
        Recebe qlqr valor de centroides(No caso, valores aleatorios), atribui cada ponto ao centroides recebido
        baseado na dist_euclidiana. Usa da estrutura de dados do Python defaultdict para rastrear cada centroide e
        pontos associados em uma unica iteracao
        
        Parametros
        ----------
        centroides: list,array;
        Array do Numpy com os valores dos centroides aleatorios
        
        grupo_de_iteracao_unica: defaultdict;
        Dicionario que armazena os centroides como chave e os pontos associados como valores e atualizado
        a cada iteracao
        
        lista_dist_euclides: list;
        Recebe as distancias calculada em dist_euclides e armazena
        
        dist_euclides: double,float;
        Devolva a distancia euclidiana de dois pontos
        
        index_prox_centroide: array;
        Devolve os menores valores de um index
        
            
        prox_centroide
        Transforme o centroide em uma tupla (pq é imutavel) para ser a chave 
        do dicionario.
        
        Retorno
        -------
		
		Os centroide e os valores associados a cada cluster que o centroide pertence em uma unica iteracao
		do KMedias.
        """
        grupo_de_iteracao_unica = defaultdict(list)
        for pontos_conj in self.X:
            lista_dist_euclides=[]
            for centroide in centroides:
                dist_euclides = self.dist_euclidiana(centroide,pontos_conj)
                lista_dist_euclides.append(dist_euclides)
            index_prox_centroide=np.argmin(lista_dist_euclides)
            prox_centroide = tuple(centroides[index_prox_centroide])
            grupo_de_iteracao_unica[prox_centroide].append(pontos_conj)
        return grupo_de_iteracao_unica
        
    def exec_kmedias_init_centroide(self,nr_inic):
        """
        Atribui os dados do conjunto ao centroide mais proximo, realoca os clusters 
        baseado nas medias dos clusters atuais. Repete até os centroides pararem de mudar
        ou atingir a itera_max.
        
        Retorno
        -------
        P**** nenhuma, nada.
        """
        centroides=self.criar_centroides_aleatorios()
        self.toda_itera_cjdados.append([])
        for iteracao in range(1,self.itera_max + 1):
            print("Comecando iteracao de numero {}...".format(iteracao))
            grupo_de_iteracao_unica = self.atrib_pontos_conj_para_centroides_prox(centroides=centroides)
            self.toda_itera_cjdados[nr_inic].append(grupo_de_iteracao_unica)
            centroides_atual=[]
            for centroide in grupo_de_iteracao_unica:
                grupo_cj_dados = grupo_de_iteracao_unica[centroide]
                centroide_atual = np.mean(grupo_cj_dados,axis=0)
                centroides_atual.append(centroide_atual)
            
            
            if self.dist_euclidiana(np.array(centroides_atual),centroides).all() == 0:
                break
            centroides = centroides_atual
        return None
        
    def encaixe(self):
        """
        Funcao que atrela todas as funcoes a fim de facilitar a entrada de atributos,
        recebe o numero de vezes que troca os centroides iniciais, a cada iteracao
        renova.
        
        Retorno
        -------
        None,nada,nothing.
        """
        self.checa_dados()
        for nr_inic in range(self.nr_init_centroides):
            self.exec_kmedias_init_centroide(nr_inic=nr_inic)
            inertia_ult_cluster = self.inertia(self.toda_itera_cjdados[nr_inic][-1])
            self.valor_inertia.append(inertia_ult_cluster)
        return None
                
    def inertia(self,grupos):
        """
        Para os items dos dicionarios e os eleva ao quadrado.
        
        Parametros
        ----------
        soma_dos_quad: float,int;
        
        retorna o quadrado da dist_euclidiana
        """
        soma_dos_quad = 0
        for centroide,pontos_grupos in grupos.items():
            for ponto_grupos in pontos_grupos:
                euclides_norm = self.dist_euclidiana(ponto_grupos,centroide)
                euclides_quad = pow(euclides_norm,2)
                soma_dos_quad+=euclides_quad
        return soma_dos_quad
    
    def menor_index_cluster(self):
        """
        Na lista de inertia devolve o menor e valor e o indice agregado.
        
        Retorno
        -------
        menor_index: int;
        O menor indice menor_index
        """
        minima = self.logmin(self.valor_inertia)
        menor_index = self.valor_inertia.index(minima)
        return menor_index
    
    def itera_final_otimazacao(self):
        """
        Pega o menor_index_cluster
        
        Retorno
        -------
        Dicionario contendo os centros como chaves e os valores dos conjuntos
        de dados como item ou valores.
        """
        return self.toda_itera_cjdados[self.menor_index_cluster()][-1]
    
    def centros_clusters(self):
        """
        Funcao que devolve os valores finais dos centros dos grupos(clusters).
        
        Parametros
        ----------
        
        centros: list;
        Re-atribui os valores da funcao itera_final_otimazacao que estao em um defaultdict
        para um array do NumPy.
        
        """
        centros = [centro for centro in list(self.itera_final_otimazacao().keys())]
        return np.array(centros)
        
    def previsto(self,X):
        """
        Funcao que dado um valor realiza a checagem em dado conjunto que fora "encaixado"(fitted)
        e que cluster ele estaria posicionado.
        
        Parametros
        ----------
        
        prox_centroide: list;
        Lista vazia para adicao de valores que facam parte de um grupo "encaixado"(fitted).
        
        distancia: list;
        Faz a calculo da distancia euclidiana dos valores inseridos com os centroides e ve qual
        mais se aproximam de algum grupo.
        
        classificacao: int;
        Devolve o menor index que os valores se encaixaram.
        
        """
        X = self.checa_dados_previstos(X)
        prox_centroide = []
        for i in range(X.shape[0]):
            distancia = [self.dist_euclidiana(centroide, X[i]) for centroide in self.centros_clusters()]
            classificacao = self.all_index(distancia,min(distancia))
            prox_centroide.append(classificacao)
        return np.array(prox_centroide)
