import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1-sig)



a = sigmoid(0.5)

b = sigmoidDerivada(a)



entradas = np.array([[0,0], [0,1], [1,0], [1,1]])

saidas = np.array([0], [1], [1], [0])



#Pesos Iniciais
#pesos0 = np.array([[-0.424, -0.740, -0.961], [0.358, -0.577, -0.469]])
#pesos1 = np.array([[-0.017],[-0.893],[0.148]])

pesos0 = 2*np.random.random((2,3)) - 1 
pesos1 = 2*np.random.random((3,1)) - 1 


epocas = 10
taxaAprendizagem = 0.5
momento = 1

#Execucao da Rede Neural
for j in range(epocas):
    
    #Entrada
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)

    #Camada de Ativação
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)

    erroCamadaSaida = saidas - erroCamadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))


    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida

    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

    #Finalizando o ajuste dos pesos
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaSaida)
    pesos0 = (pesos0 * momento) + (pesosNovo0 + taxaAprendizagem)
