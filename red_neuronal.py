from numpy import *
import random

def propagacion(num_entradas, num_neuro_cap_oculta, num_neuro_cap_salida,x,w,v):
    
    S=[]
    H=[]
    for z in range(num_neuro_cap_oculta):
        S.append(0.0)
        H.append(0.0)
    
    
    for j in range(num_neuro_cap_oculta):
        sumatoria = 0.0
        for i in range(num_entradas):
            sumatoria = sumatoria + w[i][j]*x[i]
        S[j] = sumatoria + w[i+1][j]
        H[j] = 1/(1+exp(S[j]*(-1.0)))
    
    R=[]
    y=[]
    for z in range(num_neuro_cap_salida):
        R.append(0.0)
        y.append(0.0)
        
    for k in range(num_neuro_cap_salida):
        sumatoria = 0.0
        for j in range(num_neuro_cap_oculta):
            sumatoria = sumatoria + v[j][k]*H[j]
        R[k] = sumatoria + v[j+1][k]
        y[k] = R[k] #1/(1+exp(R[k]*(-1.0)))
         
    return y,H


num_entradas = 2
num_neuro_cap_oculta = 8
num_neuro_cap_salida = 1

patrones = 4
error_permitido = 0.001

n=0.07

x = [[0.0, 0.0],
     [0.0, 1.0],
     [1.0, 0.0],
     [1.0, 1.0]]

yr =[[0.0],
     [1.0],
     [1.0],
     [0.0]]


v = []
for i in range(num_neuro_cap_oculta+1):
    v.append([])
    for j in range(num_neuro_cap_salida):
        v[i].append(random.random()*2.0 - 1.0 )

w = []
for i in range(num_entradas+1):
    w.append([])
    for j in range(num_neuro_cap_oculta):
        w[i].append(random.random()*2.0 - 1.0)
        
        
error = []
de_dv = []
for k in range(num_neuro_cap_salida):
    error.append(0.0)
    de_dv.append(0.0)
    
grad_cap_salida = []
de_dw = []
for j in range(num_neuro_cap_oculta + 1):
    de_dw.append(0.0)
    grad_cap_salida.append([])
    for k in range(num_neuro_cap_salida):
        grad_cap_salida[j].append(0.0)

grad_cap_oculta = []
for i in range(num_entradas + 1):
    grad_cap_oculta.append([])
    for j in range(num_neuro_cap_oculta):
        grad_cap_oculta[i].append(0.0)

ec = 10.0
it = 0

while ec > error_permitido and it < 10000:

    ec=0.0
    it += 1

    for patron in range(patrones):

        y,H = propagacion(num_entradas, num_neuro_cap_oculta, num_neuro_cap_salida,x[patron],w,v)

        for k in range(num_neuro_cap_salida):
            error[k] = yr[patron][k] - y[k]
            ec = ec+(error[k]**2)  
            de_dv[k] = error[k] # * y[k] *(1 - y[k])
            for j in range(num_neuro_cap_oculta):    
                grad_cap_salida[j][k] = de_dv[k] * H[j]
                v[j][k] = v[j][k] + n * grad_cap_salida[j][k]
            v[j+1][k] = v[j+1][k] + n * de_dv[k] * v[j+1][k]
            
        for i in range(num_entradas):
            for j in range(num_neuro_cap_oculta):
                sumatoria = 0
                for k in range(num_neuro_cap_salida):
                    sumatoria = sumatoria + de_dv[k] + v[j+1][k]
                de_dw[j] = sumatoria * H[j] * (1 - H[j])
                grad_cap_oculta[i][j] = de_dw[j] * x[patron][i]
                w[i][j] = w[i][j] + n*grad_cap_oculta[i][j]

        for j in range(num_neuro_cap_oculta):
            sumatoria = 0
            for k in range(num_neuro_cap_salida):
                sumatoria = sumatoria + de_dv[k] + v[j+1][k]
            de_dw[j] = sumatoria * H[j] * (1 - H[j])
            w[i+1][j] = w[i+1][j] + n * de_dw[j] * w[i+1][j]
    ec = ec *0.5
    if it%100==0:
        print('iteracion: ',it,'  error cuadratico medio: ',ec)

print('------------------------')
print('iteracion final: ',it,'  error cuadratico medio final: ',ec)
print('########################')
print('entrenamiento terminado')
print('Entradas          Salidas')
y,H = propagacion(num_entradas, num_neuro_cap_oculta, num_neuro_cap_salida,x[0],w,v)
print('[',int(x[0][0]),'     ',int(x[0][1]),']    ',y[0] )
y,H = propagacion(num_entradas, num_neuro_cap_oculta, num_neuro_cap_salida,x[1],w,v)
print('[',int(x[1][0]),'     ',int(x[1][1]),']    ',y[0] )
y,H = propagacion(num_entradas, num_neuro_cap_oculta, num_neuro_cap_salida,x[2],w,v)
print('[',int(x[2][0]),'     ',int(x[2][1]),']    ',y[0] )
y,H = propagacion(num_entradas, num_neuro_cap_oculta, num_neuro_cap_salida,x[3],w,v)
print('[',int(x[3][0]),'     ',int(x[3][1]),']    ',y[0] )


      