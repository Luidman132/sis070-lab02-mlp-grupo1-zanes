# SIS070 - Guia de laboratorio 04: MLP y backpropagation

**Estudiante:** Luidman Zanes  
**Grupo:** 1  
**Repositorio publico:** [sis070-lab02-mlp-grupo1-zanes](https://github.com/Luidman132/sis070-lab02-mlp-grupo1-zanes)

Implementacion de un perceptron multicapa (MLP) con retropropagacion manual en NumPy. El conjunto XOR permite comprobar la propagacion hacia adelante, el calculo de gradientes y la actualizacion de pesos y sesgos.

## Estructura

```text
sis070-lab02-mlp-grupo1-zanes/
|-- src/
|   |-- __init__.py
|   `-- mlp_implementation.py
|-- README.md
|-- requirements.txt
`-- .gitignore
```

El nombre sigue la nomenclatura solicitada en la guia (`sis070-lab02-mlp-[grupo]-[apellido]`), usando el grupo 1 y el apellido Zanes.

## Requisitos y ejecucion

- Python 3.9 o superior.
- NumPy, instalado con `pip install -r requirements.txt`.

Desde la carpeta raiz del proyecto, ejecutar el ejemplo base:

```bash
python3 src/mlp_implementation.py
```

Para ejecutar tambien los experimentos de tasa de aprendizaje, activacion y arquitectura:

```bash
python3 src/mlp_implementation.py --experiments
```

La implementacion no usa frameworks de redes neuronales ni diferenciacion automatica; NumPy se utiliza para operaciones matriciales.

## Como funciona

La clase `SimpleMLP` admite una o varias capas ocultas. Para cada capa oculta calcula `Z = A @ W + b` y aplica ReLU o Sigmoide. La capa de salida es lineal, como en el ejemplo de la guia. El entrenamiento calcula la perdida MSE sobre todo el conjunto y aplica la regla de la cadena hacia atras para actualizar cada `W` y `b` por descenso de gradiente.

La derivada de la perdida implementada es `dMSE/dY = 2 * (Y_pred - Y) / N`. La derivada de ReLU es 1 para entradas positivas y 0 para entradas no positivas; la de Sigmoide se calcula como `sigmoid(z) * (1 - sigmoid(z))`.

Para reproducir la escala de pesos `0.01` del fragmento de la guia, se puede construir la clase con `initialization="guide"`. La inicializacion predeterminada usa He para capas ocultas ReLU y Xavier para Sigmoide y la salida lineal.

## Actividades y resultados observados

Los resultados siguientes se obtuvieron ejecutando `python3 src/mlp_implementation.py --experiments` con semilla 42. Para comparar convergencia se considero estabilizada una ejecucion cuando mantiene MSE <= 0.001 y 100% de acierto durante 20 epocas completas consecutivas. Las pruebas de tasa de aprendizaje se limitaron a 1000 epocas.

### 1. Tasa de aprendizaje

| Tasa | MSE inicial | MSE pico | MSE final | Epoca de estabilizacion | Acierto |
|---:|---:|---:|---:|---:|---:|
| 0.9 | 2.0042091 | 22.118697 | 0.25 | No alcanzo el criterio en 1000 | 50% |
| 0.1 | 2.0042091 | 2.0042091 | 0.0004642985 | 395 | 100% |
| 0.0001 | 2.0042091 | 2.0042091 | 0.56284829 | No alcanzo el criterio en 1000 | 50% |

Con `0.9`, la perdida subio con fuerza al inicio y termino en 0.25, sin separar las cuatro combinaciones de XOR. Con `0.0001`, la perdida bajo lentamente y no llego al criterio dentro de las 1000 epocas. En esta ejecucion, `0.1` encontro una solucion estable en 395 epocas.

### 2. ReLU frente a Sigmoide

| Activacion oculta | Inicializacion | Tasa | Epocas hasta estabilizar | MSE final | Acierto |
|---|---|---:|---:|---:|---:|
| ReLU | He | 0.1 | 395 | 0.0004642985 | 100% |
| Sigmoide | Xavier | 0.1 | 2996 | 0.00087810198 | 100% |

Con el criterio y las semillas indicados, Sigmoide necesito 2601 epocas adicionales (7.58 veces el numero de epocas de ReLU). Se usaron inicializaciones escaladas apropiadas para cada activacion; por ello, esta comparacion no atribuye a la activacion el problema de arrancar todos los pesos con magnitud muy pequena.

### 3. Segunda capa oculta

| Capas ocultas | Arquitectura | Epocas hasta estabilizar | MSE final | Acierto |
|---|---|---:|---:|---:|
| Una: 4 neuronas | 2 -> 4 -> 1 | 395 | 0.0004642985 | 100% |
| Dos: 4 y 4 neuronas | 2 -> 4 -> 4 -> 1 | 214 | 0.00018581857 | 100% |

La segunda capa se implementa pasando `(4, 4)` a `hidden_size`. En esta ejecucion de XOR estabilizo antes; es un resultado de este conjunto pequeno, semilla y configuracion, no una garantia de que agregar capas acelere otros problemas.

## Observacion sobre el fragmento inicial

El diagnostico de `--experiments` tambien ejecuta el ejemplo con la inicializacion de escala `0.01`, sesgos cero y semilla 42. En esta implementacion obtuvo MSE final `0.16666667` y 75% de acierto despues de 1000 epocas; por tanto, no resolvio XOR. Se mantiene esa opcion para poder reproducir la escala del material, pero el ejemplo funcional y los experimentos usan He/Xavier para evitar que una inicializacion debil impida aprender.

El fragmento de la guia define MSE con el promedio del error cuadratico, pero en el retroceso usa `(output - y) / m`, sin el factor 2 de la derivada exacta de MSE. El codigo usa `2 * (output - y) / N`; ese factor cambia la magnitud del paso (equivale a reducir la tasa a la mitad si se omite), no su direccion.

## Entrega en GitHub

La guia solicita publicar un repositorio publico y entregar su enlace en el aula virtual. El repositorio de este trabajo es [https://github.com/Luidman132/sis070-lab02-mlp-grupo1-zanes](https://github.com/Luidman132/sis070-lab02-mlp-grupo1-zanes); copia ese enlace en el espacio de entrega del curso.
