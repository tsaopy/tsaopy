# Repositorio adscripción

Importante: es necesario correr los scripts desde una computadora con Linux; los paquetes `f2py`, `emcee` y `multiprocessing` no funcionan bien en Windows. 

## Archivos

### backend
Este script contiene los subalgoritmos principales.

### notebook
Este archivo tiene un ejemplo mínimo de funcionamiento donde se llaman a los subalgoritmos de backend para resolver un problema modelo. 

### integrator_f2py
Este archivo contiene el código Fortran, al ser ejecutado compila dicho código y produce un paquete de Python con las funciones que usa el main en la simulación.

### maketestdata
Este archivo contiene un script que produce una serie temporal de prueba con los parámetros definidos por el usuario. 
