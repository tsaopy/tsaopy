Estamos desarrollando una web con información en: https://tsaopy.github.io/.

## Archivos

### backend
Este script contiene los subalgoritmos principales.

### notebook
Este archivo tiene un ejemplo mínimo de funcionamiento donde se llaman a los subalgoritmos de backend para resolver un problema modelo. 

### integrator_f2py
Este archivo contiene el código Fortran, al ser ejecutado compila dicho código y produce un paquete de Python con funciones que usa el backend en la simulación.

### maketestdata
Este archivo contiene un script que produce una serie temporal de prueba con la EDO definida por el usuario. 
