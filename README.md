Importante: es necesario correr los scripts desde una computadora con Linux.

Es necesario que haya un compilador de fortran instalado, por ejemplo correr `sudo apt-get install gfortran`. También son necesarios los paquetes de Python `emcee`, `multiprocessing` y `corner`.

## Archivos

### backend
Este script contiene los subalgoritmos principales.

### notebook
Este archivo tiene un ejemplo mínimo de funcionamiento donde se llaman a los subalgoritmos de backend para resolver un problema modelo. 

### integrator_f2py
Este archivo contiene el código Fortran, al ser ejecutado compila dicho código y produce un paquete de Python con funciones que usa el backend en la simulación.

### maketestdata
Este archivo contiene un script que produce una serie temporal de prueba con los parámetros definidos por el usuario. 
