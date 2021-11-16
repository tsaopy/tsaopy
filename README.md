<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

# Repositorio adscripción

Importante: es necesario correr los scripts desde una computadora con Linux; los paquetes `f2py`, `emcee` y `multiprocessing` no funcionan bien en Windows. 

## Archivos

### main
Es el script principal para correr la cadena MCMC y producir gráficas con los resultados. Se cargan los datos a partir de un archivo definido por el usuario. Parámetros de la simulación, información previa sobre los coeficientes a ajustar, parámetros del modelo, etc se cargan a mano por el usuario. 



### integrator_f2py
Este archivo contiene el código Fortran, al ser ejecutado compila dicho código y produce un paquete de Python con las funciones que usa el main en la simulación.

### maketestdata
Este script producen una serie temporal de prueba con los parámetros definidos por el usuario. 

### process_expdata
Este script procesa datos experimentales aplicando filtros para reducir el ruido, convertir formatos, etc, de forma que sean utilizables por el main. Está diseñado para trabajar los datos en crudo de las mediciones del grupo de Horacio. 
