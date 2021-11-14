# Repositorio adscripcion

Cosas para agregar:
1) Crear dos versiones más del main y del integrador Fortran, una agregando la driving force, y otra agregando términos cruzados de x y v.
2) Agregar una carpeta con distintos modelos para testear? Ahora mismo está el de Duffing.

Cosas para mejorar/corregir/revisar:
1) Revisar el método para estimar x0.
2) Revisar el método para hacer el fitteo inicial.
3) Por algún motivo las funciones para Python que genera f2py no respetan el orden de ingreso de las variables del código de Fortran. Puede ser un problema de la libería f2py en sí, habría que revisar la documentación con máś detalle.
4) Revisar la función ''. 
