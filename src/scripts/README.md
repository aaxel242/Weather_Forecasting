# Explicación Carpeta Scripts

Aquí van los programas ejecutables o tareas específicas que se corren directamente.

- script que entrena un modelo. (train_model.py)
- script que procesa un dataset concreto. (data_cleaning.py)
- script para levantar un servidor. (run_server.sh)
- script para automatizar despliegues. (deploy.py)

- Características:
- Se ejecutan de principio a fin (no se importan como librerías).
- Suelen tener un if __name__ == "__main__": en Python.

Ayuda:

- Piensa en ellos como recetas completas que usan ingredientes de otras partes del proyecto.
