# C2S-Review-Generator
Context-aware natural language generation for amazon movie reviews

## Sobre el uso
Para entrenar se utilizan los archivos 
implementacion_train.py y
modificacion_train.py

para predecir se usan
implementacion_generate.py y
modificacion_generate.py

Para correr los archivos
Instalar pipenv (ambiente virtual)

    pip install pipenv

Para instalar dependencias

    pipenv install
    
Para correr los archivos

    pipenv run python archivo.py

Si por alguna razon llegase a salir error por no encontrar "reviews_Movies_and_TV_5.json" (QUE NO DEBERÍA), se puede descargar desde
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV.json.gz

Se descomprime y se agrega a la carpeta "resources". No está subido aquí porque pesa mucho.

para correr los archivos:


## Sobre el código:

La gran parte de las modificaciones ocurre en la función get_contexts_and_reviews() pues ahí es donde itero sobre los reviews del JSON,
decido cual se queda, cual no, si filtro stopwords o no, y ahí también parto el dataset en train, valid, test.

El modelo está en la función create_C2S_model

Todos los archivos pickle de resources son para ahorrar tiempo en el tratamiento de los datos. Nada se procesa dos veces si ya ha sido
procesado antes.

En la carpeta pesos se encuentran algunos pesos de los entrenamientos de implementación que menciono en el reporte.

pesos_implementacion.hdf5 obtiene el accuracy y loss que reporto de 20 y 6 aprox.
pesos_modificacion.hdf5 obtiene el accuracy y loss que reporto de 50 y 3 aprox.
