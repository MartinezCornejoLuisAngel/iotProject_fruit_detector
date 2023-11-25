import firebase_admin
from datetime import datetime
from firebase_admin import credentials,firestore,storage
from firebase_admin import db 
import pandas as pd

import matplotlib.pyplot as plt

#cARGO EL CERTIFICADO DEL PROYECTO DE FIREBASE
firebase_sdk = credentials.Certificate('.json')
#Haecemos referencias a la base de datos en tiempo real de firebase
firebase_admin.initialize_app(firebase_sdk,{'databaseURL':'url'})
current_datetime = datetime.now()
formatted_string = current_datetime.strftime("%Y-%m-%d")
#formatted_string = "2023-11-17"

ref = db.reference('/productos/')
datos = ref.get()

# Imprimir los datos
print("Datos de la Realtime Database:", datos)

# Convertir los datos a un DataFrame
df = pd.DataFrame(datos).transpose()

# Convertir las fechas a formato de fecha
#df.index = pd.to_datetime(df.index)
# Visualizar el DataFrame
print("DataFrame:")
print(df)
# Graficar los datos
df.plot(kind='bar', stacked=True)
plt.title('Datos de la Realtime Database')
plt.xlabel('Fecha')
plt.ylabel('Cantidad')
plt.show()


# Guardar la imagen localmente
imagen_local = 'grafica.png'
plt.savefig(imagen_local)


dbf = firestore.client()

ruta_imagen_local = "./grafica.png"

with open(ruta_imagen_local, "rb") as image_file:
    # Lee los bytes de la imagen
    bytes_imagen = image_file.read()

datos_a_guardar = {
    formatted_string: bytes_imagen,
    # Agrega más campos según sea necesario
}

# Agrega los datos a una colección llamada "nombre_coleccion"
nombre_coleccion = "graficas"
#dbf.collection(nombre_coleccion).add(datos_a_guardar)

print("Datos guardados exitosamente en Firestore.")




"""if formatted_string in ref.get():
    # El producto existe, actualizar los datos
    producto_ref = ref.child(formatted_string)
    producto_ref.update({'bananas':3,'apples':1,'carrots':9})
    print("sasan")
else:
    # El producto no existe, crear un nuevo producto
    ref.child(formatted_string).set({'bananas': 0, 'apples': 0, 'carrots': 0})"""


#crea una coleccion con el nombre de productos con varios frutas
#ref.push({'bananas':'0','apples':'0','carrots':'0'})

#modificar datos de un producto
#ref = db.reference('productos')
#producto_ref = ref.child()
#producto_ref.update({'bananas':3,'apples':'1','carrots':'9'})


#agregar un producto mas 
#ref =  db.reference('productos')
#producto = {'tipo':'sa','modelo','sa'}
#product_ref  = ref.push(producto)

#modificar varios datos diferentes

#ref = db.reference('productos')
#ref.update({
#    '-NiDdwNVR2yeXPpcLEdQ/banana':'3',
#    '-NdfDdwNVR2yeXPpcLdsQ/apple':'2'    
#})S