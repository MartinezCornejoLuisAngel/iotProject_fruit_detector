import numpy as np #Importa la biblioteca numpy para operaciones numéricas eficientes
import cv2 #Importa la biblioteca OpenCV para procesamiento de imágenes y visión por computadora.
import os # Importa la biblioteca os para interactuar con el sistema operativo y acceder a archivos y directorios.
import imutils #Importa la biblioteca imutils, que proporciona una serie de funciones de utilidad para trabajar con imágenes.
import psycopg2
import schedule
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db 
import threading
from datetime import datetime


#cARGO EL CERTIFICADO DEL PROYECTO DE FIREBASE
firebase_sdk = credentials.Certificate('.json')
#Haecemos referencias a la base de datos en tiempo real de firebase
firebase_admin.initialize_app(firebase_sdk,{'databaseURL':'url'})



NMS_THRESHOLD=0.3 # Establece el umbral de supresión no máxima para eliminar detecciones superpuestas débiles.
MIN_CONFIDENCE=0.2 # Establece la confianza mínima requerida para considerar una detección como válida.
# Define una función llamada pedestrian_detection que toma una 
#imagen, un modelo de detección, el nombre de una capa y un 
#identificador de clase (por defecto, 0 para "persona"). 
#Esta función se utiliza para detectar peatones en una imagen 
#dada utilizando el modelo proporcionado.
long_banana = 0
long_apple = 0
long_carrots = 0
long_broccoli = 0
long_orange = 0

def update_firebase(ora_num,broc_num,app_num,ban_num,car_num):
	current_datetime = datetime.now()
	formatted_string = current_datetime.strftime("%Y-%m-%d")
	ref = db.reference('/productos')

	if formatted_string in ref.get():	
    # El producto existe, actualizar los datos
		producto_ref = ref.child("-NiDdwNVR2yeXPpcLEdQ")
		producto_ref.update({'orange':long_orange,'broccoli':long_broccoli,'bananas':ban_num,'apples':app_num,'carrots':car_num})
		
	else:
		ref.child(formatted_string).set({'orange':long_orange,'broccoli':long_broccoli,'bananas':ban_num,'apples':app_num,'carrots':car_num})



def pedestrian_detection(image, model, layer_name, personidz=0):
	(H, W) = image.shape[:2] #Obtiene el alto (H) y ancho (W) de la imagen.
	results = [] #Inicializa una lista vacía para almacenar los resultados de la detección.


	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False) #Preprocesa la imagen para que sea compatible con el modelo de detección. Convierte la imagen a un "blob" y realiza una serie de transformaciones, como la normalización y el redimensionamiento.


	model.setInput(blob) #Establece la entrada del modelo con el blob de la imagen preprocesada.
	layerOutputs = model.forward(layer_name) #Realiza una inferencia hacia adelante en el modelo y obtiene las salidas de la capa especificada.
	#Inicializa listas vacías para almacenar las coordenadas de las cajas delimitadoras, los centroides y las confianzas de las detecciones.
	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personidz and confidence > MIN_CONFIDENCE:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	#Aplica la supresión no máxima a las cajas delimitadoras 
	#para eliminar las detecciones superpuestas débiles. 
	#Devuelve los índices de las cajas delimitadoras seleccionadas.
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
	# ensure at least one detection exists
	if len(idzs) > 0:
		# loop over the indexes we are keeping
		for i in idzs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			res = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(res)
	# return the list of results
	return results


def main():	
	labelsPath = "coco.names" #Especifica la ruta del archivo que contiene las etiquetas de las clases.
	LABELS = open(labelsPath).read().strip().split("\n") #Lee el archivo de etiquetas y crea una lista de etiquetas de clases, eliminando espacios en blanco y separando las líneas

	#Especifica las rutas de los archivos de pesos y configuración del modelo YOLOv4-Tiny.
	weights_path = "yolov4-tiny.weights"
	config_path = "yolov4-tiny.cfg"

	# Carga el modelo YOLOv4-Tiny utilizando los archivos de pesos y configuración.
	model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
	'''
	model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	'''

	layer_name = model.getLayerNames() #Obtiene los nombres de todas las capas del modelo.
	layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()] #Obtiene los nombres de las capas de salida no conectadas y los almacena en la lista layer_name.


	cap = cv2.VideoCapture(1) # Abre la transmisión de video desde la cámara web (0 indica el índice de la cámara).
	writer = None #Inicializa una variable para escribir el video de salida (no se está utilizando actualmente).

	while True:
		(grabbed, image) = cap.read()

		if not grabbed:
			break
		image = imutils.resize(image, width=700) #Redimensiona la imagen para que tenga un ancho de 700 píxeles utilizando la función resize de la biblioteca imutils.
		# Llama a la función pedestrian_detection para detectar peatones en la imagen actual. Los resultados se almacenan en la variable results.
		results = pedestrian_detection(image, model, layer_name,
			personidz=LABELS.index("banana"))

		results2 = pedestrian_detection(image, model, layer_name,
			personidz=LABELS.index("apple"))
		
		results3 = pedestrian_detection(image, model, layer_name,
			personidz=LABELS.index("carrot"))

		results4 = pedestrian_detection(image, model, layer_name,
			personidz=LABELS.index("orange"))
		
		results5 = pedestrian_detection(image, model, layer_name,
			personidz=LABELS.index("broccoli"))
		

		long_banana = len(results)
		print(f'Bananas: {long_banana}')
		long_apple = len(results2)
		print(f'Apples: {long_apple}')
		long_carrots = len(results3)
		print(f'Carrots: {long_carrots}')
		long_orange = len(results4)
		print(f'Orange: {long_orange}')
		long_broccoli = len(results5)
		print(f'Broccoli: {long_broccoli}')

		schedule.every(10).seconds.do(lambda: update_firebase(long_orange,long_broccoli, long_apple,long_banana,long_carrots))

		for res in results:
			cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)
		
		for res in results2:
			cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 0, 255), 2)
		
		for res in results3:
			cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (255, 0, 0), 2)

		cv2.imshow("Detection",image) # Muestra la imagen con las detecciones en una ventana titulada "Detection".

		key = cv2.waitKey(1) #Espera la pulsación de una tecla durante 1 milisegundo.
		#Si la tecla presionada es 'Esc' (código ASCII 27), se rompe el bucle y se sale del programa.
		if key == 27: 
			break

	cap.release() # Libera la transmisión de
	cv2.destroyAllWindows() # Cierra todas las ventanas abiertas por OpenCV.



thread_bucle_principal = threading.Thread(target=main)
thread_bucle_principal.daemon = True
thread_bucle_principal.start()


while True:
    schedule.run_pending()
    time.sleep(1)