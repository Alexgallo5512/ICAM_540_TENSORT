
import os
import cv2
import numpy as np

import time
import threading
import os
from pathlib import Path
from ultralytics import YOLO
from CamNavi2 import CamNavi2

# ================= CONFIG =================
SAVE_PATH = "/home/icam-540/capturas"
FILE_NAME = "Imagen_trigger.jpg"
MODEL_PATH = "/home/icam-540/Proyectos/Github/Entrenamiento_Deteccion_Objeto/best.pt"
# Resolución cámara (reducida para mejor rendimiento)
WIDTH  = 640   
HEIGHT = 640   
# Tamaño YOLO (pequeño = procesamiento rápido)
YOLO_SIZE_W = 640 
YOLO_SIZE_H = 480 
YOLO_CONF = 0.3  # Confianza mínima (> 0.5 = más rápido)

MUESTRA_IMAGEN = False
detection_event = threading.Event()
# =========================================
model = YOLO(MODEL_PATH)
os.makedirs(SAVE_PATH, exist_ok=True)

image_arr = None
resized_2 = None

lista_confi= []
lista_conteo= []

def lectura_Archivo_Conteo():
    global lista_conteo
    try:
        lista_conteo.clear()
        with open("/home/icam-540/Conteo_Objetos.txt","r", encoding="utf-8") as archivo_C:
            for linea in archivo_C:
                linea = linea.replace('\n','')
                print(linea)
                lista_conteo.append(linea)
    except Exception as ex:
        print(f"Error lectura Archivo CONTEO {ex}")


def lectura_Confisistema():
    global lista_confi
    try:
        lista_confi.clear()
        with open("/home/icam-540/CONFISISTEMA.txt","r", encoding="utf-8") as archivo:
            for linea in archivo:
                linea = linea.replace('\n','')
                print(linea)
                lista_confi.append(linea)
    except Exception as ex:
        print(f"Error lectura CONFISISTEMA {ex}")

def actualizar_linea_archivo(linea,valor):
    global lista_conteo
    try:
        lista_conteo.clear()
        with open("/home/icam-540/Conteo_Objetos.txt","r", encoding="utf-8") as archivo:
            lineas = archivo.readlines()

        while len(lineas) <= linea:
            lineas.append("\n")
            
        lineas[linea] = str(valor) + "\n"
        with open("/home/icam-540/Conteo_Objetos.txt","w", encoding="utf-8") as archivo:
            archivo.writelines(lineas)
        print("Modificacion linea Archivo")
    except Exception as e: 
        print(f"Actualizar archivo {e}") 

# ---------- Convert GST buffer to OpenCV ----------
def gst_to_opencv(sample):
    buf = sample.get_buffer()
    buffer = buf.extract_dup(0, buf.get_size())
    arr = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

# ---------- IMAGE CALLBACK (se llama por TRIGGER HARDWARE) ----------
def new_image_handler(sample):
    global image_arr
    global MUESTRA_IMAGEN
    if sample is None:
        return

    img = gst_to_opencv(sample)
    image_arr = img

def guardar_deteccion(frame, class_name="Sin_Tapa"):
    """Guarda imagen de detección"""
    try:
        cv2.imwrite(str(SAVE_PATH)+"/"+ str(FILE_NAME), frame)
        print(f"✅ Detección guardada: {SAVE_PATH}")
        detection_event.set()
    except Exception as e:
        print(f"❌ Error al guardar: {e}")

def calculo_posicion_obj(x1,y1,x2,y2,class_name):
    global frame_yolo
   

    cx = int((x1 + x2 )/ 2)
    cy = int((y1 + y2 )/ 2)

    if class_name == "Sin_Tapa":
        cv2.circle(frame_yolo,(int(x1),int(y1)),10,(27,34,234), 2)
        cv2.circle(frame_yolo,(int(x2),int(y2)),10,(19,4,10), 2)
        cv2.circle(frame_yolo,(cx,cy),10,(255,0,0), 2)
        print(f"Objeto sin tapa en X:{cx}, Y {cy}")
    else:
        cv2.circle(frame_yolo,(int(x1),int(y1)),10,(255,34,234), 2)
        cv2.circle(frame_yolo,(int(x2),int(y2)),10,(195,4,10), 2)
        cv2.circle(frame_yolo,(cx,cy),10,(255,0,0), 2)
        print(f"Objeto con tapa en X:{cx}, Y {cy}")
    
if __name__ == '__main__':
    lectura_Confisistema()
    lectura_Archivo_Conteo()
    bandera_Yolo = False
    count_rechazo = 0
    count_con_tapa = int(lista_conteo[0])
    count_sin_tapa = int(lista_conteo[1])
    linea_cero = 0
    linea_uno = 1
    try:
        cn2 = CamNavi2.CamNavi2()
    except:
        cn2 = CamNavi2()


    # time.sleep(1)

    # for i in range(10):
   #      if os.path.exists("dev/video0"):
     #        print("Video0:")
    #         break
    #     time.sleep(1)
    
    # Enumerar cámaras
    camera_dict = cn2.enum_camera_list()
    print("Cámaras detectadas:", camera_dict)

    camera = cn2.get_device_by_name('iCam500')  # iCAM-540 usa este driver
    icam_color = int(cn2.advcam_query_fw_sku(camera))

    # ---------- PIPELINE ----------
    pipe_params = {
        "acq_mode": 2,
        "width": WIDTH,
        "height": HEIGHT,
        "enable_infer": 0
    }

    if icam_color == 1:
        pipe_params["format"] = "YUY2"

    cn2.advcam_config_pipeline(camera, **pipe_params)
    cn2.advcam_open(camera, -1)
    # Setting do0 parameters
   #  camera.dio.do0.op_mode = 0 # DO op mode: user output
   #  camera.dio.do0.reverse = 0
    camera.dio.do0.user_output = 0 # DO low, DI high
    print("DO lOW " + str(camera.dio.do0.user_output))
   


    # ---------- REGISTER CALLBACK ----------
    cn2.advcam_register_new_image_handler(camera, new_image_handler)


    camera.hw_trigger_delay = 0
    print("Delay " + str(camera.hw_trigger_delay))

    camera.lighting.selector = int(lista_confi[0])
    #camera.lighting.selector = 2
    
    #camera.lighting.gain = 50
    camera.lighting.gain = int(lista_confi[1])

    #cn2.advcam_set_img_sharpness(camera, 5)
    #cn2.advcam_set_img_brightness(camera, 250)
    #cn2.advcam_set_img_gain(camera, 6)
    camera.image.saturation = int(lista_confi[2])
    camera.image.gamma = int(lista_confi[3])

    cn2.advcam_set_img_sharpness(camera, int(lista_confi[4]))
    cn2.advcam_set_img_brightness(camera,  int(lista_confi[5]))
    cn2.advcam_set_img_gain(camera, int(lista_confi[6]))

    camera.focus.pos_zero()

    #camera.focus.distance = 65
    camera.focus.distance = int(lista_confi[7])
    i = 0
    while i < 7:
            camera.focus.direction = 1 # lens focusing motor backward
            try:
                camera.focus.distance = 100
                print("lens motor posistion: ", camera.focus.position())
                time.sleep(0.1) 
                i+=1
                print("valor ", i)
            except ValueError:
                print("lens position out of index")
     #camera.focus.distance = 10
     #camera.focus.direction = 1
    
     #print("lens motor posistion: ", camera.focus.position())
    # ---------- START STREAM ----------
    cn2.advcam_play(camera)

    print("✅ iCAM-540 listo. Esperando trigger hardware en PIN 10...")
    ultimo_frame = None
    frame_yolo = None
    resized = None
    try:
        while True:
            if image_arr  is not None:
                
                resized = cv2.resize(image_arr, (YOLO_SIZE_W, YOLO_SIZE_H))

                if resized_2 is not None:

                    diff = cv2.absdiff(resized,resized_2)
                    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
                    #blur = cv2.GaussianBlur(gray,(5,5),0)
                    #_, thresh = cv2.threshold(blur,15,255,cv2.THRESH_BINARY)
                    _, thresh = cv2.threshold(gray,15,255,cv2.THRESH_BINARY)
                    porcentaje = np.sum(thresh > 0) / thresh.size
                    #print(porcentaje)
                    if porcentaje > 0.07:
                        bandera_Yolo = False

               
                #cv2.imshow("Vista Camara",image_arr) 
                image_arr = None

                
                if bandera_Yolo == False:
                    
                    results = model( resized, verbose=False, conf=YOLO_CONF)    # conf -> Confianza mínima = más rápido

                    frame_yolo = results[0].plot()
                    bandera_Yolo = True
                    resized_2 = resized.copy() 


                    for i,cls_id in enumerate(results[0].boxes.cls.tolist()):
                        class_name = results[0].names[int(cls_id)]
                        if class_name == "Sin_Tapa":
                            print(f"🎯 Objeto detectado: {class_name}")
                            #save_detection(frame_yolo, class_name)
                            count_sin_tapa+=1
                            count_rechazo = 0
                            #camera.dio.do0.user_output = 1
                            #salida  = str(camera.dio.do0.user_output)
                            #print("DO high " + salida)
                            x1,y1,x2,y2 = results[0].boxes.xyxy[i].tolist()
                            calculo_posicion_obj(x1,y1,x2,y2,class_name)
                        if class_name == "Con_Tapa":
                            print(f"🎯 Objeto detectado: {class_name}")
                            count_con_tapa+=1
                            x1,y1,x2,y2 = results[0].boxes.xyxy[i].tolist()
                            calculo_posicion_obj(x1,y1,x2,y2,class_name)
                            

                    cv2.imshow("Vista Camara",frame_yolo)
                    
                    actualizar_linea_archivo(linea_cero,count_con_tapa)
                    actualizar_linea_archivo(linea_uno,count_sin_tapa)
                    print(f"Con tapa :{count_con_tapa}, Sin Tapa {count_sin_tapa}")

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break
                elif key == ord('-'): 
                    cv2.destroyAllWindows()
                    image_arr = None
                elif key == ord('t'):  
                    
                    camera.dio.do0.user_output = 1
                    salida  = str(camera.dio.do0.user_output)
                    print("DO high " + salida)
                    #camera.dio.do0.user_output = 1 # DO high, DI low
                    level =  camera.dio.di0.level
                    print(level)
                elif key == ord('r'):  
                    camera.dio.do0.user_output = 0
                    salida  = str(camera.dio.do0.user_output)
                    print("DO low " + salida)
                    #camera.dio.do0.user_output = 1 # DO high, DI low
                    level =  camera.dio.di0.level
                    print(level)
            else:
                count_rechazo+=1
                if count_rechazo == 2 and  str(camera.dio.do0.user_output) == "1":
                    camera.dio.do0.user_output = 0
                    salida  = str(camera.dio.do0.user_output)
                    print("DO Low " + salida)

                if frame_yolo is not None:
                    ultimo_frame = frame_yolo.copy()

                if resized is not None  and frame_yolo is None:
                    ultimo_frame = resized.copy()

                if ultimo_frame is not None :
                    cv2.imshow("Vista Camara",ultimo_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('-'): 
                    cv2.destroyAllWindows()
                    image_arr = None
                elif key == ord('t'):  
                    count_rechazo = 0
                    camera.dio.do0.user_output = 1
                    salida  = str(camera.dio.do0.user_output)
                    print("DO high " + salida)
                    #camera.dio.do0.user_output = 1 # DO high, DI low
                    level =  camera.dio.di0.level
                    print(level)
                elif key == ord('r'):  
                    camera.dio.do0.user_output = 0
                    salida  = str(camera.dio.do0.user_output)
                    print("DO low " + salida)
                    #camera.dio.do0.user_output = 1 # DO high, DI low
                    level =  camera.dio.di0.level
                    print(level)


                if  cv2.waitKey(1) & 0xFF == 27:
                    break

                
                time.sleep(0.1)  

    except KeyboardInterrupt:
        pass

    # ---------- CLEANUP ----------
    cv2.destroyAllWindows()
    cn2.advcam_register_new_image_handler(camera, None)
    cn2.advcam_close(camera)
