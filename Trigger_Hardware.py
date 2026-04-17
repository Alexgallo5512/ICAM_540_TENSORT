
import os
import cv2
import numpy as np

import time
import threading
import queue       # OPTIMIZACIÓN 1: cola para escritura asíncrona a disco
import os
from pathlib import Path
import torch

# ================= CONFIGURACIÓN =================
print("CUDA:", torch.cuda.is_available())
print("CUDA conteo:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No hay CUDA disponible")
# ===== Patch compatibilidad torchvision con torch NVIDIA Jetson (CUDA 12.6) =====

# Patch 1: evita error al registrar operadores meta de torchvision
import torch._library.fake_impl as _fake_impl_module
_orig_fake_register = _fake_impl_module.FakeImplHolder.register
def _patched_fake_register(self, func, source):
    try:
        return _orig_fake_register(self, func, source)
    except RuntimeError:
        return None
_fake_impl_module.FakeImplHolder.register = _patched_fake_register

# Patch 2: reemplaza torchvision.ops.nms con implementacion pura PyTorch
#          (el .so de torchvision no es compatible con el build NVIDIA)
import torchvision.ops as _tv_ops
def _nms_puro(boxes, scores, iou_threshold):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = boxes[order[1:], 0].clamp(min=boxes[i, 0])
        yy1 = boxes[order[1:], 1].clamp(min=boxes[i, 1])
        xx2 = boxes[order[1:], 2].clamp(max=boxes[i, 2])
        yy2 = boxes[order[1:], 3].clamp(max=boxes[i, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        areas = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + areas - inter)
        order = order[1:][iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
_tv_ops.nms = _nms_puro

# ============================================================================

from ultralytics import YOLO
from CamNavi2 import CamNavi2


# ================= CONFIG =================
SAVE_PATH = "/home/icam-540/capturas"
FILE_NAME = "Imagen_trigger.jpg"
PT_PATH     = "/home/icam-540/Proyectos/ICAM_540_TENSORT/best.pt"
ENGINE_PATH = "/home/icam-540/Proyectos/ICAM_540_TENSORT/best.engine"

# Resolución cámara (reducida para mejor rendimiento)
WIDTH  = 3840
HEIGHT = 2160  
# Tamaño YOLO (pequeño = procesamiento rápido)
YOLO_SIZE_W = 640 
YOLO_SIZE_H = 480 
YOLO_CONF = 0.7  # Confianza mínima (> 0.5 = más rápido)

MUESTRA_IMAGEN = False
detection_event = threading.Event()

# OPTIMIZACIÓN 2: evento para despertar el loop EXACTO cuando llega un frame
# Evita el time.sleep(0.1) fijo y reduce latencia de respuesta al trigger
frame_event = threading.Event()

_frame_lock = threading.Lock()
# =========================================
#  Exporta a TensorRT Engine solo si no existe; de lo contrario carga directo
#  OPTIMIZACIÓN 3: half=True genera engine FP16 → ~30-50% más rápido en Jetson Orin
if not Path(ENGINE_PATH).exists():
    print(f"[YOLO] best.engine no encontrado — exportando desde {PT_PATH} ...")
    _tmp = YOLO(PT_PATH)
    _tmp.export(format="engine", device=0, half=True)  # FP16 para Jetson GPU
    del _tmp

model = YOLO(ENGINE_PATH)

# OPTIMIZACIÓN 4: Warmup del modelo — hace 1 inferencia dummy al arrancar
# La primera inferencia real de TensorRT inicializa contextos CUDA internos (~500ms).
# Con el warmup ese costo ocurre aquí y no en el primer trigger de producción.
_dummy = np.zeros((YOLO_SIZE_H, YOLO_SIZE_W, 3), dtype=np.uint8)
model(_dummy, verbose=False, half=True)
print("✅ Modelo TensorRT calentado y listo")

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

# OPTIMIZACIÓN 1: Cola de escritura asíncrona a disco
# El loop principal ya no se bloquea esperando I/O del archivo.
# Las escrituras se procesan en un hilo separado de fondo (daemon).
_file_queue = queue.Queue(maxsize=10)

def _writer_thread():
    """Hilo daemon: consume la cola y escribe Conteo_Objetos.txt sin bloquear el loop."""
    while True:
        linea, valor = _file_queue.get()   # espera hasta que haya algo
        try:
            with open("/home/icam-540/Conteo_Objetos.txt", "r", encoding="utf-8") as archivo:
                lineas = archivo.readlines()
            while len(lineas) <= linea:
                lineas.append("\n")
            lineas[linea] = str(valor) + "\n"
            with open("/home/icam-540/Conteo_Objetos.txt", "w", encoding="utf-8") as archivo:
                archivo.writelines(lineas)
        except Exception as e:
            print(f"Actualizar archivo {e}")

# Inicia el hilo de escritura como daemon (se cierra solo al terminar el programa)
threading.Thread(target=_writer_thread, daemon=True, name="FileWriter").start()

def actualizar_linea_archivo(linea, valor):
    """Encola la escritura en lugar de escribir directamente — no bloquea el loop."""
    try:
        _file_queue.put_nowait((linea, valor))
    except queue.Full:
        print("?? Cola escritura llena  escritura descartada")

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
    with _frame_lock:
        image_arr = img
    # OPTIMIZACIÓN 2: notifica al loop principal que llegó un frame nuevo.
    # El loop deja de dormir inmediatamente en lugar de esperar el sleep fijo.
    frame_event.set()

def guardar_deteccion(frame, class_name="Sin_Tapa"):
    """Guarda imagen de detección"""
    try:
        cv2.imwrite(str(SAVE_PATH)+"/"+ str(FILE_NAME), frame)
        print(f"✅ Detección guardada [{class_name}]: {SAVE_PATH}")
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
            # OPTIMIZACIÓN 2: espera hasta que llegue un frame nuevo (máx 500ms)
                # Reemplaza el time.sleep(0.1) fijo — el loop despierta exacto con el trigger
            frame_event.wait(timeout=0.5)
            frame_event.clear()


            with _frame_lock:
                frame_local = image_arr
                image_arr = None

            if frame_local  is not None:
                
                resized = cv2.resize(frame_local, (YOLO_SIZE_W, YOLO_SIZE_H))

                if resized_2 is not None:

                      #diff = cv2.absdiff(resized,resized_2)
                     # gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
                     #blur = cv2.GaussianBlur(gray,(5,5),0)
                     # _, thresh = cv2.threshold(gray,15,255,cv2.THRESH_BINARY)
                      #porcentaje = np.sum(thresh > 0) / thresh.size
                      #print(porcentaje)
                      #if porcentaje > 0.07:
                      #    bandera_Yolo = False

                    # OPTIMIZACIÓN 3: diff directo en gris (3x más rápido que BGR→diff→gray)
                    # OPTIMIZACIÓN 3: diff en gris con blur para eliminar ruido de sensor
                    gray_now  = cv2.GaussianBlur(cv2.cvtColor(resized,   cv2.COLOR_BGR2GRAY), (5, 5), 0)
                    gray_prev = cv2.GaussianBlur(cv2.cvtColor(resized_2, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                    gray = cv2.absdiff(gray_now, gray_prev)

                    # Umbral 25 (antes 15): ignora fluctuaciones de exposición y ruido de sensor
                    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

                    porcentaje = np.sum(thresh > 0) / thresh.size
                    print(f"diff: {porcentaje:.3f}")  # temporal: ajusta umbrales según tu cámara
                    # Área 0.15 (antes 0.07): requiere cambio real en >15% del frame
                    if porcentaje > 0.08:
                        bandera_Yolo = False

               
                if bandera_Yolo == False:
                    # OPTIMIZACIÓN 3: half=True activa inferencia FP16 en cada frame
                    # Aprovecha el engine compilado con half=True → menor latencia por inferencia
                    results = model(resized, verbose=False, conf=YOLO_CONF, half=True)

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
                    with _frame_lock:
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
                    with _frame_lock:
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


                

    except KeyboardInterrupt:
        pass

    # ---------- CLEANUP ----------
    cv2.destroyAllWindows()
    cn2.advcam_register_new_image_handler(camera, None)
    cn2.advcam_close(camera)
