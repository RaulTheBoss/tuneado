# -*- coding: utf-8 -*-
import av
import cv2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit as st
import pandas as pd
from datetime import datetime
import random
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import euclidean
import time
import json
import os

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="Detector de Residuos con GPS", 
    layout="wide"
)

st.title("🗑️ Detector de Residuos con IA - Conteo Inteligente")

# ==================== ARCHIVO PARA COMPARTIR DATOS ====================
ARCHIVO_DATOS = "datos_detecciones.json"

def guardar_datos(total, por_tipo, registro, ultima, gps, fps):
    """Guarda los datos en un archivo JSON"""
    try:
        datos = {
            'total': total,
            'por_tipo': dict(por_tipo),
            'registro': registro[-100:],
            'ultima': ultima,
            'gps': gps,
            'fps': fps,
            'timestamp': datetime.now().isoformat()
        }
        with open(ARCHIVO_DATOS, 'w') as f:
            json.dump(datos, f)
        return True
    except Exception as e:
        print(f"Error guardando datos: {e}")
        return False

def cargar_datos():
    """Carga los datos del archivo JSON"""
    try:
        if os.path.exists(ARCHIVO_DATOS):
            with open(ARCHIVO_DATOS, 'r') as f:
                datos = json.load(f)
            return datos
        else:
            return {
                'total': 0,
                'por_tipo': {},
                'registro': [],
                'ultima': None,
                'gps': None,
                'fps': 0,
                'timestamp': None
            }
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return {
            'total': 0,
            'por_tipo': {},
            'registro': [],
            'ultima': None,
            'gps': None,
            'fps': 0,
            'timestamp': None
        }

def reiniciar_archivo():
    """Reinicia el archivo de datos"""
    try:
        if os.path.exists(ARCHIVO_DATOS):
            os.remove(ARCHIVO_DATOS)
        return True
    except:
        return False

# ==================== SISTEMA DE SEGUIMIENTO ====================
class RastreadorObjetos:
    def __init__(self, distancia_maxima=50, frames_historial=10, iou_threshold=0.5):
        """
        Args:
            distancia_maxima: Distancia máxima en píxeles para considerar mismo objeto
            frames_historial: Número de frames que se recuerda un objeto
            iou_threshold: Umbral de IoU para considerar mismo objeto (0-1)
        """
        self.objetos_activos = {}
        self.distancia_maxima = distancia_maxima
        self.frames_historial = frames_historial
        self.iou_threshold = iou_threshold
        self.id_contador = 0
        self.ultimo_frame = 0
        
    def calcular_iou(self, box1, box2):
        """Calcula el Intersection over Union entre dos bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
            
        area_interseccion = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        iou = area_interseccion / (area1 + area2 - area_interseccion)
        return iou
    
    def obtener_centro(self, bbox):
        """Obtiene el centro de un bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def procesar_detecciones(self, detecciones, frame_actual, confianza_minima=0.4):
        """Procesa nuevas detecciones y retorna solo las nuevas"""
        nuevas_detecciones = []
        self.ultimo_frame += 1
        
        for deteccion in detecciones:
            bbox = deteccion['bbox']
            tipo = deteccion['tipo']
            confianza = deteccion['confianza']
            
            if confianza < confianza_minima:
                continue
            
            objeto_encontrado = False
            id_encontrado = None
            
            for obj_id, obj_data in self.objetos_activos.items():
                # Verificar por IoU
                iou = self.calcular_iou(bbox, obj_data['bbox'])
                
                if iou > self.iou_threshold and obj_data['tipo'] == tipo:
                    objeto_encontrado = True
                    id_encontrado = obj_id
                    break
                
                # Verificar por distancia de centros
                centro1 = self.obtener_centro(bbox)
                centro2 = self.obtener_centro(obj_data['bbox'])
                distancia = euclidean(centro1, centro2)
                
                if distancia < self.distancia_maxima and obj_data['tipo'] == tipo:
                    objeto_encontrado = True
                    id_encontrado = obj_id
                    break
            
            if objeto_encontrado and id_encontrado is not None:
                # Actualizar objeto existente
                self.objetos_activos[id_encontrado]['bbox'] = bbox
                self.objetos_activos[id_encontrado]['ultimo_frame'] = self.ultimo_frame
                self.objetos_activos[id_encontrado]['confianza'] = max(
                    self.objetos_activos[id_encontrado]['confianza'],
                    confianza
                )
                self.objetos_activos[id_encontrado]['contador_frames'] += 1
            else:
                # Es un objeto nuevo
                nuevo_id = self.id_contador
                self.id_contador += 1
                
                self.objetos_activos[nuevo_id] = {
                    'bbox': bbox,
                    'tipo': tipo,
                    'confianza': confianza,
                    'ultimo_frame': self.ultimo_frame,
                    'contado': False,
                    'contador_frames': 1  # Contador de frames vistos
                }
                
                deteccion['id'] = nuevo_id
                nuevas_detecciones.append(deteccion)
        
        # Limpiar objetos antiguos
        for obj_id, obj_data in list(self.objetos_activos.items()):
            if self.ultimo_frame - obj_data['ultimo_frame'] > self.frames_historial:
                del self.objetos_activos[obj_id]
        
        return nuevas_detecciones
    
    def marcar_como_contado(self, obj_id):
        """Marca un objeto como ya contado"""
        if obj_id in self.objetos_activos:
            self.objetos_activos[obj_id]['contado'] = True
    
    def reiniciar(self):
        """Reinicia el rastreador"""
        self.objetos_activos = {}
        self.id_contador = 0
        self.ultimo_frame = 0
    
    def actualizar_parametros(self, distancia_maxima=None, frames_historial=None, iou_threshold=None):
        """Actualiza los parámetros del rastreador en tiempo real"""
        if distancia_maxima is not None:
            self.distancia_maxima = distancia_maxima
        if frames_historial is not None:
            self.frames_historial = frames_historial
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold

# ==================== FUNCIONES GPS ====================
def obtener_gps():
    """Obtiene ubicación GPS simulada"""
    lat = 4.7110 + random.uniform(-0.005, 0.005)
    lon = -74.0721 + random.uniform(-0.005, 0.005)
    
    return {
        "latitud": lat,
        "longitud": lon,
        "timestamp": datetime.now().isoformat()
    }

# ==================== PROCESADOR DE VIDEO ====================
class Detector(VideoProcessorBase):
    def __init__(self):
        self.ultima_ubicacion = None
        self.frame_count = 0
        self.fps = 0
        self.start_time = datetime.now()
        self.ultima_actualizacion_gps = datetime.now()
        self.ultima_guardada = datetime.now()
        
        # Contadores locales
        self.total_contado = 0
        self.por_tipo_contado = defaultdict(int)
        self.registro_contado = []
        self.ultima_deteccion = None
        
        # Parámetros del rastreador (valores por defecto)
        self.distancia_maxima = 50
        self.frames_historial = 10
        self.iou_threshold = 0.5
        
        # Rastreador local
        self.rastreador = RastreadorObjetos(
            distancia_maxima=self.distancia_maxima,
            frames_historial=self.frames_historial,
            iou_threshold=self.iou_threshold
        )
        
    def recv(self, frame):
        try:
            # 1. Procesar frame
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            # 2. Calcular FPS
            tiempo_transcurrido = (datetime.now() - self.start_time).seconds
            if tiempo_transcurrido > 0:
                self.fps = self.frame_count / tiempo_transcurrido
            
            # 3. Obtener parámetros actualizados
            try:
                if 'distancia_maxima' in st.session_state:
                    self.distancia_maxima = st.session_state['distancia_maxima']
                if 'frames_historial' in st.session_state:
                    self.frames_historial = st.session_state['frames_historial']
                if 'iou_threshold' in st.session_state:
                    self.iou_threshold = st.session_state['iou_threshold']
                
                # Actualizar rastreador con nuevos parámetros
                self.rastreador.actualizar_parametros(
                    distancia_maxima=self.distancia_maxima,
                    frames_historial=self.frames_historial,
                    iou_threshold=self.iou_threshold
                )
            except:
                pass
            
            # 4. Obtener GPS cada 3 segundos
            if (datetime.now() - self.ultima_actualizacion_gps).seconds >= 3:
                self.ultima_ubicacion = obtener_gps()
                self.ultima_actualizacion_gps = datetime.now()
            
            # 5. Realizar predicción
            confianza = 0.4
            try:
                if 'confianza' in st.session_state:
                    confianza = st.session_state['confianza']
            except:
                pass
            
            resultados = self.modelo.predict(img, conf=confianza, verbose=False)
            
            # 6. Preparar detecciones
            detecciones = []
            if len(resultados) > 0 and resultados[0].boxes is not None:
                for box in resultados[0].boxes:
                    clase_id = int(box.cls)
                    tipo_residuo = self.modelo.names[clase_id]
                    confianza_det = float(box.conf)
                    
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    
                    detecciones.append({
                        'bbox': bbox,
                        'tipo': tipo_residuo,
                        'confianza': confianza_det
                    })
            
            # 7. Procesar nuevas detecciones
            nuevas_detecciones = self.rastreador.procesar_detecciones(
                detecciones,
                self.frame_count,
                confianza_minima=confianza
            )
            
            # 8. Registrar NUEVAS detecciones
            for deteccion in nuevas_detecciones:
                registro = {
                    "id": len(self.registro_contado) + 1,
                    "obj_id": deteccion['id'],
                    "tipo": deteccion['tipo'],
                    "confianza": round(deteccion['confianza'] * 100, 2),
                    "timestamp": datetime.now().isoformat(),
                    "fecha": datetime.now().strftime("%Y-%m-%d"),
                    "hora": datetime.now().strftime("%H:%M:%S"),
                    "latitud": self.ultima_ubicacion["latitud"] if self.ultima_ubicacion else None,
                    "longitud": self.ultima_ubicacion["longitud"] if self.ultima_ubicacion else None,
                    "frames_vistos": self.rastreador.objetos_activos[deteccion['id']]['contador_frames']
                }
                
                self.total_contado += 1
                self.por_tipo_contado[deteccion['tipo']] += 1
                self.registro_contado.append(registro)
                self.ultima_deteccion = registro
                
                self.rastreador.marcar_como_contado(deteccion['id'])
            
            # 9. Guardar datos en archivo cada 1 segundo
            if (datetime.now() - self.ultima_guardada).seconds >= 1:
                guardar_datos(
                    total=self.total_contado,
                    por_tipo=self.por_tipo_contado,
                    registro=self.registro_contado,
                    ultima=self.ultima_deteccion,
                    gps=self.ultima_ubicacion,
                    fps=self.fps
                )
                self.ultima_guardada = datetime.now()
            
            # 10. Dibujar resultados
            salida = resultados[0].plot()
            self.dibujar_ids_objetos(salida)
            self.agregar_info_frame(salida)
            
            return av.VideoFrame.from_ndarray(salida, format="bgr24")
            
        except Exception as e:
            print(f"Error en procesamiento: {e}")
            return frame
    
    def dibujar_ids_objetos(self, img):
        """Dibuja IDs y estado de los objetos"""
        for obj_id, obj_data in self.rastreador.objetos_activos.items():
            bbox = obj_data['bbox']
            x1, y1, x2, y2 = bbox
            
            # Color: Verde = contado, Rojo = no contado, Amarillo = en seguimiento
            if obj_data['contado']:
                color = (0, 255, 0)  # Verde
            elif obj_data['contador_frames'] > 3:
                color = (0, 255, 255)  # Amarillo (en seguimiento)
            else:
                color = (0, 0, 255)  # Rojo (nuevo)
            
            # Mostrar ID y frames vistos
            texto = f"ID:{obj_id} ({obj_data['contador_frames']}f)"
            cv2.putText(img, texto, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 2)
    
    def agregar_info_frame(self, img):
        """Agrega información en el frame"""
        info_text = f"Total: {self.total_contado} | Activos: {len(self.rastreador.objetos_activos)} | FPS: {self.fps:.1f}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        # Mostrar parámetros actuales
        params_text = f"Dist:{self.distancia_maxima}px | Frames:{self.frames_historial} | IoU:{self.iou_threshold:.2f}"
        cv2.putText(img, params_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (255, 255, 0), 1)
        
        if self.ultima_ubicacion:
            gps_text = f"📍 {self.ultima_ubicacion['latitud']:.6f}, {self.ultima_ubicacion['longitud']:.6f}"
            cv2.putText(img, gps_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 0), 1)
        
        y_pos = 110
        if self.por_tipo_contado:
            tipos_mostrar = sorted(
                self.por_tipo_contado.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for tipo, count in tipos_mostrar:
                texto = f"{tipo}: {count}"
                cv2.putText(img, texto, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 255, 255), 1)
                y_pos += 25

# ==================== CARGAR MODELO ====================
@st.cache_resource
def cargar_modelo():
    try:
        modelo = YOLO("best.pt")
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# ==================== FUNCIONES DE CONTROL ====================
def reiniciar_sistema():
    """Reinicia todo el sistema"""
    reiniciar_archivo()
    st.success("✅ Sistema reiniciado correctamente")
    time.sleep(0.5)
    st.rerun()

# ==================== INTERFAZ DE USUARIO ====================

# Cargar modelo
modelo = cargar_modelo()
if modelo:
    Detector.modelo = modelo
else:
    st.error("❌ No se pudo cargar el modelo 'best.pt'")
    st.stop()

# Inicializar parámetros en session_state
if 'distancia_maxima' not in st.session_state:
    st.session_state['distancia_maxima'] = 50
if 'frames_historial' not in st.session_state:
    st.session_state['frames_historial'] = 10
if 'iou_threshold' not in st.session_state:
    st.session_state['iou_threshold'] = 0.5
if 'confianza' not in st.session_state:
    st.session_state['confianza'] = 0.4

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Control de confianza
    confianza = st.slider(
        "🎯 Nivel de confianza",
        0.1, 1.0, st.session_state['confianza'], 0.05,
        key="slider_confianza",
        help="Ajusta la sensibilidad de detección. Valores más altos = menos falsos positivos"
    )
    st.session_state['confianza'] = confianza
    
    st.markdown("---")
    
    st.header("🎯 Ajustes de Seguimiento")
    st.caption("Ajusta estos parámetros para evitar conteos duplicados")
    
    # Control de frames de historial
    frames_historial = st.slider(
        "🔄 Frames de historial",
        3, 30, st.session_state['frames_historial'], 1,
        key="slider_frames",
        help="Número de frames que se recuerda un objeto. Mayor = más memoria"
    )
    st.session_state['frames_historial'] = frames_historial
    
    # Control de distancia máxima
    distancia_maxima = st.slider(
        "📏 Distancia máxima (píxeles)",
        10, 200, st.session_state['distancia_maxima'], 5,
        key="slider_distancia",
        help="Distancia máxima para considerar que dos objetos son el mismo"
    )
    st.session_state['distancia_maxima'] = distancia_maxima
    
    # Control de IoU threshold
    iou_threshold = st.slider(
        "📐 Umbral IoU",
        0.1, 0.9, st.session_state['iou_threshold'], 0.05,
        key="slider_iou",
        help="Superposición mínima para considerar mismo objeto. Mayor = más estricto"
    )
    st.session_state['iou_threshold'] = iou_threshold
    
    st.markdown("---")
    
    # Información de parámetros
    with st.expander("ℹ️ Guía de Ajuste"):
        st.markdown("""
        **🔧 Cómo ajustar para evitar conteos duplicados:**
        
        1. **Frames de historial** (recomendado: 8-15)
           - *Alto (15-30)*: Más memoria, evita duplicados
           - *Bajo (3-7)*: Olvida rápido, puede contar duplicados
        
        2. **Distancia máxima** (recomendado: 30-70)
           - *Alta (70-200)*: Más tolerante, agrupa objetos cercanos
           - *Baja (10-30)*: Más preciso, puede perder objetos
        
        3. **Umbral IoU** (recomendado: 0.4-0.6)
           - *Alto (0.6-0.9)*: Más estricto, considera objetos distintos
           - *Bajo (0.1-0.4)*: Más tolerante, agrupa objetos similares
        
        4. **Nivel de confianza** (recomendado: 0.3-0.6)
           - *Alto (>0.6)*: Solo detecciones seguras
           - *Bajo (<0.3)*: Más detecciones pero más falsos
        """)
    
    st.markdown("---")
    
    # Selección de cámara
    camara = st.selectbox(
        "📷 Selecciona la cámara",
        ["Trasera", "Delantera"],
        key="select_camara"
    )
    
    if camara == "Trasera":
        video_constraints = {
            "video": {"facingMode": {"ideal": "environment"}},
            "audio": False
        }
    else:
        video_constraints = {
            "video": {"facingMode": "user"},
            "audio": False
        }
    
    st.markdown("---")
    
    # --- Estadísticas en Vivo ---
    st.header("📊 Estadísticas en Vivo")
    
    datos = cargar_datos()
    total = datos.get('total', 0)
    tipos = len(datos.get('por_tipo', {}))
    ultima = datos.get('ultima')
    gps = datos.get('gps')
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔄 Total", total)
    with col2:
        st.metric("📋 Tipos", tipos)
    
    if gps:
        st.info(f"""
        📍 Ubicación
        - Lat: {gps['latitud']:.6f}
        - Lon: {gps['longitud']:.6f}
        """)
    
    if ultima:
        st.success(f"""
        ✅ Última Detección
        - Tipo: {ultima['tipo']}
        - Confianza: {ultima['confianza']}%
        - ID: {ultima['obj_id']}
        - Hora: {ultima['hora']}
        """)
    else:
        st.info("⏳ Esperando detecciones...")
    
    st.markdown("---")
    
    # --- Controles ---
    st.header("🎮 Controles")
    
    if st.button("🔄 Reiniciar Sistema", use_container_width=True):
        reiniciar_sistema()
    
    if st.button("💾 Exportar CSV", use_container_width=True):
        registro = datos.get('registro', [])
        if registro:
            df = pd.DataFrame(registro)
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name=f"detecciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No hay datos para exportar")

# --- Video ---
st.markdown("### 🎥 Video en Tiempo Real")

webrtc_streamer(
    key="detector",
    video_processor_factory=Detector,
    media_stream_constraints=video_constraints,
    async_processing=True
)

# --- Tabla de Detecciones ---
st.markdown("---")
st.markdown("### 📋 Detecciones Registradas")

datos = cargar_datos()
registro = datos.get('registro', [])
por_tipo = datos.get('por_tipo', {})
total = datos.get('total', 0)

if registro:
    # Mostrar resumen por tipo
    st.markdown("#### 📊 Conteo por Tipo")
    if por_tipo:
        tipos_data = list(por_tipo.items())
        cols = st.columns(min(4, len(tipos_data)))
        for idx, (tipo, count) in enumerate(tipos_data[:4]):
            with cols[idx]:
                st.metric(
                    label=tipo,
                    value=count,
                    delta=f"{count/total*100:.1f}%" if total > 0 else "0%"
                )
    
    # Mostrar últimas detecciones
    st.markdown("#### 🕒 Últimas Detecciones")
    df_reciente = pd.DataFrame(registro[-10:])
    columnas_mostrar = ['id', 'obj_id', 'tipo', 'confianza', 'hora', 'latitud', 'longitud', 'frames_vistos']
    df_reciente = df_reciente[columnas_mostrar]
    df_reciente.columns = ['ID', 'ObjID', 'Tipo', 'Confianza %', 'Hora', 'Latitud', 'Longitud', 'Frames Vistos']
    st.dataframe(df_reciente, use_container_width=True)
    
    # Mostrar todas las detecciones
    with st.expander("📋 Ver todas las detecciones"):
        df_completo = pd.DataFrame(registro)
        st.dataframe(df_completo, use_container_width=True)
    
else:
    st.info("⏳ No hay detecciones registradas aún. Apunta la cámara a algún residuo.")

# --- Información del Sistema ---
st.markdown("---")
col_info1, col_info2, col_info3, col_info4 = st.columns(4)

with col_info1:
    st.metric("🗑️ Total Detecciones", total)

with col_info2:
    st.metric("📋 Tipos Únicos", len(por_tipo))

with col_info3:
    st.metric("📏 Distancia", f"{st.session_state['distancia_maxima']}px")

with col_info4:
    st.metric("🔄 Frames", f"{st.session_state['frames_historial']}")

# Auto-refresh
if st.button("🔄 Actualizar datos"):
    st.rerun()