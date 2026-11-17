import base64
import io
import os
import cv2
import ezdxf
import numpy as np
import streamlit as st
from PIL import Image
from rembg import remove, new_session
from shapely.geometry import Polygon

st.set_page_config(page_title="Convertidor 300 DPI y Plotter", page_icon="📐", layout="centered")

st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

st.title("📐 Convertidor Multiformato 300 DPI y Plotter")
st.write("Convierte imágenes a **DXF, SVG, PNG, JPG, WEBP y PDF** calibrados exactamente a **300 DPI reales** para impresión y corte.")

DPI_OBJETIVO = 300.0
MM_POR_PULGADA = 25.4

@st.cache_resource
def cargar_modelo_ia():
    return new_session("u2netp")

def px_a_mm(pixeles):
    return (pixeles / DPI_OBJETIVO) * MM_POR_PULGADA

def mm_a_px(milimetros):
    return int(round((milimetros / MM_POR_PULGADA) * DPI_OBJETIVO))

archivo_subido = st.file_uploader("Sube tu imagen", type=["png", "jpg", "jpeg", "webp", "bmp"])

formato_salida = st.selectbox(
    "Selecciona el formato de exportación (300 DPI):",
    [
        "DXF (Vector en Milímetros Reales - Silhouette / AutoCAD)",
        "SVG (Vector en Milímetros Reales - Cricut / Illustrator)",
        "PNG (Imagen sin fondo a 300 DPI)",
        "JPG (Imagen con fondo blanco a 300 DPI)",
        "WEBP (Imagen optimizada a 300 DPI)",
        "PDF (Documento listo para impresión a 300 DPI)"
    ]
)

col1, col2 = st.columns(2)
with col1:
    quitar_fondo = st.checkbox("Quitar fondo automáticamente con IA", value=True)
    if "DXF" in formato_salida:
        modo_corte = st.radio(
            "Modo de trazado DXF:",
            ("Vinilo de Corte (Todo el diseño)", "Solo Contorno Exterior (Troquel)")
        )
    else:
        modo_corte = "Solo Contorno Exterior (Troquel)"

with col2:
    if "DXF" in formato_salida or "SVG" in formato_salida:
        margen_mm = st.slider("Margen de corte (milímetros reales)", 0.0, 10.0, 2.5, 0.5)
        suavizado = st.slider("Suavizado de nodos", 0.5, 8.0, 1.5, 0.5)
    else:
        margen_mm = 0.0
        suavizado = 1.5

def obtener_contornos(img_rgba, modo, margen_en_mm):
    ancho, alto = img_rgba.size
    margen_px = mm_a_px(margen_en_mm)
    pad = margen_px + 30
    lienzo = Image.new("RGBA", (ancho + pad * 2, alto + pad * 2), (0, 0, 0, 0))
    lienzo.paste(img_rgba, (pad, pad))
    nuevo_ancho, nuevo_alto = lienzo.size

    matriz_img = np.array(lienzo)
    canal_alfa = matriz_img[:, :, 3]

    if np.min(canal_alfa) == 255:
        gris = cv2.cvtColor(matriz_img, cv2.COLOR_RGBA2GRAY)
        _, mascara = cv2.threshold(gris, 240, 255, cv2.THRESH_BINARY_INV)
    else:
        _, mascara = cv2.threshold(canal_alfa, 127, 255, cv2.THRESH_BINARY)

    if "Troquel" in modo:
        contornos_ext, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mascara_solida = np.zeros_like(mascara)
        cv2.drawContours(mascara_solida, contornos_ext, -1, 255, thickness=cv2.FILLED)
        if margen_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margen_px * 2 + 1, margen_px * 2 + 1))
            mascara_solida = cv2.dilate(mascara_solida, kernel, iterations=1)
        contornos, _ = cv2.findContours(mascara_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contornos, _ = cv2.findContours(mascara, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    return lienzo, contornos, nuevo_ancho, nuevo_alto

if archivo_subido is not None:
    imagen_original = Image.open(archivo_subido)
    w_px, h_px = imagen_original.size
    w_cm = px_a_mm(w_px) / 10.0
    h_cm = px_a_mm(h_px) / 10.0

    st.image(imagen_original, caption=f"Original: {w_px}x{h_px} px | Tamaño físico a 300 DPI: {w_cm:.2f} x {h_cm:.2f} cm", width=260)

    if st.button("🚀 Convertir a 300 DPI", type="primary"):
        with st.spinner("Calibrando a 300 DPI y procesando archivo..."):
            try:
                input_bytes = archivo_subido.getvalue()
                nombre_base = os.path.splitext(archivo_subido.name)[0]

                if quitar_fondo:
                    sesion = cargar_modelo_ia()
                    sin_fondo_bytes = remove(input_bytes, session=sesion)
                    img_procesada = Image.open(io.BytesIO(sin_fondo_bytes)).convert("RGBA")
                else:
                    img_procesada = imagen_original.convert("RGBA")

                if "DXF" in formato_salida:
                    _, contornos, _, nuevo_alto = obtener_contornos(img_procesada, modo_corte, margen_mm)
                    doc = ezdxf.new("R2000")
                    doc.header["$INSUNITS"] = 4
                    doc.header["$MEASUREMENT"] = 1
                    msp = doc.modelspace()

                    trazos = 0
                    for c in contornos:
                        if cv2.contourArea(c) < 25:
                            continue
                        pts = c.squeeze()
                        if len(pts.shape) < 2 or len(pts) < 3:
                            continue
                        poli = Polygon(pts)
                        if not poli.is_valid:
                            poli = poli.buffer(0)
                        poli_suave = poli.simplify(suavizado, preserve_topology=True)
                        if poli_suave.geom_type == "Polygon":
                            coords = list(poli_suave.exterior.coords)
                            pts_dxf_mm = [(px_a_mm(x), px_a_mm(nuevo_alto - y)) for x, y in coords]
                            msp.add_lwpolyline(pts_dxf_mm, close=True, dxfattribs={"color": 1})
                            trazos += 1

                    if trazos == 0:
                        st.error("No se encontraron bordes claros para vectorizar.")
                    else:
                        buf = io.StringIO()
                        doc.write(buf)
                        st.success(f"¡DXF generado en milímetros reales (300 DPI) con {trazos} trazos!")
                        st.download_button("📥 Descargar DXF (300 DPI)", buf.getvalue(), f"{nombre_base}_300dpi.dxf", "application/dxf")

                elif "SVG" in formato_salida:
                    lienzo, contornos, nuevo_ancho, nuevo_alto = obtener_contornos(img_procesada, "Troquel", margen_mm)
                    if not contornos:
                        st.error("No se detectó contorno.")
                    else:
                        ancho_mm = px_a_mm(nuevo_ancho)
                        alto_mm = px_a_mm(nuevo_alto)

                        c_max = max(contornos, key=cv2.contourArea)
                        poli = Polygon(c_max.squeeze()).simplify(suavizado, preserve_topology=True)
                        coords = list(poli.exterior.coords)
                        ruta_d = "M " + " L ".join([f"{px_a_mm(x):.3f},{px_a_mm(y):.3f}" for x, y in coords]) + " Z"

                        buf_img = io.BytesIO()
                        lienzo.save(buf_img, format="PNG", dpi=(300, 300))
                        b64 = base64.b64encode(buf_img.getvalue()).decode("utf-8")

                        svg_str = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{ancho_mm:.3f}mm" height="{alto_mm:.3f}mm" viewBox="0 0 {ancho_mm:.3f} {alto_mm:.3f}" xmlns="http://w3.org" xmlns:xlink="http://w3.org">
  <g id="Capa_Impresion"><image x="0" y="0" width="{ancho_mm:.3f}" height="{alto_mm:.3f}" xlink:href="data:image/png;base64,{b64}" /></g>
  <g id="Capa_Corte"><path d="{ruta_d}" fill="none" stroke="#FF0000" stroke-width="0.2" /></g>
</svg>'''
                        st.success(f"¡SVG generado a escala real ({ancho_mm/10:.2f} x {alto_mm/10:.2f} cm a 300 DPI)!")
                        st.download_button("📥 Descargar SVG (300 DPI)", svg_str, f"{nombre_base}_300dpi.svg", "image/svg+xml")

                elif "PNG" in formato_salida:
                    buf = io.BytesIO()
                    img_procesada.save(buf, format="PNG", dpi=(300, 300))
                    st.image(img_procesada, caption="Resultado PNG a 300 DPI", width=250)
                    st.download_button("📥 Descargar PNG (300 DPI)", buf.getvalue(), f"{nombre_base}_300dpi.png", "image/png")

                elif "JPG" in formato_salida:
                    fondo_blanco = Image.new("RGB", img_procesada.size, (255, 255, 255))
                    fondo_blanco.paste(img_procesada, mask=img_procesada.split()[3])
                    buf = io.BytesIO()
                    fondo_blanco.save(buf, format="JPEG", quality=98, dpi=(300, 300))
                    st.image(fondo_blanco, caption="Resultado JPG a 300 DPI", width=250)
                    st.download_button("📥 Descargar JPG (300 DPI)", buf.getvalue(), f"{nombre_base}_300dpi.jpg", "image/jpeg")

                elif "WEBP" in formato_salida:
                    buf = io.BytesIO()
                    img_procesada.save(buf, format="WEBP", quality=95, dpi=(300, 300))
                    st.image(img_procesada, caption="Resultado WEBP a 300 DPI", width=250)
                    st.download_button("📥 Descargar WEBP (300 DPI)", buf.getvalue(), f"{nombre_base}_300dpi.webp", "image/webp")

                elif "PDF" in formato_salida:
                    fondo_blanco = Image.new("RGB", img_procesada.size, (255, 255, 255))
                    fondo_blanco.paste(img_procesada, mask=img_procesada.split()[3])
                    buf = io.BytesIO()
                    fondo_blanco.save(buf, format="PDF", resolution=300.0)
                    st.success(f"¡PDF generado a 300 DPI ({w_cm:.2f} x {h_cm:.2f} cm)!")
                    st.download_button("📥 Descargar PDF (300 DPI)", buf.getvalue(), f"{nombre_base}_300dpi.pdf", "application/pdf")

            except Exception as e:
                st.error(f"Ocurrió un error en la conversión: {e}")
