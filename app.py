import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import time
import requests
import os
from datetime import datetime
import threading

# ---------------------------------------
# INTENTO DE IMPORTAR PYTORCH (PARTE 1-B)
# ---------------------------------------
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("✅ PyTorch detectado: el filtro avanzado estará disponible.")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch NO está instalado. El filtro PyTorch usará un blur Gaussiano como respaldo.")

app = Flask(__name__)

# ==================================================
# CONFIGURA AQUÍ LA IP DE LA ESP32-CAM (ENDPOINT /capture)
# ==================================================
ESP32_URL = "http://192.168.1.77/capture"

# ============================
# CARPETAS DE CAPTURAS (1-A)
# ============================
BASE_CAPTURAS = "capturas"
CATEGORIAS = ["iluminacion", "distancia", "orientacion", "fondo", "enfoque"]

if not os.path.exists(BASE_CAPTURAS):
    os.makedirs(BASE_CAPTURAS)

for cat in CATEGORIAS:
    ruta = os.path.join(BASE_CAPTURAS, cat)
    if not os.path.exists(ruta):
        os.makedirs(ruta)

# ============================
# BACKGROUND SUBTRACTOR (1-A)
# ============================
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=True
)

# ============================
# FRAME GLOBAL (PARTE 1-A y 1-B)
# ============================
latest_frame = None
frame_lock = threading.Lock()

# -----------------------------------------------
# HILO QUE VA PIDIENDO FRAMES A /capture
# -----------------------------------------------
def capture_loop():
    global latest_frame
    print("🔁 Hilo de captura iniciado (ESP32_URL =", ESP32_URL, ")")

    while True:
        try:
            resp = requests.get(ESP32_URL, timeout=3)
            img_array = np.frombuffer(resp.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is not None:
                with frame_lock:
                    latest_frame = frame
        except Exception as e:
            print(f"❌ ERROR en capture_loop: {e}")
            time.sleep(1)

        time.sleep(0.1)


# -----------------------------------------------
# PROCESAMIENTO 1-A
# -----------------------------------------------
def procesar_frame_1a(frame, tipo):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = bg_subtractor.apply(frame)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    hist_eq = cv2.equalizeHist(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    gamma_val = 1.5
    gamma_lut = np.array([((i / 255.0) ** (1.0 / gamma_val)) * 255
                          for i in range(256)]).astype("uint8")
    gamma_img = cv2.LUT(gray, gamma_lut)

    if tipo == "original":
        return frame
    if tipo == "mask":
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if tipo == "foreground":
        return foreground
    if tipo == "hist":
        return cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)
    if tipo == "clahe":
        return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    if tipo == "gamma":
        return cv2.cvtColor(gamma_img, cv2.COLOR_GRAY2BGR)

    return frame


# -----------------------------------------------
# GENERADOR DE VIDEO 1-A
# -----------------------------------------------
def generate_frames_1a(tipo):
    prev_time = time.time()

    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.1)
            continue

        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out = procesar_frame_1a(frame, tipo)

        ret, buffer = cv2.imencode('.jpg', out)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        time.sleep(0.05)


# -----------------------------------------------
# STREAMING 1-A
# -----------------------------------------------
@app.route("/video/<tipo>")
def video(tipo):
    return Response(generate_frames_1a(tipo),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# -----------------------------------------------
# CAPTURAS INDIVIDUALES (1-A)
# -----------------------------------------------
@app.route("/capturar", methods=["POST"])
def capturar():
    data = request.get_json()
    tipo = data.get("tipo", "original")
    categoria = data.get("categoria", "iluminacion")
    detalle = data.get("detalle", "").strip()

    if categoria not in CATEGORIAS:
        return jsonify({"status": "error", "msg": "Categoría inválida."})

    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()

    if frame is None:
        return jsonify({"status": "error", "msg": "No hay frame disponible."})

    procesada = procesar_frame_1a(frame, tipo)

    carpeta = os.path.join(BASE_CAPTURAS, categoria)
    os.makedirs(carpeta, exist_ok=True)

    base = detalle + "_" + tipo if detalle else tipo
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = os.path.join(carpeta, f"{base}_{ts}.jpg")

    cv2.imwrite(nombre, procesada)

    return jsonify({"status": "ok", "msg": f"Captura guardada: {nombre}"})


# -----------------------------------------------
# CAPTURAR TODOS (1-A)
# -----------------------------------------------
@app.route("/capturar_all", methods=["POST"])
def capturar_all():
    data = request.get_json()
    categoria = data.get("categoria", "iluminacion")
    detalle = data.get("detalle", "").strip()

    if categoria not in CATEGORIAS:
        return jsonify({"status": "error", "msg": "Categoría inválida."})

    with frame_lock:
        frame = None if latest_frame is None else latest_frame.copy()

    if frame is None:
        return jsonify({"status": "error", "msg": "No hay frame disponible."})

    tipos = ["original", "mask", "foreground", "hist", "clahe", "gamma"]
    carpeta = os.path.join(BASE_CAPTURAS, categoria)
    os.makedirs(carpeta, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    guardados = []

    for t in tipos:
        procesada = procesar_frame_1a(frame, t)
        base = detalle + "_" + t if detalle else t
        nombre = os.path.join(carpeta, f"{base}_{ts}.jpg")
        cv2.imwrite(nombre, procesada)
        guardados.append(nombre)

    return jsonify({"status": "ok", "guardados": guardados})


# ==================================================
# ================  PARTE 1-B  ======================
# Ruido Gaussiano / Speckle + Filtros + PyTorch
# ==================================================

params_b = {
    "ruido": "gauss",
    "filtro": "median",
    "gauss_media": 0.0,
    "gauss_sigma": 20.0,
    "speckle_var": 0.1,
    "kernel_size": 3
}

# --------------------------------------------------
# FUNCIÓN ROBUSTA (ARREGLA TU ERROR)
# --------------------------------------------------
@app.route("/update_params_b", methods=["POST"])
def update_params_b():
    data = request.get_json()

    def to_float(x):
        try: return float(x)
        except: return 0.0

    def to_int(x):
        try: return int(x)
        except: return 3

    if "ruido" in data:
        params_b["ruido"] = str(data["ruido"])

    if "filtro" in data:
        params_b["filtro"] = str(data["filtro"])

    if "gauss_media" in data:
        params_b["gauss_media"] = to_float(data["gauss_media"])

    if "gauss_sigma" in data:
        params_b["gauss_sigma"] = to_float(data["gauss_sigma"])

    if "speckle_var" in data:
        params_b["speckle_var"] = to_float(data["speckle_var"])

    if "kernel_size" in data:
        k = to_int(data["kernel_size"])
        if k < 3: k = 3
        if k % 2 == 0: k += 1
        params_b["kernel_size"] = k

    return jsonify({"status": "ok", "params": params_b})


# --------------------------------------------------
# Ruido Gaussiano / Speckle
# --------------------------------------------------
def aplicar_ruido(frame, p):

    img = frame.astype(np.float32) / 255.0

    out = img.copy()

    if p["ruido"] in ["gauss", "ambos"]:
        gauss_noise = np.random.normal(
            p["gauss_media"] / 255.0,
            p["gauss_sigma"] / 255.0,
            img.shape
        ).astype(np.float32)
        out += gauss_noise

    if p["ruido"] in ["speckle", "ambos"]:
        speckle_noise = np.random.normal(
            0.0,
            np.sqrt(p["speckle_var"]),
            img.shape
        ).astype(np.float32)
        out += img * speckle_noise

    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


# --------------------------------------------------
# Filtro PyTorch / Backup
# --------------------------------------------------
def aplicar_filtro_pytorch(noise, kernel_size):

    if not TORCH_AVAILABLE:
        return cv2.GaussianBlur(noise, (kernel_size, kernel_size), 0)

    img = noise.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    k = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.0
    weight = k.repeat(3, 1, 1, 1)

    img_f = F.conv2d(img_t, weight, padding=1, groups=3)

    out = img_f.squeeze(0).permute(1, 2, 0).detach().numpy()
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


# --------------------------------------------------
# Filtros + Bordes
# --------------------------------------------------
def aplicar_filtros_b(noise, p):

    k = p["kernel_size"]

    if p["filtro"] == "median":
        filtered = cv2.medianBlur(noise, k)
    elif p["filtro"] == "blur":
        filtered = cv2.blur(noise, (k, k))
    elif p["filtro"] == "gauss":
        filtered = cv2.GaussianBlur(noise, (k, k), 0)
    elif p["filtro"] == "pytorch":
        filtered = aplicar_filtro_pytorch(noise, k)
    else:
        filtered = noise

    # Edges
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(gray, 100, 200)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    if np.max(sobel) > 0:
        sobel = (sobel / np.max(sobel)) * 255
    sobel = sobel.astype(np.uint8)

    edges = cv2.bitwise_or(canny, sobel)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(filtered, 0.8, edges_color, 0.2, 0)


# --------------------------------------------------
# STREAMING 1-B
# --------------------------------------------------
def generate_frames_b(modo):
    prev_time = time.time()

    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        p = params_b.copy()

        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now

        original = frame.copy()
        cv2.putText(original, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        noisy = aplicar_ruido(original, p)
        filtered = aplicar_filtros_b(noisy, p)

        if modo == "original":
            out = original
        elif modo == "noise":
            out = noisy
        elif modo == "filtered":
            out = filtered
        elif modo == "filtered_noborders":
            out = aplicar_filtro_pytorch(noisy, p["kernel_size"])
        else:
            out = original

        ret, buffer = cv2.imencode('.jpg', out)
        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

        time.sleep(0.05)


@app.route("/video_b/<modo>")
def video_b(modo):
    return Response(generate_frames_b(modo),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# -----------------------------------------------
# PÁGINA PRINCIPAL
# -----------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", categorias=CATEGORIAS)


# -----------------------------------------------
# MAIN
# -----------------------------------------------
if __name__ == "__main__":
    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
