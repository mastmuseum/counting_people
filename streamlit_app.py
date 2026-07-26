#!/usr/bin/env python3
"""
App locale (Streamlit) per il conteggio persone su una spiaggia lunga,
partendo da piu' scatti drone che si sovrappongono parzialmente.

COME SI USA:
    1. Installazione (una sola volta):
       pip install streamlit ultralytics opencv-python numpy --break-system-packages

    2. Avvio:
       streamlit run app_conteggio_spiaggia.py

    3. Si apre una pagina nel browser: carica le foto NELL'ORDINE in cui
       sono state scattate lungo la spiaggia (es. da sinistra verso destra),
       premi "Analizza spiaggia" e aspetta.

COSA FA:
    - Per ogni foto: rileva persone isolate (YOLO), ombrelloni e zone di
      folla densa (CV classica), come lo script conta_persone_ibrido.py.
    - Tra una foto e la successiva: trova automaticamente quanto si
      sovrappongono (feature matching ORB) ed esclude dal conteggio della
      foto successiva l'area gia' coperta dalla precedente, cosi' ogni
      persona/ombrellone viene contato una sola volta lungo tutta la
      spiaggia anche se appare in piu' scatti.

LIMITI DA CONOSCERE:
    - Il matching automatico funziona bene se le foto hanno abbastanza
      "riferimenti visivi" in comune (ombrelloni, dune, barche...): se due
      scatti si toccano solo su acqua o sabbia vuota puo' fallire - in tal
      caso l'app te lo segnala e non applica la deduplica per quella coppia.
    - Il conteggio di ombrelloni/folla densa resta una STIMA (vedi le note
      nello script conta_persone_ibrido.py): utile per un ordine di
      grandezza, non per un dato ufficiale preciso.
"""

import io
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

try:
    from ultralytics import YOLO
except ImportError:
    st.error("Manca 'ultralytics'. Installa con: pip install ultralytics --break-system-packages")
    st.stop()

APP_NAME = "Counting Things — Beta 1"

# Contesti disponibili: per ognuno definiamo le classi COCO da rilevare,
# se applicare la stima per oggetti occlusi/densi, e che tipo di maschera
# di sfondo usare per individuare quegli oggetti occlusi.
# id classi COCO utili: 0=persona, 1=bici, 2=auto, 3=moto, 5=bus, 7=camion, 8=barca
CONTESTI = {
    "🏖️ Spiaggia": {
        "classi": [0],
        "usa_stima_oggetti_occlusi": True,
        "tipo_maschera_scena": "spiaggia",   # usa le maschere colore acqua/sabbia/vegetazione
        "etichetta_oggetto_occluso": "ombrellone",
        "etichetta_diretto": "persone",
    },
    "🏙️ Assembramenti urbani": {
        "classi": [0],
        "usa_stima_oggetti_occlusi": True,
        "tipo_maschera_scena": "generico",   # usa la densita' di bordi, non colori specifici
        "etichetta_oggetto_occluso": "gruppo compatto",
        "etichetta_diretto": "persone",
    },
    "🚗 Auto parcheggiate": {
        "classi": [2, 3, 5, 7],
        "usa_stima_oggetti_occlusi": False,
        "tipo_maschera_scena": None,
        "etichetta_oggetto_occluso": "gruppo veicoli",
        "etichetta_diretto": "veicoli",
    },
    "❓ Altro (personalizzato)": {
        "classi": None,          # scelto dall'utente nell'interfaccia
        "usa_stima_oggetti_occlusi": None,   # scelto dall'utente nell'interfaccia
        "tipo_maschera_scena": "generico",
        "etichetta_oggetto_occluso": "oggetto occluso",
        "etichetta_diretto": "oggetti",
    },
}

# Classi COCO comuni selezionabili nel contesto "Altro (personalizzato)"
COCO_CLASSI_DISPONIBILI = {
    "persona": 0, "bicicletta": 1, "auto": 2, "moto": 3, "autobus": 5,
    "camion": 7, "barca": 8, "uccello": 14, "gatto": 15, "cane": 16,
    "cavallo": 17, "pecora": 18, "mucca": 19,
}

st.set_page_config(page_title=APP_NAME, layout="wide")


# ---------------------------------------------------------------------------
# Modello YOLO (caricato una sola volta, restra in cache tra le run)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(model_name="yolov8n.pt"):
    return YOLO(model_name)


# ---------------------------------------------------------------------------
# Rilevamento diretto con tiling
# ---------------------------------------------------------------------------

def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedup_boxes(boxes, iou_threshold=0.4):
    boxes_sorted = sorted(boxes, key=lambda b: b["conf"], reverse=True)
    kept = []
    for b in boxes_sorted:
        if not any(iou(b["box"], k["box"]) > iou_threshold for k in kept):
            kept.append(b)
    return kept


def detect_objects_tiled(model, image, target_classes, tile_size=640, overlap=120, conf=0.15):
    h, w = image.shape[:2]
    step = tile_size - overlap
    all_boxes = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            y1, x1 = max(0, y2 - tile_size), max(0, x2 - tile_size)
            tile = image[y1:y2, x1:x2]
            if tile.size == 0:
                continue
            results = model.predict(tile, classes=target_classes, conf=conf, verbose=False)
            for r in results:
                for b in r.boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    all_boxes.append({
                        "box": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                        "conf": float(b.conf[0]),
                    })
    return dedup_boxes(all_boxes)


# ---------------------------------------------------------------------------
# Maschere di scena + rilevamento ombrelloni / folla densa
# ---------------------------------------------------------------------------

def build_scene_masks(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    water_mask = cv2.inRange(hsv, (75, 40, 40), (130, 255, 255))
    veg_mask = cv2.inRange(hsv, (35, 40, 20), (85, 255, 200))
    sand_mask = cv2.inRange(hsv, (0, 0, 140), (180, 60, 255))
    kernel = np.ones((5, 5), np.uint8)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_CLOSE, kernel)
    sand_mask = cv2.morphologyEx(sand_mask, cv2.MORPH_CLOSE, kernel)
    return water_mask, sand_mask, veg_mask


def build_generic_candidate_mask(image_bgr):
    """
    Alternativa a build_scene_masks per contesti diversi dalla spiaggia
    (es. assembramenti urbani), dove non possiamo assumere colori fissi
    per lo sfondo (niente 'acqua blu' o 'sabbia beige' garantiti).
    Usa invece la densita' di bordi locale: marciapiedi/strade/tetti sono
    zone lisce con pochi bordi, una folla/oggetti densi hanno molta piu'
    texture (contorni di persone, vestiti, veicoli...). Ritorna la
    maschera dei "candidati" (aree con texture, potenzialmente oggetti).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    density = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), sigmaX=9)
    candidate = (density > 12).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    return candidate


def mask_out_boxes(mask, boxes, pad=4):
    out = mask.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["box"]]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(mask.shape[1], x2 + pad), min(mask.shape[0], y2 + pad)
        out[y1:y2, x1:x2] = 0
    return out


def split_touching_blobs(mask, typical_radius):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    local_max_thresh = max(2.0, typical_radius * 0.5)
    sure_fg = (dist > local_max_thresh).astype(np.uint8) * 255
    n_labels, markers = cv2.connectedComponents(sure_fg)
    if n_labels <= 1:
        return None
    markers = markers + 1
    unknown = cv2.subtract(mask, sure_fg)
    markers[unknown == 255] = 0
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.watershed(mask_3ch, markers.astype(np.int32))


def _classify_blob_local(image_bgr, contour, x, y, w, h, area, umbrella_min_area, umbrella_max_area, structure_min_area):
    """
    Classifica un singolo blob lavorando solo sul suo ritaglio locale
    (bounding box), non sull'intera immagine: molto piu' leggero in
    memoria/tempo quando ci sono migliaia di piccoli blob su foto grandi.
    """
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    aspect_ratio = max(w, h) / max(1, min(w, h))

    crop = image_bgr[y:y + h, x:x + w]
    local_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour - np.array([[x, y]])
    cv2.drawContours(local_mask, [shifted], -1, 255, -1)
    _, stddev = cv2.meanStdDev(crop, mask=local_mask)
    color_variance = float(np.mean(stddev))

    if area >= structure_min_area and aspect_ratio > 2.5:
        return None
    if (umbrella_min_area * 0.5 <= area <= umbrella_max_area * 1.3
            and circularity > 0.4 and aspect_ratio < 2.0 and color_variance < 50):
        return "umbrella"
    return "crowd"


def classify_blobs(image_bgr, candidate_mask, avg_person_area, umbrella_min_mult, umbrella_max_mult):
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    umbrella_min_area = avg_person_area * umbrella_min_mult
    umbrella_max_area = avg_person_area * umbrella_max_mult
    typical_radius = np.sqrt(((umbrella_min_area + umbrella_max_area) / 2) / np.pi)
    structure_min_area = avg_person_area * 60

    img_h, img_w = mask_clean.shape[:2]
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    umbrellas, crowd_blobs = [], []

    for c in contours:
        area = cv2.contourArea(c)
        if area < avg_person_area * 0.3:
            continue
        x, y, w, h = cv2.boundingRect(c)

        if area > umbrella_max_area * 1.8:
            # blob grande: lo separo lavorando su un ritaglio locale (con
            # un margine) invece che sull'intera immagine
            pad = 5
            cx1, cy1 = max(0, x - pad), max(0, y - pad)
            cx2, cy2 = min(img_w, x + w + pad), min(img_h, y + h + pad)
            local_w, local_h = cx2 - cx1, cy2 - cy1

            sub_mask = np.zeros((local_h, local_w), dtype=np.uint8)
            shifted_c = c - np.array([[cx1, cy1]])
            cv2.drawContours(sub_mask, [shifted_c], -1, 255, -1)
            markers = split_touching_blobs(sub_mask, typical_radius)

            found_any = False
            if markers is not None and markers.max() > 2:
                for label in range(2, markers.max() + 1):
                    obj_mask = np.uint8(markers == label) * 255
                    sub_c, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not sub_c:
                        continue
                    sc = max(sub_c, key=cv2.contourArea)
                    s_area = cv2.contourArea(sc)
                    if s_area < avg_person_area * 0.3:
                        continue
                    sx, sy, sw, sh = cv2.boundingRect(sc)
                    # riporto le coordinate nel sistema dell'immagine intera
                    sc_global = sc + np.array([[cx1, cy1]])
                    gx, gy = sx + cx1, sy + cy1
                    kind = _classify_blob_local(image_bgr, sc_global, gx, gy, sw, sh, s_area,
                                                 umbrella_min_area, umbrella_max_area, structure_min_area)
                    if kind == "umbrella":
                        umbrellas.append({"area": s_area, "bbox": (gx, gy, sw, sh)})
                        found_any = True
                    elif kind == "crowd":
                        crowd_blobs.append({"area": s_area, "bbox": (gx, gy, sw, sh)})
            if not found_any:
                crowd_blobs.append({"area": area, "bbox": (x, y, w, h)})
            continue

        kind = _classify_blob_local(image_bgr, c, x, y, w, h, area,
                                     umbrella_min_area, umbrella_max_area, structure_min_area)
        if kind == "umbrella":
            umbrellas.append({"area": area, "bbox": (x, y, w, h)})
        elif kind == "crowd":
            crowd_blobs.append({"area": area, "bbox": (x, y, w, h)})

    return umbrellas, crowd_blobs


# ---------------------------------------------------------------------------
# Sovrapposizione tra scatti consecutivi (ORB + omografia)
# ---------------------------------------------------------------------------

def estimate_overlap_mask(img_prev, img_curr, min_good_matches=15, min_inliers=15):
    """
    Trova quanto img_curr si sovrappone a img_prev, usando punti
    caratteristici comuni (ombrelloni, dune, barche, bordo dell'acqua...).
    Ritorna (maschera_overlap_in_curr, percentuale_overlap) oppure
    (None, None) se non trova abbastanza riferimenti in comune.
    """
    orb = cv2.ORB_create(4000)
    gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_raw = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in matches_raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < min_good_matches:
        return None, None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, None
    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < min_inliers:
        return None, None

    h_prev, w_prev = img_prev.shape[:2]
    corners_prev = np.float32([[0, 0], [w_prev, 0], [w_prev, h_prev], [0, h_prev]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners_prev, H)

    overlap_mask = np.zeros(img_curr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(overlap_mask, np.int32(warped), 255)
    # limito ai confini reali dell'immagine corrente
    bounds = np.zeros(img_curr.shape[:2], dtype=np.uint8)
    bounds[:] = 255
    overlap_mask = cv2.bitwise_and(overlap_mask, bounds)

    pct = 100.0 * np.sum(overlap_mask > 0) / (img_curr.shape[0] * img_curr.shape[1])
    return overlap_mask, pct


# ---------------------------------------------------------------------------
# Analisi completa di una foto (con esclusione opzionale area gia' contata)
# ---------------------------------------------------------------------------

def analyze_photo(model, image, params, contesto, exclusion_mask=None):
    direct_boxes = detect_objects_tiled(
        model, image, contesto["classi"],
        tile_size=params["tile_size"], overlap=params["tile_overlap"], conf=params["conf"]
    )

    if exclusion_mask is not None:
        kept = []
        for b in direct_boxes:
            cx = int((b["box"][0] + b["box"][2]) / 2)
            cy = int((b["box"][1] + b["box"][3]) / 2)
            if 0 <= cy < exclusion_mask.shape[0] and 0 <= cx < exclusion_mask.shape[1] and exclusion_mask[cy, cx] > 0:
                continue  # gia' contato nello scatto precedente
            kept.append(b)
        direct_boxes = kept

    if direct_boxes:
        areas = [(b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]) for b in direct_boxes]
        avg_obj_area = float(np.median(areas))
        calib_affidabile = True
    else:
        h, w = image.shape[:2]
        avg_obj_area = (min(h, w) / 80) ** 2
        calib_affidabile = False

    n_umbrellas, persone_ombrelloni, persone_folla = 0, 0.0, 0.0
    umbrellas, crowd_blobs = [], []

    if contesto["usa_stima_oggetti_occlusi"]:
        # Rilevante per persone/oggetti che possono essere nascosti da altri
        # elementi (ombrelloni su spiaggia, corpi ravvicinati in una folla
        # urbana...). Per i veicoli in parcheggio questo passaggio non serve.
        if contesto.get("tipo_maschera_scena") == "spiaggia":
            water_mask, sand_mask, veg_mask = build_scene_masks(image)
            non_scene = cv2.bitwise_not(cv2.bitwise_or(cv2.bitwise_or(water_mask, sand_mask), veg_mask))
        else:
            # contesto generico (es. urbano): non assumo colori fissi di
            # sfondo, uso la densita' di bordi per trovare zone con oggetti
            non_scene = build_generic_candidate_mask(image)
        candidate_mask = mask_out_boxes(non_scene, direct_boxes)

        if exclusion_mask is not None:
            candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(exclusion_mask))

        umbrellas, crowd_blobs = classify_blobs(
            image, candidate_mask, avg_obj_area,
            params["ombrellone_min_mult"], params["ombrellone_max_mult"]
        )
        n_umbrellas = len(umbrellas)
        persone_ombrelloni = n_umbrellas * params["persone_per_ombrellone"]
        area_folla = sum(b["area"] for b in crowd_blobs)
        persone_folla = (area_folla / avg_obj_area) * params["fattore_affollamento"]

    totale = len(direct_boxes) + persone_ombrelloni + persone_folla

    annotated = image.copy()
    for b in direct_boxes:
        x1, y1, x2, y2 = [int(v) for v in b["box"]]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for u in umbrellas:
        x, y, w, h = u["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 140, 0), 2)
    for c in crowd_blobs:
        x, y, w, h = c["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if exclusion_mask is not None:
        overlay = annotated.copy()
        overlay[exclusion_mask > 0] = (180, 180, 180)
        annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

    return {
        "diretti": len(direct_boxes),
        "ombrelloni": n_umbrellas,
        "persone_ombrelloni": persone_ombrelloni,
        "persone_folla_densa": persone_folla,
        "totale": totale,
        "annotated": annotated,
        "calibrazione_affidabile": calib_affidabile,
    }


# ---------------------------------------------------------------------------
# Split automatico per foto molto grandi (alta risoluzione = piu' dettaglio,
# ma anche piu' memoria: qui la gestiamo spezzando la foto in tile a griglia
# fissa con overlap noto, cosi' ogni tile e' analizzato a piena risoluzione
# nativa, senza mai ridimensionare/comprimere l'immagine originale)
# ---------------------------------------------------------------------------

def split_image_grid(image, max_dim, overlap_frac=0.15):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return [(0, 0, image)]

    step = max(1, int(max_dim * (1 - overlap_frac)))
    ys = list(range(0, max(1, h - max_dim) + 1, step)) or [0]
    xs = list(range(0, max(1, w - max_dim) + 1, step)) or [0]
    if ys[-1] + max_dim < h:
        ys.append(h - max_dim)
    if xs[-1] + max_dim < w:
        xs.append(w - max_dim)

    tiles, seen = [], set()
    for y in ys:
        for x in xs:
            y1, x1 = max(0, min(y, h - max_dim)), max(0, min(x, w - max_dim))
            y2, x2 = min(y1 + max_dim, h), min(x1 + max_dim, w)
            key = (x1, y1, x2, y2)
            if key in seen:
                continue
            seen.add(key)
            tiles.append((x1, y1, image[y1:y2, x1:x2]))
    return tiles


def analyze_uploaded_photo(model, image, params, contesto, max_dim, exclusion_mask_prev=None):
    """
    Analizza una foto caricata dall'utente. Se e' piu' grande di max_dim su
    entrambi i lati, la spezza automaticamente in tile a piena risoluzione
    (nessun ridimensionamento/perdita di dettaglio) e ne combina i risultati,
    escludendo dal conteggio dei tile successivi le aree gia' coperte dai
    precedenti - la stessa logica usata per dedupllicare piu' scatti diversi,
    ma qui applicata internamente con coordinate esatte (niente feature
    matching necessario, conosciamo gia' la geometria esatta dello split).
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return analyze_photo(model, image, params, contesto, exclusion_mask=exclusion_mask_prev), 1

    tiles = split_image_grid(image, max_dim=max_dim)
    coverage_mask = np.zeros((h, w), dtype=np.uint8)
    annotated_full = image.copy()

    agg = {"diretti": 0, "ombrelloni": 0, "persone_ombrelloni": 0.0,
           "persone_folla_densa": 0.0, "totale": 0.0, "calibrazione_affidabile": False}

    for (x1, y1, tile_img) in tiles:
        th, tw = tile_img.shape[:2]
        local_excl = coverage_mask[y1:y1 + th, x1:x1 + tw].copy()
        if exclusion_mask_prev is not None:
            local_excl = cv2.bitwise_or(local_excl, exclusion_mask_prev[y1:y1 + th, x1:x1 + tw])

        r = analyze_photo(model, tile_img, params, contesto, exclusion_mask=local_excl)

        agg["diretti"] += r["diretti"]
        agg["ombrelloni"] += r["ombrelloni"]
        agg["persone_ombrelloni"] += r["persone_ombrelloni"]
        agg["persone_folla_densa"] += r["persone_folla_densa"]
        agg["totale"] += r["totale"]
        agg["calibrazione_affidabile"] = agg["calibrazione_affidabile"] or r["calibrazione_affidabile"]

        annotated_full[y1:y1 + th, x1:x1 + tw] = r["annotated"]
        coverage_mask[y1:y1 + th, x1:x1 + tw] = 255

    agg["annotated"] = annotated_full
    return agg, len(tiles)


# ---------------------------------------------------------------------------
# INTERFACCIA
# ---------------------------------------------------------------------------

st.title("Counting Things")
st.markdown(
    """
**Come funziona, in 3 passi:**
1. 🎯 Scegli cosa contare e dai un nome al progetto
2. 📤 Carica le tue foto drone, **nell'ordine in cui le hai scattate**
3. 🔍 Premi "Analizza" e guarda il totale stimato

L'app trova da sola dove due foto consecutive si sovrappongono e non conta gli oggetti due volte.
Le foto molto grandi vengono spezzate automaticamente in tile a piena risoluzione: la qualità/dettaglio non è mai il limite.
"""
)

col_ctx, col_nome = st.columns([1, 1])
with col_ctx:
    nome_contesto = st.radio("🎯 Cosa vuoi contare?", list(CONTESTI.keys()), horizontal=False)
    contesto = dict(CONTESTI[nome_contesto])  # copia, la personalizziamo se serve
with col_nome:
    nome_progetto = st.text_input("📁 Nome del progetto", value="", placeholder="es. La Cinta - Ferragosto 2026")

if nome_contesto == "❓ Altro (personalizzato)":
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        classi_scelte = st.multiselect(
            "Cosa deve riconoscere il modello?",
            list(COCO_CLASSI_DISPONIBILI.keys()), default=["persona"]
        )
    with col_c2:
        occlusione_custom = st.checkbox(
            "Stima anche oggetti occlusi/ammassati", value=True,
            help="Attiva se gli oggetti possono nascondersi a vicenda (es. folla). "
                 "Disattiva se sono sempre ben visibili singolarmente (es. veicoli isolati)."
        )
    contesto["classi"] = [COCO_CLASSI_DISPONIBILI[c] for c in classi_scelte] or [0]
    contesto["usa_stima_oggetti_occlusi"] = occlusione_custom
    contesto["etichetta_diretto"] = ", ".join(classi_scelte) if classi_scelte else "oggetti"

etichetta = contesto["etichetta_diretto"]

with st.expander("⚙️ Impostazioni avanzate (opzionale — i valori di default vanno bene per iniziare)"):
    conf = st.slider("Sensibilità rilevamento diretto", 0.05, 0.5, 0.15, 0.05,
                      help=f"Più basso = rileva più {etichetta} ma anche più falsi positivi")
    if contesto["usa_stima_oggetti_occlusi"]:
        persone_per_ombrellone = st.slider(f"Oggetti medi per {contesto['etichetta_oggetto_occluso']}", 1.0, 5.0, 3.0, 0.5)
        fattore_affollamento = st.slider("Fattore stima zone dense", 0.1, 1.0, 0.55, 0.05,
                                          help="Aumenta se il totale ti sembra sottostimato, riduci se sovrastimato")
        col_a, col_b = st.columns(2)
        with col_a:
            ombrellone_min_mult = st.number_input(f"Area minima {contesto['etichetta_oggetto_occluso']} (x area oggetto)", value=3.0)
            tile_size = st.number_input("Dimensione tile YOLO (px)", value=640, step=64)
        with col_b:
            ombrellone_max_mult = st.number_input(f"Area massima {contesto['etichetta_oggetto_occluso']} (x area oggetto)", value=22.0)
            tile_overlap = st.number_input("Overlap tile YOLO (px)", value=120, step=20)
    else:
        persone_per_ombrellone, fattore_affollamento = 0.0, 0.0
        ombrellone_min_mult, ombrellone_max_mult = 3.0, 22.0
        tile_size = st.number_input("Dimensione tile YOLO (px)", value=640, step=64)
        tile_overlap = st.number_input("Overlap tile YOLO (px)", value=120, step=20)
        st.caption(f"ℹ️ Per questo contesto non serve la stima di oggetti occlusi: sono direttamente visibili dall'alto.")

    st.divider()
    st.caption("📐 Gestione foto ad alta risoluzione")
    max_dim_split = st.number_input(
        "Dimensione massima prima dello split automatico (px, lato più lungo)",
        value=3000, step=250,
        help="Foto più grandi di questo valore vengono spezzate internamente in tile a piena "
             "risoluzione per stare dentro la memoria disponibile, senza perdere dettaglio. "
             "Riducilo se l'app va in errore di memoria, alzalo se hai risorse disponibili."
    )

params = dict(conf=conf, persone_per_ombrellone=persone_per_ombrellone,
              fattore_affollamento=fattore_affollamento,
              ombrellone_min_mult=ombrellone_min_mult, ombrellone_max_mult=ombrellone_max_mult,
              tile_size=int(tile_size), tile_overlap=int(tile_overlap))

st.divider()
uploaded_files = st.file_uploader(
    f"📤 Carica qui le foto (in ordine di scatto)",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} foto caricate, in quest'ordine: "
               + ", ".join(f"{i+1}. {f.name}" for i, f in enumerate(uploaded_files)))

analizza = st.button("Start", type="primary", use_container_width=True,
                      disabled=not uploaded_files)

if analizza:
    model = load_model()

    photos = []
    for f in uploaded_files:
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        photos.append((f.name, img))

    progress = st.progress(0.0, text="Analisi in corso...")

    results = []
    prev_img = None
    for i, (name, img) in enumerate(photos):
        exclusion_mask = None
        overlap_pct = 0.0
        overlap_ok = True
        if prev_img is not None:
            exclusion_mask, pct = estimate_overlap_mask(prev_img, img)
            if exclusion_mask is None:
                overlap_ok = False
                overlap_pct = 0.0
            else:
                overlap_pct = pct

        r, n_tile = analyze_uploaded_photo(model, img, params, contesto, int(max_dim_split),
                                            exclusion_mask_prev=exclusion_mask)
        r["nome"] = name
        r["overlap_con_precedente_pct"] = overlap_pct
        r["overlap_rilevato"] = overlap_ok
        r["n_tile_split"] = n_tile
        results.append(r)

        prev_img = img
        testo_tile = f" (foto spezzata in {n_tile} tile per l'alta risoluzione)" if n_tile > 1 else ""
        progress.progress((i + 1) / len(photos), text=f"Analizzata {i+1}/{len(photos)}: {name}{testo_tile}")

    progress.empty()

    totale_finale = sum(r["totale"] for r in results)

    # Salvo in sessione (serve per il pulsante "Salva progetto" qui sotto,
    # cosi' resta disponibile anche dopo il refresh del pulsante Streamlit)
    st.session_state["ultimo_risultato"] = {
        "nome_progetto": nome_progetto or "Progetto senza nome",
        "contesto": nome_contesto,
        "data": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "totale": totale_finale,
        "n_foto": len(results),
        "dettaglio": [
            {"file": r["nome"], "diretti": r["diretti"], "ombrelloni": r["ombrelloni"],
             "persone_ombrelloni": round(r["persone_ombrelloni"], 1),
             "persone_folla_densa": round(r["persone_folla_densa"], 1),
             "totale": round(r["totale"], 1),
             "overlap_precedente_pct": round(r["overlap_con_precedente_pct"], 1)}
            for r in results
        ],
        "parametri": params,
    }

    st.divider()
    st.markdown(f"## 🎯 Totale stimato — {nome_progetto or 'progetto senza nome'}: **{totale_finale:.0f} {etichetta}**")

    with st.expander("Vedi il dettaglio del calcolo"):
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{etichetta.capitalize()} rilevati direttamente", sum(r["diretti"] for r in results))
        if contesto["usa_stima_oggetti_occlusi"]:
            c2.metric(f"Stima da {contesto['etichetta_oggetto_occluso']}", f"{sum(r['persone_ombrelloni'] for r in results):.0f}")
            c3.metric("Stima zone dense", f"{sum(r['persone_folla_densa'] for r in results):.0f}")

    any_overlap_failed = any(not r["overlap_rilevato"] for r in results[1:])
    if any_overlap_failed:
        st.warning(
            "⚠️ Per almeno una coppia di scatti consecutivi non ho trovato abbastanza riferimenti "
            "visivi in comune per calcolare la sovrapposizione automaticamente. Per quelle foto il "
            "conteggio potrebbe includere un doppio conteggio nella zona di sovrapposizione reale."
        )

    st.divider()
    with st.expander(f"📷 Dettaglio delle {len(results)} foto analizzate"):
        for r in results:
            st.subheader(r["nome"])
            cols = st.columns([2, 1])
            with cols[0]:
                st.image(cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB), use_container_width=True)
            with cols[1]:
                st.write(f"**Diretti (YOLO):** {r['diretti']}")
                if contesto["usa_stima_oggetti_occlusi"]:
                    st.write(f"**{contesto['etichetta_oggetto_occluso'].capitalize()} rilevati:** "
                             f"{r['ombrelloni']} (~{r['persone_ombrelloni']:.0f} {etichetta})")
                    st.write(f"**Zone dense (stima):** ~{r['persone_folla_densa']:.0f} {etichetta}")
                st.write(f"**Totale foto:** {r['totale']:.0f}")
                if r.get("n_tile_split", 1) > 1:
                    st.caption(f"🧩 Foto ad alta risoluzione: spezzata automaticamente in "
                               f"{r['n_tile_split']} tile per l'analisi, poi ricombinata.")
                if r["overlap_con_precedente_pct"] > 0:
                    st.caption(f"Sovrapposizione con lo scatto precedente: {r['overlap_con_precedente_pct']:.0f}% "
                               f"(area già esclusa dal conteggio, mostrata in grigio)")
                elif not r["overlap_rilevato"]:
                    st.caption("⚠️ Sovrapposizione con lo scatto precedente non rilevabile automaticamente")
                if not r["calibrazione_affidabile"]:
                    st.caption("⚠️ Nessun rilevamento diretto per calibrare le stime su questa foto: "
                               "numeri meno affidabili del solito")

    st.divider()
    csv_lines = ["file,diretti,ombrelloni,persone_ombrelloni,persone_folla_densa,totale,overlap_precedente_pct"]
    for r in results:
        csv_lines.append(f"{r['nome']},{r['diretti']},{r['ombrelloni']},{r['persone_ombrelloni']:.1f},"
                          f"{r['persone_folla_densa']:.1f},{r['totale']:.1f},{r['overlap_con_precedente_pct']:.1f}")
    csv_data = "\n".join(csv_lines)
    st.download_button("📥 Scarica riepilogo CSV", csv_data, file_name="riepilogo.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Salvataggio / caricamento progetti con nome
# ---------------------------------------------------------------------------
st.divider()
st.header("💾 Progetti salvati")
st.caption(
    "L'app gira su hosting gratuito e non ha un database permanente: il 'salvataggio' "
    "scarica un file di progetto (numeri e parametri, non le foto) che puoi tenere sul "
    "tuo computer e ricaricare in futuro per rivedere i risultati o confrontarli."
)

col_save, col_load = st.columns(2)

with col_save:
    st.subheader("Salva l'ultima analisi")
    if "ultimo_risultato" in st.session_state:
        payload = json.dumps(st.session_state["ultimo_risultato"], ensure_ascii=False, indent=2)
        nome_file_sicuro = "".join(c if c.isalnum() or c in "-_ " else "_" for c in
                                    st.session_state["ultimo_risultato"]["nome_progetto"]).strip() or "progetto"
        st.download_button(
            f"📥 Scarica progetto '{st.session_state['ultimo_risultato']['nome_progetto']}'",
            payload, file_name=f"{nome_file_sicuro}.json", mime="application/json"
        )
    else:
        st.caption("Esegui prima un'analisi per poterla salvare.")

with col_load:
    st.subheader("Carica un progetto salvato")
    progetto_file = st.file_uploader("Carica il file .json del progetto", type=["json"], key="carica_progetto")
    if progetto_file:
        dati = json.load(progetto_file)
        st.markdown(f"### {dati['nome_progetto']}")
        st.caption(f"Contesto: {dati['contesto']} — Analizzato il {dati['data']}")
        st.metric("Totale", f"{dati['totale']:.0f}")
        st.write(f"Foto analizzate: {dati['n_foto']}")
        with st.expander("Dettaglio per foto"):
            for d in dati["dettaglio"]:
                st.write(f"**{d['file']}** — totale: {d['totale']} "
                         f"(diretti: {d['diretti']}, ombrelloni: {d.get('persone_ombrelloni', 0)}, "
                         f"folla: {d.get('persone_folla_densa', 0)})")
