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
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

try:
    from ultralytics import YOLO
except ImportError:
    st.error("Manca 'ultralytics'. Installa con: pip install ultralytics --break-system-packages")
    st.stop()

PERSON_CLASS_ID = 0

st.set_page_config(page_title="Conteggio spiaggia da drone", layout="wide")


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


def detect_persons_tiled(model, image, tile_size=640, overlap=120, conf=0.15):
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
            results = model.predict(tile, classes=[PERSON_CLASS_ID], conf=conf, verbose=False)
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


def classify_blobs(image_bgr, candidate_mask, avg_person_area, umbrella_min_mult, umbrella_max_mult):
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    umbrella_min_area = avg_person_area * umbrella_min_mult
    umbrella_max_area = avg_person_area * umbrella_max_mult
    typical_radius = np.sqrt(((umbrella_min_area + umbrella_max_area) / 2) / np.pi)
    structure_min_area = avg_person_area * 60

    def _classify_single(area, perimeter, w, h, mask_single):
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        aspect_ratio = max(w, h) / max(1, min(w, h))
        _, stddev = cv2.meanStdDev(image_bgr, mask=mask_single)
        color_variance = float(np.mean(stddev))
        if area >= structure_min_area and aspect_ratio > 2.5:
            return None
        if (umbrella_min_area * 0.5 <= area <= umbrella_max_area * 1.3
                and circularity > 0.4 and aspect_ratio < 2.0 and color_variance < 50):
            return "umbrella"
        return "crowd"

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    umbrellas, crowd_blobs = [], []

    for c in contours:
        area = cv2.contourArea(c)
        if area < avg_person_area * 0.3:
            continue
        x, y, w, h = cv2.boundingRect(c)

        if area > umbrella_max_area * 1.8:
            sub_mask = np.zeros(mask_clean.shape, dtype=np.uint8)
            cv2.drawContours(sub_mask, [c], -1, 255, -1)
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
                    s_perim = cv2.arcLength(sc, True)
                    sx, sy, sw, sh = cv2.boundingRect(sc)
                    kind = _classify_single(s_area, s_perim, sw, sh, obj_mask)
                    if kind == "umbrella":
                        umbrellas.append({"area": s_area, "bbox": (sx, sy, sw, sh)})
                        found_any = True
                    elif kind == "crowd":
                        crowd_blobs.append({"area": s_area, "bbox": (sx, sy, sw, sh)})
            if not found_any:
                crowd_blobs.append({"area": area, "bbox": (x, y, w, h)})
            continue

        perimeter = cv2.arcLength(c, True)
        mask_single = np.zeros(mask_clean.shape, dtype=np.uint8)
        cv2.drawContours(mask_single, [c], -1, 255, -1)
        kind = _classify_single(area, perimeter, w, h, mask_single)
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

def analyze_photo(model, image, params, exclusion_mask=None):
    direct_boxes = detect_persons_tiled(
        model, image, tile_size=params["tile_size"], overlap=params["tile_overlap"], conf=params["conf"]
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
        avg_person_area = float(np.median(areas))
        calib_affidabile = True
    else:
        h, w = image.shape[:2]
        avg_person_area = (min(h, w) / 80) ** 2
        calib_affidabile = False

    water_mask, sand_mask, veg_mask = build_scene_masks(image)
    non_scene = cv2.bitwise_not(cv2.bitwise_or(cv2.bitwise_or(water_mask, sand_mask), veg_mask))
    candidate_mask = mask_out_boxes(non_scene, direct_boxes)

    if exclusion_mask is not None:
        candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(exclusion_mask))

    umbrellas, crowd_blobs = classify_blobs(
        image, candidate_mask, avg_person_area,
        params["ombrellone_min_mult"], params["ombrellone_max_mult"]
    )

    n_umbrellas = len(umbrellas)
    persone_ombrelloni = n_umbrellas * params["persone_per_ombrellone"]
    area_folla = sum(b["area"] for b in crowd_blobs)
    persone_folla = (area_folla / avg_person_area) * params["fattore_affollamento"]
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
# INTERFACCIA
# ---------------------------------------------------------------------------

st.title("🏖️ Conteggio persone spiaggia da drone")
st.markdown(
    """
**Come funziona, in 3 passi:**
1. 📤 Carica le tue foto drone, **nell'ordine in cui le hai scattate** lungo la spiaggia
2. 🔍 Premi il pulsante "Analizza spiaggia"
3. 📊 Guarda il totale stimato e il dettaglio per ogni foto

L'app trova da sola dove due foto consecutive si sovrappongono e non conta le persone due volte.
"""
)

with st.expander("⚙️ Impostazioni avanzate (opzionale — i valori di default vanno bene per iniziare)"):
    conf = st.slider("Sensibilità rilevamento diretto", 0.05, 0.5, 0.15, 0.05,
                      help="Più basso = rileva più persone ma anche più falsi positivi")
    persone_per_ombrellone = st.slider("Persone medie per ombrellone", 1.0, 5.0, 3.0, 0.5)
    fattore_affollamento = st.slider("Fattore stima folla densa", 0.1, 1.0, 0.55, 0.05,
                                      help="Aumenta se il totale ti sembra sottostimato, riduci se sovrastimato")
    col_a, col_b = st.columns(2)
    with col_a:
        ombrellone_min_mult = st.number_input("Area minima ombrellone (x area persona)", value=3.0)
        tile_size = st.number_input("Dimensione tile YOLO (px)", value=640, step=64)
    with col_b:
        ombrellone_max_mult = st.number_input("Area massima ombrellone (x area persona)", value=22.0)
        tile_overlap = st.number_input("Overlap tile YOLO (px)", value=120, step=20)

params = dict(conf=conf, persone_per_ombrellone=persone_per_ombrellone,
              fattore_affollamento=fattore_affollamento,
              ombrellone_min_mult=ombrellone_min_mult, ombrellone_max_mult=ombrellone_max_mult,
              tile_size=int(tile_size), tile_overlap=int(tile_overlap))

st.divider()
uploaded_files = st.file_uploader(
    "📤 Carica qui le foto (in ordine di scatto lungo la spiaggia)",
    type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} foto caricate, in quest'ordine: "
               + ", ".join(f"{i+1}. {f.name}" for i, f in enumerate(uploaded_files)))

analizza = st.button("🔍 Analizza spiaggia", type="primary", use_container_width=True,
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

        r = analyze_photo(model, img, params, exclusion_mask=exclusion_mask)
        r["nome"] = name
        r["overlap_con_precedente_pct"] = overlap_pct
        r["overlap_rilevato"] = overlap_ok
        results.append(r)

        prev_img = img
        progress.progress((i + 1) / len(photos), text=f"Analizzata {i+1}/{len(photos)}: {name}")

    progress.empty()

    totale_spiaggia = sum(r["totale"] for r in results)

    st.divider()
    st.markdown(f"## 👥 Totale stimato sulla spiaggia: **{totale_spiaggia:.0f} persone**")

    with st.expander("Vedi il dettaglio del calcolo"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Persone rilevate direttamente", sum(r["diretti"] for r in results))
        c2.metric("Stima da ombrelloni", f"{sum(r['persone_ombrelloni'] for r in results):.0f}")
        c3.metric("Stima folla densa", f"{sum(r['persone_folla_densa'] for r in results):.0f}")

    any_overlap_failed = any(not r["overlap_rilevato"] for r in results[1:])
    if any_overlap_failed:
        st.warning(
            "⚠️ Per almeno una coppia di scatti consecutivi non ho trovato abbastanza riferimenti "
            "visivi in comune per calcolare la sovrapposizione automaticamente (es. due foto che si "
            "toccano solo su acqua/sabbia uniforme). Per quelle foto il conteggio potrebbe includere "
            "un doppio conteggio nella zona di sovrapposizione reale."
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
                st.write(f"**Ombrelloni rilevati:** {r['ombrelloni']} (~{r['persone_ombrelloni']:.0f} persone)")
                st.write(f"**Folla densa (stima):** ~{r['persone_folla_densa']:.0f} persone")
                st.write(f"**Totale foto:** {r['totale']:.0f}")
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
    st.download_button("📥 Scarica riepilogo CSV", csv_data, file_name="riepilogo_spiaggia.csv", mime="text/csv")
