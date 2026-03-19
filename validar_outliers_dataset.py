import argparse
import datetime
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from analise_dataset import (
    PairRecord,
    build_example_panel,
    draw_bbox_on_original,
    list_images,
    load_cache_bboxes,
    resolve_pairs,
    sanitize_filename,
)
from train import MarginAwareCropModel

matplotlib.use("Agg")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    script_name = Path(__file__).stem
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = Path("logs") / f"{script_name}_{timestamp}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger("validar_outliers_dataset")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logging.captureWarnings(True)
    return logger


log = logging.getLogger("validar_outliers_dataset")


@dataclass
class SuspiciousSample:
    index: int
    key: str
    total_score: float
    ranking_mode: str
    geometry_score: float
    template_score: float
    template_match: float
    model_score: float
    model_iou: Optional[float]
    bbox_l1: Optional[float]
    margin_error: Optional[float]
    area: float
    reasons: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida o dataset e ranqueia amostras suspeitas de erro humano usando outliers "
            "geometricos, template matching e divergencia com o modelo treinado."
        )
    )
    parser.add_argument("--origin-dir", default="dataset/origin", help="Pasta de imagens originais")
    parser.add_argument("--crop-dir", default="dataset/cropped", help="Pasta de imagens editadas")
    parser.add_argument("--output-dir", default="logs", help="Pasta base para salvar o relatorio")
    parser.add_argument("--cache-path", default="models/bbox_cache.pkl", help="Cache de bounding boxes")
    parser.add_argument("--model-path", default="models/best_model.pth", help="Checkpoint do modelo")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Workers para calcular bbox quando nao houver cache",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=120,
        help="Quantidade maxima de amostras suspeitas para salvar no ranking",
    )
    parser.add_argument(
        "--save-examples",
        type=int,
        default=40,
        help="Quantidade maxima de exemplos visuais a salvar",
    )
    parser.add_argument(
        "--template-threshold",
        type=float,
        default=0.85,
        help="Template match abaixo disso aumenta suspeita",
    )
    parser.add_argument(
        "--robust-z-threshold",
        type=float,
        default=3.5,
        help="Limite de z-score robusto para destacar features anomalas",
    )
    parser.add_argument(
        "--model-iou-threshold",
        type=float,
        default=0.50,
        help="IoU abaixo disso entre label e modelo aumenta suspeita",
    )
    parser.add_argument(
        "--disable-model-check",
        action="store_true",
        help="Nao usa o modelo treinado no score de suspeita",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size da inferencia do modelo durante a revisao",
    )
    parser.add_argument(
        "--ranking-mode",
        choices=["model", "hybrid", "geometry"],
        default="model",
        help="Modo de ranking: prioriza divergencia do modelo, combinado ou apenas geometria",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k deve ser > 0.")
    if args.save_examples < 0:
        raise ValueError("--save-examples deve ser >= 0.")
    if not (0.0 < args.template_threshold <= 1.0):
        raise ValueError("--template-threshold deve estar em (0, 1].")
    if args.robust_z_threshold <= 0:
        raise ValueError("--robust-z-threshold deve ser > 0.")
    if not (0.0 <= args.model_iou_threshold <= 1.0):
        raise ValueError("--model-iou-threshold deve estar em [0, 1].")
    if args.batch_size <= 0:
        raise ValueError("--batch-size deve ser > 0.")


def bbox_to_xyxy_pixels(bbox: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(np.clip(round(float(bbox[0] * width)), 0, width - 1))
    y1 = int(np.clip(round(float(bbox[1] * height)), 0, height - 1))
    x2 = int(np.clip(round(float(bbox[2] * width)), x1 + 1, width))
    y2 = int(np.clip(round(float(bbox[3] * height)), y1 + 1, height))
    return x1, y1, x2, y2


def compute_template_match_score(record: PairRecord) -> float:
    orig = cv2.imread(str(record.orig_path))
    crop = cv2.imread(str(record.crop_path))
    if orig is None or crop is None:
        return 0.0

    orig_h, orig_w = orig.shape[:2]
    x1, y1, x2, y2 = bbox_to_xyxy_pixels(record.bbox, orig_w, orig_h)
    ref = orig[y1:y2, x1:x2]
    if ref.size == 0:
        return 0.0

    if ref.shape[:2] != crop.shape[:2]:
        ref = cv2.resize(ref, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)

    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray_ref, gray_crop, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max_val)


def robust_zscores(features: np.ndarray) -> np.ndarray:
    med = np.median(features, axis=0)
    mad = np.median(np.abs(features - med), axis=0)
    mad = np.where(mad < 1e-6, 1e-6, mad)
    return 0.6745 * (features - med) / mad


def compute_feature_matrix(records: Sequence[PairRecord]) -> Tuple[np.ndarray, List[str]]:
    rows: List[List[float]] = []
    for rec in records:
        width_n = float(rec.bbox[2] - rec.bbox[0])
        height_n = float(rec.bbox[3] - rec.bbox[1])
        center_x = float((rec.bbox[0] + rec.bbox[2]) / 2.0)
        center_y = float((rec.bbox[1] + rec.bbox[3]) / 2.0)
        left, top, right, bottom = [float(v) for v in rec.margins]
        rows.append([rec.area, width_n, height_n, center_x, center_y, left, top, right, bottom])

    feature_names = ["area", "bbox_w", "bbox_h", "center_x", "center_y", "left", "top", "right", "bottom"]
    return np.asarray(rows, dtype=np.float32), feature_names


def load_model(model_path: Path) -> Tuple[Optional[MarginAwareCropModel], Optional[int]]:
    if not model_path.exists():
        log.warning("Modelo nao encontrado em %s. Validacao usara apenas heuristicas.", model_path)
        return None, None

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = MarginAwareCropModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    img_size = int(checkpoint.get("img_size", 360))
    log.info("Modelo carregado para checagem cruzada. img_size=%d", img_size)
    return model, img_size


def predict_bbox(model: MarginAwareCropModel, img_size: int, orig_path: Path) -> np.ndarray:
    img = cv2.imread(str(orig_path))
    if img is None:
        raise ValueError(f"Falha ao ler {orig_path}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size))
    tensor = transforms.ToTensor()(resized)
    tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type="cuda"):
                pred = model(tensor)[0].detach().cpu().numpy()
        else:
            pred = model(tensor)[0].detach().cpu().numpy()

    return np.clip(pred.astype(np.float32), 0.0, 1.0)


def preprocess_image_for_model(orig_path: Path, img_size: int) -> torch.Tensor:
    img = cv2.imread(str(orig_path))
    if img is None:
        raise ValueError(f"Falha ao ler {orig_path}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size))
    tensor = transforms.ToTensor()(resized)
    tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(tensor)
    return tensor


def predict_bboxes_batch(model: MarginAwareCropModel, img_size: int, paths: Sequence[Path], batch_size: int) -> List[np.ndarray]:
    predictions: List[np.ndarray] = []
    total = len(paths)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_tensors = [preprocess_image_for_model(path, img_size) for path in paths[start:end]]
        batch = torch.stack(batch_tensors, dim=0).to(DEVICE)

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast(device_type="cuda"):
                    preds = model(batch).detach().cpu().numpy()
            else:
                preds = model(batch).detach().cpu().numpy()

        predictions.extend(np.clip(preds.astype(np.float32), 0.0, 1.0))

    return predictions


def compute_template_matches(records: Sequence[PairRecord], max_workers: int) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        scores = list(executor.map(compute_template_match_score, records))
    return np.asarray(scores, dtype=np.float32)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def compute_bbox_l1(box_a: np.ndarray, box_b: np.ndarray) -> float:
    return float(np.mean(np.abs(box_a.astype(np.float32) - box_b.astype(np.float32))))


def compute_margin_error(box_a: np.ndarray, box_b: np.ndarray) -> float:
    margins_a = np.array([box_a[0], box_a[1], 1.0 - box_a[2], 1.0 - box_a[3]], dtype=np.float32)
    margins_b = np.array([box_b[0], box_b[1], 1.0 - box_b[2], 1.0 - box_b[3]], dtype=np.float32)
    return float(np.mean(np.abs(margins_a - margins_b)))


def build_reason_text(
    abs_z: np.ndarray,
    feature_names: Sequence[str],
    robust_z_threshold: float,
    template_match: float,
    template_threshold: float,
    model_iou: Optional[float],
    model_iou_threshold: float,
    bbox_l1: Optional[float],
    margin_error: Optional[float],
) -> str:
    anomalous_features = []
    for name, value in zip(feature_names, abs_z.tolist()):
        if value >= robust_z_threshold:
            anomalous_features.append(f"{name}=z{value:.2f}")

    reasons: List[str] = []
    if anomalous_features:
        reasons.append("outlier geometrico: " + "; ".join(anomalous_features[:4]))
    if template_match < template_threshold:
        reasons.append(f"template_match baixo={template_match:.3f}")
    if model_iou is not None and model_iou < model_iou_threshold:
        reasons.append(f"label_vs_model IoU baixo={model_iou:.3f}")
    if bbox_l1 is not None and bbox_l1 >= 0.05:
        reasons.append(f"bbox_l1 alto={bbox_l1:.3f}")
    if margin_error is not None and margin_error >= 0.04:
        reasons.append(f"erro_margem alto={margin_error:.3f}")
    if not reasons:
        reasons.append("score elevado por combinacao de sinais fracos")
    return " | ".join(reasons)


def build_model_score(
    model_iou: Optional[float],
    bbox_l1: Optional[float],
    margin_error: Optional[float],
) -> float:
    if model_iou is None or bbox_l1 is None or margin_error is None:
        return 0.0
    iou_component = (1.0 - model_iou) * 12.0
    l1_component = bbox_l1 * 20.0
    margin_component = margin_error * 18.0
    return float(iou_component + l1_component + margin_component)


def rank_suspicious_samples(
    records: Sequence[PairRecord],
    feature_names: Sequence[str],
    zscores: np.ndarray,
    template_matches: np.ndarray,
    template_threshold: float,
    model_ious: Sequence[Optional[float]],
    bbox_l1s: Sequence[Optional[float]],
    margin_errors: Sequence[Optional[float]],
    model_iou_threshold: float,
    robust_z_threshold: float,
    ranking_mode: str,
) -> List[SuspiciousSample]:
    ranked: List[SuspiciousSample] = []

    abs_z = np.abs(zscores)
    for idx, rec in enumerate(records):
        geometry_score = float(np.mean(np.sort(abs_z[idx])[-3:]))

        template_match = float(template_matches[idx])
        template_score = 0.0
        if template_match < template_threshold:
            template_score = ((template_threshold - template_match) / max(template_threshold, 1e-6)) * 6.0

        model_iou = model_ious[idx]
        bbox_l1 = bbox_l1s[idx]
        margin_error = margin_errors[idx]
        model_score = build_model_score(model_iou, bbox_l1, margin_error)

        if ranking_mode == "model":
            total_score = model_score
        elif ranking_mode == "hybrid":
            total_score = model_score + (0.35 * geometry_score) + (0.25 * template_score)
        else:
            total_score = geometry_score + template_score

        reasons = build_reason_text(
            abs_z=abs_z[idx],
            feature_names=feature_names,
            robust_z_threshold=robust_z_threshold,
            template_match=template_match,
            template_threshold=template_threshold,
            model_iou=model_iou,
            model_iou_threshold=model_iou_threshold,
            bbox_l1=bbox_l1,
            margin_error=margin_error,
        )

        ranked.append(
            SuspiciousSample(
                index=idx,
                key=rec.key,
                total_score=total_score,
                ranking_mode=ranking_mode,
                geometry_score=geometry_score,
                template_score=template_score,
                template_match=template_match,
                model_score=model_score,
                model_iou=model_iou,
                bbox_l1=bbox_l1,
                margin_error=margin_error,
                area=rec.area,
                reasons=reasons,
            )
        )

    ranked.sort(key=lambda item: item.total_score, reverse=True)
    return ranked


def draw_comparison_bboxes(orig: np.ndarray, label_bbox: np.ndarray, pred_bbox: Optional[np.ndarray]) -> np.ndarray:
    img = draw_bbox_on_original(orig, label_bbox)
    cv2.putText(
        img,
        "label humano",
        (12, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (35, 220, 35),
        2,
        cv2.LINE_AA,
    )

    if pred_bbox is not None:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox_to_xyxy_pixels(pred_bbox, w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 215, 0), 3)
        cv2.putText(
            img,
            "predicao modelo",
            (12, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 215, 0),
            2,
            cv2.LINE_AA,
        )

    return img


def save_examples(
    output_dir: Path,
    records: Sequence[PairRecord],
    ranked: Sequence[SuspiciousSample],
    predicted_bboxes: Sequence[Optional[np.ndarray]],
    max_examples: int,
) -> Path:
    examples_dir = output_dir / "suspeitos"
    examples_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "suspeitos_exemplos.csv"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("rank,key,total_score,model_score,bbox_l1,margin_error,geometry_score,template_match,model_iou,reasons,orig_path,crop_path,example_path\n")

        for rank, item in enumerate(ranked[:max_examples], start=1):
            rec = records[item.index]
            orig = cv2.imread(str(rec.orig_path))
            crop = cv2.imread(str(rec.crop_path))
            if orig is None or crop is None:
                continue

            comp = draw_comparison_bboxes(orig, rec.bbox, predicted_bboxes[item.index])
            panel = build_example_panel(comp, crop, rec.area, rec.key)

            line2 = f"score={item.total_score:.2f} | tm={item.template_match:.3f} | model_iou="
            if item.model_iou is None:
                line2 += "NA"
            else:
                line2 += f"{item.model_iou:.3f}"
            cv2.putText(panel, line2, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            bbox_l1_text = "NA" if item.bbox_l1 is None else f"{item.bbox_l1:.3f}"
            margin_error_text = "NA" if item.margin_error is None else f"{item.margin_error:.3f}"
            cv2.putText(panel, f"bbox_l1={bbox_l1_text} | erro_margem={margin_error_text}", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(panel, item.reasons[:140], (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv2.LINE_AA)

            filename = sanitize_filename(f"{rank:03d}_{rec.key}.jpg")
            out_path = examples_dir / filename
            cv2.imwrite(str(out_path), panel)

            model_iou_text = "NA" if item.model_iou is None else f"{item.model_iou:.6f}"
            bbox_l1_text = "NA" if item.bbox_l1 is None else f"{item.bbox_l1:.6f}"
            margin_error_text = "NA" if item.margin_error is None else f"{item.margin_error:.6f}"
            escaped_reasons = item.reasons.replace(",", ";")
            f.write(
                f"{rank},{rec.key},{item.total_score:.6f},{item.model_score:.6f},{bbox_l1_text},{margin_error_text},{item.geometry_score:.6f},"
                f"{item.template_match:.6f},{model_iou_text},{escaped_reasons},"
                f"{rec.orig_path},{rec.crop_path},{out_path}\n"
            )

    return csv_path


def write_ranking_csv(output_path: Path, records: Sequence[PairRecord], ranked: Sequence[SuspiciousSample]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("rank,key,ranking_mode,total_score,model_score,model_iou,bbox_l1,margin_error,geometry_score,template_score,template_match,area,reasons,orig_path,crop_path\n")
        for rank, item in enumerate(ranked, start=1):
            rec = records[item.index]
            model_iou_text = "NA" if item.model_iou is None else f"{item.model_iou:.6f}"
            bbox_l1_text = "NA" if item.bbox_l1 is None else f"{item.bbox_l1:.6f}"
            margin_error_text = "NA" if item.margin_error is None else f"{item.margin_error:.6f}"
            escaped_reasons = item.reasons.replace(",", ";")
            f.write(
                f"{rank},{item.key},{item.ranking_mode},{item.total_score:.6f},{item.model_score:.6f},{model_iou_text},{bbox_l1_text},{margin_error_text},"
                f"{item.geometry_score:.6f},{item.template_score:.6f},{item.template_match:.6f},{item.area:.6f},{escaped_reasons},{rec.orig_path},{rec.crop_path}\n"
            )


def write_summary(
    output_path: Path,
    total_pairs: int,
    ranked: Sequence[SuspiciousSample],
    template_threshold: float,
    model_used: bool,
    robust_z_threshold: float,
    ranking_mode: str,
) -> None:
    top_10 = ranked[:10]
    if ranking_mode == "model":
        high_suspicion = [item for item in ranked if item.model_iou is not None and item.model_iou <= 0.85]
    else:
        high_suspicion = [item for item in ranked if item.total_score >= 8.0]
    with output_path.open("w", encoding="utf-8") as f:
        f.write("=== VALIDACAO DE AMOSTRAS SUSPEITAS ===\n\n")
        f.write(f"Total de pares avaliados: {total_pairs}\n")
        f.write(f"Modo de ranking: {ranking_mode}\n")
        f.write(f"Regra de template matching: sinaliza abaixo de {template_threshold:.3f}\n")
        f.write(f"Regra de z-score robusto: destaca acima de {robust_z_threshold:.2f}\n")
        f.write(f"Checagem com modelo treinado: {'SIM' if model_used else 'NAO'}\n")
        if ranking_mode == "model":
            f.write(f"Amostras com IoU label x modelo <= 0.85: {len(high_suspicion)}\n\n")
        else:
            f.write(f"Amostras com score >= 8.0: {len(high_suspicion)}\n\n")

        f.write("Top 10 suspeitas:\n")
        for pos, item in enumerate(top_10, start=1):
            model_iou_text = "NA" if item.model_iou is None else f"{item.model_iou:.3f}"
            bbox_l1_text = "NA" if item.bbox_l1 is None else f"{item.bbox_l1:.3f}"
            margin_error_text = "NA" if item.margin_error is None else f"{item.margin_error:.3f}"
            f.write(
                f"- #{pos} key={item.key} | score={item.total_score:.2f} | tm={item.template_match:.3f} "
                f"| model_iou={model_iou_text} | bbox_l1={bbox_l1_text} | erro_margem={margin_error_text} | motivos={item.reasons}\n"
            )


def generate_plots(output_path: Path, ranked: Sequence[SuspiciousSample]) -> None:
    scores = np.array([item.total_score for item in ranked], dtype=np.float32)
    template_matches = np.array([item.template_match for item in ranked], dtype=np.float32)
    valid_model_ious = np.array([item.model_iou for item in ranked if item.model_iou is not None], dtype=np.float32)
    geometry_scores = np.array([item.geometry_score for item in ranked], dtype=np.float32)
    valid_bbox_l1 = np.array([item.bbox_l1 for item in ranked if item.bbox_l1 is not None], dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Validacao de Outliers do Dataset", fontsize=16)

    axes[0, 0].hist(scores, bins=30, color="#e45756", edgecolor="black", alpha=0.85)
    axes[0, 0].set_title("Distribuicao do score total de suspeita")
    axes[0, 0].set_xlabel("Score")
    axes[0, 0].set_ylabel("Quantidade")

    if len(valid_model_ious) > 0 and len(valid_bbox_l1) == len(valid_model_ious):
        axes[0, 1].scatter(valid_model_ious, valid_bbox_l1, s=22, alpha=0.55, color="#4c78a8")
        axes[0, 1].set_title("IoU label x modelo vs bbox_l1")
        axes[0, 1].set_xlabel("IoU")
        axes[0, 1].set_ylabel("bbox_l1")
    else:
        axes[0, 1].scatter(geometry_scores, template_matches, s=22, alpha=0.55, color="#4c78a8")
        axes[0, 1].set_title("Geometria x Template matching")
        axes[0, 1].set_xlabel("Geometry score")
        axes[0, 1].set_ylabel("Template match")

    top_scores = ranked[:20]
    labels = [item.key for item in top_scores]
    vals = [item.total_score for item in top_scores]
    axes[1, 0].barh(labels[::-1], vals[::-1], color="#f58518")
    axes[1, 0].set_title("Top 20 amostras suspeitas")
    axes[1, 0].set_xlabel("Score")

    if len(valid_model_ious) > 0:
        axes[1, 1].hist(valid_model_ious, bins=20, color="#72b7b2", edgecolor="black", alpha=0.85)
        axes[1, 1].set_title("Distribuicao de IoU label x modelo")
        axes[1, 1].set_xlabel("IoU")
        axes[1, 1].set_ylabel("Quantidade")
    else:
        axes[1, 1].text(0.5, 0.5, "Modelo desabilitado\nou indisponivel", ha="center", va="center", fontsize=14)
        axes[1, 1].set_axis_off()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)

    origin_dir = Path(args.origin_dir)
    crop_dir = Path(args.crop_dir)
    cache_path = Path(args.cache_path)
    output_base = Path(args.output_dir)
    model_path = Path(args.model_path)

    if not origin_dir.exists() or not crop_dir.exists():
        raise FileNotFoundError(f"Pastas nao encontradas: {origin_dir} e/ou {crop_dir}")

    origin_files = list_images(origin_dir)
    crop_files = list_images(crop_dir)
    if len(origin_files) == 0 or len(crop_files) == 0:
        raise ValueError("Dataset vazio em uma das pastas.")

    log.info("Originais encontrados: %d", len(origin_files))
    log.info("Crops encontrados: %d", len(crop_files))

    cache_map = load_cache_bboxes(cache_path)
    records, missing_crop, orphan_crop = resolve_pairs(origin_files, crop_files, cache_map, args.max_workers)

    if missing_crop or orphan_crop:
        log.warning("Foram encontrados pares incompletos: missing_crop=%d | orphan_crop=%d", len(missing_crop), len(orphan_crop))
    if len(records) == 0:
        raise ValueError("Nenhum par valido foi encontrado para validacao.")

    feature_matrix, feature_names = compute_feature_matrix(records)
    zscores = robust_zscores(feature_matrix)

    log.info("Calculando template matching para %d pares...", len(records))
    template_matches = compute_template_matches(records, args.max_workers)

    use_model = not args.disable_model_check
    model, img_size = (None, None)
    if use_model:
        model, img_size = load_model(model_path)

    predicted_bboxes: List[Optional[np.ndarray]] = [None] * len(records)
    model_ious: List[Optional[float]] = [None] * len(records)
    bbox_l1s: List[Optional[float]] = [None] * len(records)
    margin_errors: List[Optional[float]] = [None] * len(records)

    if model is not None and img_size is not None:
        log.info("Calculando divergencia label x modelo...")
        pred_list = predict_bboxes_batch(
            model=model,
            img_size=img_size,
            paths=[rec.orig_path for rec in records],
            batch_size=args.batch_size,
        )
        for idx, rec in enumerate(records):
            pred_bbox = pred_list[idx]
            predicted_bboxes[idx] = pred_bbox
            model_ious[idx] = compute_iou(rec.bbox, pred_bbox)
            bbox_l1s[idx] = compute_bbox_l1(rec.bbox, pred_bbox)
            margin_errors[idx] = compute_margin_error(rec.bbox, pred_bbox)

    ranked = rank_suspicious_samples(
        records=records,
        feature_names=feature_names,
        zscores=zscores,
        template_matches=template_matches,
        template_threshold=args.template_threshold,
        model_ious=model_ious,
        bbox_l1s=bbox_l1s,
        margin_errors=margin_errors,
        model_iou_threshold=args.model_iou_threshold,
        robust_z_threshold=args.robust_z_threshold,
        ranking_mode=args.ranking_mode,
    )

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output_base / f"validacao_outliers_{now}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "resumo.txt"
    ranking_csv_path = output_dir / "ranking_suspeitos.csv"
    plot_path = output_dir / "graficos_validacao.png"

    write_summary(
        output_path=summary_path,
        total_pairs=len(records),
        ranked=ranked[: args.top_k],
        template_threshold=args.template_threshold,
        model_used=model is not None,
        robust_z_threshold=args.robust_z_threshold,
        ranking_mode=args.ranking_mode,
    )
    write_ranking_csv(ranking_csv_path, records, ranked[: args.top_k])
    examples_csv_path = save_examples(
        output_dir=output_dir,
        records=records,
        ranked=ranked,
        predicted_bboxes=predicted_bboxes,
        max_examples=args.save_examples,
    )
    generate_plots(plot_path, ranked)

    log.info("Resumo salvo em: %s", summary_path)
    log.info("Ranking salvo em: %s", ranking_csv_path)
    log.info("Exemplos suspeitos salvos em: %s", output_dir / "suspeitos")
    log.info("Indice de exemplos salvo em: %s", examples_csv_path)
    log.info("Graficos salvos em: %s", plot_path)


if __name__ == "__main__":
    setup_logging()
    start_ts = datetime.datetime.now()
    try:
        main()
    except Exception as exc:
        log.exception("Falha na validacao de outliers: %s", exc)
        raise
    finally:
        elapsed = datetime.datetime.now() - start_ts
        elapsed_sec = max(elapsed.total_seconds(), 0.0)
        hh = math.floor(elapsed_sec // 3600)
        mm = math.floor((elapsed_sec % 3600) // 60)
        ss = math.floor(elapsed_sec % 60)
        log.info("Tempo total: %02d:%02d:%02d", hh, mm, ss)