import argparse
import datetime
import logging
import math
import os
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_BINS: Tuple[float, ...] = (0.10, 0.20, 0.35, 0.50, 0.70, 1.00)


def setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    script_name = Path(__file__).stem
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = Path("logs") / f"{script_name}_{timestamp}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger("analise_dataset")
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


log = logging.getLogger("analise_dataset")


@dataclass
class PairRecord:
    key: str
    orig_path: Path
    crop_path: Path
    width: int
    height: int
    bbox: np.ndarray

    @property
    def area(self) -> float:
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))

    @property
    def aspect_ratio(self) -> float:
        return float(self.width / max(self.height, 1))

    @property
    def margins(self) -> np.ndarray:
        return np.array([
            self.bbox[0],
            self.bbox[1],
            1.0 - self.bbox[2],
            1.0 - self.bbox[3],
        ], dtype=np.float32)


@dataclass
class BinStatus:
    label: str
    count: int
    percent: float
    ideal_count: float
    status: str
    indices: List[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analisa o dataset de recorte e gera estatisticas + graficos para identificar "
            "faixas em excesso e faixas faltando."
        )
    )
    parser.add_argument("--origin-dir", default="dataset/origin", help="Pasta de imagens originais")
    parser.add_argument("--crop-dir", default="dataset/cropped", help="Pasta de imagens recortadas")
    parser.add_argument("--output-dir", default="logs", help="Pasta base para salvar analise")
    parser.add_argument("--cache-path", default="models/bbox_cache.pkl", help="Cache de bbox (opcional)")
    parser.add_argument(
        "--bins",
        nargs="+",
        type=float,
        default=list(DEFAULT_BINS),
        help="Limites superiores de area (normalizada) para classificacao. Ex: --bins 0.1 0.2 0.35 0.5 0.7 1.0",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Tolerancia de desbalanceamento para classificar excesso/falta (padrao=0.25 = 25%%)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Workers para calcular bbox quando nao houver cache suficiente",
    )
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=8,
        help="Quantidade maxima de exemplos visuais salvos por faixa",
    )
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def base_key(path: Path) -> str:
    return path.stem.replace("_editado", "")


def map_by_key(files: Sequence[Path]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in files:
        mapping[base_key(path)] = path
    return mapping


def load_cache_bboxes(cache_path: Path) -> Dict[Tuple[str, str], np.ndarray]:
    if not cache_path.exists():
        return {}

    try:
        with cache_path.open("rb") as f:
            cache = pickle.load(f)

        orig_paths = cache.get("orig_paths", [])
        crop_paths = cache.get("crop_paths", [])
        bboxes = cache.get("bboxes", [])

        if not (len(orig_paths) == len(crop_paths) == len(bboxes)):
            log.warning("Cache de bbox invalido (tamanho inconsistente).")
            return {}

        cache_map: Dict[Tuple[str, str], np.ndarray] = {}
        for orig, crop, bbox in zip(orig_paths, crop_paths, bboxes):
            key = (str(Path(orig)), str(Path(crop)))
            cache_map[key] = np.asarray(bbox, dtype=np.float32)

        log.info("Cache de bbox carregado: %d entradas", len(cache_map))
        return cache_map
    except Exception as exc:
        log.warning("Falha ao ler cache de bbox (%s): %s", cache_path, exc)
        return {}


def compute_bbox_for_pair(orig_path: Path, crop_path: Path) -> np.ndarray:
    try:
        orig = cv2.imread(str(orig_path))
        crop = cv2.imread(str(crop_path))
        if orig is None or crop is None:
            return np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32)

        orig_h, orig_w = orig.shape[:2]
        gray_orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(gray_orig, gray_crop, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        x1, y1 = max_loc
        x2, y2 = x1 + crop.shape[1], y1 + crop.shape[0]

        bbox = np.array([x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h], dtype=np.float32)
        return np.clip(bbox, 0.0, 1.0)
    except Exception as exc:
        log.error("Erro ao calcular bbox para %s: %s", orig_path, exc)
        return np.array([0.05, 0.05, 0.95, 0.95], dtype=np.float32)


def resolve_pairs(
    origin_files: Sequence[Path],
    crop_files: Sequence[Path],
    cache_map: Dict[Tuple[str, str], np.ndarray],
    max_workers: int,
) -> Tuple[List[PairRecord], List[Path], List[Path]]:
    origin_map = map_by_key(origin_files)
    crop_map = map_by_key(crop_files)

    common_keys = sorted(set(origin_map) & set(crop_map))
    missing_crop_keys = sorted(set(origin_map) - set(crop_map))
    orphan_crop_keys = sorted(set(crop_map) - set(origin_map))

    missing_crop = [origin_map[k] for k in missing_crop_keys]
    orphan_crop = [crop_map[k] for k in orphan_crop_keys]

    planned_pairs: List[Tuple[str, Path, Path]] = []
    for key in common_keys:
        planned_pairs.append((key, origin_map[key], crop_map[key]))

    records: List[Optional[PairRecord]] = [None] * len(planned_pairs)
    to_compute: List[Tuple[int, str, Path, Path]] = []

    for idx, (key, orig, crop) in enumerate(planned_pairs):
        cache_key = (str(orig), str(crop))
        bbox = cache_map.get(cache_key)
        img = cv2.imread(str(orig))
        if img is None:
            log.warning("Nao foi possivel ler imagem original: %s", orig)
            continue
        h, w = img.shape[:2]

        if bbox is not None:
            records[idx] = PairRecord(key=key, orig_path=orig, crop_path=crop, width=w, height=h, bbox=bbox)
        else:
            to_compute.append((idx, key, orig, crop))

    if to_compute:
        log.info("Calculando %d bboxes fora do cache...", len(to_compute))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(compute_bbox_for_pair, orig, crop): (idx, key, orig, crop)
                for idx, key, orig, crop in to_compute
            }
            for future in futures:
                idx, key, orig, crop = futures[future]
                bbox = future.result()
                img = cv2.imread(str(orig))
                if img is None:
                    continue
                h, w = img.shape[:2]
                records[idx] = PairRecord(key=key, orig_path=orig, crop_path=crop, width=w, height=h, bbox=bbox)

    valid_records = [r for r in records if r is not None]
    return valid_records, missing_crop, orphan_crop


def summarize_numeric(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def classify_bins(values: np.ndarray, bins: Sequence[float], threshold: float) -> List[BinStatus]:
    bins_sorted = sorted(float(v) for v in bins)
    if not bins_sorted or bins_sorted[-1] < 1.0:
        bins_sorted.append(1.0)

    edges = [0.0] + bins_sorted
    total = len(values)
    ideal_count = total / max(len(edges) - 1, 1)

    result: List[BinStatus] = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == len(edges) - 2:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        indices = np.where(mask)[0].tolist()

        count = int(mask.sum())
        pct = (count / total * 100.0) if total else 0.0

        upper = ideal_count * (1.0 + threshold)
        lower = ideal_count * (1.0 - threshold)
        if count > upper:
            status = "EXCESSO"
        elif count < lower:
            status = "FALTANDO"
        else:
            status = "OK"

        result.append(
            BinStatus(
                label=f"{lo:.2f}-{hi:.2f}",
                count=count,
                percent=pct,
                ideal_count=ideal_count,
                status=status,
                indices=indices,
            )
        )

    return result


def write_text_report(
    output_path: Path,
    total_origin: int,
    total_crop: int,
    records: Sequence[PairRecord],
    missing_crop: Sequence[Path],
    orphan_crop: Sequence[Path],
    area_stats: Dict[str, float],
    aspect_stats: Dict[str, float],
    margin_stats: Dict[str, Dict[str, float]],
    area_balance: Sequence[BinStatus],
    ext_counter: Counter,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("=== ANALISE DO DATASET ===\n\n")
        f.write(f"Total de originais: {total_origin}\n")
        f.write(f"Total de crops: {total_crop}\n")
        f.write(f"Pares validos: {len(records)}\n")
        f.write(f"Originais sem crop correspondente: {len(missing_crop)}\n")
        f.write(f"Crops orfaos (sem original): {len(orphan_crop)}\n\n")

        if missing_crop:
            f.write("Exemplos de originais sem crop:\n")
            for path in list(missing_crop)[:10]:
                f.write(f"- {path}\n")
            f.write("\n")

        if orphan_crop:
            f.write("Exemplos de crops orfaos:\n")
            for path in list(orphan_crop)[:10]:
                f.write(f"- {path}\n")
            f.write("\n")

        f.write("Distribuicao por extensao (originais):\n")
        for ext, count in ext_counter.most_common():
            f.write(f"- {ext}: {count}\n")
        f.write("\n")

        f.write("Estatisticas de area do bbox:\n")
        for key in ["min", "p25", "median", "mean", "p75", "max", "std"]:
            f.write(f"- {key}: {area_stats[key]:.6f}\n")
        f.write("\n")

        f.write("Estatisticas de aspect ratio (largura/altura):\n")
        for key in ["min", "p25", "median", "mean", "p75", "max", "std"]:
            f.write(f"- {key}: {aspect_stats[key]:.6f}\n")
        f.write("\n")

        f.write("Estatisticas de margens normalizadas:\n")
        for side in ["left", "top", "right", "bottom"]:
            f.write(f"- {side}:\n")
            for key in ["min", "p25", "median", "mean", "p75", "max", "std"]:
                f.write(f"    - {key}: {margin_stats[side][key]:.6f}\n")
        f.write("\n")

        f.write("Balanceamento por faixa de area:\n")
        for item in area_balance:
            f.write(
                f"- faixa {item.label}: {item.count} ({item.percent:.2f}%) | "
                f"ideal={item.ideal_count:.2f} | status={item.status}\n"
            )
            if item.indices:
                sample_idx = item.indices[: min(5, len(item.indices))]
                f.write("  exemplos:\n")
                for idx in sample_idx:
                    rec = records[idx]
                    f.write(
                        f"    - key={rec.key} | area={rec.area:.4f} | orig={rec.orig_path} | crop={rec.crop_path}\n"
                    )
            else:
                f.write("  exemplos: nenhuma imagem nesta faixa\n")


def sanitize_filename(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "sample"


def draw_bbox_on_original(orig: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    img = orig.copy()
    h, w = img.shape[:2]
    x1 = int(max(0, min(w - 1, bbox[0] * w)))
    y1 = int(max(0, min(h - 1, bbox[1] * h)))
    x2 = int(max(0, min(w - 1, bbox[2] * w)))
    y2 = int(max(0, min(h - 1, bbox[3] * h)))

    cv2.rectangle(img, (x1, y1), (x2, y2), (35, 220, 35), 3)
    cv2.putText(
        img,
        "bbox",
        (max(8, x1), max(28, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (35, 220, 35),
        2,
        cv2.LINE_AA,
    )
    return img


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_h, 3), dtype=np.uint8)
    scale = target_h / h
    target_w = max(1, int(w * scale))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def build_example_panel(orig_with_bbox: np.ndarray, crop_img: np.ndarray, area: float, key: str) -> np.ndarray:
    target_h = 360
    left = resize_to_height(orig_with_bbox, target_h)
    right = resize_to_height(crop_img, target_h)
    separator = np.full((target_h, 12, 3), 28, dtype=np.uint8)
    panel = np.concatenate([left, separator, right], axis=1)

    cv2.putText(
        panel,
        f"key={key} | area={area:.4f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "original com bbox",
        (12, target_h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (225, 225, 225),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "crop editado",
        (left.shape[1] + separator.shape[1] + 12, target_h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (225, 225, 225),
        2,
        cv2.LINE_AA,
    )
    return panel


def select_evenly_spaced(indices: Sequence[int], max_count: int) -> List[int]:
    if max_count <= 0:
        return []
    if len(indices) <= max_count:
        return list(indices)

    positions = np.linspace(0, len(indices) - 1, num=max_count, dtype=int)
    return [indices[p] for p in positions]


def save_example_images(
    output_dir: Path,
    records: Sequence[PairRecord],
    area_balance: Sequence[BinStatus],
    samples_per_bin: int,
) -> Path:
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    index_csv_path = output_dir / "examples_index.csv"

    with index_csv_path.open("w", encoding="utf-8") as f:
        f.write("faixa,status,rank,key,area,orig_path,crop_path,example_image\n")

        for bin_item in area_balance:
            folder_name = sanitize_filename(f"{bin_item.label}_{bin_item.status}")
            bin_dir = examples_dir / folder_name
            bin_dir.mkdir(parents=True, exist_ok=True)

            selected_indices = select_evenly_spaced(bin_item.indices, samples_per_bin)

            if not selected_indices:
                continue

            for rank, rec_idx in enumerate(selected_indices, start=1):
                rec = records[rec_idx]
                orig = cv2.imread(str(rec.orig_path))
                crop = cv2.imread(str(rec.crop_path))
                if orig is None or crop is None:
                    continue

                orig_bbox = draw_bbox_on_original(orig, rec.bbox)
                panel = build_example_panel(orig_bbox, crop, rec.area, rec.key)

                filename = sanitize_filename(f"{rank:02d}_{rec.key}.jpg")
                out_file = bin_dir / filename
                cv2.imwrite(str(out_file), panel)

                f.write(
                    f"{bin_item.label},{bin_item.status},{rank},{rec.key},{rec.area:.6f},"
                    f"{rec.orig_path},{rec.crop_path},{out_file}\n"
                )

    return index_csv_path


def write_pair_catalog_csv(output_path: Path, records: Sequence[PairRecord], area_balance: Sequence[BinStatus]) -> None:
    row_status: Dict[int, Tuple[str, str]] = {}
    for bin_item in area_balance:
        for idx in bin_item.indices:
            row_status[idx] = (bin_item.label, bin_item.status)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("key,orig_path,crop_path,width,height,area,aspect_ratio,bin,status\n")
        for idx, rec in enumerate(records):
            bin_label, status = row_status.get(idx, ("NA", "NA"))
            f.write(
                f"{rec.key},{rec.orig_path},{rec.crop_path},{rec.width},{rec.height},"
                f"{rec.area:.6f},{rec.aspect_ratio:.6f},{bin_label},{status}\n"
            )


def write_csv_balance(output_path: Path, rows: Sequence[BinStatus]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("faixa,count,percent,ideal_count,status\n")
        for row in rows:
            f.write(f"{row.label},{row.count},{row.percent:.4f},{row.ideal_count:.4f},{row.status}\n")


def generate_plots(
    output_path: Path,
    total_origin: int,
    total_crop: int,
    records: Sequence[PairRecord],
    missing_crop: Sequence[Path],
    orphan_crop: Sequence[Path],
    area_balance: Sequence[BinStatus],
) -> None:
    areas = np.array([r.area for r in records], dtype=np.float32)
    widths = np.array([r.width for r in records], dtype=np.float32)
    heights = np.array([r.height for r in records], dtype=np.float32)

    margins = np.vstack([r.margins for r in records])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Diagnostico do Dataset", fontsize=16)

    labels_pairs = ["Pares validos", "Orig sem crop", "Crop orfao"]
    values_pairs = [len(records), len(missing_crop), len(orphan_crop)]
    colors_pairs = ["#4c78a8", "#f58518", "#e45756"]
    axes[0, 0].bar(labels_pairs, values_pairs, color=colors_pairs)
    axes[0, 0].set_title("Integridade de pareamento")
    axes[0, 0].set_ylabel("Quantidade")

    axes[0, 1].hist(areas, bins=20, color="#72b7b2", edgecolor="black", alpha=0.9)
    axes[0, 1].set_title("Distribuicao de area do bbox")
    axes[0, 1].set_xlabel("Area normalizada")
    axes[0, 1].set_ylabel("Quantidade")

    bin_labels = [b.label for b in area_balance]
    bin_counts = [b.count for b in area_balance]
    ideal = [b.ideal_count for b in area_balance]
    color_map = {"EXCESSO": "#e45756", "FALTANDO": "#f58518", "OK": "#54a24b"}
    bar_colors = [color_map[b.status] for b in area_balance]
    axes[1, 0].bar(bin_labels, bin_counts, color=bar_colors)
    axes[1, 0].plot(bin_labels, ideal, color="#4c78a8", linestyle="--", linewidth=2, label="ideal")
    axes[1, 0].set_title("Excesso/Falta por faixa de area")
    axes[1, 0].set_xlabel("Faixas")
    axes[1, 0].set_ylabel("Quantidade")
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].legend()

    axes[1, 1].scatter(widths, heights, c=areas, cmap="viridis", alpha=0.65, s=26)
    axes[1, 1].set_title("Resolucao das imagens (cor = area bbox)")
    axes[1, 1].set_xlabel("Largura")
    axes[1, 1].set_ylabel("Altura")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.boxplot(
        [margins[:, 0], margins[:, 1], margins[:, 2], margins[:, 3]],
        labels=["left", "top", "right", "bottom"],
        patch_artist=True,
        boxprops={"facecolor": "#72b7b2", "alpha": 0.8},
    )
    ax2.set_title("Distribuicao das margens normalizadas")
    ax2.set_ylabel("Margem")
    fig2.tight_layout()
    margins_path = output_path.with_name("margins_boxplot.png")
    fig2.savefig(margins_path, dpi=180)
    plt.close(fig2)

    log.info("Grafico principal salvo em: %s", output_path)
    log.info("Grafico de margens salvo em: %s", margins_path)

    log.info(
        "Resumo rapido: originais=%d | crops=%d | pares=%d | faltando_crop=%d | crop_orfao=%d",
        total_origin,
        total_crop,
        len(records),
        len(missing_crop),
        len(orphan_crop),
    )


def validate_bins(bins: Sequence[float]) -> List[float]:
    unique_sorted = sorted(set(float(v) for v in bins))
    if not unique_sorted:
        raise ValueError("Informe ao menos um valor de bin.")
    if unique_sorted[0] <= 0:
        raise ValueError("Os bins devem ser > 0.")
    if unique_sorted[-1] > 1.0:
        raise ValueError("Os bins devem estar no intervalo (0, 1].")
    return unique_sorted


def main() -> None:
    args = parse_args()

    origin_dir = Path(args.origin_dir)
    crop_dir = Path(args.crop_dir)
    output_base = Path(args.output_dir)
    cache_path = Path(args.cache_path)

    if not origin_dir.exists() or not crop_dir.exists():
        raise FileNotFoundError(
            f"Pastas nao encontradas. Esperado: {origin_dir} e {crop_dir}"
        )

    bins = validate_bins(args.bins)
    if args.threshold < 0 or args.threshold >= 1:
        raise ValueError("--threshold deve estar no intervalo [0, 1).")
    if args.samples_per_bin < 0:
        raise ValueError("--samples-per-bin deve ser >= 0.")

    origin_files = list_images(origin_dir)
    crop_files = list_images(crop_dir)

    log.info("Originais encontrados: %d", len(origin_files))
    log.info("Crops encontrados: %d", len(crop_files))

    if len(origin_files) == 0 or len(crop_files) == 0:
        raise ValueError("Dataset vazio em uma das pastas.")

    cache_map = load_cache_bboxes(cache_path)
    records, missing_crop, orphan_crop = resolve_pairs(origin_files, crop_files, cache_map, args.max_workers)

    if len(records) == 0:
        raise ValueError("Nenhum par valido foi encontrado para analise.")

    areas = np.array([r.area for r in records], dtype=np.float32)
    aspects = np.array([r.aspect_ratio for r in records], dtype=np.float32)
    margins = np.vstack([r.margins for r in records])

    area_stats = summarize_numeric(areas)
    aspect_stats = summarize_numeric(aspects)
    margin_stats = {
        "left": summarize_numeric(margins[:, 0]),
        "top": summarize_numeric(margins[:, 1]),
        "right": summarize_numeric(margins[:, 2]),
        "bottom": summarize_numeric(margins[:, 3]),
    }

    area_balance = classify_bins(areas, bins=bins, threshold=args.threshold)
    ext_counter = Counter(p.suffix.lower() for p in origin_files)

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir = output_base / f"dataset_analise_{now}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    report_path = run_output_dir / "resumo.txt"
    csv_path = run_output_dir / "balanceamento_faixas.csv"
    catalog_csv_path = run_output_dir / "catalogo_pares.csv"
    plot_path = run_output_dir / "diagnostico_dataset.png"

    write_text_report(
        output_path=report_path,
        total_origin=len(origin_files),
        total_crop=len(crop_files),
        records=records,
        missing_crop=missing_crop,
        orphan_crop=orphan_crop,
        area_stats=area_stats,
        aspect_stats=aspect_stats,
        margin_stats=margin_stats,
        area_balance=area_balance,
        ext_counter=ext_counter,
    )
    write_csv_balance(csv_path, area_balance)
    write_pair_catalog_csv(catalog_csv_path, records, area_balance)
    examples_csv_path = save_example_images(
        output_dir=run_output_dir,
        records=records,
        area_balance=area_balance,
        samples_per_bin=args.samples_per_bin,
    )
    generate_plots(
        output_path=plot_path,
        total_origin=len(origin_files),
        total_crop=len(crop_files),
        records=records,
        missing_crop=missing_crop,
        orphan_crop=orphan_crop,
        area_balance=area_balance,
    )

    log.info("Relatorio salvo em: %s", report_path)
    log.info("CSV salvo em: %s", csv_path)
    log.info("Catalogo de pares salvo em: %s", catalog_csv_path)
    log.info("Indice de exemplos salvo em: %s", examples_csv_path)
    log.info("Exemplos visuais salvos em: %s", run_output_dir / "examples")
    log.info("Analise concluida em: %s", run_output_dir)


if __name__ == "__main__":
    setup_logging()
    start_ts = datetime.datetime.now()
    try:
        main()
    except Exception as exc:
        log.exception("Falha na analise do dataset: %s", exc)
        raise
    finally:
        elapsed = datetime.datetime.now() - start_ts
        elapsed_sec = max(elapsed.total_seconds(), 0.0)
        hh = math.floor(elapsed_sec // 3600)
        mm = math.floor((elapsed_sec % 3600) // 60)
        ss = math.floor(elapsed_sec % 60)
        log.info("Tempo total: %02d:%02d:%02d", hh, mm, ss)
