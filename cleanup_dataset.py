from pathlib import Path
from typing import List, Set, Tuple

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def listar_arquivos(pasta: Path) -> List[Path]:
    if not pasta.exists():
        return []
    return [
        p for p in sorted(pasta.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

def agrupar_por_stem(arquivos: List[Path]) -> dict:
    d = {}
    for f in arquivos:
        d.setdefault(f.stem, []).append(f)
    return d

def encontrar_editado(cropped_por_stem: dict, stem_original: str) -> List[Path]:
    # Procura exatamente stem+"_editado"
    chave = f"{stem_original}_editado"
    return cropped_por_stem.get(chave, [])

def encontrar_original(origin_por_stem: dict, stem_editado: str) -> List[Path]:
    # stem_editado termina com _editado
    if not stem_editado.endswith("_editado"):
        return []
    base = stem_editado[: -len("_editado")]
    return origin_por_stem.get(base, [])

def analisar(dataset_dir: Path) -> Tuple[List[Path], List[Path]]:
    origin_dir = dataset_dir / "origin"
    cropped_dir = dataset_dir / "cropped"

    origin_files = listar_arquivos(origin_dir)
    cropped_files = listar_arquivos(cropped_dir)

    cropped_por_stem = agrupar_por_stem(cropped_files)

    originais_sem_editado: List[Path] = []
    for arquivo in origin_files:
        correspondentes = encontrar_editado(cropped_por_stem, arquivo.stem)
        if correspondentes:
            correspondentes.pop()
        else:
            originais_sem_editado.append(arquivo)

    editados_sem_original: List[Path] = []
    for stem, arquivos in cropped_por_stem.items():
        if stem.endswith("_editado") and arquivos:
            editados_sem_original.extend(arquivos)

    return originais_sem_editado, editados_sem_original

def imprimir_lista(titulo: str, arquivos: List[Path]):
    print(f"\n{titulo} ({len(arquivos)}):")
    for f in arquivos:
        print(f"  - {f.name}")

def excluir(arquivos: List[Path]):
    for f in arquivos:
        try:
            f.unlink()
            print(f"Excluído: {f}")
        except OSError as e:
            print(f"Falha ao excluir {f}: {e}")

def main():
    project_root = Path(__file__).resolve().parent
    dataset_dir = project_root / "dataset"
    if not dataset_dir.exists():
        print("Pasta 'dataset' não encontrada.")
        return

    originais_sem_editado, editados_sem_original = analisar(dataset_dir)

    if not originais_sem_editado and not editados_sem_original:
        print("Nenhum arquivo órfão encontrado. Nada a fazer.")
        return

    if originais_sem_editado:
        imprimir_lista("Originais sem correspondente editado", originais_sem_editado)
    if editados_sem_original:
        imprimir_lista("Editados sem correspondente original", editados_sem_original)

    total = len(originais_sem_editado) + len(editados_sem_original)
    resp = input(f"\nDeseja excluir estes {total} arquivos órfãos? (s/N): ").strip().lower()
    if resp != "s":
        print("Nenhuma exclusão realizada.")
        return

    print("\nIniciando exclusão...")
    excluir(originais_sem_editado)
    excluir(editados_sem_original)
    print("\nConcluído.")

if __name__ == "__main__":
    main()
