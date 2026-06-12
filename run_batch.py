import argparse
import json
import logging
import sys
import os
from dotenv import load_dotenv
load_dotenv()
import time
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("batch_run.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT    = Path(__file__).parent
CHECKPOINT_FILE = PROJECT_ROOT / "cache" / "batch_checkpoint.json"
IMAGES_DIR      = PROJECT_ROOT / "data" / "images"

CATEGORIES = ["normal", "empty", "planogram_violation", "dirty_messy", "ambiguous"]
ZONE_MAP = {
    "normal": "Z_S1", "empty": "Z_S2", "planogram_violation": "Z_S3",
    "dirty_messy": "Z_S4", "ambiguous": "Z_S1",
}
BATCH_SIZE    = 10
PAUSE_SECONDS = 45


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"processed": [], "failed": [], "quota_hit": False, "last_run": None, "total_api_calls": 0}


def save_checkpoint(checkpoint):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["last_run"] = datetime.now().isoformat()
    CHECKPOINT_FILE.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_checkpoint():
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    print("Checkpoint apagado.")


def collect_pending_images(checkpoint):
    processed_set = set(checkpoint["processed"])
    pending = []
    for category in CATEGORIES:
        cat_dir = IMAGES_DIR / category
        if not cat_dir.exists():
            continue
        zone_id = ZONE_MAP[category]
        images = sorted(f for f in cat_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for img_path in images:
            if str(img_path) not in processed_set:
                pending.append((img_path, zone_id))
    return pending


def print_status(checkpoint=None):
    if checkpoint is None:
        checkpoint = load_checkpoint()
    total_available = sum(
        len([f for f in (IMAGES_DIR / cat).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        for cat in CATEGORIES if (IMAGES_DIR / cat).exists()
    )
    processed = len(checkpoint["processed"])
    print(f"\n{'='*55}")
    print(f"  Imagens disponíveis:  {total_available}")
    print(f"  Processadas (cache):  {processed}")
    print(f"  Por processar:        {total_available - processed}")
    print(f"  Falhadas:             {len(checkpoint['failed'])}")
    print(f"  Chamadas API totais:  {checkpoint.get('total_api_calls', 0)}")
    print(f"  Último run:           {checkpoint.get('last_run', 'nunca')}")
    if checkpoint.get("quota_hit"):
        print(f"  Parou por quota — retoma às 08h00 (hora PT)")
    print(f"{'='*55}\n")


def run_batch(dry_run=False):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from shelf_inspector import ShelfInspector, PromptStrategy
    from rag_memory import RAGMemory

    checkpoint = load_checkpoint()
    if checkpoint["quota_hit"]:
        print("\n O último run parou por quota. A retomar...\n")
        checkpoint["quota_hit"] = False

    pending = collect_pending_images(checkpoint)

    print(f"\n{'='*55}")
    print(f"  Imagens já processadas:  {len(checkpoint['processed'])}")
    print(f"  Imagens por processar:   {len(pending)}")
    print(f"  Lotes de {BATCH_SIZE} com pausa de {PAUSE_SECONDS}s")
    print(f"  Estratégia:              chain_of_thought")
    print(f"  Modelo:                  {os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}")
    print(f"{'='*55}\n")

    if dry_run:
        print("  (dry-run: nenhuma chamada à API será feita)")
        return

    if not pending:
        print("Todas as imagens já foram processadas.")
        return

    inspector = ShelfInspector(cache_enabled=True)
    rag = RAGMemory()
    success_count = 0

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Lote {batch_num}/{total_batches} — {len(batch)} imagens...")

        for img_path, zone_id in batch:
            try:
                result = inspector.inspect(img_path, zone_id, PromptStrategy.CHAIN_OF_THOUGHT)
                if result.get("overall_status") != "error":
                    rag.index_inspection(result)
                checkpoint["processed"].append(str(img_path))
                checkpoint["total_api_calls"] += 1
                success_count += 1
                save_checkpoint(checkpoint)
            except RuntimeError as exc:
                if "quota" in str(exc).lower() or "esgotada" in str(exc).lower():
                    checkpoint["quota_hit"] = True
                    save_checkpoint(checkpoint)
                    print(f"\n QUOTA ESGOTADA após {success_count} imagens.")
                    print(f"   Total processado: {len(checkpoint['processed'])} imagens.")
                    print(f"   Retoma amanhã às 08h00 com: python run_batch.py")
                    return
                else:
                    checkpoint["failed"].append(str(img_path))
                    save_checkpoint(checkpoint)
            except KeyboardInterrupt:
                save_checkpoint(checkpoint)
                print(f"\n  Interrompido. Progresso guardado: {len(checkpoint['processed'])} imagens.")
                return
            except Exception as exc:
                logger.error("Erro em %s: %s", img_path.name, exc)
                checkpoint["failed"].append(str(img_path))
                save_checkpoint(checkpoint)

        if batch_start + BATCH_SIZE < len(pending):
            remaining = len(pending) - batch_start - BATCH_SIZE
            eta = (remaining / BATCH_SIZE) * (PAUSE_SECONDS / 60)
            print(f"    ✓ Lote concluído. Pausa de {PAUSE_SECONDS}s... (restam ~{remaining}, ETA ~{eta:.0f} min)")
            time.sleep(PAUSE_SECONDS)

    print(f"\n Processamento concluído! {success_count} imagens processadas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.reset:
        reset_checkpoint()
    elif args.status:
        print_status()
    else:
        run_batch(dry_run=args.dry_run if hasattr(args, "dry_run") else False)
