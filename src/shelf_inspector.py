import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class PromptStrategy(str, Enum):
    ZERO_SHOT = "zero_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"


class ShelfInspector:
    """
    Analisa imagens de prateleiras usando Gemini 1.5 Flash.

    Responsabilidades:
    - Carregar e redimensionar imagens para a API
    - Aplicar a estratégia de prompting selecionada
    - Gerir cache local por hash MD5 para evitar chamadas redundantes
    - Implementar rate limiting com backoff exponencial
    - Retornar JSON de inspeção estruturado e validado
    """

    PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
    CACHE_DIR = Path(os.getenv("CACHE_PATH", "./cache"))
    MAX_IMAGE_DIMENSION = 1536  # Gemini recomenda <= 1536px por lado
    MAX_RETRIES = 6
    INITIAL_BACKOFF = 2.0  # segundos

    def __init__(self, cache_enabled: bool = True):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY não encontrada. "
                "Copia .env.example para .env e preenche a chave."
            )
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
        self.client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature
        self.cache_enabled = cache_enabled
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._prompts: dict[PromptStrategy, str] = {}
        self._load_prompts()
        self._last_call_ts: float = 0.0
        self._rpm_limit = int(os.getenv("RATE_LIMIT_RPM", "15"))
        self._min_interval = 60.0 / self._rpm_limit
        logger.info(
            "ShelfInspector iniciado | modelo=%s | cache=%s | rpm_limit=%d",
            model_name, cache_enabled, self._rpm_limit,
        )

    # Carregamento de prompts

    def _load_prompts(self) -> None:
        mapping = {
            PromptStrategy.ZERO_SHOT: "inspect_zero_shot.txt",
            PromptStrategy.CHAIN_OF_THOUGHT: "inspect_chain_of_thought.txt",
            PromptStrategy.FEW_SHOT: "inspect_few_shot.txt",
        }
        for strategy, filename in mapping.items():
            path = self.PROMPTS_DIR / filename
            if not path.exists():
                raise FileNotFoundError(f"Prompt não encontrado: {path}")
            self._prompts[strategy] = path.read_text(encoding="utf-8")
        logger.debug("Prompts carregados: %s", list(mapping.values()))

    # Cache

    def _image_md5(self, image_path: Path) -> str:
        """Calcula o MD5 do ficheiro de imagem para identificar a versão."""
        h = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _cache_key(self, image_path: Path, strategy: PromptStrategy) -> str:
        md5 = self._image_md5(image_path)
        return f"{md5}_{strategy.value}"

    def _cache_path(self, key: str) -> Path:
        return self.CACHE_DIR / f"{key}.json"

    def _load_from_cache(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                logger.debug("Cache hit: %s", key)
                return data
            except json.JSONDecodeError:
                logger.warning("Cache corrompido, ignorando: %s", key)
        return None

    def _save_to_cache(self, key: str, result: dict) -> None:
        self._cache_path(key).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.debug("Resultado guardado em cache: %s", key)

    # Rate limiting

    def _wait_for_rate_limit(self) -> None:
        """Espera o tempo mínimo necessário entre chamadas à API."""
        elapsed = time.time() - self._last_call_ts
        wait = self._min_interval - elapsed
        if wait > 0:
            logger.debug("Rate limit: a aguardar %.2fs", wait)
            time.sleep(wait)

    # Preparação de imagem

    def _prepare_image(self, image_path: Path) -> Image.Image:
        """Carrega e redimensiona a imagem se necessário."""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > self.MAX_IMAGE_DIMENSION:
            scale = self.MAX_IMAGE_DIMENSION / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            logger.debug(
                "Imagem redimensionada de %dx%d para %dx%d",
                w, h, img.width, img.height,
            )
        return img

    # Chamada à API com retry

    def _call_api_with_retry(self, prompt: str, image: Image.Image) -> str:
        """
        Faz a chamada à API com backoff exponencial em caso de erro 429.
        Lança RuntimeError se a quota diária estiver esgotada.
        """
        backoff = self.INITIAL_BACKOFF
        for attempt in range(1, self.MAX_RETRIES + 1):
            self._wait_for_rate_limit()
            try:
                logger.debug("Chamada à API (tentativa %d/%d)", attempt, self.MAX_RETRIES)
                response = self.client.models.generate_content(
                    model=self._model_name,
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(temperature=self._temperature),
                )
                self._last_call_ts = time.time()
                return response.text
            except Exception as exc:
                error_str = str(exc).lower()
                if (
                        "429" in error_str
                        or "quota" in error_str
                        or "resource_exhausted" in error_str
                ):
                    if attempt == self.MAX_RETRIES:
                        raise RuntimeError(
                            "Quota diária da API esgotada. O sistema continua a "
                            "funcionar para imagens em cache. Tenta novamente amanhã."
                        ) from exc
                    logger.warning(
                        "Rate limit (429) na tentativa %d. Backoff de %.1fs.",
                        attempt, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise

    # Parsing e validação do JSON retornado

    def _parse_response(self, raw_text: str, image_path: Path, zone_id: str) -> dict:
        """
        Extrai e valida o JSON da resposta do modelo.
        Garante que os campos obrigatórios estão presentes.
        """
        # Remove eventuais fences de markdown que o modelo possa incluir
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Resposta não é JSON válido: %s", raw_text[:300])
            raise ValueError(f"O modelo não retornou JSON válido: {exc}") from exc

        # Preenche campos que o modelo pode ter omitido
        now = datetime.now(timezone.utc).isoformat()
        data.setdefault("inspection_id", self._generate_inspection_id())
        data.setdefault("timestamp", now)
        data.setdefault("image_path", str(image_path))
        data.setdefault("zone_id", zone_id)
        data.setdefault("overall_status", "ok")
        data.setdefault("issues", [])
        data.setdefault("shelf_fill_rate", 1.0)
        data.setdefault("products_detected", [])
        data.setdefault("model_reasoning", "")

        # Garante issue_ids únicos caso o modelo não os tenha gerado
        for i, issue in enumerate(data.get("issues", []), start=1):
            issue.setdefault("issue_id", f"ISS_{i:03d}")
            issue.setdefault("confidence", 0.5)
            issue.setdefault("affected_area_pct", 0.0)

        return data

    @staticmethod
    def _generate_inspection_id() -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"INS_{ts}"

    # Método principal


    def inspect(
        self,
        image_path: str | Path,
        zone_id: str,
        strategy: PromptStrategy = PromptStrategy.CHAIN_OF_THOUGHT,
    ) -> dict:
        """
        Analisa uma imagem de prateleira e retorna o relatório de inspeção.

        Args:
            image_path: Caminho para a imagem.
            zone_id: Identificador da zona (ex: "Z_S3").
            strategy: Estratégia de prompting a usar.

        Returns:
            Dicionário com o relatório de inspeção no schema definido.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        # Verificar cache
        cache_key = self._cache_key(image_path, strategy)
        if self.cache_enabled:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        logger.info(
            "A inspecionar: %s | zona=%s | estratégia=%s",
            image_path.name, zone_id, strategy.value,
        )

        prompt = self._prompts[strategy]
        image = self._prepare_image(image_path)

        raw_response = self._call_api_with_retry(prompt, image)
        result = self._parse_response(raw_response, image_path, zone_id)

        # Actualiza campos dinâmicos
        result["image_path"] = str(image_path)
        result["zone_id"] = zone_id

        if self.cache_enabled:
            self._save_to_cache(cache_key, result)

        logger.info(
            "Inspeção concluída: %s | status=%s | fill_rate=%.0f%% | issues=%d",
            result["inspection_id"],
            result["overall_status"],
            result["shelf_fill_rate"] * 100,
            len(result["issues"]),
        )
        return result

    def inspect_batch(
        self,
        images_dir: str | Path,
        zone_id: str,
        strategy: PromptStrategy = PromptStrategy.CHAIN_OF_THOUGHT,
    ) -> list[dict]:
        """
        Inspeciona todas as imagens de uma diretoria.

        Args:
            images_dir: Diretoria com imagens (.jpg, .jpeg, .png).
            zone_id: Zona a associar a todas as imagens.
            strategy: Estratégia de prompting.

        Returns:
            Lista de resultados de inspeção.
        """
        images_dir = Path(images_dir)
        supported = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in supported]

        if not image_files:
            logger.warning("Nenhuma imagem encontrada em: %s", images_dir)
            return []

        logger.info("Batch: %d imagens a processar em %s", len(image_files), images_dir)
        results = []
        for img_path in sorted(image_files):
            try:
                result = self.inspect(img_path, zone_id, strategy)
                results.append(result)
            except RuntimeError as exc:
                # Quota esgotada — parar graciosamente
                logger.error("Quota esgotada durante batch: %s", exc)
                print(f"\n {exc}")
                break
            except Exception as exc:
                logger.error("Erro a processar %s: %s", img_path.name, exc)
                results.append({
                    "inspection_id": ShelfInspector._generate_inspection_id(),
                    "image_path": str(img_path),
                    "zone_id": zone_id,
                    "overall_status": "error",
                    "error": str(exc),
                    "issues": [],
                    "shelf_fill_rate": None,
                    "products_detected": [],
                    "model_reasoning": "",
                })

        return results


# Execução directa para testes

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if len(sys.argv) < 3:
        print("Uso: python shelf_inspector.py <imagem> <zona_id> [estrategia]")
        print("Estratégias: zero_shot | chain_of_thought | few_shot")
        sys.exit(1)

    img = sys.argv[1]
    zone = sys.argv[2]
    strat = PromptStrategy(sys.argv[3]) if len(sys.argv) > 3 else PromptStrategy.CHAIN_OF_THOUGHT

    inspector = ShelfInspector()
    result = inspector.inspect(img, zone, strat)
    print(json.dumps(result, ensure_ascii=False, indent=2))
