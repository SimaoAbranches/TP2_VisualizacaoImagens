import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)


class RAGMemory:
    """
    Base de dados vetorial para histórico de inspeções de prateleiras.

    Usa sentence-transformers para embeddings locais (sem chamadas de API adicionais)
    e ChromaDB para armazenamento persistente em disco.

    A estratégia híbrida indexa:
    1. Summary semântico gerado por LLM (para matching por conteúdo)
    2. Metadata estruturada (zona, data, fill_rate) para filtragem eficiente
    """

    PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
    VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", "./vectorstore"))
    COLLECTION_NAME = "shelf_inspections"
    DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY não encontrada.")

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
        self.client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature

        # Modelo de embeddings local — suporta português nativamente
        embedding_model = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        logger.info("A carregar modelo de embeddings: %s", embedding_model)
        self._embedder = SentenceTransformer(embedding_model)

        # ChromaDB persistente em disco
        self.VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(
            path=str(self.VECTORSTORE_PATH)
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Prompts
        self._summary_prompt = (
            (self.PROMPTS_DIR / "rag_summary.txt").read_text(encoding="utf-8")
        )
        self._query_prompt = (
            (self.PROMPTS_DIR / "rag_query.txt").read_text(encoding="utf-8")
        )

        logger.info(
            "RAGMemory iniciado | colecção=%s | documentos=%d",
            self.COLLECTION_NAME,
            self._collection.count(),
        )

    # Geração de summary semântico

    def _generate_summary(self, inspection: dict) -> str:
        """
        Usa o LLM para gerar um summary semântico rico para indexação.
        O summary é otimizado para recuperação futura por similaridade.
        """
        prompt = self._summary_prompt.replace(
            "{INSPECTION_JSON}",
            json.dumps(inspection, ensure_ascii=False, indent=2),
        )
        response = self.client.models.generate_content(model=self._model_name, contents=prompt, config=types.GenerateContentConfig(temperature=self._temperature))
        summary = response.text.strip()

        # Fallback se o LLM retornar algo muito curto
        if len(summary) < 20:
            zone = inspection.get("zone_id", "?")
            fill = inspection.get("shelf_fill_rate", 0)
            status = inspection.get("overall_status", "?")
            n_issues = len(inspection.get("issues", []))
            summary = (
                f"Zona {zone}, status {status}, fill rate {fill*100:.0f}%, "
                f"{n_issues} problema(s) detectado(s)."
            )

        logger.debug("Summary gerado: %s", summary[:100])
        return summary

    # Indexação

    def _build_metadata(self, inspection: dict) -> dict:
        """
        Extrai metadata estruturada para filtragem pre-retrieval.
        ChromaDB só aceita tipos primitivos (str, int, float, bool).
        """
        timestamp_str = inspection.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            weekday = ts.strftime("%A")  # Monday, Tuesday, ...
            hour = ts.hour
            date_str = ts.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            weekday = ""
            hour = -1
            date_str = ""

        issue_types = list({
            i.get("type", "") for i in inspection.get("issues", [])
        })

        return {
            "inspection_id": inspection.get("inspection_id", ""),
            "zone_id": inspection.get("zone_id", ""),
            "overall_status": inspection.get("overall_status", ""),
            "fill_rate": float(inspection.get("shelf_fill_rate", 1.0)),
            "n_issues": int(len(inspection.get("issues", []))),
            "issue_types": ", ".join(issue_types),  # ChromaDB não suporta listas
            "timestamp": timestamp_str,
            "date": date_str,
            "weekday": weekday,
            "hour": hour,
        }

    def index_inspection(self, inspection: dict) -> str:
        """
        Indexa uma inspeção na vector store.

        Args:
            inspection: Dicionário de inspeção no schema padrão.

        Returns:
            ID do documento indexado.
        """
        inspection_id = inspection.get("inspection_id")
        if not inspection_id:
            raise ValueError("inspection_id em falta no registo de inspeção.")

        # Verifica se já está indexado
        existing = self._collection.get(ids=[inspection_id])
        if existing["ids"]:
            logger.debug("Inspeção já indexada, a saltar: %s", inspection_id)
            return inspection_id

        summary = self._generate_summary(inspection)
        embedding = self._embedder.encode(summary).tolist()
        metadata = self._build_metadata(inspection)

        # Guarda também o JSON completo no documento para contexto
        document_text = f"{summary}\n\n[raw_json]\n{json.dumps(inspection, ensure_ascii=False)}"

        self._collection.add(
            ids=[inspection_id],
            documents=[document_text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

        logger.info(
            "Inspeção indexada: %s | zona=%s | status=%s",
            inspection_id,
            metadata["zone_id"],
            metadata["overall_status"],
        )
        return inspection_id

    def index_batch(self, inspections: list[dict]) -> list[str]:
        """Indexa uma lista de inspeções."""
        ids = []
        for insp in inspections:
            try:
                doc_id = self.index_inspection(insp)
                ids.append(doc_id)
            except Exception as exc:
                logger.error(
                    "Erro ao indexar %s: %s",
                    insp.get("inspection_id", "?"), exc,
                )
        logger.info("Batch indexado: %d/%d inspeções", len(ids), len(inspections))
        return ids

    # Retrieval

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        zone_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Recupera os documentos mais relevantes para uma query.

        Args:
            query: Texto de pesquisa em linguagem natural (português).
            top_k: Número de resultados a retornar.
            zone_filter: Se fornecido, restringe a uma zona específica.

        Returns:
            Lista de dicionários com: inspection_id, summary, distance, metadata.
        """
        k = top_k or self.DEFAULT_TOP_K
        query_embedding = self._embedder.encode(query).tolist()

        where_filter = None
        if zone_filter:
            where_filter = {"zone_id": {"$eq": zone_filter}}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, max(self._collection.count(), 1)),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i, doc_id in enumerate(results["ids"][0]):
            retrieved.append({
                "inspection_id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        logger.debug(
            "Retrieval para '%s': %d resultados (top_k=%d)",
            query[:60], len(retrieved), k,
        )
        return retrieved

    # RAG: query com síntese por LLM

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        zone_filter: Optional[str] = None,
    ) -> dict:
        """
        Responde a uma pergunta sobre o histórico usando RAG.

        Args:
            question: Pergunta em linguagem natural do gestor.
            top_k: Número de documentos a recuperar.
            zone_filter: Restrição de zona opcional.

        Returns:
            Dicionário com: answer, sources, retrieved_count.
        """
        if self._collection.count() == 0:
            return {
                "answer": "Ainda não existem inspeções indexadas. "
                          "Realiza inspeções primeiro.",
                "sources": [],
                "retrieved_count": 0,
            }

        retrieved = self.retrieve(question, top_k=top_k, zone_filter=zone_filter)

        # Constrói o contexto para o LLM
        context_parts = []
        for r in retrieved:
            meta = r["metadata"]
            context_parts.append(
                f"[{r['inspection_id']}] {meta.get('date', '')} "
                f"Zona {meta.get('zone_id', '?')} — "
                f"{r['document'].split('[raw_json]')[0].strip()}"
            )
        context = "\n\n".join(context_parts)

        prompt = (
            self._query_prompt
            .replace("{RETRIEVED_CONTEXT}", context)
            .replace("{QUERY}", question)
        )

        response = self.client.models.generate_content(model=self._model_name, contents=prompt, config=types.GenerateContentConfig(temperature=self._temperature))
        answer = response.text.strip()

        logger.info("Query RAG: '%s' → %d docs recuperados", question[:60], len(retrieved))

        return {
            "answer": answer,
            "sources": [
                {
                    "inspection_id": r["inspection_id"],
                    "date": r["metadata"].get("date"),
                    "zone_id": r["metadata"].get("zone_id"),
                    "distance": round(r["distance"], 4),
                }
                for r in retrieved
            ],
            "retrieved_count": len(retrieved),
        }

    # Utilitários

    def count(self) -> int:
        """Retorna o número de documentos indexados."""
        return self._collection.count()

    def get_stats(self) -> dict:
        """Retorna estatísticas sobre a colecção indexada."""
        total = self._collection.count()
        if total == 0:
            return {"total": 0}

        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        zones = {}
        statuses = {}
        for m in all_meta:
            zone = m.get("zone_id", "?")
            status = m.get("overall_status", "?")
            zones[zone] = zones.get(zone, 0) + 1
            statuses[status] = statuses.get(status, 0) + 1

        return {
            "total_inspections": total,
            "by_zone": zones,
            "by_status": statuses,
        }


# Execução directa

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    rag = RAGMemory()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = rag.query(question)
        print(f"\nPergunta: {question}")
        print(f"\nResposta: {result['answer']}")
        print(f"\nFontes ({result['retrieved_count']}):")
        for s in result["sources"]:
            print(f"  • {s['inspection_id']} | {s['date']} | Zona {s['zone_id']}")
    else:
        stats = rag.get_stats()
        print("=== Estatísticas RAG ===")
        print(json.dumps(stats, ensure_ascii=False, indent=2))