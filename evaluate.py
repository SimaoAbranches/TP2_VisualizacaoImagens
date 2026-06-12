import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from shelf_inspector import ShelfInspector, PromptStrategy
from rule_engine import RuleEngine
from rag_memory import RAGMemory

logger = logging.getLogger(__name__)


# Ground truth loader

def load_ground_truth(images_dir: Path) -> dict:
    """
    Carrega o ground truth de um ficheiro ground_truth.json na diretoria de teste.

    Formato esperado:
    {
      "image_name.jpg": {
        "zone_id": "Z_S1",
        "overall_status": "warning",
        "issues": [{"type": "empty_shelf", "severity": "medium", "location": "..."}],
        "shelf_fill_rate": 0.65
      }
    }
    """
    gt_path = images_dir / "ground_truth.json"
    if not gt_path.exists():
        logger.warning(
            "ground_truth.json não encontrado em %s. "
            "A usar avaliação sem ground truth.", images_dir
        )
        return {}
    return json.loads(gt_path.read_text(encoding="utf-8"))


# Métricas de análise visual

class VisualMetrics:
    """Calcula métricas de qualidade da análise visual."""

    @staticmethod
    def issue_detection_rate(predicted: list[dict], ground_truth: list[dict]) -> float:
        """
        Recall: % de issues do ground truth correctamente identificados.
        Um issue é considerado detectado se o tipo corresponde.
        """
        if not ground_truth:
            return 1.0 if not predicted else 0.0

        gt_types = [i.get("type") for i in ground_truth]
        pred_types = [i.get("type") for i in predicted]
        detected = sum(1 for t in gt_types if t in pred_types)
        return detected / len(gt_types)

    @staticmethod
    def false_positive_rate(predicted: list[dict], ground_truth: list[dict]) -> float:
        """% de issues reportados que não existem no ground truth."""
        if not predicted:
            return 0.0

        gt_types = set(i.get("type") for i in ground_truth)
        false_positives = sum(1 for p in predicted if p.get("type") not in gt_types)
        return false_positives / len(predicted)

    @staticmethod
    def severity_accuracy(predicted: list[dict], ground_truth: list[dict]) -> float:
        """% de issues com severidade correctamente classificada."""
        if not ground_truth or not predicted:
            return 0.0

        correct = 0
        matched = 0
        for gt_issue in ground_truth:
            for pred_issue in predicted:
                if gt_issue.get("type") == pred_issue.get("type"):
                    matched += 1
                    if gt_issue.get("severity") == pred_issue.get("severity"):
                        correct += 1
                    break

        return correct / matched if matched > 0 else 0.0

    @staticmethod
    def json_parse_rate(raw_responses: list[str]) -> float:
        """% de respostas do modelo que são JSON válido parseável."""
        if not raw_responses:
            return 0.0
        valid = 0
        for r in raw_responses:
            text = r.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(l for l in lines if not l.startswith("```")).strip()
            try:
                json.loads(text)
                valid += 1
            except json.JSONDecodeError:
                pass
        return valid / len(raw_responses)


# Métricas de RAG

class RAGMetrics:
    """Avalia qualidade do sistema de recuperação."""

    def __init__(self, rag: RAGMemory):
        self.rag = rag

    def recall_at_k(self, queries_with_relevant: list[dict], k: int = 3) -> float:
        """
        Recall@k: % de queries onde o documento relevante está nos top-k resultados.

        queries_with_relevant: [{"query": "...", "relevant_inspection_id": "INS_..."}]
        """
        if not queries_with_relevant:
            return 0.0

        hits = 0
        for item in queries_with_relevant:
            results = self.rag.retrieve(item["query"], top_k=k)
            retrieved_ids = [r["inspection_id"] for r in results]
            if item["relevant_inspection_id"] in retrieved_ids:
                hits += 1

        return hits / len(queries_with_relevant)


# Métricas do Rule Engine

class RuleEngineMetrics:
    """Avalia qualidade da conversão e execução de regras."""

    def __init__(self, engine: RuleEngine):
        self.engine = engine

    def rule_parse_rate(self, test_rules: list[str]) -> float:
        """% de regras convertidas em JSON válido."""
        if not test_rules:
            return 0.0
        parsed = 0
        for rule_text in test_rules:
            try:
                self.engine.parse_rule(rule_text)
                parsed += 1
            except Exception:
                pass
        return parsed / len(test_rules)

    def ambiguity_detection_rate(
            self,
            rules_with_labels: list[dict],
    ) -> float:
        """
        % de regras ambíguas corretamente identificadas.
        rules_with_labels: [{"text": "...", "is_ambiguous": True/False}]
        """
        if not rules_with_labels:
            return 0.0

        correct = 0
        for item in rules_with_labels:
            try:
                rule = self.engine.parse_rule(item["text"])
                predicted_ambiguous = not rule["validation"].get("is_valid", True)
                if predicted_ambiguous == item["is_ambiguous"]:
                    correct += 1
            except Exception:
                pass

        return correct / len(rules_with_labels)


# LLM-as-Judge

class LLMJudge:
    """
    Usa Gemini Flash como juiz para avaliação qualitativa.
    Avalia relatórios e respostas RAG com critérios definidos.
    """

    PROMPTS_DIR = Path(__file__).parent / "prompts"

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config=genai.GenerationConfig(temperature=0.0),
        )
        self._judge_prompt = (
            (self.PROMPTS_DIR / "llm_judge.txt").read_text(encoding="utf-8")
        )

    def evaluate(
            self,
            output: str,
            criterion: str,
            reference: str = "N/A",
    ) -> dict:
        """
        Avalia um output com um critério específico.
        Retorna: {score, justification, strengths, weaknesses, agrees_with_human}
        """
        prompt = (
            self._judge_prompt
            .replace("{CRITERION}", criterion)
            .replace("{OUTPUT}", output[:2000])  # limita para não exceder contexto
            .replace("{REFERENCE}", reference[:500])
        )
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(l for l in lines if not l.startswith("```")).strip()
            return json.loads(text)
        except Exception as exc:
            logger.error("LLM Judge falhou: %s", exc)
            return {"score": -1, "justification": str(exc), "error": True}


# Harness principal

def run_evaluation(images_dir: Path, output_path: Path) -> dict:
    """
    Executa o harness de avaliação completo.
    Retorna o dicionário com todos os resultados.
    """
    logger.info("=== Início da Avaliação ===")
    logger.info("Directoria de teste: %s", images_dir)

    ground_truth = load_ground_truth(images_dir)

    # Inicializa componentes
    inspector = ShelfInspector(cache_enabled=True)
    rule_engine = RuleEngine()
    rag = RAGMemory()
    judge = LLMJudge()

    # 1. Avaliação da análise visual (3 estratégias)

    logger.info("--- Avaliação Visual ---")

    visual_results = {}
    strategies = [
        PromptStrategy.ZERO_SHOT,
        PromptStrategy.CHAIN_OF_THOUGHT,
        PromptStrategy.FEW_SHOT,
    ]

    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
           and f.name != "ground_truth.json"
    ])

    for strategy in strategies:
        logger.info("Estratégia: %s", strategy.value)
        detection_rates, fp_rates, severity_accs = [], [], []

        for img_path in image_files:
            gt = ground_truth.get(img_path.name, {})
            gt_issues = gt.get("issues", [])

            try:
                result = inspector.inspect(img_path, gt.get("zone_id", "Z_TEST"), strategy)
                pred_issues = result.get("issues", [])

                if gt_issues or pred_issues:
                    detection_rates.append(
                        VisualMetrics.issue_detection_rate(pred_issues, gt_issues)
                    )
                    fp_rates.append(
                        VisualMetrics.false_positive_rate(pred_issues, gt_issues)
                    )
                    severity_accs.append(
                        VisualMetrics.severity_accuracy(pred_issues, gt_issues)
                    )

            except Exception as exc:
                logger.warning("Erro na imagem %s: %s", img_path.name, exc)

        visual_results[strategy.value] = {
            "issue_detection_rate": (
                sum(detection_rates) / len(detection_rates) if detection_rates else 0.0
            ),
            "false_positive_rate": (
                sum(fp_rates) / len(fp_rates) if fp_rates else 0.0
            ),
            "severity_accuracy": (
                sum(severity_accs) / len(severity_accs) if severity_accs else 0.0
            ),
            "n_images_evaluated": len(detection_rates),
        }
        logger.info(
            "%s → IDR=%.2f | FPR=%.2f | SEV=%.2f",
            strategy.value,
            visual_results[strategy.value]["issue_detection_rate"],
            visual_results[strategy.value]["false_positive_rate"],
            visual_results[strategy.value]["severity_accuracy"],
        )

    # 2. Avaliação do RAG

    logger.info("--- Avaliação RAG ---")

    # Queries de teste com ground truth manual

    rag_test_queries = [
        {
            "query": "prateleira vazia zona Z_S1",
            "relevant_inspection_id": "PLACEHOLDER_ID_1",
        },
        {
            "query": "produto danificado prateleira inferior",
            "relevant_inspection_id": "PLACEHOLDER_ID_2",
        },
        {
            "query": "fill rate baixo terça-feira",
            "relevant_inspection_id": "PLACEHOLDER_ID_3",
        },
    ]
    # Filtra placeholders não substituídos
    valid_rag_queries = [
        q for q in rag_test_queries
        if not q["relevant_inspection_id"].startswith("PLACEHOLDER")
    ]

    rag_metrics_calc = RAGMetrics(rag)
    recall_at_3 = (
        rag_metrics_calc.recall_at_k(valid_rag_queries, k=3)
        if valid_rag_queries else None
    )

    rag_results = {
        "recall_at_3": recall_at_3,
        "total_indexed": rag.count(),
        "note": (
            "Preenche os relevant_inspection_id no harness com IDs reais para Recall@3 válido."
            if not valid_rag_queries else ""
        ),
    }

    # 3. Avaliação do Rule Engine

    logger.info("--- Avaliação Rule Engine ---")

    test_rules_clear = [
        "Avisa-me quando a zona Z_S1 tiver fill rate abaixo de 50%.",
        "Se houver um produto tombado, é sempre severidade alta.",
        "Na zona Z_S3, se a prateleira inferior estiver vazia entre as 9h e as 12h, alerta crítico.",
    ]
    test_rules_ambiguous = [
        "Avisa-me quando a prateleira estiver vazia.",  # ambíguo: que prateleira? que nível?
        "Se houver problemas, diz-me.",  # muito vago
    ]
    all_test_rules_labeled = (
            [{"text": r, "is_ambiguous": False} for r in test_rules_clear]
            + [{"text": r, "is_ambiguous": True} for r in test_rules_ambiguous]
    )

    re_metrics = RuleEngineMetrics(rule_engine)
    rule_parse_rate = re_metrics.rule_parse_rate(test_rules_clear + test_rules_ambiguous)
    ambiguity_rate = re_metrics.ambiguity_detection_rate(all_test_rules_labeled)

    rule_results = {
        "rule_parse_rate": rule_parse_rate,
        "ambiguity_detection_rate": ambiguity_rate,
        "n_rules_tested": len(all_test_rules_labeled),
    }


    # 4. LLM-as-Judge (qualitativo)

    logger.info("--- LLM-as-Judge ---")

    # Avalia um exemplo de resposta RAG
    if rag.count() > 0:
        sample_query = "Que zonas tiveram mais problemas recentemente?"
        rag_response = rag.query(sample_query)
        judge_rag = judge.evaluate(
            output=rag_response["answer"],
            criterion="A resposta responde directamente à pergunta com referências a inspeções específicas (IDs e datas)?",
            reference=sample_query,
        )
    else:
        judge_rag = {"score": -1, "justification": "Sem dados no RAG para avaliar."}

    judge_results = {
        "rag_answer_relevance": judge_rag,
    }


    # Compila e guarda resultados

    evaluation_report = {
        "evaluation_date": datetime.now().isoformat(),
        "images_dir": str(images_dir),
        "n_test_images": len(image_files),
        "visual_analysis": visual_results,
        "rag_evaluation": rag_results,
        "rule_engine_evaluation": rule_results,
        "llm_judge": judge_results,
        "summary": {
            "best_visual_strategy": max(
                visual_results,
                key=lambda s: visual_results[s]["issue_detection_rate"],
            ) if visual_results else "N/A",
            "recall_at_3": rag_results.get("recall_at_3"),
            "rule_parse_rate": rule_results["rule_parse_rate"],
            "ambiguity_detection_rate": rule_results["ambiguity_detection_rate"],
        },
    }

    output_path.write_text(
        json.dumps(evaluation_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Relatório de avaliação guardado: %s", output_path)

    # Print sumário
    print("\n=== RESULTADOS DE AVALIAÇÃO ===")
    print(f"\nImagens testadas: {len(image_files)}")
    print("\n[ Análise Visual ]")
    for strat, metrics in visual_results.items():
        print(
            f"  {strat:25s} → IDR={metrics['issue_detection_rate']:.2f} | "
            f"FPR={metrics['false_positive_rate']:.2f} | "
            f"SEV={metrics['severity_accuracy']:.2f}"
        )
    print(f"\n[ RAG ] Recall@3: {recall_at_3}")
    print(f"\n[ Rule Engine ] Parse Rate: {rule_parse_rate:.2f} | "
          f"Ambiguity Detection: {ambiguity_rate:.2f}")
    print(f"\n[ LLM Judge ] Resposta RAG: {judge_rag.get('score', 'N/A')}/5")
    print(f"\nRelatório completo: {output_path}")

    return evaluation_report



# Entry point


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Harness de avaliação do Retail Vision Intelligence System"
    )
    parser.add_argument(
        "--_sku110k_raw-dir",
        type=Path,
        default=Path("test_images/"),
        help="Directoria com imagens de teste e ground_truth.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_report.json"),
        help="Ficheiro de saída com os resultados",
    )
    args = parser.parse_args()

    if not args.images_dir.exists():
        print(f"Erro: directoria não encontrada: {args.images_dir}")
        sys.exit(1)

    run_evaluation(args.images_dir, args.output)