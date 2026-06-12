import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Motor de regras que traduz linguagem natural em lógica executável.

    Responsabilidades:
    - Converter regras em texto para JSON estruturado via LLM
    - Detectar ambiguidades e solicitar esclarecimentos
    - Persistir regras em disco
    - Executar regras sobre resultados de inspeção
    - Gerar logs de execução detalhados
    """

    PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
    RULES_DIR = Path(os.getenv("RULES_PATH", "./data/rules"))
    RULES_FILE = "rules.json"

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY não encontrada.")

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
        self.client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature

        self.RULES_DIR.mkdir(parents=True, exist_ok=True)

        self._rule_prompt_template = (
            (self.PROMPTS_DIR / "rule_convert.txt").read_text(encoding="utf-8")
        )

        # Carrega regras persistidas
        self._rules: list[dict] = self._load_rules_from_disk()

        logger.info(
            "RuleEngine iniciado | %d regras carregadas", len(self._rules)
        )

    # Persistência

    def _rules_path(self) -> Path:
        return self.RULES_DIR / self.RULES_FILE

    def _load_rules_from_disk(self) -> list[dict]:
        path = self._rules_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Erro ao carregar regras: %s", exc)
        return []

    def _save_rules_to_disk(self) -> None:
        self._rules_path().write_text(
            json.dumps(self._rules, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("Regras guardadas em disco (%d regras)", len(self._rules))

    # Conversão de linguagem natural → JSON

    def _call_llm(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self._model_name, contents=prompt, config=types.GenerateContentConfig(temperature=self._temperature))
        return response.text.strip()

    def _generate_rule_id(self) -> str:
        existing_ids = {r.get("rule_id", "") for r in self._rules}
        n = len(self._rules) + 1
        while f"RULE_{n:03d}" in existing_ids:
            n += 1
        return f"RULE_{n:03d}"

    def parse_rule(self, natural_language: str) -> dict:
        """
        Converte uma regra em linguagem natural para JSON estruturado.

        Retorna o dicionário da regra. Se houver ambiguidades, o campo
        validation.is_valid será False e ambiguities listará os problemas.
        Não persiste automaticamente — usa add_rule() para guardar.
        """
        prompt = self._rule_prompt_template.replace("{RULE_TEXT}", natural_language)
        raw = self._call_llm(prompt)

        # Remove fences de markdown
        text = raw
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()

        try:
            rule = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("LLM não retornou JSON válido para regra: %s", raw[:300])
            raise ValueError(f"Conversão de regra falhou: {exc}") from exc

        # Garante campos obrigatórios
        rule["rule_id"] = self._generate_rule_id()
        rule["created_at"] = datetime.now(timezone.utc).isoformat()
        rule.setdefault("natural_language", natural_language)
        rule.setdefault("validation", {
            "is_valid": True,
            "ambiguities": [],
            "assumptions": [],
        })

        logger.info(
            "Regra convertida: %s | válida=%s | ambiguidades=%d",
            rule["rule_id"],
            rule["validation"].get("is_valid"),
            len(rule["validation"].get("ambiguities", [])),
        )
        return rule

    # Gestão de regras

    def add_rule(self, rule: dict) -> str:
        """
        Persiste uma regra já convertida. Retorna o rule_id.
        Só aceita regras com validation.is_valid == True.
        """
        if not rule.get("validation", {}).get("is_valid", False):
            ambiguities = rule["validation"].get("ambiguities", [])
            raise ValueError(
                "Regra inválida ou ambígua. Resolve as ambiguidades antes de guardar:\n"
                + "\n".join(f"  • {a}" for a in ambiguities)
            )

        # Evita duplicados
        for existing in self._rules:
            if existing.get("natural_language") == rule.get("natural_language"):
                logger.warning("Regra já existe com o mesmo texto: %s", existing["rule_id"])

        self._rules.append(rule)
        self._save_rules_to_disk()
        logger.info("Regra guardada: %s", rule["rule_id"])
        return rule["rule_id"]

    def delete_rule(self, rule_id: str) -> bool:
        """Remove uma regra pelo ID. Retorna True se encontrada e removida."""
        original_count = len(self._rules)
        self._rules = [r for r in self._rules if r.get("rule_id") != rule_id]
        if len(self._rules) < original_count:
            self._save_rules_to_disk()
            logger.info("Regra removida: %s", rule_id)
            return True
        logger.warning("Regra não encontrada: %s", rule_id)
        return False

    def list_rules(self) -> list[dict]:
        """Retorna todas as regras guardadas."""
        return list(self._rules)

    def get_rule(self, rule_id: str) -> Optional[dict]:
        """Retorna uma regra por ID."""
        for rule in self._rules:
            if rule.get("rule_id") == rule_id:
                return rule
        return None


    # Execução de regras sobre resultados de inspeção

    def _matches_conditions(self, rule: dict, inspection: dict) -> bool:
        """
        Verifica se um resultado de inspeção satisfaz as condições de uma regra.
        Retorna True se a regra deve disparar.
        """
        cond = rule.get("conditions", {})

        # Filtro de zona
        zone_filter = cond.get("zone_filter", [])
        if zone_filter and inspection.get("zone_id") not in zone_filter:
            return False

        # Filtro de hora
        time_filter = cond.get("time_filter", {})
        if time_filter and (time_filter.get("hours_start") is not None):
            try:
                ts = datetime.fromisoformat(
                    inspection.get("timestamp", "").replace("Z", "+00:00")
                )
                hour = ts.hour
                h_start = time_filter["hours_start"]
                h_end = time_filter.get("hours_end", 23)
                if not (h_start <= hour <= h_end):
                    return False
            except (ValueError, AttributeError):
                pass  # Se não conseguirmos parsear a hora, ignoramos o filtro

        # Filtro de fill rate
        fill_threshold = cond.get("fill_rate_threshold")
        if fill_threshold is not None:
            if inspection.get("shelf_fill_rate", 1.0) >= fill_threshold:
                return False

        # Filtro de tipo de issue
        issue_types = cond.get("issue_types", [])
        if issue_types:
            detected_types = {i.get("type") for i in inspection.get("issues", [])}
            if not detected_types.intersection(set(issue_types)):
                return False

        # Filtro de severidade
        severity_threshold = cond.get("severity_threshold")
        severity_order = {"low": 1, "medium": 2, "high": 3}
        if severity_threshold:
            threshold_val = severity_order.get(severity_threshold, 0)
            max_severity = max(
                (severity_order.get(i.get("severity", "low"), 0)
                 for i in inspection.get("issues", [])),
                default=0,
            )
            if max_severity < threshold_val:
                return False

        # Filtro de localização (location_filter)
        location_filter = cond.get("location_filter", "any")
        if location_filter and location_filter != "any":
            matching_issues = [
                i for i in inspection.get("issues", [])
                if location_filter.lower() in i.get("location", "").lower()
            ]
            if not matching_issues:
                return False

        return True

    def _render_notification(self, rule: dict, inspection: dict) -> str:
        """Preenche o template de notificação com dados reais da inspeção."""
        template = rule.get("action", {}).get(
            "notification_message", "Alerta da regra {rule_id} disparado."
        )

        # Encontra o issue mais severo para incluir na mensagem
        issues = inspection.get("issues", [])
        severity_order = {"high": 3, "medium": 2, "low": 1}
        worst_issue = max(
            issues,
            key=lambda i: severity_order.get(i.get("severity", "low"), 0),
            default={},
        )

        return template.format(
            zone_id=inspection.get("zone_id", "N/A"),
            issue_type=worst_issue.get("type", "N/A"),
            severity=worst_issue.get("severity", "N/A"),
            fill_rate=f"{inspection.get('shelf_fill_rate', 0) * 100:.0f}%",
            timestamp=inspection.get("timestamp", "N/A"),
            rule_id=rule.get("rule_id", "N/A"),
        )

    def evaluate(self, inspection: dict) -> list[dict]:
        """
        Avalia todas as regras activas contra um resultado de inspeção.

        Returns:
            Lista de alertas gerados (um por regra disparada).
            Cada alerta contém: rule_id, alert_level, message, triggered_at.
        """
        if not self._rules:
            logger.debug("Nenhuma regra para avaliar.")
            return []

        alerts = []
        log_lines = []

        for rule in self._rules:
            rule_id = rule.get("rule_id", "?")

            if not rule.get("validation", {}).get("is_valid", True):
                log_lines.append(f"  [SKIP] {rule_id} — regra inválida, a ignorar")
                continue

            triggered = self._matches_conditions(rule, inspection)
            status = "TRIGGERED" if triggered else "NOT_TRIGGERED"
            log_lines.append(f"  [{status}] {rule_id}")

            if triggered:
                message = self._render_notification(rule, inspection)
                alert = {
                    "rule_id": rule_id,
                    "alert_level": rule.get("action", {}).get("alert_level", "info"),
                    "message": message,
                    "triggered_at": datetime.now(timezone.utc).isoformat(),
                    "inspection_id": inspection.get("inspection_id"),
                    "zone_id": inspection.get("zone_id"),
                }
                alerts.append(alert)
                logger.info("Regra disparada: %s | level=%s", rule_id, alert["alert_level"])

        # Log de execução completo
        execution_log = "\n".join(log_lines)
        logger.debug(
            "Execução de regras para %s:\n%s",
            inspection.get("inspection_id"), execution_log,
        )

        return alerts

    def test_rule(self, rule_id: str, inspection: dict) -> dict:
        """
        Testa uma regra específica contra uma inspeção (sem persistência).
        Útil para verificar antes de guardar.
        """
        rule = self.get_rule(rule_id)
        if not rule:
            raise ValueError(f"Regra não encontrada: {rule_id}")

        triggered = self._matches_conditions(rule, inspection)
        return {
            "rule_id": rule_id,
            "triggered": triggered,
            "message": self._render_notification(rule, inspection) if triggered else None,
            "conditions_checked": rule.get("conditions", {}),
        }



# Execução directa

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if len(sys.argv) < 2:
        print("Uso: python rule_engine.py <regra em texto>")
        sys.exit(1)

    rule_text = " ".join(sys.argv[1:])
    engine = RuleEngine()
    rule = engine.parse_rule(rule_text)

    print("\nRegra Convertida")
    print(json.dumps(rule, ensure_ascii=False, indent=2))

    if not rule["validation"]["is_valid"]:
        print("\nRegra ambígua. Ambiguidades detectadas:")
        for amb in rule["validation"]["ambiguities"]:
            print(f"  • {amb}")
    else:
        rule_id = engine.add_rule(rule)
        print(f"\nRegra guardada com ID: {rule_id}")