import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Gera relatórios de inspeção em Markdown com:
    - Sumário executivo accionável
    - Problemas por zona com histórico RAG
    - Regras disparadas
    - Recomendações ordenadas por urgência
    """

    REPORTS_DIR = Path(os.getenv("INSPECTIONS_PATH", "./data/inspections")) / "reports"

    def __init__(self, rag_memory=None, rule_engine=None):
        """
        Args:
            rag_memory: Instância de RAGMemory para contexto histórico.
            rule_engine: Instância de RuleEngine para alertas.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY não encontrada.")

        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))
        self.client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature

        self.rag = rag_memory
        self.rule_engine = rule_engine

        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("ReportGenerator iniciado")

    # Helpers

    def _severity_emoji(self, status: str) -> str:
        return {"ok", "warning", "critical"}.get(status)

    def _alert_emoji(self, level: str) -> str:
        return {"info", "warning", "critical"}.get(level)

    def _format_fill_rate(self, rate: Optional[float]) -> str:
        if rate is None:
            return "N/A"
        return f"{rate * 100:.0f}%"

    def _format_timestamp(self, ts_str: str) -> str:
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return ts.strftime("%d/%m/%Y às %H:%M")
        except (ValueError, AttributeError):
            return ts_str

    # Secções do relatório

    def _section_executive_summary(
        self, inspections: list[dict], all_alerts: list[dict]
    ) -> str:
        """Gera o sumário executivo via LLM (máx. 150 palavras)."""
        n_zones = len({i.get("zone_id") for i in inspections})
        n_critical = sum(1 for i in inspections if i.get("overall_status") == "critical")
        n_warning = sum(1 for i in inspections if i.get("overall_status") == "warning")
        n_ok = sum(1 for i in inspections if i.get("overall_status") == "ok")
        avg_fill = (
            sum(i.get("shelf_fill_rate", 0) for i in inspections) / len(inspections)
            if inspections else 0
        )
        total_issues = sum(len(i.get("issues", [])) for i in inspections)

        context = (
            f"Sessão de inspeção: {len(inspections)} imagens, {n_zones} zonas. "
            f"Estados: {n_critical} críticos, {n_warning} avisos, {n_ok} OK. "
            f"Fill rate médio: {avg_fill*100:.0f}%. Total de issues: {total_issues}. "
            f"Alertas gerados: {len(all_alerts)}."
        )

        prompt = (
            "Escreve um sumário executivo de no máximo 150 palavras para um gestor de loja "
            "de retalho, com linguagem directa e accionável. Usa o seguinte contexto:\n\n"
            f"{context}\n\n"
            "Começa directamente com os factos mais relevantes. Não uses listas."
        )

        response = self.client.models.generate_content(model=self._model_name, contents=prompt, config=types.GenerateContentConfig(temperature=self._temperature))
        return response.text.strip()

    def _section_issues_by_zone(
        self, inspections: list[dict]
    ) -> str:
        """Agrupa e formata problemas por zona, com histórico RAG se disponível."""
        by_zone: dict[str, list[dict]] = defaultdict(list)
        for insp in inspections:
            zone = insp.get("zone_id", "Desconhecida")
            by_zone[zone].append(insp)

        lines = []
        for zone_id in sorted(by_zone.keys()):
            zone_inspections = by_zone[zone_id]
            lines.append(f"\n### Zona {zone_id}")

            for insp in zone_inspections:
                status_emoji = self._severity_emoji(insp.get("overall_status", ""))
                fill = self._format_fill_rate(insp.get("shelf_fill_rate"))
                ts = self._format_timestamp(insp.get("timestamp", ""))
                lines.append(
                    f"\n**Inspeção {insp.get('inspection_id', '?')}** "
                    f"| {ts} | {status_emoji} {insp.get('overall_status', '?').upper()} "
                    f"| Fill rate: {fill}"
                )

                issues = insp.get("issues", [])
                if not issues:
                    lines.append("\n_Nenhum problema detectado._")
                else:
                    for issue in issues:
                        sev_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                            issue.get("severity", ""), "⚪"
                        )
                        lines.append(
                            f"\n- {sev_emoji} **{issue.get('type', '?')}** "
                            f"— {issue.get('location', '?')}: "
                            f"{issue.get('description', '')} "
                            f"_(confiança: {issue.get('confidence', 0)*100:.0f}%, "
                            f"área: {issue.get('affected_area_pct', 0)*100:.0f}%)_"
                        )

            # Contexto histórico via RAG
            if self.rag and self.rag.count() > 0:
                try:
                    hist_query = f"histórico de problemas na zona {zone_id}"
                    hist = self.rag.retrieve(hist_query, top_k=2, zone_filter=zone_id)
                    if hist:
                        lines.append("\n** Contexto histórico (RAG):**")
                        for h in hist:
                            meta = h["metadata"]
                            summary = h["document"].split("[raw_json]")[0].strip()[:200]
                            lines.append(
                                f"\n> `{h['inspection_id']}` ({meta.get('date', '?')}): "
                                f"{summary}..."
                            )
                except Exception as exc:
                    logger.warning("Erro ao recuperar histórico para %s: %s", zone_id, exc)

        return "\n".join(lines)

    def _section_triggered_rules(self, all_alerts: list[dict]) -> str:
        """Lista as regras que dispararam com dados concretos."""
        if not all_alerts:
            return "\n_Nenhuma regra disparou nesta sessão._"

        lines = []
        for alert in all_alerts:
            emoji = self._alert_emoji(alert.get("alert_level", "info"))
            lines.append(
                f"\n- {emoji} **{alert['rule_id']}** `{alert['alert_level'].upper()}`\n"
                f"  - Zona: {alert.get('zone_id', 'N/A')}\n"
                f"  - Mensagem: {alert.get('message', '')}\n"
                f"  - Inspeção: `{alert.get('inspection_id', 'N/A')}`"
            )

        return "\n".join(lines)

    def _section_historical_context(self, inspections: list[dict]) -> str:
        """Padrões históricos relevantes recuperados do RAG."""
        if not self.rag or self.rag.count() == 0:
            return "\n_Base de dados histórica ainda sem dados suficientes._"

        queries = [
            "padrões recorrentes de prateleira vazia",
            "zonas com mais issues de planograma",
            "problemas detectados em dias e horas semelhantes",
        ]

        lines = []
        seen_ids = set()

        for q in queries:
            try:
                results = self.rag.retrieve(q, top_k=2)
                for r in results:
                    if r["inspection_id"] not in seen_ids:
                        seen_ids.add(r["inspection_id"])
                        meta = r["metadata"]
                        summary = r["document"].split("[raw_json]")[0].strip()[:250]
                        lines.append(
                            f"\n> **`{r['inspection_id']}`** ({meta.get('date', '?')}, "
                            f"Zona {meta.get('zone_id', '?')}):\n> {summary}"
                        )
            except Exception as exc:
                logger.warning("Erro em query histórica: %s", exc)

        return "\n".join(lines) if lines else "\n_Sem padrões históricos relevantes encontrados._"

    def _section_recommendations(
        self, inspections: list[dict], all_alerts: list[dict]
    ) -> str:
        """Gera até 5 recomendações concretas ordenadas por urgência via LLM."""
        critical_issues = [
            f"Zona {i['zone_id']}: {', '.join(x['type'] for x in i.get('issues', []))}"
            for i in inspections
            if i.get("overall_status") == "critical"
        ]
        warning_issues = [
            f"Zona {i['zone_id']}: fill rate {self._format_fill_rate(i.get('shelf_fill_rate'))}"
            for i in inspections
            if i.get("overall_status") == "warning"
        ]

        context = "\n".join(
            [f"- CRÍTICO: {x}" for x in critical_issues]
            + [f"- AVISO: {x}" for x in warning_issues]
            + [f"- ALERTA ({a['rule_id']}): {a['message']}" for a in all_alerts]
        )

        if not context.strip():
            return "\n_Nenhuma acção recomendada — loja em bom estado._"

        prompt = (
            "Com base nos seguintes problemas detectados numa loja de retalho, "
            "gera exatamente 5 recomendações concretas e accionáveis, "
            "ordenadas por urgência decrescente. "
            "Cada recomendação deve ser específica o suficiente para ser executada "
            "sem interpretação adicional. Usa formato de lista numerada.\n\n"
            f"{context}"
        )

        response = self.client.models.generate_content(model=self._model_name, contents=prompt, config=types.GenerateContentConfig(temperature=self._temperature))
        return "\n" + response.text.strip()

    # Geração do relatório completo

    def generate(
        self,
        inspections: list[dict],
        session_id: Optional[str] = None,
        save_to_disk: bool = True,
    ) -> str:
        """
        Gera o relatório Markdown completo para uma sessão de inspeção.

        Args:
            inspections: Lista de resultados de inspeção.
            session_id: Identificador da sessão (gerado automaticamente se omitido).
            save_to_disk: Se True, guarda o relatório em ficheiro .md.

        Returns:
            String com o relatório em Markdown.
        """
        if not inspections:
            return "# Relatório de Inspeção\n\n_Nenhuma inspeção para reportar._"

        if not session_id:
            session_id = f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        now_str = datetime.now(timezone.utc).strftime("%d/%m/%Y às %H:%M UTC")

        # Avalia regras para todas as inspeções
        all_alerts: list[dict] = []
        if self.rule_engine:
            for insp in inspections:
                alerts = self.rule_engine.evaluate(insp)
                all_alerts.extend(alerts)

        logger.info(
            "A gerar relatório %s | %d inspeções | %d alertas",
            session_id, len(inspections), len(all_alerts),
        )

        # Constrói as secções
        exec_summary = self._section_executive_summary(inspections, all_alerts)
        issues_by_zone = self._section_issues_by_zone(inspections)
        rules_section = self._section_triggered_rules(all_alerts)
        history_section = self._section_historical_context(inspections)
        recommendations = self._section_recommendations(inspections, all_alerts)

        report = f"""# Relatório de Inspeção — {session_id}

**Gerado em:** {now_str}
**Inspeções incluídas:** {len(inspections)}
**Alertas gerados:** {len(all_alerts)}

---

## 1. Sumário Executivo

{exec_summary}

---

## 2. Problemas por Zona

{issues_by_zone}

---

## 3. Regras Disparadas

{rules_section}

---

## 4. Contexto Histórico Relevante

{history_section}

---

## 5. Recomendações

{recommendations}
"""

        if save_to_disk:
            report_path = self.REPORTS_DIR / f"{session_id}.md"
            report_path.write_text(report, encoding="utf-8")
            logger.info("Relatório guardado: %s", report_path)

        return report



# Execução direta

if __name__ == "__main__":
    import sys
    import glob

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    inspections_dir = Path(os.getenv("INSPECTIONS_PATH", "./data/inspections"))

    # Carrega as inspeções JSON da sessão de hoje
    json_files = sorted(inspections_dir.glob("INS_*.json"))
    if not json_files:
        print("Nenhuma inspeção encontrada em:", inspections_dir)
        sys.exit(1)

    inspections = []
    for f in json_files:
        try:
            inspections.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Erro ao carregar %s: %s", f, exc)

    generator = ReportGenerator()
    report = generator.generate(inspections)
    print(report)