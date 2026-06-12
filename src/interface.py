import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Adiciona o diretório src ao path para imports locais
sys.path.insert(0, str(Path(__file__).parent))

from shelf_inspector import ShelfInspector, PromptStrategy
from rule_engine import RuleEngine
from rag_memory import RAGMemory
from report_generator import ReportGenerator

load_dotenv()

logger = logging.getLogger(__name__)
console = Console()

INSPECTIONS_PATH = Path(os.getenv("INSPECTIONS_PATH", "./data/inspections"))
INSPECTIONS_PATH.mkdir(parents=True, exist_ok=True)


# Orquestrador de sessão

class RetailVisionSystem:
    """
    Orquestra os 5 componentes do sistema numa interface unificada.
    Mantém estado de sessão entre comandos.
    """

    def __init__(self):
        console.print("[bold cyan]A inicializar Retail Vision Intelligence System...[/]")
        try:
            self.inspector = ShelfInspector()
            self.rule_engine = RuleEngine()
            self.rag = RAGMemory()
            self.report_generator = ReportGenerator(
                rag_memory=self.rag,
                rule_engine=self.rule_engine,
            )
        except EnvironmentError as exc:
            console.print(f"[bold red]Erro de configuração:[/] {exc}")
            sys.exit(1)

        self.session_inspections: list[dict] = []
        console.print("[bold green]✓ Sistema iniciado com sucesso.[/]\n")

    # Comandos de inspeção

    def cmd_inspect(self, args: list[str]) -> None:
        """
        Uso: inspect <zona_id> --image <caminho>
             inspect all --_sku110k_raw-dir <directoria>
        """
        if "--image" in args:
            zone_id = args[0]
            img_idx = args.index("--image") + 1
            if img_idx >= len(args):
                console.print("[red]Erro: caminho de imagem em falta após --image[/]")
                return

            strategy_arg = PromptStrategy.CHAIN_OF_THOUGHT
            if "--strategy" in args:
                s_idx = args.index("--strategy") + 1
                try:
                    strategy_arg = PromptStrategy(args[s_idx])
                except ValueError:
                    console.print(f"[yellow]Estratégia desconhecida. A usar chain_of_thought.[/]")

            img_path = args[img_idx]
            console.print(f"[cyan]A inspecionar {img_path} (zona {zone_id})...[/]")
            try:
                result = self.inspector.inspect(img_path, zone_id, strategy_arg)
                self._save_inspection(result)
                self.session_inspections.append(result)
                self.rag.index_inspection(result)
                self._display_inspection(result)
            except FileNotFoundError as exc:
                console.print(f"[red]Ficheiro não encontrado:[/] {exc}")
            except RuntimeError as exc:
                console.print(f"[yellow]{exc}[/]")
            except Exception as exc:
                console.print(f"[red]Erro inesperado:[/] {exc}")
                logger.exception("Erro em cmd_inspect")

        elif "--_sku110k_raw-dir" in args:
            zone_id = args[0] if args[0] != "all" else "BATCH"
            dir_idx = args.index("--_sku110k_raw-dir") + 1
            if dir_idx >= len(args):
                console.print("[red]Erro: caminho de directoria em falta[/]")
                return

            images_dir = args[dir_idx]
            console.print(f"[cyan]A processar batch em {images_dir}...[/]")
            results = self.inspector.inspect_batch(images_dir, zone_id)
            for r in results:
                self._save_inspection(r)
                self.session_inspections.append(r)
                if r.get("overall_status") != "error":
                    self.rag.index_inspection(r)
            console.print(
                f"[green]Batch concluído: {len(results)} imagens processadas.[/]"
            )
        else:
            console.print(
                "[yellow]Uso:[/] inspect <zona> --image <img.jpg> [--strategy zero_shot|chain_of_thought|few_shot]\n"
                "       inspect <zona> --_sku110k_raw-dir <dir/>"
            )

    def _save_inspection(self, inspection: dict) -> None:
        insp_id = inspection.get("inspection_id", "UNKNOWN")
        path = INSPECTIONS_PATH / f"{insp_id}.json"
        path.write_text(
            json.dumps(inspection, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _display_inspection(self, inspection: dict) -> None:
        status = inspection.get("overall_status", "?")
        color = {"ok": "green", "warning": "yellow", "critical": "red"}.get(status, "white")
        fill = inspection.get("shelf_fill_rate", 0)

        table = Table(show_header=True, header_style="bold")
        table.add_column("Campo")
        table.add_column("Valor")
        table.add_row("ID", inspection.get("inspection_id", "?"))
        table.add_row("Zona", inspection.get("zone_id", "?"))
        table.add_row("Status", f"[{color}]{status.upper()}[/]")
        table.add_row("Fill Rate", f"{fill*100:.0f}%")
        table.add_row("Issues", str(len(inspection.get("issues", []))))

        console.print(table)

        issues = inspection.get("issues", [])
        if issues:
            for issue in issues:
                sev = issue.get("severity", "low")
                sev_color = {"high": "red", "medium": "yellow", "low": "green"}.get(sev, "white")
                console.print(
                    f"  [{sev_color}]●[/] [{issue.get('issue_id')}] "
                    f"{issue.get('type')} — {issue.get('description', '')[:100]}"
                )

        # Avalia regras imediatamente
        if self.rule_engine:
            alerts = self.rule_engine.evaluate(inspection)
            for alert in alerts:
                alert_color = {"critical": "red", "warning": "yellow", "info": "blue"}.get(
                    alert["alert_level"], "white"
                )
                console.print(
                    Panel(
                        f"[bold]Regra {alert['rule_id']}:[/] {alert['message']}",
                        title=f"[{alert_color}]ALERTA {alert['alert_level'].upper()}[/]",
                        border_style=alert_color,
                    )
                )

    # ------------------------------------------------------------------
    # Comandos de regras
    # ------------------------------------------------------------------

    def cmd_add_rule(self, rule_text: str) -> None:
        """Converte e guarda uma regra em linguagem natural."""
        console.print("[cyan]A converter regra...[/]")
        try:
            rule = self.rule_engine.parse_rule(rule_text)

            if not rule["validation"]["is_valid"]:
                console.print("[yellow] Regra ambígua. Clarifica:[/]")
                for amb in rule["validation"]["ambiguities"]:
                    console.print(f"  • {amb}")
                console.print(
                    "\n[dim]Reformula a regra incluindo estas informações e tenta novamente.[/]"
                )
                return

            rule_id = self.rule_engine.add_rule(rule)
            console.print(f"[green]✓ Regra guardada: {rule_id}[/]")

            # Mostra os pressupostos assumidos
            assumptions = rule["validation"].get("assumptions", [])
            if assumptions:
                console.print("[dim]Pressupostos assumidos:[/]")
                for a in assumptions:
                    console.print(f"  [dim]→ {a}[/]")

        except ValueError as exc:
            console.print(f"[red]Erro:[/] {exc}")
        except Exception as exc:
            console.print(f"[red]Erro inesperado:[/] {exc}")
            logger.exception("Erro em cmd_add_rule")

    def cmd_list_rules(self) -> None:
        """Lista todas as regras activas."""
        rules = self.rule_engine.list_rules()
        if not rules:
            console.print("[dim]Nenhuma regra definida.[/]")
            return

        table = Table(title=f"Regras Activas ({len(rules)})", show_header=True)
        table.add_column("ID", style="bold")
        table.add_column("Descrição")
        table.add_column("Alert Level")
        table.add_column("Válida")

        for rule in rules:
            is_valid = rule.get("validation", {}).get("is_valid", True)
            table.add_row(
                rule.get("rule_id", "?"),
                rule.get("description", rule.get("natural_language", "?"))[:60],
                rule.get("action", {}).get("alert_level", "?"),
                "✓" if is_valid else "✗",
            )
        console.print(table)

    def cmd_delete_rule(self, rule_id: str) -> None:
        """Remove uma regra pelo ID."""
        if self.rule_engine.delete_rule(rule_id):
            console.print(f"[green]✓ Regra {rule_id} removida.[/]")
        else:
            console.print(f"[red]Regra não encontrada: {rule_id}[/]")

    def cmd_test_rule(self, args: list[str]) -> None:
        """Testa uma regra contra uma imagem."""
        if len(args) < 3 or "--image" not in args:
            console.print("[yellow]Uso:[/] test rule <RULE_ID> --image <img.jpg>")
            return

        rule_id = args[0]
        img_idx = args.index("--image") + 1
        img_path = args[img_idx]

        try:
            result = self.inspector.inspect(img_path, "TEST", PromptStrategy.CHAIN_OF_THOUGHT)
            test_result = self.rule_engine.test_rule(rule_id, result)
            triggered = test_result["triggered"]
            color = "green" if triggered else "dim"
            console.print(
                f"[{color}]Regra {rule_id}: {'DISPARARIA ✓' if triggered else 'não dispararia'}[/]"
            )
            if triggered:
                console.print(f"  Mensagem: {test_result['message']}")
        except Exception as exc:
            console.print(f"[red]Erro:[/] {exc}")

    # Comandos de histórico (RAG)

    def cmd_history(self, question: str) -> None:
        """Responde a uma pergunta sobre o histórico."""
        console.print("[cyan]A consultar histórico...[/]")
        try:
            result = self.rag.query(question)
            console.print(Panel(result["answer"], title="📚 Resposta", border_style="blue"))
            if result["sources"]:
                console.print("[dim]Fontes:[/]")
                for s in result["sources"]:
                    console.print(
                        f"  [dim]• {s['inspection_id']} | {s['date']} | "
                        f"Zona {s['zone_id']} | dist={s['distance']:.3f}[/]"
                    )
        except Exception as exc:
            console.print(f"[red]Erro:[/] {exc}")
            logger.exception("Erro em cmd_history")

    def cmd_compare(self, args: list[str]) -> None:
        """Compara duas zonas num período."""
        if len(args) < 2:
            console.print("[yellow]Uso:[/] compare <Z_S1> <Z_S2> [--period '7 days']")
            return

        zone1, zone2 = args[0], args[1]
        question = f"Comparação entre zona {zone1} e zona {zone2}: quais tiveram mais problemas?"
        self.cmd_history(question)

    # Comandos de relatório
    def cmd_report(self, args: list[str]) -> None:
        """Gera um relatório de inspeção."""
        inspections = []

        if "--session" in args:
            # Usa inspeções da sessão atual
            inspections = self.session_inspections

        elif "--zone" in args:
            zone_idx = args.index("--zone") + 1
            zone_id = args[zone_idx] if zone_idx < len(args) else None
            # Carrega inspeções do disco filtradas por zona
            inspections = self._load_inspections_from_disk(zone_filter=zone_id)

        else:
            # Por defeito, usa a sessão atual
            inspections = self.session_inspections

        if not inspections:
            console.print("[yellow]Nenhuma inspeção disponível para o relatório.[/]")
            return

        console.print(f"[cyan]A gerar relatório ({len(inspections)} inspeções)...[/]")
        try:
            report = self.report_generator.generate(inspections)
            console.print(Markdown(report))
        except Exception as exc:
            console.print(f"[red]Erro ao gerar relatório:[/] {exc}")
            logger.exception("Erro em cmd_report")

    def _load_inspections_from_disk(
        self, zone_filter: str = None
    ) -> list[dict]:
        results = []
        for f in sorted(INSPECTIONS_PATH.glob("INS_*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if zone_filter and data.get("zone_id") != zone_filter:
                    continue
                results.append(data)
            except Exception as exc:
                logger.warning("Erro ao carregar %s: %s", f, exc)
        return results

    # CLI loop
    def run_cli(self) -> None:
        """Inicia o loop interativo da CLI."""
        console.print(
            Panel(
                "[bold]Retail Vision Intelligence System[/]\n"
                "Comandos: [cyan]inspect[/] | [cyan]add rule[/] | [cyan]list rules[/] | "
                "[cyan]delete rule[/] | [cyan]history[/] | [cyan]report[/] | [cyan]help[/] | [cyan]exit[/]",
                border_style="cyan",
            )
        )

        while True:
            try:
                user_input = console.input("\n[bold green]>[/] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]A sair...[/]")
                break

            if not user_input:
                continue

            tokens = user_input.split()
            cmd = tokens[0].lower()
            rest = tokens[1:]

            try:
                if cmd == "exit" or cmd == "quit":
                    console.print("[dim]Sessão terminada.[/]")
                    break

                elif cmd == "help":
                    self._print_help()

                elif cmd == "inspect":
                    self.cmd_inspect(rest)

                elif cmd == "add" and rest and rest[0] == "rule":
                    rule_text = " ".join(rest[1:])
                    if not rule_text:
                        console.print("[yellow]Escreve a regra a seguir a 'add rule'.[/]")
                    else:
                        self.cmd_add_rule(rule_text)

                elif cmd == "list" and rest and rest[0] == "rules":
                    self.cmd_list_rules()

                elif cmd == "delete" and rest and rest[0] == "rule":
                    if len(rest) < 2:
                        console.print("[yellow]Uso:[/] delete rule <RULE_ID>")
                    else:
                        self.cmd_delete_rule(rest[1])

                elif cmd == "test" and rest and rest[0] == "rule":
                    self.cmd_test_rule(rest[1:])

                elif cmd == "history":
                    question = " ".join(rest)
                    if not question:
                        console.print('[yellow]Uso:[/] history "pergunta sobre histórico"')
                    else:
                        self.cmd_history(question.strip('"\''))

                elif cmd == "compare":
                    self.cmd_compare(rest)

                elif cmd == "report":
                    self.cmd_report(rest)

                elif cmd == "stats":
                    stats = self.rag.get_stats()
                    console.print_json(json.dumps(stats, ensure_ascii=False))

                else:
                    console.print(
                        f"[yellow]Comando desconhecido:[/] '{user_input}'. "
                        "Escreve 'help' para ver os comandos disponíveis."
                    )

            except Exception as exc:
                console.print(f"[red]Erro ao executar comando:[/] {exc}")
                logger.exception("Erro no loop CLI para comando: %s", user_input)

    def _print_help(self) -> None:
        help_text = """
## Comandos Disponíveis

**Inspeção**
- `inspect Z_S3 --image shelf.jpg [--strategy zero_shot|chain_of_thought|few_shot]`
- `inspect Z_S3 --_sku110k_raw-dir ./fotos/`

**Regras**
- `add rule "Avisa-me quando a prateleira inferior estiver mais de 40% vazia"`
- `list rules`
- `delete rule RULE_001`
- `test rule RULE_001 --image shelf.jpg`

**Histórico (RAG)**
- `history "quais as zonas com mais problemas esta semana?"`
- `compare Z_S1 Z_S3 --period "last 7 days"`

**Relatórios**
- `report --session today`
- `report --zone Z_S3`

**Outros**
- `stats` — estatísticas da base de dados
- `exit` — sair
"""
        console.print(Markdown(help_text))


# Interface Streamlit (opcional)

def run_streamlit():
    """Lança a interface Streamlit."""
    try:
        import streamlit as st
    except ImportError:
        print("Instala o streamlit: pip install streamlit")
        sys.exit(1)

    st.set_page_config(
        page_title="Retail Vision Intelligence",
        layout="wide",
    )

    st.title("Retail Vision Intelligence System")

    # Inicializa o sistema uma vez por sessão
    if "system" not in st.session_state:
        with st.spinner("A inicializar sistema..."):
            try:
                st.session_state.system = RetailVisionSystem()
                st.session_state.inspections = []
            except Exception as exc:
                st.error(f"Erro de inicialização: {exc}")
                st.stop()

    system: RetailVisionSystem = st.session_state.system

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Inspeção", "Regras", "Histórico", "Relatório"]
    )

    # ---- Tab: Inspeção ----
    with tab1:
        st.subheader("Inspecionar Prateleira")
        col1, col2 = st.columns(2)
        with col1:
            zone_id = st.text_input("Zona ID", value="Z_S1")
        with col2:
            strategy = st.selectbox(
                "Estratégia de prompting",
                ["chain_of_thought", "zero_shot", "few_shot"],
            )

        uploaded = st.file_uploader("Imagem de prateleira", type=["jpg", "jpeg", "png"])
        if uploaded and st.button("Inspecionar"):
            tmp_path = Path(f"/tmp/{uploaded.name}")
            tmp_path.write_bytes(uploaded.getbuffer())
            with st.spinner("A analisar..."):
                try:
                    result = system.inspector.inspect(
                        tmp_path, zone_id, PromptStrategy(strategy)
                    )
                    system._save_inspection(result)
                    st.session_state.inspections.append(result)
                    system.rag.index_inspection(result)

                    status_color = {"ok": "green", "warning": "orange", "critical": "red"}
                    status = result.get("overall_status", "?")
                    st.markdown(
                        f"**Status:** :{status_color.get(status, 'grey')}[{status.upper()}]  |  "
                        f"**Fill Rate:** {result.get('shelf_fill_rate', 0)*100:.0f}%  |  "
                        f"**Issues:** {len(result.get('issues', []))}"
                    )

                    if result.get("issues"):
                        st.json(result["issues"])

                    alerts = system.rule_engine.evaluate(result)
                    for a in alerts:
                        st.warning(f"**{a['rule_id']}** ({a['alert_level']}): {a['message']}")

                except Exception as exc:
                    st.error(str(exc))

    # Tab: Regras
    with tab2:
        st.subheader("Gestão de Regras")
        rule_text = st.text_area("Nova regra em linguagem natural:")
        if st.button("Adicionar Regra") and rule_text:
            with st.spinner("A converter regra..."):
                try:
                    rule = system.rule_engine.parse_rule(rule_text)
                    if not rule["validation"]["is_valid"]:
                        st.warning("Regra ambígua:")
                        for a in rule["validation"]["ambiguities"]:
                            st.write(f"• {a}")
                    else:
                        rid = system.rule_engine.add_rule(rule)
                        st.success(f"Regra guardada: {rid}")
                except Exception as exc:
                    st.error(str(exc))

        st.divider()
        rules = system.rule_engine.list_rules()
        if rules:
            for rule in rules:
                with st.expander(f"{rule['rule_id']} — {rule.get('description', '')[:60]}"):
                    st.json(rule)
                    if st.button(f"Eliminar {rule['rule_id']}", key=f"del_{rule['rule_id']}"):
                        system.rule_engine.delete_rule(rule["rule_id"])
                        st.rerun()

    # Tab: Histórico
    with tab3:
        st.subheader("Consulta de Histórico")
        question = st.text_input("Pergunta sobre histórico:")
        if st.button("Pesquisar") and question:
            with st.spinner("A consultar..."):
                result = system.rag.query(question)
                st.write(result["answer"])
                if result["sources"]:
                    st.caption("Fontes:")
                    for s in result["sources"]:
                        st.caption(
                            f"• {s['inspection_id']} | {s['date']} | Zona {s['zone_id']}"
                        )

    # Tab: Relatório
    with tab4:
        st.subheader("Gerar Relatório")
        inspections = system._load_inspections_from_disk()
        st.write(f"Inspeções disponíveis: {len(inspections)}")
        if st.button("Gerar Relatório") and inspections:
            with st.spinner("A gerar..."):
                try:
                    report_md = system.report_generator.generate(inspections)
                    st.markdown(report_md)
                    st.download_button("⬇ Descarregar .md", report_md, file_name="report.md")
                except Exception as exc:
                    st.error(str(exc))



# Entry point

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    if "--streamlit" in sys.argv:
        run_streamlit()
    else:
        system = RetailVisionSystem()
        system.run_cli()