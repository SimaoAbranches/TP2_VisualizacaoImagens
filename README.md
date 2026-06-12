# Retail Vision Intelligence System
**LIACD — Trabalho Prático #2 | 2025/2026**

Sistema de inspeção contínua de prateleiras com memória histórica, motor de regras em linguagem natural, e interface conversacional para gestores de loja.

---

## Arquitectura

```
IMAGENS DE PRATELEIRAS
         ↓
[1] shelf_inspector.py   — Análise visual com Gemini 2.0 Flash
         ↓
[2] rule_engine.py       — Regras em linguagem natural → JSON executável
         ↓
[3] rag_memory.py        — ChromaDB + sentence-transformers (embeddings PT)
         ↓
[4] report_generator.py  — Relatórios Markdown com contexto histórico
         ↓
[5] interface.py         — CLI interactiva + interface Streamlit
```

---

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env e inserir GEMINI_API_KEY (https://aistudio.google.com)
```

---

## Resultados Obtidos

Avaliação sobre **26 imagens reais** do dataset SKU-110K:

| Métrica | Chain-of-Thought | Zero-Shot |
|---|---|---|
| Imagens avaliadas | 17 | 9 |
| Fill rate médio | **84.4%** | 43.4% |
| Issues por imagem | **5.8** | 1.4 |
| Status Critical | 24% | 67% |
| Status Warning | 65% | 33% |
| Status OK | 12% | 0% |

**Conclusão:** Chain-of-thought supera zero-shot em todas as métricas.

---

## Dataset

Fonte: SKU-110K (Goldman et al., 2019)

| Categoria | Target | Disponível |
|---|---|---|
| normal | 150 | ~200 |
| empty | 100 | ~150 |
| planogram_violation | 100 | ~150 |
| dirty_messy | 80 | ~120 |
| ambiguous | 70 | ~100 |
| **Total** | **500** | **~720** |

```bash
python data/images/download.py --organize-sku110k
python data/images/download.py --summary
```

---

## Uso

### CLI Interactiva

```bash
python src/interface.py
```

```bash
> inspect Z_S3 --image shelf.jpg
> inspect Z_S3 --image shelf.jpg --strategy zero_shot
> add rule "Avisa-me quando a prateleira inferior estiver mais de 40% vazia"
> list rules
> history "quais as zonas com mais problemas esta semana?"
> report --session today
> stats
```

### Batch Processing

```bash
python run_batch.py          # processa tudo com checkpoint
python run_batch.py --status # ver progresso
python run_batch.py --reset  # recomeçar do zero
```

### Avaliação

```bash
python evaluate.py --images-dir test_images/ --output evaluation_report.json
```

---

## Estratégias de Prompting

| Estratégia | Ficheiro | Descrição |
|---|---|---|
| A — Zero-shot | `prompts/inspect_zero_shot.txt` | Instrução direta |
| B — Chain-of-Thought | `prompts/inspect_chain_of_thought.txt` | Raciocínio por etapas |
| C — Few-shot | `prompts/inspect_few_shot.txt` | Exemplos textuais |

---

## Estrutura

```
tp2/
├── README.md
├── requirements.txt
├── .env.example
├── run_batch.py
├── evaluate.py
├── data/
│   ├── images/
│   │   ├── download.py
│   │   ├── normal/ empty/ planogram_violation/ dirty_messy/ ambiguous/
│   ├── inspections/
│   └── rules/
├── src/
│   ├── shelf_inspector.py
│   ├── rule_engine.py
│   ├── rag_memory.py
│   ├── report_generator.py
│   └── interface.py
├── prompts/*.txt
├── vectorstore/
├── cache/
└── test_images/
    └── ground_truth.json
```

---

## Variáveis de Ambiente

| Variável | Default | Descrição |
|---|---|---|
| `GEMINI_API_KEY` | — | **Obrigatória** |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Modelo Gemini |
| `GEMINI_TEMPERATURE` | `0.0` | Reprodutibilidade |
| `RATE_LIMIT_RPM` | `15` | Requests por minuto |
| `RAG_TOP_K` | `3` | Docs recuperados por query |

---

## Notas Técnicas

- Cache por MD5 — cada imagem consome quota apenas uma vez
- Backoff exponencial em erro 429
- Checkpoint no run_batch.py — retoma onde ficou se interrompido
- Embeddings locais (MiniLM-L12-v2) — sem quota adicional para RAG
- Sem GPU necessária — processamento na cloud
- Usa biblioteca google-genai (nova API v1beta)
