# ============================================================
# PAINEL DE MERCADO - ABAS MACRO, MICRO e AI MONITOR
# Versão web (Streamlit) - rode com: streamlit run app.py
# ============================================================
# Como usar:
#   1) Cole tudo numa célula do Colab e execute.
#   2) Aba MACRO (não depende dos tickers):
#         - Inflação EUA: CPI, PPI e preços de importação (cheio e núcleo)
#         - VIX term structure + curva de juros dos Treasuries
#         - Sentimento em opções (escala ISEE / call-put)
#      Escolha a janela (2/5/10 anos) e clique em "Atualizar Macro".
#   3) Aba MICRO (por ativo):
#         - painel de preços, tabela de rendimentos e volatilidade,
#           range de P/E, matriz de correlação, preços base 100,
#           retornos diários e histograma.
#      Digite os tickers, escolha período e data de referência do P/E
#      e clique em "Atualizar Micro".
#   4) Aba AI MONITOR (ciclo de investimento em IA):
#         - baseada em "The AI Innovation Cycle: Scenarios and Signposts"
#           (Schwab Asset Management, 09/11/2026)
#         - 4 cenários (boom de produtividade, retorno adiado, boom
#           inflacionário, quebra deflacionária) x 4 signposts (capex,
#           financiamento, adoção, monetização)
#         - termômetro de cenários = sua visão (sliders) + sinais de mercado
#   5) "Exportar PDF" gera um PDF com as três abas (Macro, Micro, AI Monitor).
#
# Ações da B3 usam sufixo ".SA" (ex: PETR4.SA, VALE3.SA).
# OBS: o P/E é um proxy TTM (preço / lucro 12m), pois o P/E forward
# histórico é dado proprietário (FactSet/Bloomberg).

# Se precisar no Colab:
# !pip install yfinance ipywidgets pandas_datareader -q

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, date
import textwrap
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================
PERIOD_TO_INTERVAL = {
    '5d': '30m', '1mo': '1d', '3mo': '1d', '6mo': '1d',
    '1y': '1wk', '2y': '1wk', '5y': '1mo',
}
PERIOD_LABELS = {
    '5d': '5D', '1mo': '1M', '3mo': '3M', '6mo': '6M',
    '1y': '1A', '2y': '2A', '5y': '5A'
}

# ---- Tema (escuro) -----------------------------------------------------------
BG = '#0d1117'
PANEL = '#161b22'
BORDER = '#2a323d'
TXT = '#e6edf3'
DIM = '#8b949e'
FAINT = '#5c6470'
UP = '#2ea043'
DOWN = '#e5534b'

# Cores do gráfico de P/E
PE_10Y = '#3d4654'
PE_3Y  = '#6e7681'
PE_2Y  = '#b5673f'
PE_1Y  = '#e5534b'
PE_CUR = '#ff6b6b'   # diamante cheio (hoje)
PE_REF = BG          # diamante vazado (data de referência)

# Cores do term structure
TS_MARK = '#38b2c4'
TS_LINE = '#8b949e'
TS_PREV = '#57606a'

# ============================================================
# UTILIDADES
# ============================================================
def fmt_price(v):
    if v is None:
        return '-'
    dec = 0 if v >= 5000 else 2
    s = f'{v:,.{dec}f}'
    return s.replace(',', 'X').replace('.', ',').replace('X', '.')

def fmt_pct(v, sign=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    s = f'{v:+.1f}%' if sign else f'{v:.1f}%'
    return s.replace('.', ',')

def parse_tickers(texto):
    out, seen = [], set()
    for s in texto.replace(';', ',').split(','):
        s = s.strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append({'sym': s, 'label': s, 'name': s})
    return out

def _cmap(n):
    # plt.cm.get_cmap foi removido no matplotlib >= 3.9
    return plt.get_cmap('tab20', max(n, 3))


# ############################################################
#                         M A C R O
# ############################################################

# ============================================================
# MACRO 1: INFLAÇÃO EUA - CPI, PPI E PREÇOS DE IMPORTAÇÃO
# ============================================================
# Busca direto do CSV público do FRED (sem API key).
# Cheio = linha sólida, núcleo = tracejada.
INFL_SERIES = {
    "CPI":             "CPIAUCSL",   # CPI-U, SA
    "CPI núcleo":      "CPILFESL",   # ex alimentos e energia, SA
    "PPI":             "PPIFIS",     # demanda final, SA
    "PPI núcleo":      "PPIFES",     # demanda final ex alimentos e energia, SA
    "Import":          "IR",         # todas as importações (NSA)
    "Import ex-comb.": "IREXFDFLS",  # ex alimentos e combustíveis (NSA)
}
INFL_GRUPOS = {  # grupo -> (cheio, núcleo, cor)
    "CPI":    ("CPI",    "CPI núcleo",      '#58a6ff'),
    "PPI":    ("PPI",    "PPI núcleo",      '#e3b341'),
    "Import": ("Import", "Import ex-comb.", '#bc8cff'),
}

def _fred_csv(sid):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    df = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')
    return pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()

def get_inflation():
    series = {}
    for lab, sid in INFL_SERIES.items():
        try:
            series[lab] = _fred_csv(sid)
        except Exception as e:
            print(f"  {lab} ({sid}) falhou: {type(e).__name__}")
    return pd.DataFrame(series)

def draw_inflation(anos=5):
    lvl = get_inflation()
    if lvl.empty:
        print("Sem dados de inflação (FRED indisponível).")
        return
    yoy = lvl.pct_change(12, fill_method=None) * 100
    mom = lvl.pct_change(1,  fill_method=None) * 100
    yoy_p = yoy[yoy.index >= lvl.index.max() - pd.DateOffset(years=anos)]

    fig = plt.figure(figsize=(15, 8.5), facecolor=BG)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 2.3], hspace=0.30,
                  wspace=0.12, left=0.05, right=0.97, top=0.87, bottom=0.07)
    fig.text(0.05, 0.955, 'Inflação EUA - CPI, PPI e preços de importação',
             color=TXT, fontsize=17, fontweight='bold')
    fig.text(0.05, 0.92, 'Fonte: FRED / BLS - YoY sobre o nível do índice; '
             'MoM de importação é sem ajuste sazonal', color=DIM, fontsize=9.5)

    # ---- cards (último dado) -------------------------------------------
    for i, (grp, (head, core, cor)) in enumerate(INFL_GRUPOS.items()):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(BORDER)
        ax.text(0.04, 0.86, grp, transform=ax.transAxes, color=cor,
                fontsize=13, fontweight='bold', va='top')
        if head not in yoy or yoy[head].dropna().shape[0] < 2:
            ax.text(0.5, 0.45, 'sem dados', transform=ax.transAxes,
                    color=FAINT, ha='center')
            continue
        s = yoy[head].dropna()
        last, prev, dt = s.iloc[-1], s.iloc[-2], s.index[-1]
        delta = last - prev
        ax.text(0.96, 0.86, dt.strftime('%b/%y'), transform=ax.transAxes,
                color=FAINT, fontsize=9, ha='right', va='top')
        ax.text(0.04, 0.58, fmt_pct(last), transform=ax.transAxes, color=TXT,
                fontsize=24, fontweight='bold', va='center')
        ax.text(0.04, 0.33, f'a/a  ({fmt_pct(delta).rstrip("%")} p.p. vs mês ant.)',
                transform=ax.transAxes, va='center', fontsize=9,
                color=(DOWN if delta > 0 else UP))   # inflação subindo = vermelho
        m = mom[head].dropna()
        c = yoy[core].dropna() if core in yoy else pd.Series(dtype=float)
        rod = (f"MoM {fmt_pct(m.iloc[-1]) if not m.empty else '-'}   |   "
               f"núcleo a/a {fmt_pct(c.iloc[-1]) if not c.empty else '-'}")
        ax.text(0.04, 0.12, rod, transform=ax.transAxes, color=DIM, fontsize=9)

    # ---- linhas YoY ------------------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor(PANEL)
    for grp, (head, core, cor) in INFL_GRUPOS.items():
        if head in yoy_p:
            ax.plot(yoy_p.index, yoy_p[head], color=cor, lw=1.8,
                    label=f'{head} (cheio)')
        if core in yoy_p:
            ax.plot(yoy_p.index, yoy_p[core], color=cor, lw=1.2, ls='--',
                    alpha=0.85, label=core)
    ax.axhline(0, color=FAINT, lw=0.8)
    ax.axhline(2, color=DIM, lw=0.8, ls=':', alpha=0.8)
    ax.text(yoy_p.index[0], 2.15, ' meta Fed 2%', color=DIM, fontsize=8)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.set_ylabel('Variação 12 meses (%)', color=DIM, fontsize=9)
    ax.grid(linestyle='--', alpha=0.2, color=FAINT)
    ax.set_title(f'Variação anual - últimos {anos} anos', color=TXT, fontsize=12,
                 fontweight='bold', loc='left', pad=10)
    ax.legend(loc='upper right', ncol=3, frameon=False, labelcolor=TXT,
              fontsize=8.5)
    plt.show()

# ============================================================
# MACRO 2: VIX TERM STRUCTURE + CURVA DE JUROS (UST)
# ============================================================
MOSTRAR_COMPARACAO = True   # False = só a curva atual

# "manual" = valores digitados abaixo (futuros do VIX não estão no Yahoo)
# "yahoo"  = curva-proxy com os índices VIX de vários prazos
VIX_FONTE = "manual"

VIX_TERM_MANUAL = {
    "Spot":   16.64,
    "Aug-26": 18.51,
    "Sep-26": 19.60,
    "Oct-26": 20.45,
    "Nov-26": 20.79,
    "Dec-26": 20.90,
    "Jan-27": 21.88,
    "Feb-27": 22.28,
}

VIX_YAHOO = {
    "1 dia":   "^VIX1D",
    "9 dias":  "^VIX9D",
    "30 dias": "^VIX",
    "3 meses": "^VIX3M",
    "6 meses": "^VIX6M",
    "1 ano":   "^VIX1Y",
}

def get_vix_term():
    """Devolve (labels, valores, fonte_txt)."""
    if VIX_FONTE == "yahoo":
        labels, vals = [], []
        for lab, sym in VIX_YAHOO.items():
            try:
                h = yf.Ticker(sym).history(period="10d")["Close"].dropna()
                if not h.empty:
                    labels.append(lab)
                    vals.append(float(h.iloc[-1]))
            except Exception:
                continue
        if len(vals) >= 3:
            return labels, vals, "Índices de volatilidade Cboe (proxy) - Yahoo Finance"
        print("VIX via Yahoo indisponível - usando os valores manuais.")
    return (list(VIX_TERM_MANUAL.keys()),
            list(VIX_TERM_MANUAL.values()),
            "Futuros do VIX - valores inseridos manualmente (relatório Cboe)")

FRED_SERIES = {
    "3M":  "DGS3MO", "6M":  "DGS6MO", "1A":  "DGS1",
    "3A":  "DGS3",   "5A":  "DGS5",   "7A":  "DGS7",
    "10A": "DGS10",  "20A": "DGS20",  "30A": "DGS30",
}
TREASURY_COLS = {
    "3M": "3 Mo", "6M": "6 Mo", "1A": "1 Yr", "3A": "3 Yr", "5A": "5 Yr",
    "7A": "7 Yr", "10A": "10 Yr", "20A": "20 Yr", "30A": "30 Yr",
}
# usado só se as fontes online falharem -> EDITE se precisar
YIELD_FALLBACK = {
    "3M": 4.30, "6M": 4.20, "1A": 4.05, "3A": 3.90, "5A": 3.95,
    "7A": 4.05, "10A": 4.20, "20A": 4.60, "30A": 4.70,
}

def _yield_from_fred_csv():
    """Curva via CSV público do FRED (mesma fonte da inflação, sem dependências)."""
    start = pd.Timestamp.today() - pd.Timedelta(days=420)
    cols = {}
    for k, sid in FRED_SERIES.items():
        s = _fred_csv(sid)
        cols[k] = s[s.index >= start]
    df = pd.DataFrame(cols)[list(FRED_SERIES.keys())]
    return df.dropna(how="all").ffill(), "FRED (Treasury constant maturity)"

def _yield_from_treasury():
    ano = pd.Timestamp.today().year
    url = ("https://home.treasury.gov/resource-center/data-chart-center/"
           f"interest-rates/daily-treasury-rates.csv/{ano}/all?"
           f"type=daily_treasury_yield_curve&field_tdr_date_value={ano}"
           "&page&_format=csv")
    raw = pd.read_csv(url)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").sort_index()
    cols = {k: v for k, v in TREASURY_COLS.items() if v in raw.columns}
    df = raw[list(cols.values())]
    df.columns = list(cols.keys())
    return df.ffill(), "U.S. Department of the Treasury"

def get_yield_curve():
    """Devolve (labels, atual, mes_passado|None, data_ref, fonte_txt)."""
    df, fonte = None, None
    for fn in (_yield_from_fred_csv, _yield_from_treasury):
        try:
            df, fonte = fn()
            if df is not None and not df.empty:
                break
        except Exception as e:
            print(f"  {fn.__name__} falhou: {type(e).__name__}")
            df = None
    if df is None or df.empty:
        print("Curva de juros online indisponível - usando YIELD_FALLBACK (edite os valores).")
        return (list(YIELD_FALLBACK.keys()), list(YIELD_FALLBACK.values()),
                None, None, "valores de exemplo (offline) - edite YIELD_FALLBACK")

    atual_row = df.iloc[-1]
    data_ref = df.index[-1]
    labels = [c for c in FRED_SERIES if c in df.columns and pd.notna(atual_row[c])]
    atual = [float(atual_row[c]) for c in labels]

    prev = None
    if MOSTRAR_COMPARACAO:
        alvo = data_ref - pd.DateOffset(months=1)
        janela = df[df.index <= alvo]
        if not janela.empty:
            prow = janela.iloc[-1]
            if all(pd.notna(prow.get(c, np.nan)) for c in labels):
                prev = [float(prow[c]) for c in labels]
    return labels, atual, prev, data_ref, fonte

def _nice_step(rng):
    for s in (0.1, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0):
        if rng / s <= 6.5:
            return s
    return 20.0

def _ts_axis(ax, labels, values, titulo, ylab, prev=None, prev_label=None,
             fmt="%.1f"):
    x = np.arange(len(values))
    ax.set_facecolor(PANEL)

    if prev is not None:
        ax.plot(x, prev, color=TS_PREV, lw=1.2, ls="--", zorder=2)
        ax.plot(x, prev, ls="none", marker="D", ms=4.5,
                markerfacecolor=PANEL, markeredgecolor=TS_PREV, zorder=3)

    ax.plot(x, values, color=TS_LINE, lw=1.4, zorder=4)
    ax.plot(x, values, ls="none", marker="D", ms=7,
            markerfacecolor=TS_MARK, markeredgecolor=TS_MARK, zorder=5)

    todos = list(values) + (list(prev) if prev is not None else [])
    lo, hi = min(todos), max(todos)
    step = _nice_step(max(hi - lo, 1e-6))
    lo = np.floor(lo / step) * step - step
    hi = np.ceil(hi / step) * step + step
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=9, color=TXT)
    ax.tick_params(axis="y", labelsize=9, colors=DIM, length=0)
    ax.tick_params(axis="x", length=0)

    ax.grid(axis="y", color=FAINT, lw=0.7, alpha=0.35, zorder=1)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_edgecolor(BORDER)
        s.set_linewidth(0.8)

    ax.set_ylabel(ylab, color=DIM, fontsize=9)
    ax.text(0.0, 1.03, f" {titulo} ", transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color=TS_MARK, va="bottom", ha="left",
            bbox=dict(facecolor=BORDER, edgecolor="none", pad=3.5))

    if prev is not None and prev_label:
        ax.legend(handles=[
            Line2D([0], [0], marker="D", color=TS_LINE, markerfacecolor=TS_MARK,
                   markeredgecolor=TS_MARK, ms=7, lw=1.4, label="Atual"),
            Line2D([0], [0], marker="D", color=TS_PREV, markerfacecolor=PANEL,
                   markeredgecolor=TS_PREV, ms=5, lw=1.2, ls="--",
                   label=prev_label),
        ], loc="lower right", frameon=False, labelcolor=TXT, fontsize=8.5)

def draw_term_structures():
    """VIX term structure + curva de juros dos Treasuries, lado a lado."""
    v_labels, v_vals, v_fonte = get_vix_term()
    y_labels, y_now, y_prev, y_date, y_fonte = get_yield_curve()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6), facecolor=BG)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.20, wspace=0.20)

    _ts_axis(ax1, v_labels, v_vals, "VIX Futures Term Structure",
             "Volatilidade implícita")
    _ts_axis(ax2, y_labels, y_now, "U.S. Treasury Yield Curve",
             "Yield (%)", prev=y_prev, prev_label="1 mês atrás", fmt="%.2f")

    fig.text(0.055, 0.955, 'Term structure & curva de juros', color=TXT,
             fontsize=16, fontweight='bold')
    ref = f' - referência {y_date:%d/%m/%Y}' if y_date is not None else ''
    fig.text(0.055, 0.915, f'{v_fonte}  |  {y_fonte}{ref}',
             color=DIM, fontsize=9)
    plt.show()

# ============================================================
# MACRO 3: ISEE / CALL-PUT RATIO (sentimento em opções)
# ------------------------------------------------------------
# ESCALA ISEE:  (volume de calls / volume de puts) x 100
#   > 150 muito otimista (contrário: cautela) | ~100 equilibrado
#   <  80 pessimista (contrário: possível fundo)
# "proxy"  -> volume TOTAL do dia na cadeia de opções do Yahoo (nível NÃO
#             comparável ao ISEE oficial; use para leitura relativa)
# "manual" -> valores oficiais digitados (Cboe / ISEE)
# ============================================================
ISEE_FONTE = "proxy"
ISEE_UNDERLYINGS = ["SPY", "QQQ", "IWM"]
ISEE_MAX_VENCIMENTOS = 6

ISEE_MANUAL = {
    "ISEE (oficial)":        95.0,
    "Cboe Equity P/C (inv)": 110.0,
}

ISEE_ZONAS = [
    (0,   80,  DOWN,      "pessimista"),
    (80,  120, '#8b949e', "neutro"),
    (120, 150, '#3fb950', "otimista"),
    (150, 400, '#e3b341', "muito otimista"),
]

def _isee_zona(v):
    for lo, hi, cor, nm in ISEE_ZONAS:
        if lo <= v < hi:
            return cor, nm
    return ISEE_ZONAS[-1][2], ISEE_ZONAS[-1][3]   # acima de 400

def _callput_proxy(sym, max_exp=ISEE_MAX_VENCIMENTOS):
    try:
        t = yf.Ticker(sym)
        expiries = t.options[:max_exp]
        if not expiries:
            return None
        calls_v = puts_v = 0.0
        for exp in expiries:
            ch = t.option_chain(exp)
            calls_v += float(ch.calls["volume"].fillna(0).sum())
            puts_v  += float(ch.puts["volume"].fillna(0).sum())
        if puts_v <= 0:
            return None
        return calls_v / puts_v * 100.0
    except Exception:
        return None

def get_isee_valores():
    if ISEE_FONTE == "proxy":
        vals = {}
        for sym in ISEE_UNDERLYINGS:
            v = _callput_proxy(sym)
            if v is not None:
                vals[f"{sym} (call/put x100)"] = v
        if vals:
            return vals, ("Proxy: volume total de opções do dia - Yahoo Finance "
                          "(nível NÃO comparável ao ISEE oficial)")
        print("Proxy via Yahoo indisponível - usando valores manuais.")
    return dict(ISEE_MANUAL), "Valores inseridos manualmente (Cboe / ISEE)"

def draw_isee():
    vals, fonte = get_isee_valores()
    if not vals:
        print("Sem dados para o indicador ISEE/call-put.")
        return

    labels = list(vals.keys())
    n = len(labels)
    x_max = max(200, max(vals.values()) * 1.15)

    fig, ax = plt.subplots(figsize=(12, 1.0 * n + 2.2), facecolor=BG)
    ax.set_facecolor(PANEL)

    for lo, hi, cor, _ in ISEE_ZONAS:
        ax.axvspan(lo, min(hi, x_max), color=cor, alpha=0.13, zorder=1)
    ax.axvline(100, color=DIM, lw=1.0, ls="--", alpha=0.7, zorder=2)

    for i, lab in enumerate(labels):
        v = vals[lab]
        cor, leitura = _isee_zona(v)
        ax.barh(i, v, height=0.45, color=cor, alpha=0.9, zorder=3)
        ax.scatter(v, i, marker="D", s=80, color=cor,
                   edgecolors=TXT, linewidths=0.8, zorder=4)
        txt = f"{v:,.0f}".replace(",", ".")
        ax.text(v + x_max * 0.015, i, f"{txt}  ({leitura})",
                va="center", color=TXT, fontsize=10, fontweight="bold")

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, color=TXT, fontsize=10)
    ax.set_xlim(0, x_max)
    ax.tick_params(axis="x", colors=DIM, labelsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.grid(axis="x", linestyle="--", alpha=0.2, color=FAINT)
    ax.set_axisbelow(True)

    ax.set_title("Sentimento em opções - escala ISEE (calls/puts x 100)",
                 color=TXT, fontsize=13, fontweight="bold", pad=26, loc="left")

    legend_elements = [Patch(facecolor=c, alpha=0.5, label=f"{nm} ({lo}-{hi})")
                       for lo, hi, c, nm in ISEE_ZONAS]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=False,
              labelcolor=TXT, fontsize=8.5)

    fig.text(0.99, 0.02, fonte, color=FAINT, fontsize=8, ha="right")
    fig.text(0.01, 0.02, ">150 = excesso de otimismo (leitura contrária)",
             color=FAINT, fontsize=8, ha="left")
    plt.tight_layout()
    plt.show()


# ############################################################
#                         M I C R O
# ############################################################

# ============================================================
# MICRO 1: PAINEL DE PREÇOS
# ============================================================
def fetch(sym, period):
    interval = PERIOD_TO_INTERVAL.get(period, '1d')
    t = yf.Ticker(sym)
    hist = t.history(period=period, interval=interval)
    if hist is None or hist.empty:
        return None
    closes = hist['Close'].dropna()
    if closes.empty:
        return None
    try:
        info = t.fast_info if hasattr(t, 'fast_info') else {}
        currency = info.get('currency', '') or 'USD'
    except Exception:
        currency = 'USD'
    name = sym
    try:
        meta = t.info
        name = meta.get('shortName') or meta.get('longName') or sym
    except Exception:
        pass
    return {'x': closes.index, 'y': closes.values, 'currency': currency, 'name': name}

def draw_panel(period, tickers):
    n = len(tickers)
    cols = 3
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(15, max(5, rows * 4)), facecolor=BG)
    gs = GridSpec(rows, cols, figure=fig, hspace=0.50, wspace=0.18,
                  left=0.04, right=0.97, top=0.84, bottom=0.06)
    fig.text(0.04, 0.965, 'Painel de mercado', color=TXT, fontsize=18, fontweight='bold')
    fig.text(0.04, 0.925, f'Fonte: Yahoo Finance - período {PERIOD_LABELS[period]}',
             color=DIM, fontsize=10)
    ok = 0
    for idx, item in enumerate(tickers):
        r, c = divmod(idx, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color(BORDER)
        ax.set_xticks([]); ax.set_yticks([])
        data = fetch(item['sym'], period)
        ax.text(0.02, 1.30, item['label'], transform=ax.transAxes, color=TXT,
                fontsize=13, fontweight='bold', va='top')
        if not data:
            ax.text(0.02, 1.17, item['name'], transform=ax.transAxes, color=FAINT,
                    fontsize=9, va='top')
            ax.text(0.98, 1.28, 'indisponível', transform=ax.transAxes, color=FAINT,
                    fontsize=10, ha='right', va='top')
            ax.text(0.5, 0.5, 'sem dados', transform=ax.transAxes, color=FAINT,
                    fontsize=11, ha='center', va='center')
            continue
        ok += 1
        ax.text(0.02, 1.17, data['name'], transform=ax.transAxes, color=FAINT,
                fontsize=9, va='top')
        y = data['y']; x = range(len(y))
        first, last = y[0], y[-1]
        chg = ((last - first) / first) * 100
        color = UP if chg >= 0 else DOWN
        sign = '+' if chg >= 0 else ''
        ax.text(0.98, 1.28, f'{sign}{chg:.2f}%', transform=ax.transAxes, color=color,
                fontsize=11, fontweight='bold', ha='right', va='top')
        ax.text(0.02, 0.92, fmt_price(last), transform=ax.transAxes, color=TXT,
                fontsize=18, fontweight='bold', va='top')
        ax.text(0.33, 0.92, data['currency'], transform=ax.transAxes, color=FAINT,
                fontsize=9, va='top')
        ax.plot(x, y, color=color, linewidth=1.8)
        ax.fill_between(x, y, min(y), color=color, alpha=0.18)
        ax.set_xlim(0, max(len(y) - 1, 1))
        pad = (max(y) - min(y)) * 0.08 or 1
        ax.set_ylim(min(y) - pad, max(y) + pad * 3)
        ax.text(0.02, -0.06, 'min ' + fmt_price(min(y)), transform=ax.transAxes,
                color=FAINT, fontsize=8, va='top')
        ax.text(0.98, -0.06, 'max ' + fmt_price(max(y)), transform=ax.transAxes,
                color=FAINT, fontsize=8, va='top', ha='right')
    fig.text(0.97, 0.01, f'{ok}/{len(tickers)} carregados', color=FAINT,
             fontsize=9, ha='right')
    plt.show()

# ============================================================
# MICRO 2: TABELA DE RENDIMENTOS E VOLATILIDADE
# ============================================================
def get_perf_stats(sym):
    t = yf.Ticker(sym)
    hist = t.history(period="5y", interval="1d")
    if hist is None or hist.empty:
        return None
    closes = hist['Close'].dropna()
    if len(closes) < 5:
        return None
    closes.index = closes.index.tz_localize(None)
    last = closes.iloc[-1]
    now = closes.index.max()

    def ret_since(dt, tol_days):
        w = closes[closes.index >= dt]
        if len(w) < 2:
            return np.nan
        if (w.index[0] - dt).days > tol_days:
            return np.nan
        return (last / w.iloc[0] - 1) * 100

    r1w = ret_since(now - pd.DateOffset(weeks=1),  tol_days=5)
    r1m = ret_since(now - pd.DateOffset(months=1), tol_days=10)
    ytd = ret_since(pd.Timestamp(year=now.year, month=1, day=1), tol_days=15)
    r3y = ret_since(now - pd.DateOffset(years=3),  tol_days=30)
    r5y = ret_since(now - pd.DateOffset(years=5),  tol_days=30)

    daily = closes.pct_change().dropna()
    daily_1y = daily[daily.index >= now - pd.DateOffset(years=1)]
    vol = daily_1y.std() * np.sqrt(252) * 100 if len(daily_1y) > 20 else np.nan

    return {'1w': r1w, '1m': r1m, 'ytd': ytd, '3y': r3y, '5y': r5y, 'vol': vol}

def draw_returns_table(tickers):
    col_labels = ['1 semana', '1 mês', 'YTD', '3 anos', '5 anos', 'Volatilidade']
    labels, values = [], []
    for item in tickers:
        stats = get_perf_stats(item['sym'])
        labels.append(item['label'])
        if stats is None:
            values.append([np.nan] * 6)
        else:
            values.append([stats['1w'], stats['1m'], stats['ytd'],
                           stats['3y'], stats['5y'], stats['vol']])

    n = len(labels)
    fig, ax = plt.subplots(figsize=(12, 0.42 * n + 1.6), facecolor=BG)
    ax.axis('off')
    ax.set_title('Rendimento por período e volatilidade (anualizada, 1a)',
                 color=TXT, fontsize=13, fontweight='bold', pad=14, loc='left')

    cell_text = []
    for row in values:
        txt = [fmt_pct(v) for v in row[:5]] + [fmt_pct(row[5], sign=False)]
        cell_text.append(txt)

    table = ax.table(cellText=cell_text, rowLabels=labels, colLabels=col_labels,
                     cellLoc='center', rowLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_facecolor(PANEL)
        if r == 0:
            cell.set_facecolor(BG)
            cell.set_text_props(color=DIM, fontweight='bold')
        elif c == -1:
            cell.set_facecolor(BG)
            cell.set_text_props(color=TXT, fontweight='bold')
        else:
            v = values[r - 1][c]
            if c == 5:
                cell.set_text_props(color=TXT)
            elif isinstance(v, float) and np.isnan(v):
                cell.set_text_props(color=FAINT)
            else:
                cell.set_text_props(color=(UP if v >= 0 else DOWN), fontweight='bold')

    fig.text(0.99, 0.02, 'Retornos de preço (sem dividendos ajustados) - Yahoo Finance',
             color=FAINT, fontsize=8, ha='right')
    plt.tight_layout()
    plt.show()

# ============================================================
# MICRO 3: RANGE DE P/E (proxy TTM)
# ============================================================
def get_pe_series(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="10y")["Close"]
    if hist.empty:
        return None
    hist.index = hist.index.tz_localize(None)

    eps_ttm = None

    def _income_df():
        for attr in ["quarterly_income_stmt", "quarterly_incomestmt", "quarterly_financials"]:
            try:
                df_inc = getattr(t, attr)
                if df_inc is not None and not df_inc.empty:
                    return df_inc.T
            except Exception:
                continue
        return None

    inc = _income_df()
    if inc is not None:
        inc.index = pd.to_datetime(inc.index)
        eps_col = next((c for c in ["Diluted EPS", "Basic EPS"] if c in inc.columns), None)
        if eps_col is not None:
            eps_q = pd.to_numeric(inc[eps_col], errors="coerce").dropna().sort_index()
            if len(eps_q) >= 4:
                eps_ttm = eps_q.rolling(4).sum().dropna()
        if eps_ttm is None:
            ni_col = next((c for c in ["Net Income", "Net Income Common Stockholders",
                                       "Net Income Continuous Operations"] if c in inc.columns), None)
            shares = None
            try:
                shares = t.info.get("sharesOutstanding")
            except Exception:
                shares = None
            if ni_col is not None and shares:
                ni_q = pd.to_numeric(inc[ni_col], errors="coerce").dropna().sort_index()
                if len(ni_q) >= 4:
                    eps_ttm = (ni_q / shares).rolling(4).sum().dropna()

    if eps_ttm is None or eps_ttm.empty:
        return None

    eps_daily = eps_ttm.reindex(hist.index, method="ffill")
    pe = (hist / eps_daily).dropna()
    pe = pe[pe > 0]
    return pe if len(pe) >= 30 else None

def _range(pe, years):
    cutoff = pe.index.max() - pd.DateOffset(years=years)
    w = pe[pe.index >= cutoff]
    return (w.min(), w.max()) if not w.empty else None

def _point(pe, date_str):
    target = pd.to_datetime(date_str)
    idx = pe.index.get_indexer([target], method="nearest")[0]
    return pe.iloc[idx]

def draw_pe_chart(tickers, reference_date):
    today_label = datetime.today().strftime("%d-%b-%y")
    rows = []
    for item in tickers:
        pe = get_pe_series(item['sym'])
        if pe is None:
            continue
        try:
            ref = _point(pe, reference_date)
        except Exception:
            ref = np.nan
        rows.append({
            "ticker": item['label'],
            "r10": _range(pe, 10), "r3": _range(pe, 3),
            "r2": _range(pe, 2),  "r1": _range(pe, 1),
            "current": pe.iloc[-1], "ref": ref,
        })

    if not rows:
        plt.figure(figsize=(12, 3), facecolor=BG)
        plt.figtext(0.5, 0.5, "Sem dados de lucro suficientes para calcular o P/E\n"
                              "dos ativos selecionados (índices/ETFs não têm P/E).",
                    color=DIM, ha='center', va='center', fontsize=12)
        plt.axis('off'); plt.show()
        return

    df = pd.DataFrame(rows).set_index("ticker")[::-1]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * len(df))), facecolor=BG)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER)

    for i, (_, row) in enumerate(df.iterrows()):
        if row["r10"]:
            ax.barh(i, row["r10"][1]-row["r10"][0], left=row["r10"][0],
                    height=0.50, color=PE_10Y, zorder=1)
        if row["r3"]:
            ax.barh(i, row["r3"][1]-row["r3"][0], left=row["r3"][0],
                    height=0.32, color=PE_3Y, zorder=2)
        if row["r2"]:
            ax.barh(i, row["r2"][1]-row["r2"][0], left=row["r2"][0],
                    height=0.20, color=PE_2Y, zorder=3)
        if row["r1"]:
            ax.plot([row["r1"][0], row["r1"][1]], [i, i], color=PE_1Y, lw=3, zorder=4)
        if not np.isnan(row["current"]):
            ax.scatter(row["current"], i, marker="D", s=70, color=PE_CUR, zorder=5)
        if not np.isnan(row["ref"]):
            ax.scatter(row["ref"], i, marker="D", s=70, facecolors=PE_REF,
                       edgecolors=PE_CUR, linewidths=1.5, zorder=6)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index, color=TXT, fontsize=10)
    ax.tick_params(axis='x', colors=DIM, labelsize=9)
    ax.set_xlabel("P/E ratio (x) - proxy TTM via yfinance", color=DIM, fontsize=10)
    ax.set_title("Ranges de P/E das ações selecionadas", color=TXT,
                 fontsize=14, fontweight="bold", pad=30)
    ax.grid(axis="x", linestyle="--", alpha=0.25, color=FAINT)

    legend_elements = [
        Patch(facecolor=PE_10Y, label="10a range"),
        Patch(facecolor=PE_3Y,  label="3a range"),
        Patch(facecolor=PE_2Y,  label="2a range"),
        Line2D([0],[0], color=PE_1Y, lw=3, label="1a range"),
        Line2D([0],[0], marker="D", color=PANEL, markerfacecolor=PE_CUR,
               markersize=8, label=today_label),
        Line2D([0],[0], marker="D", color=PANEL, markerfacecolor=PE_REF,
               markeredgecolor=PE_CUR, markersize=8, label=str(reference_date)),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              ncol=3, frameon=False, labelcolor=TXT, fontsize=9)
    plt.tight_layout()
    plt.show()

# ============================================================
# MICRO 4: MATRIZ DE CORRELAÇÃO + PREÇOS/RETORNOS/HISTOGRAMA
# ============================================================
def _batch_closes(tickers, period="5y"):
    """Fechamentos ajustados em lote (colunas = rótulos), só ativos com dados."""
    syms = [it['sym'] for it in tickers]
    labels = dict(zip(syms, [it['label'] for it in tickers]))
    try:
        raw = yf.download(syms, period=period, interval="1d",
                          progress=False, auto_adjust=True)
    except Exception as e:
        print(f"Falha ao baixar dados: {e}")
        return None
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw['Close']
    else:
        closes = raw[['Close']]
        closes.columns = [syms[0]]
    closes = closes.dropna(axis=1, how='all')
    valid = [s for s in syms if s in closes.columns
             and closes[s].dropna().shape[0] >= 30]
    if not valid:
        return None
    closes = closes[valid]
    closes.columns = [labels[s] for s in valid]
    return closes

def draw_correlation_matrix(tickers, period="1y"):
    closes = _batch_closes(tickers, period)
    if closes is None or closes.shape[1] < 2:
        print("Ativos com dados insuficientes para calcular a correlação "
              "(mínimo de 2 ativos com ~30 pregões).")
        return
    rets = closes.pct_change().dropna(how='all')
    corr = rets.corr(min_periods=30)

    n = len(corr)
    size = max(6.5, 0.62 * n + 2.2)
    fig, ax = plt.subplots(figsize=(size, size * 0.88), facecolor=BG)
    ax.set_facecolor(PANEL)

    im = ax.imshow(corr.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', color=TXT, fontsize=9)
    ax.set_yticklabels(corr.index, color=TXT, fontsize=9)
    ax.tick_params(colors=DIM, length=0)
    for s in ax.spines.values():
        s.set_color(BORDER)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color=BG, linewidth=1.2)
    ax.tick_params(which='minor', length=0)

    fs = 9 if n <= 10 else (7.5 if n <= 16 else 6)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if np.isnan(v):
                ax.text(j, i, '-', ha='center', va='center', color=FAINT, fontsize=fs)
                continue
            txt = f'{v:.2f}'.replace('.', ',')
            tcolor = '#0d1117' if abs(v) < 0.65 else '#ffffff'
            ax.text(j, i, txt, ha='center', va='center', color=tcolor, fontsize=fs,
                    fontweight='bold' if i != j else 'normal')

    ax.set_title(f'Matriz de correlação - retornos diários ({period})',
                 color=TXT, fontsize=14, fontweight='bold', pad=14, loc='left')
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(colors=DIM, labelsize=8)
    cbar.outline.set_edgecolor(BORDER)
    fig.text(0.99, 0.01, 'Correlação de Pearson dos retornos diários - Yahoo Finance',
             color=FAINT, fontsize=8, ha='right')
    plt.tight_layout()
    plt.show()

def _line_chart(df, title, ylabel):
    n = len(df.columns)
    cmap = _cmap(n)
    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=BG)
    ax.set_facecolor(PANEL)
    for i, col in enumerate(df.columns):
        ax.plot(df.index, df[col], lw=1.1, color=cmap(i), label=col)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.set_ylabel(ylabel, color=DIM, fontsize=9)
    ax.set_title(title, color=TXT, fontsize=13, fontweight='bold', pad=12, loc='left')
    ax.grid(linestyle='--', alpha=0.2, color=FAINT)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False,
              labelcolor=TXT, fontsize=8)
    plt.tight_layout()
    plt.show()

def draw_prices_and_returns(tickers, period="5y"):
    closes = _batch_closes(tickers, period)
    if closes is None or closes.empty:
        print("Sem dados para preços de fechamento / retornos.")
        return

    norm = closes.apply(lambda s: s / s.dropna().iloc[0] * 100)
    _line_chart(norm, f'Preços de fechamento ajustados - base 100 ({period})',
                'Índice (início do período = 100)')

    rets = closes.pct_change().dropna(how='all') * 100
    _line_chart(rets, f'Retornos diários ({period})', 'Retorno diário (%)')

    n = len(rets.columns)
    cmap = _cmap(n)
    fig, ax = plt.subplots(figsize=(13, 5.5), facecolor=BG)
    ax.set_facecolor(PANEL)
    lo = np.nanpercentile(rets.values, 0.2)
    hi = np.nanpercentile(rets.values, 99.8)
    bins = np.linspace(lo, hi, 120)
    for i, col in enumerate(rets.columns):
        ax.hist(rets[col].dropna(), bins=bins, alpha=0.45,
                color=cmap(i), label=col, edgecolor='none')
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.set_xlabel('Retorno diário (%)', color=DIM, fontsize=9)
    ax.set_ylabel('Frequência (dias)', color=DIM, fontsize=9)
    ax.set_title(f'Histograma dos retornos diários ({period})', color=TXT,
                 fontsize=13, fontweight='bold', pad=12, loc='left')
    ax.grid(axis='y', linestyle='--', alpha=0.2, color=FAINT)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False,
              labelcolor=TXT, fontsize=8)
    plt.tight_layout()
    plt.show()


# ############################################################
#                   A I   M O N I T O R
# ############################################################
# Baseado em "The AI Innovation Cycle: Scenarios and Signposts"
# Schwab Asset Management (Ferrarone & Gibley), 09/11/2026.
#
# 4 cenários:  boom de produtividade | retorno adiado |
#              boom inflacionário    | quebra deflacionária
# 4 signposts: capex | financiamento | adoção | monetização
#
# O que é AO VIVO (Yahoo Finance / FRED):
#   - capex, fluxo de caixa operacional, FCF, dívida e margem operacional
#     dos hyperscalers (demonstrativos trimestrais e anuais via yfinance)
#   - spreads de crédito IG e HY, juro real de 10a, CPI de eletricidade,
#     produtividade do trabalho, produção industrial de semicondutores (FRED)
#   - cesta equal-weight da cadeia de IA (proxy do Bloomberg AI Value Chain
#     EW), memória, software vs semis, ETFs regionais
# O que é MANUAL (edite abaixo):
#   - setas dos signposts e notas (tabela do artigo, 04/11/2026)
#   - obrigações fora do balanço (10-Q/10-K, set/2026)
#   - sua probabilidade-base de cada cenário (sliders na aba)
#
# O termômetro de cenários é uma HEURÍSTICA própria deste painel
# (regras simples abaixo), não um modelo da Schwab.
# ============================================================

# ---- universo -----------------------------------------------------------
# A Schwab inclui também a SpaceX no grupo de hyperscalers; adicione o ticker
# aqui se/quando estiver listado no Yahoo.
AI_HYPERSCALERS = {
    'GOOGL': 'Alphabet', 'AMZN': 'Amazon', 'MSFT': 'Microsoft',
    'META': 'Meta', 'ORCL': 'Oracle', 'CRWV': 'CoreWeave', 'NBIS': 'Nebius',
}
# Proxy equal-weight da cadeia de valor de IA (chips, memória, equipamentos,
# rede, servidores, energia, neoclouds). Edite à vontade.
AI_CESTA = ['NVDA', 'AVGO', 'AMD', 'TSM', 'ASML', 'MU', 'LRCX', 'AMAT', 'KLAC',
            'MRVL', 'ANET', 'DELL', 'SMCI', 'VRT', 'ETN', 'GEV', 'CEG', 'VST',
            'CRWV', 'NBIS']
AI_MEMORIA = {'MU': 'Micron', '000660.KS': 'SK Hynix', '005930.KS': 'Samsung',
              'SNDK': 'SanDisk'}
AI_BENCH = {'SPY': 'S&P 500', 'SMH': 'Semis (SMH)', 'IGV': 'Software (IGV)'}
AI_REGIOES = {   # rótulo -> (ticker, signposts-chave do artigo)
    'EUA':            ('SPY',  'monetização, disciplina de capex, crédito'),
    'Taiwan':         ('EWT',  'utilização, preços, estoques, exportações'),
    'Coreia do Sul':  ('EWY',  'HBM, exportações, concentração de clientes'),
    'Japão':          ('EWJ',  'pedidos, backlog, investimento em energia'),
    'Europa':         ('VGK',  'pedidos, exportações, adoção'),
    'China':          ('MCHI', 'acesso a chips, preço de modelos, política'),
    'Brasil':         ('EWZ',  'baixa exposição direta a IA (referência)'),
}

# ---- cenários -----------------------------------------------------------
AI_CENARIOS = {   # chave -> (nome, cor)
    'PB': ('Boom de produtividade', '#3fb950'),
    'DP': ('Retorno adiado',        '#58a6ff'),
    'IB': ('Boom inflacionário',    '#e3b341'),
    'DB': ('Quebra deflacionária',  '#e5534b'),
}
AI_CURTO = {'PB': 'Produtividade', 'DP': 'Ret. adiado',
            'IB': 'Inflacionário', 'DB': 'Deflacionária'}
AI_CENARIO_TXT = {   # resumo do quadro "Possible outcomes" do artigo
    'PB': ["Modelos melhoram, custos caem; adoção e uso se espalham rápido.",
           "Capex desacelera, mas dura mais pelo ROIC alto.",
           "Crescimento real maior, lucros mais amplos, desinflação.",
           "Sinais: economias quantificadas, produtividade/hora e margens sobem."],
    'DP': ["Adoção mais lenta (retorno, regulação, cibersegurança); gargalos.",
           "Capex e backlog crescem; preço de compute cai; monetização atrasa.",
           "Capex sustenta o PIB, mas pressiona caixa; mais dispersão.",
           "Sinais: poucas empresas quantificam ganho; backlog converte devagar."],
    'IB': ["Energia, rede, equipamentos, memória e obras ficam atrás da adoção.",
           "Capex depende de dívida, equity e fora do balanço; custo de capital sobe.",
           "Crescimento nominal forte, inflação persistente, juro real alto.",
           "Sinais: escassez de energia, lead times, custos de obra, funding externo."],
    'DB': ["Capacidade supera a demanda durável; preço cai mais que volume.",
           "Utilização e receita frustram; projetos cortados ou adiados.",
           "Queda do capex, condições financeiras apertadas, desinflação.",
           "Sinais: aluguel de GPU cai, cancelamentos, spreads abrem, baixas contábeis."],
}

# Probabilidade-base (SUA visão, editável nos sliders). O artigo NÃO publica
# números: diz que retorno adiado e boom inflacionário são os mais prováveis,
# e que a eleição de meio de mandato elevou retorno adiado e quebra deflacionária.
AI_PRIOR = {'PB': 15, 'DP': 35, 'IB': 30, 'DB': 20}
AI_PESO_SINAIS = 0.5   # 0 = só sua visão | 1 = só os sinais de mercado

# ---- signposts (tabela "AI scenario signposts", Schwab, 04/11/2026) -------
# tendência: 'up' melhora/sobe | 'down' piora | 'flat' lateral/inconclusivo
AI_SIGNPOSTS = [
    ('Capex', 'up', [
        'Capex deve passar de US$1 tri em 2026 e crescer mais devagar em 2027',
        'Importações de chips, equipamentos e materiais de data center fortes',
        'Obras aceleradas, mas energia e regulação podem frear',
        'Modelos melhores podem trazer inovação e monetização']),
    ('Financiamento', 'down', [
        'Hyperscalers gastam mais que o FCF -> funding externo',
        'Emissão de dívida e equity forte, mas o custo subiu',
        'Financiamento fora do balanço ~4x o do balanço',
        'Spreads de crédito ligados a IA abriram mais que o mercado']),
    ('Adoção', 'up', [
        'Preço por token caindo (bom p/ adoção, ruim p/ os labs)',
        'Volume de tokens acelerando',
        'Gasto com tokens estável (preço menor x volume maior)',
        'Gasto por funcionário sobe, mas top 1% muito acima da mediana']),
    ('Monetização', 'flat', [
        'RPO dos provedores de nuvem (receita contratada futura)',
        'Margens dos hyperscalers',
        'Produtividade do usuário final',
        'Ainda inconclusivo: evidência ampla limitada além de código']),
]
AI_SETA = {'up': ('▲', UP, 'subindo'), 'down': ('▼', DOWN, 'piorando'),
           'flat': ('◆', '#58a6ff', 'inconclusivo')}

# ---- obrigações on/off-balance (US$ bi, 10-Q/10-K set/2026; pirâmide do artigo)
AI_OFF_BALANCE = {
    #             dívida LP, leasing (balanço), leasing não iniciado, compromissos de compra
    'Alphabet':  (125.1, 20.6, 130.5, 924.4),
    'Amazon':    (180.9, 109.8, 164.9, 159.6),
    'Microsoft': (40.3,  88.5, 364.6, 228.6),
    'Meta':      (83.7,  31.2, 355.7, 454.5),
}
AI_OFF_COLS = ['Dívida LP', 'Leasing (balanço)', 'Leasing não iniciado',
               'Compromissos de compra']
AI_OFF_CORES = ['#7ee787', '#8b949e', '#38b2c4', '#1f6feb']

# ---- séries FRED ----------------------------------------------------------
AI_FRED = {
    'hy':    'BAMLH0A0HYM2',      # spread HY (OAS, %)
    'ig':    'BAMLC0A0CM',        # spread IG (OAS, %)
    'real':  'DFII10',            # juro real 10a (TIPS, %)
    'elet':  'CUSR0000SEHF01',    # CPI eletricidade (SA)
    'prod':  'OPHNFB',            # produtividade: output/hora, setor não-agrícola
    'semi':  'IPG3344S',          # produção industrial: semicondutores e componentes
}

# ============================================================
# CACHE + COLETA DE DADOS
# ============================================================
_AI_CACHE = {}

def ai_limpar_cache():
    _AI_CACHE.clear()

def _ai_cached(key, fn):
    if key not in _AI_CACHE:
        try:
            _AI_CACHE[key] = fn()
        except Exception as e:
            print(f"  {key} falhou: {type(e).__name__}")
            _AI_CACHE[key] = None
    return _AI_CACHE[key]

def _ai_fred(nome):
    return _ai_cached(('fred', nome), lambda: _fred_csv(AI_FRED[nome]))

def _ai_row(df, nomes):
    if df is None or getattr(df, 'empty', True):
        return None
    for n in nomes:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors='coerce').dropna()
            if s.empty:
                continue
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    return None

def _ai_fund_one(sym):
    t = yf.Ticker(sym)

    def g(attr):
        try:
            d = getattr(t, attr)
            return d if d is not None and not d.empty else None
        except Exception:
            return None

    qcf, acf = g('quarterly_cashflow'), g('cashflow')
    qis, qbs = g('quarterly_income_stmt'), g('quarterly_balance_sheet')
    ocf_n = ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities']
    out = {
        'capex_q': _ai_row(qcf, ['Capital Expenditure']),
        'ocf_q':   _ai_row(qcf, ocf_n),
        'capex_a': _ai_row(acf, ['Capital Expenditure']),
        'ocf_a':   _ai_row(acf, ocf_n),
        'rev_q':   _ai_row(qis, ['Total Revenue', 'Operating Revenue']),
        'opi_q':   _ai_row(qis, ['Operating Income', 'Total Operating Income As Reported']),
        'debt_q':  _ai_row(qbs, ['Total Debt', 'Long Term Debt']),
    }
    for k in ('capex_q', 'capex_a'):
        if out[k] is not None:
            out[k] = out[k].abs()
    return out

def ai_fundamentos():
    def _load():
        res = {}
        for sym, nm in AI_HYPERSCALERS.items():
            print(f"  fundamentos {sym}...")
            try:
                res[nm] = _ai_fund_one(sym)
            except Exception as e:
                print(f"    {sym} falhou: {type(e).__name__}")
        return res
    return _ai_cached('fund', _load) or {}

def _ai_painel(fund, chave, freq='Q'):
    """DataFrame (US$ bi) período x empresa. Trimestres calendarizados."""
    cols = {}
    for nm, d in fund.items():
        s = d.get(chave) if d else None
        if s is None or s.empty:
            continue
        s = s.copy() / 1e9
        s.index = s.index.to_period('Q') if freq == 'Q' else s.index.year
        cols[nm] = s.groupby(level=0).last()
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()

def ai_precos(period='2y'):
    syms = list(dict.fromkeys(
        AI_CESTA + list(AI_MEMORIA) + list(AI_BENCH) +
        [v[0] for v in AI_REGIOES.values()]))

    def _load():
        raw = yf.download(syms, period=period, interval='1d',
                          progress=False, auto_adjust=True)
        c = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
        c = c.copy()
        c.index = pd.to_datetime(c.index)
        if c.index.tz is not None:
            c.index = c.index.tz_localize(None)
        return c.dropna(axis=1, how='all').ffill()
    return _ai_cached(('px', period), _load)

def _ai_cesta_idx(px):
    cols = [s for s in AI_CESTA if s in px.columns]
    if not cols:
        return None
    r = px[cols].pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0)
    return (1 + r).cumprod() * 100

def _ret_desde(s, dt):
    s = s.dropna()
    w = s[s.index >= dt]
    if len(w) < 2:
        return np.nan
    base = s[s.index < dt]
    b = base.iloc[-1] if not base.empty else w.iloc[0]
    return (s.iloc[-1] / b - 1) * 100

def _inicio_tri(ts):
    return pd.Timestamp(year=ts.year, month=3 * ((ts.month - 1) // 3) + 1, day=1)

def _yoy_q(df):
    """Último trimestre vs mesmo tri do ano anterior, só empresas com os 2 dados."""
    if df is None or df.empty:
        return np.nan, None
    p = df.index[-1]
    if (p - 4) not in df.index:
        return np.nan, p
    a, b = df.loc[p], df.loc[p - 4]
    m = a.notna() & b.notna()
    if m.sum() == 0 or b[m].sum() == 0:
        return np.nan, p
    return (a[m].sum() / b[m].sum() - 1) * 100, p

def _ultimo_tri_completo(df, min_frac=0.6):
    """Corta trimestres em que poucas empresas já reportaram."""
    if df is None or df.empty:
        return df
    n = df.notna().sum(axis=1)
    ok = n >= max(1, int(np.ceil(min_frac * n.max())))
    return df[ok]

# ============================================================
# MOTOR DE SINAIS -> CENÁRIOS (heurística)
# ============================================================
def ai_sinais():
    """Lista de dicts: categoria, indicador, leitura, pontos {cenário: pts}."""
    S = []

    def add(cat, nome, leitura, pts):
        S.append({'cat': cat, 'nome': nome, 'leitura': leitura, 'pts': pts})

    fund = ai_fundamentos()
    capex = _ultimo_tri_completo(_ai_painel(fund, 'capex_q'))
    ocf = _ai_painel(fund, 'ocf_q')
    debt = _ai_painel(fund, 'debt_q')
    rev, opi = _ai_painel(fund, 'rev_q'), _ai_painel(fund, 'opi_q')

    # --- CAPEX ---------------------------------------------------------
    g, p = _yoy_q(capex)
    if not np.isnan(g):
        if g > 20:   pts = {'DP': 1, 'IB': 1}
        elif g >= 0: pts = {'PB': 1, 'DP': 0.5}
        else:        pts = {'DB': 2}
        add('Capex', f'Capex hyperscalers a/a ({p})', fmt_pct(g), pts)

    s = _ai_fred('semi')
    if s is not None and len(s) > 13:
        y = (s.iloc[-1] / s.iloc[-13] - 1) * 100
        pts = {'IB': 1} if y > 5 else ({'DB': 1} if y < -3 else {'DP': 0.5})
        add('Capex', 'Produção ind. semicondutores a/a', fmt_pct(y), pts)

    # --- FINANCIAMENTO -------------------------------------------------
    if not capex.empty and not ocf.empty:
        ix = capex.index.intersection(ocf.index)[-4:]
        if len(ix) >= 2:
            c4, o4 = capex.loc[ix], ocf.loc[ix]
            m = c4.notna() & o4.notna()
            if o4[m].sum().sum() > 0:
                r = c4[m].sum().sum() / o4[m].sum().sum() * 100
                pts = {'IB': 1, 'DB': 0.5} if r > 80 else ({'DP': 1} if r > 60 else {'PB': 1})
                add('Financiamento', f'Capex / FCO (últ. {len(ix)} tri)',
                    fmt_pct(r, sign=False), pts)
        fcf = (ocf - capex).dropna(how='all')
        if not fcf.empty and (fcf.index[-1] - 4) in fcf.index:
            p2 = fcf.index[-1]
            a, b = fcf.loc[p2], fcf.loc[p2 - 4]
            m = a.notna() & b.notna()
            if m.sum():
                f0, f1 = a[m].sum(), b[m].sum()
                var = (f0 - f1) / max(abs(f1), 1e-9) * 100
                pts = ({'DP': 1, 'IB': 1} if var < -10 else
                       ({'PB': 1} if var > 10 else {'DP': 0.5}))
                add('Financiamento', f'FCF trimestral hyperscalers ({p2})',
                    f'US${f0:.0f} bi ({f0 - f1:+.0f} bi a/a)', pts)

    g3, p3 = _yoy_q(debt)
    if not np.isnan(g3):
        pts = {'IB': 1} if g3 > 20 else ({'DP': 0.5} if g3 > 0 else {'PB': 0.5})
        add('Financiamento', f'Dívida total a/a ({p3})', fmt_pct(g3), pts)

    hy = _ai_fred('hy')
    if hy is not None and len(hy) > 70:
        d = (hy.iloc[-1] - hy.iloc[-64]) * 100
        if d > 75:    pts = {'DB': 2}
        elif d > 25:  pts = {'DB': 1, 'DP': 1}
        elif d < -25: pts = {'PB': 1, 'IB': 0.5}
        else:         pts = {'DP': 0.5}
        add('Financiamento', 'Spread HY: var. 3 meses',
            f'{hy.iloc[-1]:.2f}% ({d:+.0f} bp)'.replace('.', ','), pts)

    rr = _ai_fred('real')
    if rr is not None and len(rr) > 70:
        d = rr.iloc[-1] - rr.iloc[-64]
        pts = {'IB': 1} if (rr.iloc[-1] > 2 and d > 0) else ({'PB': 0.5} if d < -0.25 else {})
        if pts:
            add('Financiamento', 'Juro real 10a (nível, var. 3m)',
                f'{rr.iloc[-1]:.2f}% ({d*100:+.0f} bp)'.replace('.', ','), pts)

    # --- ADOÇÃO / CADEIA DE SUPRIMENTO --------------------------------
    px = ai_precos()
    if px is not None and not px.empty:
        idx = _ai_cesta_idx(px)
        if idx is not None:
            last = idx.index[-1]
            dd = (idx.iloc[-1] / idx[idx.index >= last - pd.DateOffset(years=1)].max() - 1) * 100
            if dd < -25:   pts = {'DB': 2}
            elif dd < -10: pts = {'DP': 1, 'DB': 1}
            else:          pts = {'PB': 1, 'IB': 1}
            qtd = _ret_desde(idx, _inicio_tri(last))
            add('Adoção', 'Cesta IA EW: queda vs máx. 52s (QTD)',
                f'{fmt_pct(dd)} ({fmt_pct(qtd)})', pts)
        if 'MU' in px:
            r3 = _ret_desde(px['MU'], px.index[-1] - pd.DateOffset(months=3))
            pts = {'IB': 1} if r3 > 15 else ({'DB': 1} if r3 < -20 else {'DP': 0.5})
            add('Adoção', 'Micron 3m (proxy preço DRAM/HBM)', fmt_pct(r3), pts)

    el = _ai_fred('elet')
    if el is not None and len(el) > 13:
        y = (el.iloc[-1] / el.iloc[-13] - 1) * 100
        pts = {'IB': 1} if y > 4 else ({'DB': 0.5} if y < 1 else {})
        if pts:
            add('Adoção', 'CPI eletricidade a/a (escassez de energia)', fmt_pct(y), pts)

    # --- MONETIZAÇÃO ---------------------------------------------------
    if not rev.empty and not opi.empty:
        ix = rev.index.intersection(opi.index)
        if len(ix) and (ix[-1] - 4) in ix:
            def mg(pp):
                a, b = opi.loc[pp], rev.loc[pp]
                m = a.notna() & b.notna()
                return a[m].sum() / b[m].sum() * 100 if b[m].sum() else np.nan
            m0, m1 = mg(ix[-1]), mg(ix[-1] - 4)
            d = m0 - m1
            pts = {'PB': 1} if d > 1 else ({'DP': 1, 'DB': 0.5} if d < -1 else {'DP': 0.5})
            add('Monetização', f'Margem oper. agregada ({ix[-1]})',
                f'{m0:.1f}%'.replace('.', ',') + f' ({d:+.1f}'.replace('.', ',') + ' p.p. a/a)', pts)

    pr = _ai_fred('prod')
    if pr is not None and len(pr) > 5:
        y = (pr.iloc[-1] / pr.iloc[-5] - 1) * 100
        pts = {'PB': 2} if y > 2.5 else ({'PB': 1} if y > 1.5 else {'DP': 1})
        add('Monetização', f'Produtividade/hora EUA a/a ({pr.index[-1]:%m/%y})',
            fmt_pct(y), pts)

    if px is not None and {'IGV', 'SMH'} <= set(px.columns):
        rel = (px['IGV'] / px['SMH']).dropna()
        if len(rel) > 130:
            d = (rel.iloc[-1] / rel.iloc[-127] - 1) * 100
            pts = {'PB': 1} if d > 10 else ({'IB': 0.5, 'DP': 0.5} if d < -10 else {})
            if pts:
                add('Monetização', 'Software vs semis 6m (adotantes x habilitadores)',
                    fmt_pct(d), pts)
    return S

def ai_probabilidades(sinais, prior, peso):
    pts = {k: 0.0 for k in AI_CENARIOS}
    for s in sinais:
        for k, v in s['pts'].items():
            pts[k] += v
    tot = sum(pts.values())
    sig = ({k: v / tot * 100 for k, v in pts.items()} if tot > 0
           else {k: 25.0 for k in AI_CENARIOS})
    ptot = sum(prior.values()) or 1
    pri = {k: prior[k] / ptot * 100 for k in AI_CENARIOS}
    comb = {k: (1 - peso) * pri[k] + peso * sig[k] for k in AI_CENARIOS}
    return pri, sig, comb, pts

# ============================================================
# HELPERS DE ESTILO
# ============================================================
def _ai_ax(ax, titulo=None, ylab=None):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.grid(linestyle='--', alpha=0.2, color=FAINT)
    ax.set_axisbelow(True)
    if titulo:
        ax.set_title(titulo, color=TXT, fontsize=11, fontweight='bold', loc='left', pad=8)
    if ylab:
        ax.set_ylabel(ylab, color=DIM, fontsize=9)

def _ai_vazio(ax, msg='sem dados'):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, color=FAINT,
            ha='center', va='center', fontsize=11)

def _ai_header(fig, titulo, sub):
    h = fig.get_figheight()
    fig.text(0.04, 1 - 0.25 / h, titulo, color=TXT, fontsize=17, fontweight='bold', va='top')
    fig.text(0.04, 1 - 0.62 / h, sub, color=DIM, fontsize=9.5, va='top')

def _ai_legend(ax, **kw):
    kw.setdefault('fontsize', 8)
    ax.legend(frameon=False, labelcolor=TXT, **kw)

def _wrap(t, w):
    return '\n'.join(textwrap.wrap(t, w))

# ============================================================
# AI 1: QUADRO DE CENÁRIOS + SIGNPOSTS
# ============================================================
def draw_ai_quadro():
    fig = plt.figure(figsize=(15, 9.6), facecolor=BG)
    _ai_header(fig, 'AI Monitor - cenários e signposts do ciclo de IA',
               'Base: Schwab Asset Management, "The AI Innovation Cycle: Scenarios '
               'and Signposts" (09/11/2026) - tabela de signposts de 04/11/2026')
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.05, 1], hspace=0.22,
                  wspace=0.06, left=0.03, right=0.97, top=0.86, bottom=0.10)
    rot = ['Tecnologia', 'Ciclo de capital', 'Macro/mercado', '']
    for i, (k, (nm, cor)) in enumerate(AI_CENARIOS.items()):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(cor if k in ('DP', 'IB') else BORDER)
            s.set_linewidth(1.6 if k in ('DP', 'IB') else 0.8)
        ax.add_patch(plt.Rectangle((0, 0.9), 1, 0.1, transform=ax.transAxes,
                                   color=cor, alpha=0.25, lw=0))
        ax.text(0.05, 0.95, f'{i+1}. {nm}', transform=ax.transAxes, color=cor,
                fontsize=11.5, fontweight='bold', va='center')
        y = 0.84
        for j, linha in enumerate(AI_CENARIO_TXT[k]):
            if rot[j]:
                ax.text(0.05, y, rot[j].upper(), transform=ax.transAxes,
                        color=FAINT, fontsize=7, fontweight='bold', va='top')
                y -= 0.045
            ax.text(0.05, y, _wrap(linha, 40), transform=ax.transAxes,
                    color=(TXT if j < 3 else DIM), fontsize=8.3, va='top',
                    style=('italic' if j == 3 else 'normal'), linespacing=1.2)
            y -= 0.058 * (len(textwrap.wrap(linha, 40))) + 0.035

    for i, (cat, tend, itens) in enumerate(AI_SIGNPOSTS):
        seta, cor, txt = AI_SETA[tend]
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(BORDER)
        ax.text(0.05, 0.92, f'Signpost {i+1}: {cat}', transform=ax.transAxes,
                color=TXT, fontsize=11.5, fontweight='bold', va='center')
        ax.text(0.95, 0.92, seta, transform=ax.transAxes, color=cor,
                fontsize=20, ha='right', va='center')
        ax.text(0.95, 0.81, txt, transform=ax.transAxes, color=cor,
                fontsize=8, ha='right', va='center')
        y = 0.72
        for it in itens:
            ax.text(0.05, y, '• ' + _wrap(it, 38), transform=ax.transAxes,
                    color=DIM, fontsize=8.4, va='top', linespacing=1.2)
            y -= 0.075 * len(textwrap.wrap(it, 38)) + 0.05

    fig.text(0.03, 0.045,
             _wrap('Visão da Schwab: retorno adiado e boom inflacionário são os '
                   'cenários mais prováveis (bordas destacadas); boom de produtividade '
                   'é o cenário otimista e quebra deflacionária o pessimista. A eleição '
                   'de meio de mandato, a oposição a data centers e a queda de 22% do '
                   'Bloomberg AI Value Chain EW no trimestre elevam a chance de retorno '
                   'adiado e de quebra deflacionária.', 190),
             color=TXT, fontsize=9, va='center',
             bbox=dict(facecolor=BORDER, edgecolor='none', pad=6))
    plt.show()

# ============================================================
# AI 2: TERMÔMETRO DE CENÁRIOS
# ============================================================
def draw_ai_termometro(prior=None, peso=None):
    prior = prior or AI_PRIOR
    peso = AI_PESO_SINAIS if peso is None else peso
    sinais = ai_sinais()
    pri, sig, comb, pts = ai_probabilidades(sinais, prior, peso)

    n = max(len(sinais), 4)
    fig = plt.figure(figsize=(15, max(6.2, 0.36 * n + 3.2)), facecolor=BG)
    _ai_header(fig, 'Termômetro de cenários',
               f'Combinado = {1-peso:.0%} sua visão (sliders) + {peso:.0%} sinais de '
               'mercado. Heurística deste painel - não é um modelo da Schwab.')
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.55], wspace=0.08,
                  left=0.04, right=0.98, top=0.84, bottom=0.08)

    # ---- barras ----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _ai_ax(ax, 'Probabilidades (%)')
    ks = list(AI_CENARIOS)[::-1]
    y = np.arange(len(ks))
    h = 0.24
    ax.barh(y + h, [pri[k] for k in ks], height=h, color=FAINT, label='Sua visão')
    ax.barh(y, [sig[k] for k in ks], height=h, color='#38b2c4', label='Sinais')
    for i, k in enumerate(ks):
        cor = AI_CENARIOS[k][1]
        ax.barh(i - h, comb[k], height=h, color=cor, label=None)
        ax.text(comb[k] + 1, i - h, f'{comb[k]:.0f}%', color=cor, fontsize=10,
                fontweight='bold', va='center')
        ax.text(sig[k] + 1, i, f'{sig[k]:.0f}%  ({pts[k]:.1f} pts)', color=DIM,
                fontsize=8, va='center')
    ax.set_yticks(y)
    ax.set_yticklabels([AI_CENARIOS[k][0] for k in ks], color=TXT, fontsize=10)
    ax.set_xlim(0, max(max(pri.values()), max(sig.values()), max(comb.values())) * 1.35)
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[Patch(color=FAINT, label='Sua visão'),
                       Patch(color='#38b2c4', label='Sinais de mercado'),
                       Patch(color=TXT, label='Combinado (cor do cenário)')],
              loc='lower right', frameon=False, labelcolor=TXT, fontsize=8)
    lider = max(comb, key=comb.get)
    ax.text(0.0, -0.12, f'Cenário líder: {AI_CENARIOS[lider][0]} ({comb[lider]:.0f}%)',
            transform=ax.transAxes, color=AI_CENARIOS[lider][1], fontsize=11,
            fontweight='bold')

    # ---- tabela de sinais -----------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); ax.axis('off')
    ax.set_title('Sinais de mercado (ao vivo)', color=TXT, fontsize=11,
                 fontweight='bold', loc='left', pad=8)
    if not sinais:
        _ai_vazio(ax, 'Nenhum sinal disponível (Yahoo/FRED offline?)')
        plt.show(); return
    linhas, cores = [], []
    for s in sinais:
        fav = sorted(s['pts'].items(), key=lambda kv: -kv[1])
        fav_txt = ', '.join(f"{AI_CURTO[k]} +{v:g}" for k, v in fav)
        linhas.append([s['cat'], s['nome'], s['leitura'], fav_txt])
        cores.append(AI_CENARIOS[fav[0][0]][1] if fav else DIM)
    tb = ax.table(cellText=linhas, colLabels=['Signpost', 'Indicador', 'Leitura', 'Favorece'],
                  cellLoc='left', loc='upper center',
                  colWidths=[0.12, 0.37, 0.21, 0.30])
    tb.auto_set_font_size(False); tb.set_fontsize(8.3); tb.scale(1, 1.45)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_facecolor(BG if r == 0 else PANEL)
        if r == 0:
            cell.set_text_props(color=DIM, fontweight='bold')
        elif c == 3:
            cell.set_text_props(color=cores[r - 1], fontweight='bold')
        elif c == 2:
            cell.set_text_props(color=TXT, fontweight='bold')
        else:
            cell.set_text_props(color=(TXT if c == 1 else DIM))
    plt.show()

# ============================================================
# AI 3: SIGNPOST 1 - CAPEX
# ============================================================
def draw_ai_capex():
    fund = ai_fundamentos()
    cq = _ai_painel(fund, 'capex_q')
    ca = _ai_painel(fund, 'capex_a', 'A')
    oq = _ai_painel(fund, 'ocf_q')

    fig = plt.figure(figsize=(15, 6.4), facecolor=BG)
    _ai_header(fig, 'Signpost 1 - Capex dos hyperscalers',
               'US$ bilhões | trimestres calendarizados (MSFT e ORCL têm ano fiscal '
               'diferente) | Fonte: demonstrativos via Yahoo Finance')
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1], wspace=0.16,
                  left=0.05, right=0.95, top=0.83, bottom=0.12)
    nomes = list(AI_HYPERSCALERS.values())
    cmap = _cmap(len(nomes))
    cor = {nm: cmap(i) for i, nm in enumerate(nomes)}

    ax = fig.add_subplot(gs[0, 0])
    _ai_ax(ax, 'Capex trimestral por empresa', 'US$ bi')
    if cq.empty:
        _ai_vazio(ax)
    else:
        cq = cq.iloc[-8:]
        x = np.arange(len(cq)); base = np.zeros(len(cq))
        for nm in [n for n in nomes if n in cq]:
            v = cq[nm].fillna(0).values
            ax.bar(x, v, bottom=base, color=cor[nm], label=nm, width=0.65)
            base += v
        for xi, b in zip(x, base):
            ax.text(xi, b * 1.01, f'{b:.0f}', ha='center', va='bottom',
                    color=TXT, fontsize=8.5, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels([str(p) for p in cq.index], color=TXT)
        _ai_legend(ax, loc='upper left', ncol=4)
        ax.set_ylim(0, base.max() * 1.25)
        if not oq.empty:
            ix = cq.index.intersection(oq.index)
            ratio = []
            for p in cq.index:
                if p in ix:
                    a, b = cq.loc[p], oq.loc[p]
                    m = a.notna() & b.notna()
                    ratio.append(a[m].sum() / b[m].sum() * 100 if b[m].sum() > 0 else np.nan)
                else:
                    ratio.append(np.nan)
            ax2 = ax.twinx()
            ax2.plot(x, ratio, color='#ff7b72', marker='D', ms=5, lw=1.5,
                     label='Capex / FCO (%)')
            ax2.set_ylim(0, max(120, np.nanmax(ratio) * 1.2) if np.isfinite(np.nanmax(ratio)) else 120)
            ax2.axhline(100, color='#ff7b72', lw=0.8, ls=':', alpha=0.6)
            ax2.tick_params(colors='#ff7b72', labelsize=8)
            ax2.set_ylabel('Capex / fluxo de caixa operacional (%)', color='#ff7b72', fontsize=8.5)
            for s in ax2.spines.values():
                s.set_color(BORDER)
            ax2.legend(loc='upper right', frameon=False, labelcolor=TXT, fontsize=8)
        g, p = _yoy_q(_ultimo_tri_completo(cq))
        if not np.isnan(g):
            ax.text(0.01, -0.13, f'Crescimento a/a do último tri ({p}): {fmt_pct(g)}',
                    transform=ax.transAxes, color=(UP if g > 0 else DOWN),
                    fontsize=9.5, fontweight='bold')

    ax = fig.add_subplot(gs[0, 1])
    _ai_ax(ax, 'Capex anual (ano fiscal)', 'US$ bi')
    if ca.empty:
        _ai_vazio(ax)
    else:
        ca = ca.iloc[-5:]
        x = np.arange(len(ca)); base = np.zeros(len(ca))
        for nm in [n for n in nomes if n in ca]:
            v = ca[nm].fillna(0).values
            ax.bar(x, v, bottom=base, color=cor[nm], width=0.6)
            base += v
        for xi, b in zip(x, base):
            ax.text(xi, b * 1.01, f'{b:.0f}', ha='center', va='bottom',
                    color=TXT, fontsize=9, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels([str(a) for a in ca.index], color=TXT)
        ax.set_ylim(0, base.max() * 1.18)
        ax.text(0.02, 0.95, 'Schwab: capex do grupo deve\npassar de US$1 tri em 2026',
                transform=ax.transAxes, color=DIM, fontsize=8.5, va='top')
    plt.show()

# ============================================================
# AI 4: SIGNPOST 2 - FINANCIAMENTO
# ============================================================
def draw_ai_financiamento():
    fund = ai_fundamentos()
    ca, oa = _ai_painel(fund, 'capex_a', 'A'), _ai_painel(fund, 'ocf_a', 'A')
    cq, oq = _ai_painel(fund, 'capex_q'), _ai_painel(fund, 'ocf_q')

    fig = plt.figure(figsize=(15, 10), facecolor=BG)
    _ai_header(fig, 'Signpost 2 - Financiamento',
               'FCF = fluxo de caixa operacional - capex | spreads: ICE BofA OAS (FRED) | '
               'obrigações: 10-Q/10-K (Schwab, set/2026)')
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.16,
                  left=0.05, right=0.97, top=0.86, bottom=0.06)

    # (a) FCO x capex x FCF anual agregado
    ax = fig.add_subplot(gs[0, 0])
    _ai_ax(ax, 'Agregado anual: FCO, capex e FCF', 'US$ bi')
    if ca.empty or oa.empty:
        _ai_vazio(ax)
    else:
        ix = ca.index.intersection(oa.index)[-5:]
        O, C = [], []
        for a in ix:
            m = ca.loc[a].notna() & oa.loc[a].notna()
            O.append(oa.loc[a][m].sum()); C.append(ca.loc[a][m].sum())
        O, C = np.array(O), np.array(C); F = O - C
        x = np.arange(len(ix)); w = 0.27
        ax.bar(x - w, O, w, color='#8b949e', label='Fluxo de caixa operacional')
        ax.bar(x, C, w, color='#1f6feb', label='Capex')
        ax.bar(x + w, F, w, color=[UP if f >= 0 else DOWN for f in F], label='FCF')
        for xi, f in zip(x, F):
            ax.text(xi + w, f + (5 if f >= 0 else -5), f'{f:.0f}', ha='center',
                    va=('bottom' if f >= 0 else 'top'), color=TXT, fontsize=8.5)
        ax.axhline(0, color=FAINT, lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([str(a) for a in ix], color=TXT)
        _ai_legend(ax, loc='upper left')

    # (b) FCF trimestral por empresa
    ax = fig.add_subplot(gs[0, 1])
    _ai_ax(ax, 'FCF trimestral por empresa', 'US$ bi')
    fcf = (oq - cq).dropna(how='all').iloc[-6:] if not cq.empty and not oq.empty else pd.DataFrame()
    if fcf.empty:
        _ai_vazio(ax)
    else:
        nomes = [n for n in AI_HYPERSCALERS.values() if n in fcf]
        cmap = _cmap(len(AI_HYPERSCALERS))
        x = np.arange(len(fcf)); w = 0.8 / max(len(nomes), 1)
        for i, nm in enumerate(nomes):
            ax.bar(x - 0.4 + w * (i + 0.5), fcf[nm].values, w,
                   color=cmap(list(AI_HYPERSCALERS.values()).index(nm)), label=nm)
        tot = fcf.sum(axis=1, min_count=1)
        ax.plot(x, tot, color=TXT, marker='o', ms=4, lw=1.2, label='Total')
        ax.axhline(0, color=FAINT, lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([str(p) for p in fcf.index], color=TXT)
        _ai_legend(ax, loc='upper right', ncol=2, fontsize=7.5)

    # (c) spreads de crédito
    ax = fig.add_subplot(gs[1, 0])
    _ai_ax(ax, 'Spreads de crédito corporativo EUA (OAS)', 'p.p.')
    hy, ig = _ai_fred('hy'), _ai_fred('ig')
    if hy is None and ig is None:
        _ai_vazio(ax)
    else:
        corte = pd.Timestamp.today() - pd.DateOffset(years=3)
        if hy is not None:
            h = hy[hy.index >= corte]
            ax.plot(h.index, h, color='#e5534b', lw=1.4, label=f'High yield ({h.iloc[-1]:.2f})')
        if ig is not None:
            i2 = ig[ig.index >= corte]
            ax2 = ax.twinx()
            ax2.plot(i2.index, i2, color='#58a6ff', lw=1.4, label=f'Grau de investimento ({i2.iloc[-1]:.2f})')
            ax2.tick_params(colors='#58a6ff', labelsize=8)
            for s in ax2.spines.values():
                s.set_color(BORDER)
            ax2.legend(loc='upper right', frameon=False, labelcolor=TXT, fontsize=8)
        _ai_legend(ax, loc='upper left')
        ax.text(0.01, 0.03, 'Abertura de spread = sinal de quebra deflacionária',
                transform=ax.transAxes, color=FAINT, fontsize=8)

    # (d) on vs off-balance
    ax = fig.add_subplot(gs[1, 1])
    _ai_ax(ax, 'Obrigações: no balanço x fora do balanço', 'US$ bi')
    df = pd.DataFrame(AI_OFF_BALANCE, index=AI_OFF_COLS).T
    y = np.arange(len(df)); left = np.zeros(len(df))
    for c, cc in zip(AI_OFF_COLS, AI_OFF_CORES):
        ax.barh(y, df[c], left=left, color=cc, label=c, height=0.6)
        left += df[c].values
    for yi, (nm, row) in enumerate(df.iterrows()):
        on, off = row.iloc[:2].sum(), row.iloc[2:].sum()
        ax.text(left[yi] + 15, yi, f'{off/on:.1f}x fora/no', color=TXT,
                fontsize=8.5, va='center')
    ax.set_yticks(y); ax.set_yticklabels(df.index, color=TXT)
    ax.set_xlim(0, left.max() * 1.22)
    ax.set_ylabel('')
    on_t, off_t = df.iloc[:, :2].values.sum(), df.iloc[:, 2:].values.sum()
    tot_txt = (f'Total: US$ {on_t:,.0f} bi no balanço x US$ {off_t:,.0f} bi fora '
               f'({off_t/on_t:.1f}x)').replace(',', '.')
    ax.text(1.0, 1.015, tot_txt, transform=ax.transAxes, color=DIM,
            fontsize=8.5, ha='right', va='bottom')
    _ai_legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=7.5)
    plt.show()

# ============================================================
# AI 5: SIGNPOST 3 - ADOÇÃO E CADEIA DE SUPRIMENTO (proxies)
# ============================================================
def draw_ai_adocao():
    px = ai_precos()
    fig = plt.figure(figsize=(15, 10), facecolor=BG)
    _ai_header(fig, 'Signpost 3 - Adoção e cadeia de suprimento (proxies de mercado)',
               'Dados de tokens, aluguel de GPU e DRAM não são públicos/gratuitos: '
               'usamos preços de mercado e séries do FRED como proxy')
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.16,
                  left=0.05, right=0.97, top=0.86, bottom=0.06)
    corte = pd.Timestamp.today() - pd.DateOffset(years=1)

    # (a) cesta IA vs benchmarks
    ax = fig.add_subplot(gs[0, 0])
    _ai_ax(ax, 'Cesta IA equal-weight vs benchmarks (1 ano, base 100)', 'base 100')
    if px is None or px.empty:
        _ai_vazio(ax)
    else:
        p1 = px[px.index >= corte]
        idx = _ai_cesta_idx(p1)
        if idx is not None:
            ax.plot(idx.index, idx, color='#e3b341', lw=2.0,
                    label=f'Cesta IA EW ({len([s for s in AI_CESTA if s in px])} ações)')
            q0 = _inicio_tri(idx.index[-1])
            ax.axvline(q0, color=FAINT, ls=':', lw=1)
            qtd = _ret_desde(idx, q0)
            ax.text(q0, 0.97, f' QTD cesta {fmt_pct(qtd)}', color='#e3b341',
                    fontsize=9, va='top', fontweight='bold',
                    transform=ax.get_xaxis_transform())
        for sym, nm in AI_BENCH.items():
            if sym in p1:
                s = p1[sym].dropna()
                ax.plot(s.index, s / s.iloc[0] * 100, lw=1.2, label=nm,
                        color={'SPY': '#8b949e', 'SMH': '#38b2c4', 'IGV': '#bc8cff'}[sym])
        _ai_legend(ax, loc='upper left')
        ax.text(0.99, 0.03, 'Schwab: Bloomberg AI Value Chain EW -22% no tri (até 04/11)',
                transform=ax.transAxes, color=FAINT, fontsize=7.5, ha='right')

    # (b) memória
    ax = fig.add_subplot(gs[0, 1])
    _ai_ax(ax, 'Memória (proxy de preço DRAM/HBM) - 1 ano, base 100', 'base 100')
    if px is None or px.empty:
        _ai_vazio(ax)
    else:
        p1 = px[px.index >= corte]
        cols = ['#3fb950', '#e5534b', '#58a6ff', '#e3b341']
        for (sym, nm), c in zip(AI_MEMORIA.items(), cols):
            if sym in p1 and p1[sym].dropna().shape[0] > 10:
                s = p1[sym].dropna()
                ax.plot(s.index, s / s.iloc[0] * 100, lw=1.4, color=c,
                        label=f'{nm} ({fmt_pct((s.iloc[-1]/s.iloc[0]-1)*100)})')
        _ai_legend(ax, loc='upper left')

    # (c) produção de semicondutores
    ax = fig.add_subplot(gs[1, 0])
    _ai_ax(ax, 'Produção industrial EUA - semicondutores e componentes (a/a)', '%')
    s = _ai_fred('semi')
    if s is None or len(s) < 14:
        _ai_vazio(ax)
    else:
        y = (s.pct_change(12, fill_method=None) * 100).dropna()
        y = y[y.index >= pd.Timestamp.today() - pd.DateOffset(years=6)]
        ax.bar(y.index, y, width=25, color=[UP if v >= 0 else DOWN for v in y])
        ax.axhline(0, color=FAINT, lw=0.8)
        ax.text(0.01, 0.95, f'Último ({y.index[-1]:%m/%y}): {fmt_pct(y.iloc[-1])}',
                transform=ax.transAxes, color=TXT, fontsize=9, va='top')

    # (d) pressões de custo (boom inflacionário)
    ax = fig.add_subplot(gs[1, 1])
    _ai_ax(ax, 'Pressões de custo: eletricidade e juro real', 'CPI eletricidade a/a (%)')
    el, rr = _ai_fred('elet'), _ai_fred('real')
    corte5 = pd.Timestamp.today() - pd.DateOffset(years=5)
    if el is None and rr is None:
        _ai_vazio(ax)
    else:
        if el is not None:
            y = (el.pct_change(12, fill_method=None) * 100).dropna()
            y = y[y.index >= corte5]
            ax.plot(y.index, y, color='#e3b341', lw=1.6,
                    label=f'CPI eletricidade a/a ({fmt_pct(y.iloc[-1])})')
            ax.axhline(0, color=FAINT, lw=0.8)
        _ai_legend(ax, loc='upper left')
        if rr is not None:
            r = rr[rr.index >= corte5]
            ax2 = ax.twinx()
            ax2.plot(r.index, r, color='#58a6ff', lw=1.2,
                     label=f'Juro real 10a ({r.iloc[-1]:.2f}%)')
            ax2.tick_params(colors='#58a6ff', labelsize=8)
            ax2.set_ylabel('TIPS 10a (%)', color='#58a6ff', fontsize=9)
            for sp in ax2.spines.values():
                sp.set_color(BORDER)
            ax2.legend(loc='upper right', frameon=False, labelcolor=TXT, fontsize=8)
    plt.show()

# ============================================================
# AI 6: SIGNPOST 4 - MONETIZAÇÃO
# ============================================================
def draw_ai_monetizacao():
    fund = ai_fundamentos()
    rev, opi = _ai_painel(fund, 'rev_q'), _ai_painel(fund, 'opi_q')
    px = ai_precos()

    fig = plt.figure(figsize=(15, 10), facecolor=BG)
    _ai_header(fig, 'Signpost 4 - Payoff e monetização',
               'Margens subindo = retorno sobre o capital melhorando | '
               'liderança de adotantes (software) sobre habilitadores (semis) = '
               'sinal de boom de produtividade')
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.16,
                  left=0.05, right=0.97, top=0.86, bottom=0.06)

    # (a) margem operacional por empresa
    ax = fig.add_subplot(gs[0, :])
    _ai_ax(ax, 'Margem operacional trimestral dos hyperscalers', '%')
    if rev.empty or opi.empty:
        _ai_vazio(ax)
    else:
        mg = (opi / rev * 100).iloc[-8:]
        cmap = _cmap(len(AI_HYPERSCALERS))
        x = np.arange(len(mg))
        for nm in mg.columns:
            s = mg[nm]
            if s.notna().sum() < 2:
                continue
            c = cmap(list(AI_HYPERSCALERS.values()).index(nm))
            ax.plot(x, s.values, marker='o', ms=4, lw=1.5, color=c,
                    label=f'{nm} ({s.dropna().iloc[-1]:.0f}%)')
        ax.axhline(0, color=FAINT, lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([str(p) for p in mg.index], color=TXT)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False,
                  labelcolor=TXT, fontsize=8)

    # (b) produtividade
    ax = fig.add_subplot(gs[1, 0])
    _ai_ax(ax, 'Produtividade do trabalho EUA (output/hora, a/a)', '%')
    pr = _ai_fred('prod')
    if pr is None or len(pr) < 6:
        _ai_vazio(ax)
    else:
        y = (pr.pct_change(4, fill_method=None) * 100).dropna()
        y = y[y.index >= pd.Timestamp.today() - pd.DateOffset(years=12)]
        media = y.mean()
        ax.bar(y.index, y, width=70, color=[UP if v >= media else '#57606a' for v in y])
        ax.axhline(media, color=DIM, ls='--', lw=0.9)
        ax.text(y.index[0], media, f' média {media:.1f}%'.replace('.', ','),
                color=DIM, fontsize=8, va='bottom')
        ax.text(0.01, 0.95, f'Último ({y.index[-1]:%m/%y}): {fmt_pct(y.iloc[-1])}',
                transform=ax.transAxes, color=TXT, fontsize=9, va='top')

    # (c) software / semis
    ax = fig.add_subplot(gs[1, 1])
    _ai_ax(ax, 'Adotantes vs habilitadores: IGV / SMH (2 anos)', 'razão (base 100)')
    if px is None or not {'IGV', 'SMH'} <= set(px.columns):
        _ai_vazio(ax)
    else:
        rel = (px['IGV'] / px['SMH']).dropna()
        rel = rel / rel.iloc[0] * 100
        ax.plot(rel.index, rel, color='#bc8cff', lw=1.6)
        ax.plot(rel.index, rel.rolling(50).mean(), color=DIM, lw=1, ls='--', label='MM50')
        ax.text(0.01, 0.95, 'Subindo = software (adotantes) liderando',
                transform=ax.transAxes, color=TXT, fontsize=8.5, va='top')
        _ai_legend(ax, loc='lower left')
    plt.show()

# ============================================================
# AI 7: EXPOSIÇÃO REGIONAL
# ============================================================
def draw_ai_regioes():
    px = ai_precos()
    fig, ax = plt.subplots(figsize=(15, 5.8), facecolor=BG)
    fig.subplots_adjust(left=0.10, right=0.60, top=0.78, bottom=0.10)
    _ai_header(fig, 'Exposição regional ao ciclo de IA',
               'Retornos de ETFs em US$ (Yahoo Finance) | Schwab: EUA e Taiwan/Coreia '
               'têm maior peso em habilitadores de IA; Japão/Europa são casos específicos')
    _ai_ax(ax)
    if px is None or px.empty:
        _ai_vazio(ax); plt.show(); return
    last = px.index[-1]
    janelas = {'QTD': _inicio_tri(last),
               'YTD': pd.Timestamp(year=last.year, month=1, day=1),
               '1 ano': last - pd.DateOffset(years=1)}
    labs = [k for k, v in AI_REGIOES.items() if v[0] in px]
    dados = {j: [_ret_desde(px[AI_REGIOES[l][0]], dt) for l in labs]
             for j, dt in janelas.items()}
    y = np.arange(len(labs))[::-1]
    h = 0.26
    cores = {'QTD': '#e3b341', 'YTD': '#38b2c4', '1 ano': '#8b949e'}
    for i, (j, vals) in enumerate(dados.items()):
        ax.barh(y + h * (1 - i), vals, height=h, color=cores[j], label=j)
        for yi, v in zip(y + h * (1 - i), vals):
            if not np.isnan(v):
                ax.text(v + (0.6 if v >= 0 else -0.6), yi, fmt_pct(v), fontsize=7.5,
                        color=TXT, va='center', ha=('left' if v >= 0 else 'right'))
    ax.axvline(0, color=FAINT, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{l} ({AI_REGIOES[l][0]})' for l in labs], color=TXT, fontsize=9.5)
    ax.set_xlabel('Retorno (%)', color=DIM, fontsize=9)
    vals_all = [v for vv in dados.values() for v in vv if not np.isnan(v)]
    if vals_all:
        lo, hi = min(vals_all + [0]), max(vals_all + [0])
        pad = (hi - lo) * 0.18 or 5
        ax.set_xlim(lo - pad, hi + pad)
    _ai_legend(ax, loc='lower right')
    fig.text(0.63, 0.80, 'O que monitorar (Schwab)', color=TXT, fontsize=11,
             fontweight='bold', va='top')
    yy = 0.73
    for l in labs:
        fig.text(0.63, yy, l, color=TXT, fontsize=9, fontweight='bold', va='top')
        fig.text(0.73, yy, AI_REGIOES[l][1], color=DIM, fontsize=9, va='top')
        yy -= 0.085
    plt.show()

def ai_conteudo(prior=None, peso=None):
    print("AI Monitor: quadro de cenários...")
    draw_ai_quadro()
    print("AI Monitor: coletando sinais (fundamentos + FRED + preços)...")
    draw_ai_termometro(prior, peso)
    print("AI Monitor: capex...")
    draw_ai_capex()
    print("AI Monitor: financiamento...")
    draw_ai_financiamento()
    print("AI Monitor: adoção...")
    draw_ai_adocao()
    print("AI Monitor: monetização...")
    draw_ai_monetizacao()
    print("AI Monitor: regiões...")
    draw_ai_regioes()



# ############################################################
#             INTERFACE WEB (STREAMLIT)
# ############################################################
import io
import streamlit as st

st.set_page_config(page_title="Painel de Mercado - Bankshop", layout="wide",
                   page_icon="📈")

# ---- cache (evita rebaixar dados a cada clique/visitante) -------------------
_TTL = 3600  # segundos (1 hora)
_cd = lambda f: st.cache_data(ttl=_TTL, show_spinner=False)(f)
_fred_csv      = _cd(_fred_csv)
fetch          = _cd(fetch)
get_perf_stats = _cd(get_perf_stats)
get_pe_series  = _cd(get_pe_series)
_batch_closes  = _cd(_batch_closes)
_callput_proxy = _cd(_callput_proxy)

@st.cache_resource(ttl=_TTL)
def _ai_cache_persistente():
    return {}
_AI_CACHE = _ai_cache_persistente()   # cache do AI Monitor sobrevive aos reruns

# ---- cada plt.show() vira um gráfico na página -------------------------------
def _st_show(*args, **kwargs):
    fig = plt.gcf()
    try:
        st.pyplot(fig, width='stretch')            # Streamlit >= 1.46
    except TypeError:
        st.pyplot(fig, use_container_width=True)   # versões anteriores
    plt.close(fig)
plt.show = _st_show

# ---- barra lateral -------------------------------------------------------------
st.sidebar.title("Painel de Mercado")
secao = st.sidebar.radio("Seção", ["AI Monitor", "Macro", "Micro"])
st.sidebar.divider()

if secao == "Macro":
    anos = st.sidebar.select_slider("Janela da inflação (anos)", [2, 5, 10], value=5)

    def conteudo():
        draw_inflation(anos)
        draw_term_structures()
        draw_isee()

elif secao == "Micro":
    txt = st.sidebar.text_area(
        "Ativos (separados por vírgula; B3 com .SA)",
        "SPY, ^BVSP, ^VIX, AAPL, MSFT, NVDA, GOOGL, AMZN, META, MU, ASML, TSM")
    periodo = st.sidebar.select_slider(
        "Período do painel", list(PERIOD_TO_INTERVAL), value='1mo',
        format_func=lambda p: PERIOD_LABELS[p])
    ref = st.sidebar.date_input("Data de referência do P/E", date.today())

    def conteudo():
        tickers = parse_tickers(txt)
        if not tickers:
            st.warning("Digite ao menos um ativo.")
            return
        draw_panel(periodo, tickers)
        draw_returns_table(tickers)
        draw_pe_chart(tickers, ref.strftime("%Y-%m-%d"))
        draw_correlation_matrix(tickers)
        draw_prices_and_returns(tickers, period="5y")

else:  # AI Monitor
    st.sidebar.caption("Sua probabilidade-base de cada cenário (normalizada para 100%)")
    prior = {k: st.sidebar.slider(AI_CENARIOS[k][0], 0, 100, AI_PRIOR[k], 5)
             for k in AI_CENARIOS}
    peso = st.sidebar.slider("Peso dos sinais de mercado", 0.0, 1.0,
                             AI_PESO_SINAIS, 0.1)

    def conteudo():
        ai_conteudo(prior, peso)

st.sidebar.divider()
if st.sidebar.button("Recarregar dados"):
    st.cache_data.clear()
    _AI_CACHE.clear()
    st.rerun()

# ---- página --------------------------------------------------------------------
titulos = {
    "Macro": "Indicadores macro - inflação EUA, curva de juros, VIX e opções",
    "Micro": "Ativos - preços, rendimentos, P/E e correlação",
    "AI Monitor": "AI Monitor - cenários e signposts do ciclo de investimento em IA",
}
st.title(titulos[secao])
with st.spinner("Carregando dados..."):
    conteudo()

# ---- exportar PDF da seção atual ---------------------------------------------
if st.sidebar.button("Gerar PDF desta seção"):
    buf = io.BytesIO()
    pdf = PdfPages(buf)

    def _to_pdf(*a, **k):
        fig = plt.gcf()
        pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    plt.show = _to_pdf
    try:
        conteudo()
    finally:
        plt.show = _st_show
        pdf.close()
    st.sidebar.download_button("Baixar PDF", buf.getvalue(),
                               file_name=f"painel_{secao.lower().replace(' ', '_')}.pdf",
                               mime="application/pdf")

st.caption("Fontes: Yahoo Finance, FRED/BLS/ICE BofA, U.S. Treasury, Cboe; "
           "AI Monitor baseado em Schwab Asset Management, 'The AI Innovation Cycle: "
           "Scenarios and Signposts' (09/11/2026). Material informativo; não constitui "
           "recomendação de investimento.")
