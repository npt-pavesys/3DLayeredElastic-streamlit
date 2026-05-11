import sys
import os
import io
import contextlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from Main.MDA_Huang import Layer3D
from Main.Interactive_Functions import plot_interactive_heatmap

# ── Constantes ─────────────────────────────────────────────────────────────────
RESPONSE_KEYS = [
    "sigma_z", "sigma_x", "sigma_y", "sigma_xy", "sigma_yz", "sigma_xz",
    "eps_z", "eps_x", "eps_y", "eps_xy", "eps_yz", "eps_xz", "deflection_z",
]
RESPONSE_LABELS = {
    "sigma_z":       "σ_z  — Tensão vertical",
    "sigma_x":       "σ_x  — Tensão horizontal X",
    "sigma_y":       "σ_y  — Tensão horizontal Y",
    "sigma_xy":      "σ_xy — Tensão cisalhante XY",
    "sigma_yz":      "σ_yz — Tensão cisalhante YZ",
    "sigma_xz":      "σ_xz — Tensão cisalhante XZ",
    "eps_z":         "ε_z  — Deformação vertical",
    "eps_x":         "ε_x  — Deformação horizontal X",
    "eps_y":         "ε_y  — Deformação horizontal Y",
    "eps_xy":        "ε_xy — Deformação cisalhante XY",
    "eps_yz":        "ε_yz — Deformação cisalhante YZ",
    "eps_xz":        "ε_xz — Deformação cisalhante XZ",
    "deflection_z":  "w_z  — Deflexão vertical",
}
STRAIN_KEYS = {"eps_z", "eps_x", "eps_y", "eps_xy", "eps_yz", "eps_xz"}

# ── Configuração da página ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análise Elástica em Camadas 3D",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inicialização do session state ─────────────────────────────────────────────
for key in ("RS", "RS_params", "RS2", "RS2_params", "log"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Funções auxiliares ─────────────────────────────────────────────────────────

def _is_strain(resp: str) -> bool:
    return resp in STRAIN_KEYS


def _unit_label(resp: str, unit_sys: str) -> str:
    if resp == "deflection_z":
        return "polegada" if unit_sys == "Imperial" else "mm"
    if _is_strain(resp):
        return "με"
    return "psi" if unit_sys == "Imperial" else "MPa"


def _mult(resp: str) -> float:
    return 1e6 if _is_strain(resp) else 1.0


def run_analysis(E, H, nu, L, LPos, a, x, y, z, it, ZRO, tolerance, every):
    """Executa Layer3D e captura saída verbose. Retorna (RS, log_str)."""
    log_buf = io.StringIO()
    isBD = np.ones(len(E))
    with contextlib.redirect_stdout(log_buf):
        RS = Layer3D(L, LPos, a, x, y, z, H, E, nu, it, ZRO, isBD, tolerance,
                     verbose=True, every=every)
    return RS, log_buf.getvalue()


def params_tuple(E, H, nu, L, LPos, a, x, y, z, it, ZRO, tol, every):
    return (
        tuple(E), tuple(H), tuple(nu),
        tuple(L), tuple(tuple(p) for p in LPos), a,
        tuple(x), tuple(y), tuple(z),
        it, ZRO, tol, every,
    )


def validate_inputs(E, H, nu, L, LPos) -> list[str]:
    errors = []
    n_layers = len(H) + 1
    if len(E) != n_layers:
        errors.append(f"E deve ter {n_layers} valores (um por camada incluindo subleito), encontrado {len(E)}.")
    if len(nu) != n_layers:
        errors.append(f"ν deve ter {n_layers} valores, encontrado {len(nu)}.")
    if len(L) != len(LPos):
        errors.append(f"Número de cargas ({len(L)}) deve ser igual ao número de posições ({len(LPos)}).")
    if any(v <= 0 for v in E):
        errors.append("Todos os módulos E devem ser positivos.")
    if any(v <= 0 for v in H):
        errors.append("Todas as espessuras H devem ser positivas.")
    return errors


# ── Barra lateral ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛣️ 3D Layered Elastic")
    st.caption("Análise de pavimentos flexíveis")

    unit_sys = st.radio("Sistema de unidades", ["Imperial (psi / pol / lbs)", "SI (MPa / mm / kN)"],
                        index=0, horizontal=True)
    unit_sys = "Imperial" if unit_sys.startswith("Imperial") else "SI"

    E_unit = "psi" if unit_sys == "Imperial" else "MPa"
    H_unit = "pol" if unit_sys == "Imperial" else "mm"
    L_unit = "lbs" if unit_sys == "Imperial" else "kN"

    st.divider()

    # ── Camadas ───────────────────────────────────────────────────────────────
    st.subheader("Camadas do Pavimento")
    n_bound = st.number_input("Número de camadas ligadas (excl. subleito)", 1, 6, 2, 1)
    n_layers = n_bound + 1  # +1 para subleito

    st.markdown("**Propriedades das camadas** — topo a base (última linha = subleito, sem espessura)")
    E_vals, H_vals, nu_vals = [], [], []

    if unit_sys == "Imperial":
        E_defaults = [500_000, 50_000, 10_000]
        H_defaults = [6, 18]
        nu_defaults = [0.35, 0.40, 0.45]
    else:
        E_defaults = [3300, 2000, 70]
        H_defaults = [150, 375]
        nu_defaults = [0.35, 0.35, 0.35]

    for i in range(n_layers):
        label = f"Camada {i+1}" if i < n_layers - 1 else "Subleito"
        cols = st.columns([2, 2, 2] if i < n_layers - 1 else [2, 1, 2])
        E_def = E_defaults[i] if i < len(E_defaults) else E_defaults[-1]
        nu_def = nu_defaults[i] if i < len(nu_defaults) else 0.45

        with cols[0]:
            E_vals.append(st.number_input(f"E_{i+1} ({E_unit})", 1, 10_000_000,
                                          int(E_def), 1000, key=f"E_{i}",
                                          help=f"{label} — módulo de elasticidade"))
        with cols[1]:
            if i < n_layers - 1:
                H_def = H_defaults[i] if i < len(H_defaults) else 10
                H_vals.append(st.number_input(f"H_{i+1} ({H_unit})", 1, 10_000,
                                              int(H_def), 1, key=f"H_{i}",
                                              help=f"{label} — espessura"))
            else:
                st.write("")  # subleito — sem H
        with cols[2]:
            nu_vals.append(st.number_input(f"ν_{i+1}", 0.01, 0.499,
                                           float(nu_def), 0.01, key=f"nu_{i}",
                                           help=f"{label} — coeficiente de Poisson",
                                           format="%.3f"))

    st.divider()

    # ── Cargas ────────────────────────────────────────────────────────────────
    st.subheader("Cargas")
    n_loads = st.number_input("Número de cargas", 1, 10, 2, 1)

    if unit_sys == "Imperial":
        L_def, Lx_def, Ly_def, a_def = 9000, 10, 0, 4
    else:
        L_def, Lx_def, Ly_def, a_def = 21676, 0, 0, 100

    L_vals, LPos_vals = [], []
    for i in range(n_loads):
        st.markdown(f"**Carga {i+1}**")
        c1, c2, c3 = st.columns(3)
        offset_x = i * (11 if unit_sys == "Imperial" else 288)
        with c1:
            L_vals.append(st.number_input(f"L_{i+1} ({L_unit})", 1, 10_000_000,
                                          int(L_def), 100, key=f"L_{i}"))
        with c2:
            px_ = st.number_input(f"x_{i+1} ({H_unit})", -10_000, 10_000,
                                  int(Lx_def + offset_x), 1, key=f"Lx_{i}")
        with c3:
            py_ = st.number_input(f"y_{i+1} ({H_unit})", -10_000, 10_000,
                                  int(Ly_def), 1, key=f"Ly_{i}")
        LPos_vals.append((px_, py_))

    a_val = st.number_input(f"Raio de contato a ({H_unit})", 1, 1000,
                            int(a_def), 1)

    st.divider()

    # ── Grade de consulta ─────────────────────────────────────────────────────
    st.subheader("Grade de Consulta")

    if unit_sys == "Imperial":
        gx0, gx1, gxs = 0, 30, 1
        gy_raw = "0"
        gz0, gz1, gzs = 0, 30, 1
    else:
        gx0, gx1, gxs = -200, 200, 10
        gy_raw = "0"
        gz0, gz1, gzs = 0, 500, 10

    c1, c2, c3 = st.columns(3)
    with c1:
        x_min = st.number_input(f"x mín ({H_unit})", value=int(gx0), key="xmin")
        x_max = st.number_input(f"x máx ({H_unit})", value=int(gx1), key="xmax")
        x_step = st.number_input(f"passo x", value=int(gxs), min_value=1, key="xstep")
    with c2:
        y_input = st.text_input(f"valores de y ({H_unit}), separados por vírgula", value=gy_raw)
    with c3:
        z_min = st.number_input(f"z mín ({H_unit})", value=int(gz0), key="zmin")
        z_max = st.number_input(f"z máx ({H_unit})", value=int(gz1), key="zmax")
        z_step = st.number_input(f"passo z", value=int(gzs), min_value=1, key="zstep")

    st.divider()

    # ── Configurações avançadas ───────────────────────────────────────────────
    with st.expander("Configurações avançadas do solver"):
        it_val = st.number_input("Iterações máximas", 100, 10_000, 1600, 100)
        every_val = st.number_input("Verificar convergência a cada N passos", 10, 500, 100, 10)
        tol_val = st.number_input("Tolerância (%)", 0.001, 10.0, 0.05, 0.005, format="%.4f")
        ZRO_val = st.number_input("ZRO (zero numérico)", value=7e-20, format="%.2e")

    st.divider()

    run_btn = st.button("▶  Executar Análise", type="primary", use_container_width=True)

# ── Processar valores de y ─────────────────────────────────────────────────────
try:
    y_vals = [float(v.strip()) for v in y_input.split(",") if v.strip()]
    if not y_vals:
        y_vals = [0.0]
except Exception:
    y_vals = [0.0]

x_arr = np.arange(x_min, x_max + x_step, x_step, dtype=float)
z_arr = np.arange(z_min, z_max + z_step, z_step, dtype=float)

# ── Executar análise ao clicar ─────────────────────────────────────────────────
if run_btn:
    errors = validate_inputs(E_vals, H_vals, nu_vals, L_vals, LPos_vals)
    if errors:
        for e in errors:
            st.error(e)
    else:
        p_key = params_tuple(E_vals, H_vals, nu_vals, L_vals, LPos_vals,
                             a_val, x_arr, y_vals, z_arr, it_val, ZRO_val, tol_val, every_val)
        if st.session_state.RS_params != p_key:
            with st.spinner("Executando análise elástica em camadas... isso pode levar um momento."):
                RS, log = run_analysis(
                    np.array(E_vals, dtype=float),
                    H_vals,
                    nu_vals,
                    L_vals,
                    LPos_vals,
                    a_val,
                    x_arr, y_vals, z_arr,
                    it_val, ZRO_val, tol_val / 100, every_val,
                )
            st.session_state.RS = RS
            st.session_state.RS_params = p_key
            st.session_state.log = log
            st.success("Análise concluída!")
        else:
            st.info("Parâmetros inalterados — usando resultados em cache.")

# ── Área principal ─────────────────────────────────────────────────────────────
st.title("Análise Elástica em Camadas 3D para Pavimentos")
st.caption("Tensões, deformações e deflexões em sistemas de pavimentos flexíveis sob cargas circulares.")

RS = st.session_state.RS

if RS is None:
    st.info("Configure os parâmetros na barra lateral e clique em **▶ Executar Análise** para começar.")
else:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Tabela de Resultados",
        "🌡️ Mapa de Calor",
        "📉 Perfil de Profundidade",
        "📈 Perfil Horizontal",
        "🔀 Comparar Estruturas",
        "📄 Log do Solver",
    ])

    # ─── Aba 1: Tabela de Resultados ──────────────────────────────────────────
    with tab1:
        st.header("Resultados — Pontos Selecionados")
        st.markdown("Todas as deformações estão em microdeformação (με). Tensões e deflexões usam o sistema de unidades selecionado.")

        rows = []
        for xi, xv in enumerate(x_arr):
            for yi, yv in enumerate(y_vals):
                for zi, zv in enumerate(z_arr):
                    row = {"x": xv, "y": yv, "z": zv}
                    for k in RESPONSE_KEYS:
                        mult = 1e6 if _is_strain(k) else 1.0
                        row[k] = RS[k][yi, xi, zi] * mult
                    rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=400)

        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇ Baixar CSV", csv, "3DLEA_resultados.csv", "text/csv",
                               use_container_width=True)
        with col2:
            buf = io.BytesIO()
            try:
                df.to_excel(buf, index=False, engine="openpyxl")
                st.download_button("⬇ Baixar Excel", buf.getvalue(),
                                   "3DLEA_resultados.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            except Exception:
                st.caption("Instale o openpyxl para exportação em Excel.")

    # ─── Aba 2: Mapa de Calor ─────────────────────────────────────────────────
    with tab2:
        st.header("Mapa de Calor (plano x–z)")
        col1, col2 = st.columns([2, 1])
        with col1:
            resp_hm = st.selectbox("Resposta a plotar", RESPONSE_KEYS,
                                   format_func=lambda k: RESPONSE_LABELS[k], key="hm_resp")
        with col2:
            y_idx_hm = st.selectbox("Índice de fatia y", range(len(y_vals)),
                                    format_func=lambda i: f"y = {y_vals[i]}", key="hm_y")
            interpolate_hm = st.checkbox("Interpolar", value=True, key="hm_interp")

        data = RS[resp_hm][y_idx_hm, :, :] * _mult(resp_hm)
        label = _unit_label(resp_hm, unit_sys)

        if len(x_arr) > 1 and len(z_arr) > 1:
            fig_hm = plot_interactive_heatmap(
                RESPONSE_LABELS[resp_hm], data, label, x_arr, z_arr, H_vals,
                aspect=(16, 8), interpolate=interpolate_hm,
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.warning("São necessários pelo menos 2 pontos em x e 2 em z para renderizar o mapa de calor.")

    # ─── Aba 3: Perfil de Profundidade ────────────────────────────────────────
    with tab3:
        st.header("Resposta vs Profundidade para dado x")
        col1, col2 = st.columns([2, 1])
        with col1:
            resp_d = st.selectbox("Resposta", RESPONSE_KEYS,
                                  format_func=lambda k: RESPONSE_LABELS[k], key="dp_resp")
        with col2:
            y_idx_d = st.selectbox("Fatia y", range(len(y_vals)),
                                   format_func=lambda i: f"y = {y_vals[i]}", key="dp_y")

        default_x_idx = int(np.argmin(np.abs(x_arr - LPos_vals[0][0]))) if len(x_arr) > 0 else 0
        x_idx_d = st.slider("Índice de x", 0, len(x_arr) - 1, default_x_idx,
                            format=f"x = %d {H_unit}", key="dp_x_slider")
        x_val_d = x_arr[x_idx_d]
        st.caption(f"Plotando em x = {x_val_d} {H_unit}")

        A = np.transpose(RS[resp_d][y_idx_d, :, :]) * _mult(resp_d)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=A[:, x_idx_d], y=-z_arr,
            mode="lines+markers", marker=dict(size=6),
            name=RESPONSE_LABELS[resp_d],
        ))
        cumH = np.cumsum(H_vals)
        for h in cumH:
            if h <= z_arr.max():
                fig.add_hline(y=-h, line_dash="dash", line_color="black",
                              annotation_text=f"Limite de camada @ {h} {H_unit}")
        fig.update_layout(
            xaxis_title=f"{RESPONSE_LABELS[resp_d]} ({_unit_label(resp_d, unit_sys)})",
            yaxis_title=f"Profundidade ({H_unit})",
            height=500, template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ─── Aba 4: Perfil Horizontal ─────────────────────────────────────────────
    with tab4:
        st.header("Resposta vs x para dada profundidade")
        col1, col2 = st.columns([2, 1])
        with col1:
            resp_h = st.selectbox("Resposta", RESPONSE_KEYS,
                                  format_func=lambda k: RESPONSE_LABELS[k], key="hp_resp")
        with col2:
            y_idx_h = st.selectbox("Fatia y", range(len(y_vals)),
                                   format_func=lambda i: f"y = {y_vals[i]}", key="hp_y")

        z_idx_h = st.slider("Índice de z (profundidade)", 0, len(z_arr) - 1, min(3, len(z_arr) - 1),
                            format=f"z = %d {H_unit}", key="hp_z_slider")
        z_val_h = z_arr[z_idx_h]
        st.caption(f"Plotando na profundidade z = {z_val_h} {H_unit}")

        A = np.transpose(RS[resp_h][y_idx_h, :, :]) * _mult(resp_h)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x_arr, y=A[z_idx_h, :],
            mode="lines+markers", marker=dict(size=6),
            name=RESPONSE_LABELS[resp_h],
        ))
        for lp in LPos_vals:
            fig2.add_vline(x=lp[0], line_dash="dot", line_color="red",
                           annotation_text=f"Carga @ x={lp[0]}")
        fig2.update_layout(
            xaxis_title=f"x ({H_unit})",
            yaxis_title=f"{RESPONSE_LABELS[resp_h]} ({_unit_label(resp_h, unit_sys)})",
            height=500, template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ─── Aba 5: Comparar Estruturas ───────────────────────────────────────────
    with tab5:
        st.header("Comparar Duas Estruturas")
        st.markdown(
            "Defina uma segunda estrutura (modificada) abaixo, execute-a e compare as respostas lado a lado. "
            "A estrutura **base** é a que você executou na barra lateral."
        )

        with st.expander("Parâmetros da segunda estrutura", expanded=True):
            n_layers2 = n_layers
            E2_vals, H2_vals, nu2_vals = [], [], []
            for i in range(n_layers2):
                c1c, c2c, c3c = st.columns([2, 2, 2] if i < n_layers2 - 1 else [2, 1, 2])
                E_def2 = E_vals[i] if i < len(E_vals) else 10_000
                nu_def2 = nu_vals[i] if i < len(nu_vals) else 0.45
                with c1c:
                    E2_vals.append(st.number_input(f"E2_{i+1} ({E_unit})", 1, 10_000_000,
                                                   int(E_def2), 1000, key=f"E2_{i}"))
                with c2c:
                    if i < n_layers2 - 1:
                        H_def2 = H_vals[i] if i < len(H_vals) else 6
                        H2_vals.append(st.number_input(f"H2_{i+1} ({H_unit})", 1, 10_000,
                                                       int(H_def2), 1, key=f"H2_{i}"))
                    else:
                        st.write("")
                with c3c:
                    nu2_vals.append(st.number_input(f"ν2_{i+1}", 0.01, 0.499,
                                                    float(nu_def2), 0.01, key=f"nu2_{i}",
                                                    format="%.3f"))

            run_btn2 = st.button("▶  Executar Segunda Estrutura", key="run2", type="secondary")

        if run_btn2:
            errors2 = validate_inputs(E2_vals, H2_vals, nu2_vals, L_vals, LPos_vals)
            if errors2:
                for e in errors2:
                    st.error(e)
            else:
                p_key2 = params_tuple(E2_vals, H2_vals, nu2_vals, L_vals, LPos_vals,
                                      a_val, x_arr, y_vals, z_arr, it_val, ZRO_val, tol_val, every_val)
                if st.session_state.RS2_params != p_key2:
                    with st.spinner("Executando segunda análise..."):
                        RS2, log2 = run_analysis(
                            np.array(E2_vals, dtype=float), H2_vals, nu2_vals,
                            L_vals, LPos_vals, a_val,
                            x_arr, y_vals, z_arr,
                            it_val, ZRO_val, tol_val / 100, every_val,
                        )
                    st.session_state.RS2 = RS2
                    st.session_state.RS2_params = p_key2
                    st.success("Segunda análise concluída!")
                else:
                    st.info("Segunda estrutura inalterada — usando resultados em cache.")

        if st.session_state.RS2 is not None:
            RS2 = st.session_state.RS2
            st.divider()

            col1, col2 = st.columns([2, 1])
            with col1:
                resp_cmp = st.selectbox("Resposta a comparar", RESPONSE_KEYS,
                                        format_func=lambda k: RESPONSE_LABELS[k], key="cmp_resp")
            with col2:
                y_idx_cmp = st.selectbox("Fatia y", range(len(y_vals)),
                                         format_func=lambda i: f"y = {y_vals[i]}", key="cmp_y")

            default_x_cmp = int(np.argmin(np.abs(x_arr - LPos_vals[0][0])))
            x_idx_cmp = st.slider("Índice de x (para perfil de profundidade)", 0, len(x_arr) - 1,
                                  default_x_cmp, key="cmp_x_slider")

            A1 = np.transpose(RS[resp_cmp][y_idx_cmp, :, :]) * _mult(resp_cmp)
            A2 = np.transpose(RS2[resp_cmp][y_idx_cmp, :, :]) * _mult(resp_cmp)

            col_a, col_b = st.columns(2)
            for col, Ai, lbl in [(col_a, A1, "Base"), (col_b, A2, "Modificada")]:
                with col:
                    st.subheader(lbl)
                    fig_c = go.Figure()
                    fig_c.add_trace(go.Scatter(
                        x=Ai[:, x_idx_cmp], y=-z_arr,
                        mode="lines+markers", marker=dict(size=6),
                        name=lbl,
                    ))
                    Hv = H_vals if lbl == "Base" else H2_vals
                    for h in np.cumsum(Hv):
                        if h <= z_arr.max():
                            fig_c.add_hline(y=-h, line_dash="dash", line_color="black")
                    fig_c.update_layout(
                        xaxis_title=f"{RESPONSE_LABELS[resp_cmp]} ({_unit_label(resp_cmp, unit_sys)})",
                        yaxis_title=f"Profundidade ({H_unit})",
                        height=450, template="plotly_white",
                        title=f"{lbl} — x={x_arr[x_idx_cmp]:.1f}",
                    )
                    st.plotly_chart(fig_c, use_container_width=True)

            st.subheader("Diferença (Modificada − Base) por profundidade")
            diff = A2[:, x_idx_cmp] - A1[:, x_idx_cmp]
            fig_diff = go.Figure()
            fig_diff.add_trace(go.Scatter(x=diff, y=-z_arr, mode="lines+markers",
                                          line=dict(color="purple"), marker=dict(size=6)))
            fig_diff.add_vline(x=0, line_dash="solid", line_color="gray")
            fig_diff.update_layout(
                xaxis_title=f"Δ {RESPONSE_LABELS[resp_cmp]} ({_unit_label(resp_cmp, unit_sys)})",
                yaxis_title=f"Profundidade ({H_unit})",
                height=400, template="plotly_white",
            )
            st.plotly_chart(fig_diff, use_container_width=True)
        else:
            st.info("Execute a segunda estrutura acima para habilitar a comparação.")

    # ─── Aba 6: Log do Solver ─────────────────────────────────────────────────
    with tab6:
        st.header("Log do Solver")
        log = st.session_state.log
        if log:
            st.code(log, language=None)
        else:
            st.info("Sem log ainda. Execute a análise primeiro.")
