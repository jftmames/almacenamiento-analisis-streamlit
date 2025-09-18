import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Análisis de Almacenamiento", layout="wide")

# =========================
# Utilidades
# =========================
DATA_PATH = Path("data") / "comparativa.csv"

@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Limpieza ligera: nube tiene MTBF=0 y Consumo_W=0 por ser "N/A" en concepto on-prem
    # Para el radar, convertiremos MTBF a escala 1–5 con min-max simple (tratando 0 como mínimo).
    return df

def min_max_scale(series: pd.Series, out_min=1.0, out_max=5.0) -> pd.Series:
    s = series.astype(float)
    s_min, s_max = float(s.min()), float(s.max())
    if s_max == s_min:
        return pd.Series([ (out_min + out_max)/2 ] * len(s), index=s.index)
    return out_min + (s - s_min) * (out_max - out_min) / (s_max - s_min)

def hours_to_read(volume_tb: float, read_mb_s: float) -> float:
    """
    Estimación didáctica del tiempo (horas) para leer 'volume_tb' una vez,
    asumiendo transferencia secuencial.
    1 TB ≈ 1e6 MB (aprox. doc.).
    """
    if read_mb_s <= 0:
        return float("inf")
    mb_total = volume_tb * 1_000_000.0
    seconds = mb_total / read_mb_s
    return seconds / 3600.0

# =========================
# Carga de datos
# =========================
st.title("Análisis Técnico de Soluciones de Almacenamiento")
st.caption("Proyecto educativo 100% web (GitHub + Streamlit Cloud). Algunos valores están marcados como [Unverified] por ser estimaciones didácticas.")

if not DATA_PATH.exists():
    st.error("No se encontró `data/comparativa.csv`. Verifica que el archivo exista en el repositorio.")
    st.stop()

df = load_data(DATA_PATH)

# =========================
# Portada / contexto
# =========================
st.header("Contexto y objetivos")
st.write(
    """
    Esta aplicación compara **HDD, SSD, Cinta y Nube** en métricas clave y simula el impacto en tiempos de lectura
    bajo crecimiento de datos. Es un **material didáctico**: los valores con **[Unverified]** son aproximaciones para enseñar
    conceptos de **coste, rendimiento y escalabilidad**.
    """
)

# =========================
# Tabla comparativa
# =========================
st.header("1) Tabla comparativa")
st.dataframe(df, use_container_width=True)

with st.expander("Notas sobre las columnas"):
    st.markdown(
        "- **Lectura/Escritura (MB/s):** transferencia secuencial aproximada.\n"
        "- **Costo USD/GB:** coste de almacenamiento puro; en nube **no** incluye transferencias/solicitudes.\n"
        "- **MTBF horas:** indicador de fiabilidad. Para **Nube** no aplica del mismo modo que hardware on-prem.\n"
        "- **Seguridad/Escalabilidad (1–5):** evaluación cualitativa para el radar.\n"
        "- **[Unverified]:** cifras educativas sin verificación externa."
    )

# =========================
# 2) Gráficos (matplotlib)
# =========================
st.header("2) Gráficos comparativos (matplotlib)")

# --- Barras: Lectura ---
st.subheader("a) Velocidad de lectura (MB/s) — Barras")
fig1, ax1 = plt.subplots()
ax1.bar(df["Tecnologia"], df["Lectura_MB_s"])
ax1.set_title("Velocidad de lectura")
ax1.set_xlabel("Tecnología")
ax1.set_ylabel("MB/s")
st.pyplot(fig1, clear_figure=True)

# --- Barras: Escritura ---
st.subheader("b) Velocidad de escritura (MB/s) — Barras")
fig2, ax2 = plt.subplots()
ax2.bar(df["Tecnologia"], df["Escritura_MB_s"])
ax2.set_title("Velocidad de escritura")
ax2.set_xlabel("Tecnología")
ax2.set_ylabel("MB/s")
st.pyplot(fig2, clear_figure=True)

# --- Barras: Costo/GB y coste anual estimado (100 TB) ---
st.subheader("c) Coste por GB y coste anual estimado — Barras")
col_c1, col_c2 = st.columns(2)
with col_c1:
    fig3, ax3 = plt.subplots()
    ax3.bar(df["Tecnologia"], df["Costo_USD_GB"])
    ax3.set_title("Costo por GB (USD)")
    ax3.set_xlabel("Tecnología")
    ax3.set_ylabel("USD/GB")
    st.pyplot(fig3, clear_figure=True)

with col_c2:
    volumen_base_tb = 100.0
    # coste anual estimado simple: volumen * 1e6 * costo/GB (para nube, esto ignora requests/egress)
    coste_anual = df["Costo_USD_GB"].astype(float) * volumen_base_tb * 1_000_000.0
    fig4, ax4 = plt.subplots()
    ax4.bar(df["Tecnologia"], coste_anual)
    ax4.set_title(f"Coste anual estimado (para {int(volumen_base_tb)} TB)")
    ax4.set_xlabel("Tecnología")
    ax4.set_ylabel("USD/año (aprox.)")
    st.pyplot(fig4, clear_figure=True)

# --- Radar: Fiabilidad, Seguridad, Escalabilidad ---
st.subheader("d) Radar: comparación cualitativa (1–5)")
st.markdown(
    "La **Fiabilidad** se deriva de MTBF con una normalización min-max a escala **1–5**. "
    "Seguridad y Escalabilidad ya vienen en 1–5 desde el CSV."
)

# Preparar datos radar
techs = df["Tecnologia"].tolist()
fiabilidad_scaled = min_max_scale(df["MTBF_horas"])
seguridad = df["Seguridad_1_5"].astype(float)
escalabilidad = df["Escalabilidad_1_5"].astype(float)

# Radar por tecnología (una sola figura con varias líneas)
labels = ["Fiabilidad", "Seguridad", "Escalabilidad"]
n_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
angles += angles[:1]  # cerrar el polígono

fig5 = plt.figure()
ax5 = plt.subplot(111, polar=True)

for i, tech in enumerate(techs):
    values = [fiabilidad_scaled.iloc[i], seguridad.iloc[i], escalabilidad.iloc[i]]
    values += values[:1]
    ax5.plot(angles, values, linewidth=1)
    ax5.fill(angles, values, alpha=0.1)
# ejes
ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(labels)
ax5.set_yticklabels([])
ax5.set_title("Radar (1–5): Fiabilidad (MTBF normalizado), Seguridad, Escalabilidad")

st.pyplot(fig5, clear_figure=True)

# =========================
# 3) Simulación
# =========================
st.header("3) Simulación: crecimiento de volumen y tiempo aproximado de lectura")
with st.sidebar:
    st.markdown("### Parámetros de simulación")
    vol_ini = st.number_input("Volumen inicial (TB)", min_value=1.0, value=100.0, step=1.0)
    crecimiento = st.number_input("Crecimiento anual (%)", min_value=0.0, value=20.0, step=1.0)
    anios = st.number_input("Horizonte (años)", min_value=1, max_value=15, value=5, step=1)

# construir tabla de simulación
rows = []
for year in range(1, anios + 1):
    vol_tb = vol_ini * ((1 + (crecimiento / 100.0)) ** (year - 1))
    row = {"Año": year, "Volumen_TB": round(vol_tb, 2)}
    for _, r in df.iterrows():
        tech = r["Tecnologia"]
        read = float(r["Lectura_MB_s"])
        row[f"Tiempo_{tech}_h"] = round(hours_to_read(vol_tb, read), 2)
    rows.append(row)
sim_df = pd.DataFrame(rows)

st.dataframe(sim_df, use_container_width=True)

st.markdown(
    """
**Interpretación (ejemplo didáctico):** si el volumen crece y la **velocidad de lectura** no acompaña,
los **tiempos** aumentan rápidamente. En cargas de consulta intensiva, **SSD** y (según red) **Nube**
suelen acercarse más a objetivos de **SLA**, mientras **HDD** y **Cinta** tienden a quedar para capas
de menor criticidad o archivado.
"""
)

with st.expander("Limitaciones del modelo (importante)"):
    st.markdown(
        "- Aproximación *secuencial* (no considera IOPS, colas, concurrencia, cachés ni compresión).\n"
        "- Para **Nube**, el throughput real depende de la **red** y las **APIs** del servicio.\n"
        "- Los valores marcados **[Unverified]** son docentes y no deben tomarse como benchmarking real."
    )

# =========================
# 4) Arquitectura propuesta (texto)
# =========================
st.header("4) Arquitectura propuesta (híbrida)")
st.markdown(
    """
**Patrón:** SSD on-prem para transaccional; Nube para backups/datos fríos/escala analítica; Cinta para archivo a largo plazo.

**Flujo de datos:** Ingesta → Procesamiento en SSD → Replicación en Nube → Archivado en Cinta.

**Redundancias:** RAID en SSD; replicación multi-región (cloud); dos juegos de cintas en ubicaciones separadas.

**RPO/RTO (referenciales):** RPO ≤ 15 min; RTO ≤ 2 h (*[Unverified]*).

**Seguridad:** Cifrado en reposo (p. ej., AES-256) y en tránsito (TLS), IAM de mínimo privilegio, MFA, logging/auditoría.

**Cumplimiento:** Control de residencia de datos (p. ej., región UE) y trazabilidad de accesos (GDPR).
"""
)

# =========================
# 5) Riesgos y mitigaciones
# =========================
st.header("5) Riesgos, mitigaciones y oportunidades")
col_r1, col_r2, col_r3 = st.columns(3)
with col_r1:
    st.subheader("Riesgos")
    st.markdown(
        """
- CAPEX alto en SSD.
- Desgaste en escrituras intensivas (SSD).
- Dependencia de conectividad (Nube).
- Riesgos regulatorios (residencia de datos).
        """
    )
with col_r2:
    st.subheader("Mitigaciones")
    st.markdown(
        """
- SSD como caché + backend económico.
- Políticas de rotación/borrado y monitoreo TBW.
- Doble ISP / enlaces dedicados.
- Selección de región UE + auditoría de accesos.
        """
    )
with col_r3:
    st.subheader("Oportunidades")
    st.markdown(
        """
- Elasticidad y pago por uso (Nube).
- Mejora de SLA con SSD.
- Reducción de OPEX en archivado (Cinta).
- Continuidad de negocio multi-site.
        """
    )

# =========================
# 6) Conclusión y próximos pasos
# =========================
st.header("6) Conclusión y próximos pasos")
st.markdown(
    """
**Conclusión:** Un enfoque **híbrido** equilibra rendimiento (SSD), elasticidad/continuidad (Nube) y coste de archivo (Cinta).  
Valores **[Unverified]**: utilízalos como guía didáctica, no como especificación de compra.

**Próximos pasos sugeridos:**
1) PoC con ~5 TB y cargas representativas.  
2) Medir throughput/latencia real y **validar costes** con proveedores.  
3) Definir políticas de archivado/retención y **RPO/RTO** medibles.  
4) Establecer métricas de éxito: coste efectivo por TB, latencia p95, tasa de fallos, tiempo de restauración.
"""
)
