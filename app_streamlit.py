# app_streamlit.py
import io, time, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import BeautifyIcon
from streamlit_folium import st_folium
import vrp_routes as core

# ── App config ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VRP Route Balancer", layout="wide")
st.title("VRP Route Balancer")
st.caption("FREE setup: solve with haversine; optionally refine per-leg with LOCAL OSRM (no keys).")

# ── Persist results across reruns ─────────────────────────────────────────────
for k in ["last_routes_csv", "last_routes_xlsx", "last_summary", "last_routes_df"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── File + delimiter ──────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload CSV", type=["csv"])
delim = st.selectbox(
    "CSV delimiter",
    ["Auto", ", (comma)", "; (semicolon)", "\\t (tab)", "| (pipe)"],
    index=1
)
delim_map = {"Auto": None, ", (comma)": ",", "; (semicolon)": ";", "\\t (tab)": "\t", "| (pipe)": "|"}
chosen_sep = delim_map[delim]

# ── Top controls ──────────────────────────────────────────────────────────────
col_top = st.columns(4)
with col_top[0]:
    vehicles = st.number_input("Number of routes (vehicles)", min_value=1, value=2, step=1)
with col_top[1]:
    equality_basis = st.selectbox("Balance by", ["time","distance"], index=0)
with col_top[2]:
    per_stop_min = st.number_input("Default service time per stop (min)", min_value=0.0, value=2.0, step=0.5)
with col_top[3]:
    avg_speed = st.number_input("Avg speed (km/h) for haversine", min_value=1.0, value=35.0, step=1.0)

with st.expander("OR-Tools (advanced)"):
    solver = st.selectbox("Solver", ["ortools","cluster2opt"], index=0)
    first_solution = st.text_input("First solution strategy", value="PATH_CHEAPEST_ARC")
    metaheuristic = st.text_input("Local search metaheuristic", value="GUIDED_LOCAL_SEARCH")
    time_limit_s = st.number_input("Time limit (s)", min_value=1, value=20, step=1)
    cap_enabled = st.checkbox("Enable per-route minute cap (time)", value=False)
    max_route_min = st.number_input("Max route minutes", min_value=0.0, value=0.0, step=5.0)
    st.caption("Tip: For big instances, try PARALLEL_CHEAPEST_INSERTION and 120–300s.")

with st.expander("Local OSRM refinement (optional, free)"):
    use_osrm = st.checkbox("Refine with local OSRM per leg", value=False)
    osrm_base = st.text_input("Local OSRM URL", value="http://localhost:5000")
    backend_costs = st.selectbox(
        "Solver cost matrix",
        ["Haversine (fast, approximate)", "OSRM (road-accurate, local)"],
        index=0
    )
    osrm_block = st.number_input(
        "OSRM /table block size (tile)",
        min_value=50, max_value=200, value=100, step=10,
        help="Bigger = fewer requests but longer URLs. 100 is a good default."
    )
    if use_osrm:
        st.info("Refines leg km/min after solving via your local OSRM. One request per leg.")

with st.expander("Depot & filters (optional)"):
    use_first_row_as_depot = st.checkbox("Treat first row in CSV as depot (service=0)", value=False)
    use_manual_depot = st.checkbox("Add manual depot (prepend)", value=False)
    depot_lat = st.number_input("Depot latitude", value=0.0, step=0.0001, format="%.6f", disabled=not use_manual_depot)
    depot_lon = st.number_input("Depot longitude", value=0.0, step=0.0001, format="%.6f", disabled=not use_manual_depot)

    filter_active = st.checkbox("Filter only active rows", value=False)
    active_col = st.text_input("Active column name (case-insensitive)", value="Status active")
    active_values = st.text_input("Active values (comma-separated, case-insensitive)", value="yes, active, true, 1")

# ── Helpers ──────────────────────────────────────────────────────────────────
def robust_read(uploaded_file, default_sep=None):
    raw = uploaded_file.getvalue()
    seps = [default_sep] if default_sep else [",",";","\t","|"]
    encs = ["utf-8-sig","utf-8","cp1252","latin1","iso-8859-1","utf-16","utf-16le","utf-16be"]
    for sep in seps:
        for enc in encs:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep, engine="python")
            except Exception:
                continue
    return pd.read_csv(io.BytesIO(raw), encoding="latin1", sep=",", engine="python", encoding_errors="replace")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # clean headers
    cleaned = {}
    for c in df.columns:
        cc = str(c).replace("\ufeff","").strip()
        cc = " ".join(cc.split())
        cleaned[c] = cc
    df = df.rename(columns=cleaned)
    # normalized lookup: lower, remove spaces/underscores/slashes
    norm = {c: c.lower().replace(" ", "").replace("_", "").replace("/", "") for c in df.columns}
    rev = {v: k for k, v in norm.items()}
    # lat/lon aliases
    lat_src = next((rev[a] for a in ["lat","latitude","breddegrad","y"] if a in rev), None)
    lon_src = next((rev[a] for a in ["lon","longitude","lengdegrad","longitud","x"] if a in rev), None)
    if lat_src and "lat" not in df.columns: df = df.rename(columns={lat_src: "lat"})
    if lon_src and "lon" not in df.columns: df = df.rename(columns={lon_src: "lon"})
    return df

def sanitize_coords(df: pd.DataFrame, per_stop_min: float) -> pd.DataFrame:
    for col in ["lat","lon"]:
        df[col] = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "service_min" not in df.columns:
        df["service_min"] = per_stop_min
    df["service_min"] = pd.to_numeric(df["service_min"], errors="coerce").fillna(per_stop_min).clip(lower=0)
    bad = (df["lat"].isna() | df["lon"].isna()
           | ~df["lat"].between(-90,90) | ~df["lon"].between(-180,180)
           | ~np.isfinite(df["lat"]) | ~np.isfinite(df["lon"]))
    if bad.any():
        st.error(f"Found {bad.sum()} invalid coordinate rows — showing sample below.")
        st.dataframe(df.loc[bad].head(50), use_container_width=True)
        st.stop()
    df["lat"] = df["lat"].round(7); df["lon"] = df["lon"].round(7)
    return df

def draw_map(df_routes: pd.DataFrame):
    if df_routes.empty: return None
    mid_lat = df_routes["from_lat"].mean()
    mid_lon = df_routes["from_lon"].mean()
    m = folium.Map(location=[mid_lat, mid_lon], zoom_start=11)
    for rid in sorted(df_routes["route_id"].unique()):
        sub = df_routes[df_routes["route_id"] == rid]
        coords = list(zip(sub["from_lat"], sub["from_lon"])) + [(sub.iloc[-1]["to_lat"], sub.iloc[-1]["to_lon"])]
        folium.PolyLine(coords, tooltip=f"Route {rid}").add_to(m)
        for _, row in sub.iterrows():
            folium.Marker(
                [row["from_lat"], row["from_lon"]],
                tooltip=f"R{rid} • {row['from_name']} → {row['to_name']}",
                icon=BeautifyIcon(number=int(row["order"]), border_color="#333", text_color="#000")
            ).add_to(m)
    return m

# ── Run button ────────────────────────────────────────────────────────────────
run = st.button("🚀 Build Routes", type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    t0 = time.perf_counter()
    with st.status("Starting…", expanded=False) as st_status:
        try:
            # READ
            st_status.update(label="Reading CSV…", state="running")
            df = robust_read(uploaded, default_sep=chosen_sep)
            st.write(f"✅ CSV read. Rows: {len(df)}, Columns: {len(df.columns)}")

            # NORMALIZE
            st_status.update(label="Normalizing headers…", state="running")
            df = df.loc[:, ~df.columns.duplicated()]
            df = normalize_columns(df)
            # last-chance exact-case fallback for common names
            if "lat" not in df.columns:
                exact_lat = [c for c in df.columns if c.strip().lower() == "latitude"]
                if exact_lat: df = df.rename(columns={exact_lat[0]: "lat"})
            if "lon" not in df.columns:
                exact_lon = [c for c in df.columns if c.strip().lower() == "longitude"]
                if exact_lon: df = df.rename(columns={exact_lon[0]: "lon"})
            # map id/name from your business fields
            if "Locations ID" in df.columns and "id" not in df.columns:
                df["id"] = df["Locations ID"].astype(str)
            elif "Panel id" in df.columns and "id" not in df.columns:
                df["id"] = df["Panel id"].astype(str)
            if "name" not in df.columns:
                for cand in ["Name/Street", "Name/Address", "Name", "Address 1"]:
                    if cand in df.columns:
                        df["name"] = df[cand].astype(str); break
            if "name" not in df.columns: df["name"] = [f"S{i}" for i in range(len(df))]
            if "id" not in df.columns:   df["id"]   = [f"S{i}" for i in range(len(df))]

            if "lat" not in df.columns or "lon" not in df.columns:
                st_status.update(label="Could not find latitude/longitude columns.", state="error")
                st.error("Could not find latitude/longitude columns after normalization.")
                st.write("Detected columns:", list(df.columns))
                st.stop()

            # FILTER
            if filter_active:
                st_status.update(label="Filtering active rows…", state="running")
                col = next((c for c in df.columns if c.lower() == active_col.strip().lower()), None)
                if col:
                    allowed = {v.strip().lower() for v in active_values.split(",") if v.strip()}
                    df = df[df[col].astype(str).str.lower().isin(allowed)]
                    if df.empty:
                        st_status.update(label="All rows filtered out by 'active' filter.", state="error")
                        st.error("All rows filtered out by 'active' filter.")
                        st.stop()
                else:
                    st.warning(f"Active column '{active_col}' not found; skipping filter.")

            # COORDS
            st_status.update(label="Validating coordinates…", state="running")
            df = sanitize_coords(df, per_stop_min)
            if use_first_row_as_depot and len(df) > 0:
                df.loc[df.index[0], "service_min"] = 0.0
            st.info(f"🧭 Stops detected (pre-depot): {len(df)}")

            # STOPS
            st_status.update(label="Building stop list…", state="running")
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                df.to_csv(tmp.name, index=False); csv_path = tmp.name
            stops = core.load_stops(
                csv_path, per_stop_min,
                depot_lat if use_manual_depot else None,
                depot_lon if use_manual_depot else None
            )
            st.write(f"📍 Total locations passed to solver (incl. depot): {len(stops)}")

            # MATRIX
            if backend_costs.startswith("OSRM"):
                st_status.update(label="Building OSRM (road-accurate) matrix…", state="running")
                prog = st.progress(0.0)


                def _cb(done, total):
                    prog.progress(done / total)


                matrix = core.build_osrm_matrix_batched(
                    stops,
                    base_url=osrm_base,
                    block=int(osrm_block),
                    sleep=0.0,
                    progress_cb=_cb
                )
                st.write("✅ OSRM matrix built")
            else:
                st_status.update(label="Building offline (haversine) matrix…", state="running")
                matrix = core.build_haversine_matrix(stops, avg_speed)
                st.write("✅ Matrix built")

            # SOLVE
            st_status.update(label="Solving routes…", state="running")
            if solver == "ortools":
                max_limit = max_route_min if (cap_enabled and equality_basis=="time" and max_route_min > 0) else None
                sol = core.solve_vrp_ortools(
                    stops, int(vehicles), matrix, equality_basis, per_stop_min,
                    max_limit, first_solution, metaheuristic, int(time_limit_s)
                )
            else:
                sol = core.solve_vrp_cluster2opt(stops, int(vehicles), matrix, equality_basis, per_stop_min)

            # REFINE / EXPORT
            if use_osrm:
                st_status.update(label="Refining legs with local OSRM…", state="running")
            else:
                st_status.update(label="Exporting results…", state="running")

            use_refine = use_osrm and not backend_costs.startswith("OSRM")
            df_routes, summary = core.export_outputs(
                stops, matrix, sol, equality_basis,
                export_csv=None, export_xlsx=None, per_stop_min=per_stop_min,
                use_osrm=use_refine, osrm_base=osrm_base
            )

            # Persist outputs (so they survive reruns)
            xls_buf = io.BytesIO()
            with pd.ExcelWriter(xls_buf, engine="openpyxl") as w:
                df_routes.to_excel(w, sheet_name="stops", index=False)
                summary.to_excel(w, sheet_name="summary", index=False)
            xls_buf.seek(0)
            csv_buf = io.StringIO(); df_routes.to_csv(csv_buf, index=False)

            st.session_state["last_routes_xlsx"] = xls_buf.getvalue()
            st.session_state["last_routes_csv"]  = csv_buf.getvalue()
            st.session_state["last_routes_df"]   = df_routes
            st.session_state["last_summary"]     = summary

            st_status.update(label="Done!", state="complete")
            st.toast("✅ Routes built. Scroll to the downloads below.", icon="✅")

        except Exception as e:
            st_status.update(label="Error", state="error")
            st.exception(e)
            st.error(str(e))

# ── Results area (always shown; survives reruns) ──────────────────────────────
routes_df = st.session_state.get("last_routes_df")
summary_df = st.session_state.get("last_summary")
xlsx_bytes = st.session_state.get("last_routes_xlsx")
csv_text  = st.session_state.get("last_routes_csv")

if routes_df is not None and summary_df is not None:
    st.success("Routes ready")
    col_dl = st.columns(2)
    with col_dl[0]:
        st.download_button(
            "⬇️ Download Excel (routes.xlsx)",
            data=xlsx_bytes,
            file_name="routes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_xlsx_persist",
        )
    with col_dl[1]:
        st.download_button(
            "⬇️ Download CSV (routes.csv)",
            data=csv_text,
            file_name="routes.csv",
            mime="text/csv",
            key="dl_csv_persist",
        )

    st.subheader("Summary");   st.dataframe(summary_df, use_container_width=True)
    st.subheader("Per-leg details"); st.dataframe(routes_df, use_container_width=True, height=420)

    st.subheader("Map preview")
    m = draw_map(routes_df)
    if m is not None:
        st_folium(m, width=1100)
