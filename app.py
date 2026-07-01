from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


st.set_page_config(
    page_title="Road Safety Medallion",
    page_icon="🛣️",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
        .title-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: linear-gradient(90deg, #17324d, #2f5d7c);
            color: #f7fbff;
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }
        .hero {
            background: linear-gradient(135deg, rgba(23,50,77,0.08), rgba(221,170,72,0.10));
            border: 1px solid rgba(23,50,77,0.10);
            border-radius: 1.1rem;
            padding: 1.25rem 1.3rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, sep=";", low_memory=False)


def parse_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    parsed = df.copy()
    for column in columns:
        if column in parsed.columns:
            parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    return parsed


def select_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def build_model(caract: pd.DataFrame, lieux: pd.DataFrame, usagers: pd.DataFrame, vehicules: pd.DataFrame) -> dict[str, pd.DataFrame]:
    accidents = parse_numeric(
        caract,
        ["Num_Acc", "jour", "mois", "an", "lum", "agg", "int", "atm", "col", "lat", "long"],
    )
    accidents["date"] = pd.to_datetime(
        accidents[["an", "mois", "jour"]].rename(columns={"an": "year", "mois": "month", "jour": "day"}),
        errors="coerce",
    )
    accidents["hour"] = pd.to_datetime(accidents["hrmn"].astype("string"), format="%H:%M", errors="coerce").dt.hour
    accidents["minute"] = pd.to_datetime(accidents["hrmn"].astype("string"), format="%H:%M", errors="coerce").dt.minute
    accidents["time_of_day"] = pd.cut(
        accidents["hour"],
        bins=[-1, 5, 11, 15, 19, 23],
        labels=["Night", "Morning peak", "Afternoon slack", "Evening peak", "Late evening"],
    )

    locations = parse_numeric(
        lieux,
        ["Num_Acc", "catr", "v1", "circ", "nbv", "vosp", "prof", "pr", "pr1", "plan", "larrout", "surf", "infra", "situ", "vma"],
    )
    if "voie" in locations.columns:
        locations["road_name"] = locations["voie"].astype("string").fillna("Unknown")
    else:
        locations["road_name"] = "Unknown"
    location_summary = (
        locations.sort_values(["Num_Acc", "road_name"])
        .groupby("Num_Acc", as_index=False)
        .agg(
            road_name=("road_name", "first"),
            road_category=("catr", "first"),
            lanes=("nbv", "first"),
            speed_limit=("vma", "first"),
            surface=("surf", "first"),
            infrastructure=("infra", "first"),
            situation=("situ", "first"),
        )
    )

    vehicles = parse_numeric(vehicules, ["Num_Acc", "senc", "catv", "obs", "obsm", "choc", "manv", "motor", "occutc"])
    vehicle_summary = (
        vehicles.groupby("Num_Acc", as_index=False)
        .agg(
            vehicle_count=("id_vehicule", "nunique"),
            vehicle_types=("catv", "nunique"),
            dominant_vehicle_category=("catv", lambda series: series.dropna().mode().iat[0] if not series.dropna().empty else np.nan),
        )
    )

    users = parse_numeric(
        usagers,
        ["Num_Acc", "place", "catu", "grav", "sexe", "an_nais", "trajet", "secu1", "secu2", "secu3", "locp", "actp", "etatp"],
    )
    users = users.merge(accidents[["Num_Acc", "an"]], on="Num_Acc", how="left")
    users["age"] = users["an"] - users["an_nais"]
    users["age_band"] = pd.cut(
        users["age"],
        bins=[0, 17, 24, 34, 49, 64, 120],
        labels=["0-17", "18-24", "25-34", "35-49", "50-64", "65+"],
        include_lowest=True,
    )
    users["severity_flag"] = users["grav"].isin([2, 3, 4]).astype("Int64")
    user_summary = (
        users.groupby("Num_Acc", as_index=False)
        .agg(
            victim_count=("id_usager", "nunique"),
            severe_cases=("severity_flag", "sum"),
            average_age=("age", "mean"),
        )
    )

    fact = (
        accidents[["Num_Acc", "date", "hour", "minute", "time_of_day", "dep", "com", "agg", "int", "atm", "col", "adr", "lat", "long"]]
        .merge(location_summary, on="Num_Acc", how="left")
        .merge(vehicle_summary, on="Num_Acc", how="left")
        .merge(user_summary, on="Num_Acc", how="left")
    )
    fact["victim_count"] = fact["victim_count"].fillna(0)
    fact["vehicle_count"] = fact["vehicle_count"].fillna(0)
    fact["severe_cases"] = fact["severe_cases"].fillna(0)
    fact["severity_index"] = np.where(fact["victim_count"] > 0, fact["severe_cases"] / fact["victim_count"], 0)

    dim_time = accidents[["Num_Acc", "date", "hour", "minute", "time_of_day"]].copy()
    dim_location = fact[["Num_Acc", "dep", "com", "adr", "road_name", "road_category", "lanes", "speed_limit", "surface", "infrastructure", "situation", "lat", "long"]].copy()
    dim_vehicle = vehicles[select_existing_columns(vehicles, ["Num_Acc", "id_vehicule", "num_veh", "senc", "catv", "obs", "obsm", "choc", "manv", "motor", "occutc"])].copy()
    dim_user = users[select_existing_columns(users, ["Num_Acc", "id_usager", "id_vehicule", "num_veh", "place", "catu", "grav", "sexe", "an_nais", "age", "age_band", "trajet", "secu1", "secu2", "secu3", "locp", "actp", "etatp"])].copy()

    model_summary = pd.DataFrame(
        [
            {"table": "fact_accidents", "rows": len(fact), "columns": len(fact.columns)},
            {"table": "dim_time", "rows": len(dim_time), "columns": len(dim_time.columns)},
            {"table": "dim_location", "rows": len(dim_location), "columns": len(dim_location.columns)},
            {"table": "dim_vehicle", "rows": len(dim_vehicle), "columns": len(dim_vehicle.columns)},
            {"table": "dim_user", "rows": len(dim_user), "columns": len(dim_user.columns)},
        ]
    )

    return {
        "fact": fact,
        "dim_time": dim_time,
        "dim_location": dim_location,
        "dim_vehicle": dim_vehicle,
        "dim_user": dim_user,
        "accidents": accidents,
        "locations_raw": locations,
        "vehicles_raw": vehicles,
        "users_raw": users,
        "model_summary": model_summary,
    }


def format_ratio(value: float) -> str:
    return f"{value:.1%}" if pd.notna(value) else "n/a"


def main() -> None:
    st.markdown('<div class="title-badge">Medallion architecture for road safety</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0 0 0.35rem 0;">Road-Safety Analytical Model</h1>
            <p style="margin:0; font-size:1.02rem; line-height:1.5;">
                A Streamlit dashboard built directly on the cleaned CSV files, then shaped into a gold star schema with a
                central accident fact table and reusable dimensions for analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading and shaping the datasets..."):
        model = build_model(
            load_csv("caract-2024.csv"),
            load_csv("lieux-2024.csv"),
            load_csv("usagers-2024.csv"),
            load_csv("vehicules-2024.csv"),
        )

    fact = model["fact"].copy()
    accidents = model["accidents"].copy()
    users = model["users_raw"].copy()
    locations = model["locations_raw"].copy()
    vehicles = model["vehicles_raw"].copy()
    model_summary = model["model_summary"].copy()

    min_date = fact["date"].min()
    max_date = fact["date"].max()
    date_value = None
    if pd.notna(min_date) and pd.notna(max_date):
        date_value = (min_date.date(), max_date.date())

    with st.sidebar:
        st.header("Filters")
        selected_departments = st.multiselect(
            "Department",
            sorted(fact["dep"].dropna().astype(str).unique().tolist()),
        )
        selected_time_bands = st.multiselect(
            "Time of day",
            sorted(fact["time_of_day"].dropna().astype(str).unique().tolist()),
            default=sorted(fact["time_of_day"].dropna().astype(str).unique().tolist()),
        )
        date_range = st.date_input("Date range", value=date_value)
        selected_hours = st.slider("Hour range", 0, 23, (0, 23))
        show_missing_locations = st.checkbox("Keep missing road names", value=True)

    filtered = fact.copy()
    if selected_departments:
        filtered = filtered[filtered["dep"].astype(str).isin(selected_departments)]
    if selected_time_bands:
        filtered = filtered[filtered["time_of_day"].astype(str).isin(selected_time_bands)]
    if isinstance(date_range, tuple) and len(date_range) == 2 and date_range[0] and date_range[1]:
        filtered = filtered[(filtered["date"].dt.date >= date_range[0]) & (filtered["date"].dt.date <= date_range[1])]
    filtered = filtered[(filtered["hour"].fillna(-1) >= selected_hours[0]) & (filtered["hour"].fillna(-1) <= selected_hours[1])]
    if not show_missing_locations:
        filtered = filtered[filtered["road_name"].notna() & filtered["road_name"].ne("Unknown")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accidents", f"{len(filtered):,}".replace(",", " "))
    c2.metric("Victims", f"{int(filtered['victim_count'].sum()):,}".replace(",", " "))
    c3.metric("Vehicles", f"{int(filtered['vehicle_count'].sum()):,}".replace(",", " "))
    c4.metric("Severe case share", format_ratio(filtered["severity_index"].mean()))

    tab_overview, tab_model, tab_raw = st.tabs(["Overview", "Analytical model", "Raw samples"])

    with tab_overview:
        left, right = st.columns([1.15, 0.85])
        with left:
            by_dep = (
                filtered.groupby("dep", as_index=False)
                .agg(accidents=("Num_Acc", "nunique"), victims=("victim_count", "sum"))
                .sort_values("accidents", ascending=False)
                .head(15)
            )
            fig = px.bar(by_dep, x="dep", y="accidents", color="victims", color_continuous_scale="Blues", title="Top departments by accident count")
            fig.update_layout(xaxis_title="Department", yaxis_title="Accidents", coloraxis_colorbar_title="Victims")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            by_hour = filtered.dropna(subset=["hour"]).groupby("hour", as_index=False).agg(accidents=("Num_Acc", "nunique"), severity=("severity_index", "mean"))
            fig = px.line(by_hour, x="hour", y="accidents", markers=True, title="Accidents by hour")
            fig.update_layout(xaxis_title="Hour of day", yaxis_title="Accidents")
            st.plotly_chart(fig, use_container_width=True)

        bottom_left, bottom_right = st.columns([1, 1])
        with bottom_left:
            severity = (
                users.assign(gravity=users["grav"].map({1: "Uninjured", 2: "Injured", 3: "Hospitalized", 4: "Killed"}).fillna("Unknown"))
                .groupby("gravity", as_index=False)
                .size()
                .sort_values("size", ascending=False)
            )
            fig = px.pie(severity, values="size", names="gravity", title="User severity mix")
            st.plotly_chart(fig, use_container_width=True)
        with bottom_right:
            geo = filtered.dropna(subset=["lat", "long"])
            if len(geo):
                fig = px.scatter_mapbox(
                    geo,
                    lat="lat",
                    lon="long",
                    size="victim_count",
                    color="severity_index",
                    color_continuous_scale="OrRd",
                    zoom=4,
                    height=500,
                    title="Spatial distribution of accidents",
                )
                fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 40, "l": 0, "b": 0})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No coordinates available for the current filter set.")

        st.subheader("Accident patterns by time of day")
        by_time = filtered.groupby("time_of_day", as_index=False).agg(accidents=("Num_Acc", "nunique"), severity=("severity_index", "mean"))
        fig = px.bar(by_time, x="time_of_day", y="accidents", color="severity", color_continuous_scale="OrRd", title="Accidents by time of day")
        fig.update_layout(xaxis_title="Time of day", yaxis_title="Accidents", coloraxis_colorbar_title="Severity")
        st.plotly_chart(fig, use_container_width=True)

    with tab_model:
        if not filtered.empty:
            top_left, top_right = st.columns(2)

            with top_left:
                age_band = (
                    users.dropna(subset=["age_band"])
                    .groupby("age_band", as_index=False)
                    .agg(users=("id_usager", "nunique"), severe_cases=("grav", lambda series: series.isin([2, 3, 4]).sum()))
                )
                fig = px.bar(
                    age_band,
                    x="age_band",
                    y="users",
                    color="severe_cases",
                    color_continuous_scale="Blues",
                    title="Age demographic analysis",
                )
                fig.update_layout(xaxis_title="Age band", yaxis_title="Users", coloraxis_colorbar_title="Severe cases")
                st.plotly_chart(fig, use_container_width=True)

            with top_right:
                collision_type = (
                    filtered.groupby("col", as_index=False)
                    .agg(accidents=("Num_Acc", "nunique"), severity=("severity_index", "mean"))
                    .sort_values("accidents", ascending=False)
                )
                fig = px.bar(
                    collision_type,
                    x="col",
                    y="accidents",
                    color="severity",
                    color_continuous_scale="OrRd",
                    title="Collision types from `col`",
                )
                fig.update_layout(xaxis_title="Collision type", yaxis_title="Accidents", coloraxis_colorbar_title="Severity")
                st.plotly_chart(fig, use_container_width=True)

            bottom_left, bottom_right = st.columns(2)

            with bottom_left:
                atmospheric_severity = (
                    filtered.groupby("atm", as_index=False)
                    .agg(accidents=("Num_Acc", "nunique"), severity=("severity_index", "mean"))
                    .sort_values("accidents", ascending=False)
                )
                fig = px.bar(
                    atmospheric_severity,
                    x="atm",
                    y="accidents",
                    color="severity",
                    color_continuous_scale="Viridis",
                    title="Severity by atmospheric conditions",
                )
                fig.update_layout(xaxis_title="Atmospheric condition", yaxis_title="Accidents", coloraxis_colorbar_title="Severity")
                st.plotly_chart(fig, use_container_width=True)

            with bottom_right:
                lane_surface = (
                    filtered.groupby(["road_category", "surface"], as_index=False)
                    .agg(accidents=("Num_Acc", "nunique"), severity=("severity_index", "mean"))
                    .sort_values("accidents", ascending=False)
                )
                fig = px.bar(
                    lane_surface,
                    x="road_category",
                    y="accidents",
                    color="severity",
                    facet_col="surface",
                    color_continuous_scale="Magma",
                    title="Severity by lane category and road surface",
                )
                fig.update_layout(xaxis_title="Lane / road category", yaxis_title="Accidents", coloraxis_colorbar_title="Severity")
                st.plotly_chart(fig, use_container_width=True)

            hourly_severity = filtered.dropna(subset=["hour"]).groupby("hour", as_index=False).agg(severity=("severity_index", "mean"), accidents=("Num_Acc", "nunique"))
            fig = px.bar(hourly_severity, x="hour", y="severity", title="Average severity index by hour")
            fig.update_layout(xaxis_title="Hour", yaxis_title="Severity index")
            st.plotly_chart(fig, use_container_width=True)

            road_category_mix = filtered.groupby("road_category", as_index=False).agg(accidents=("Num_Acc", "nunique"), victims=("victim_count", "sum"))
            fig = px.bar(road_category_mix, x="road_category", y="accidents", color="victims", title="Accidents by road category")
            fig.update_layout(xaxis_title="Road category", yaxis_title="Accidents")
            st.plotly_chart(fig, use_container_width=True)

    with tab_raw:
        st.subheader("Silver layer samples")
        s1, s2 = st.columns(2)
        with s1:
            st.caption("Accident header")
            st.dataframe(accidents.head(15), use_container_width=True)
            st.caption("Location details")
            st.dataframe(locations.head(15), use_container_width=True)
        with s2:
            st.caption("Vehicle details")
            st.dataframe(vehicles.head(15), use_container_width=True)
            st.caption("User details")
            st.dataframe(users.head(15), use_container_width=True)


if __name__ == "__main__":
    main()
