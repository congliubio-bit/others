# app.py

import time
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go  # for smooth browser-side animation


# ============================
# 1. Synthetic data generator
# ============================

def generate_daily_log(
    days: int,
    countries: List[str],
    n_sites_per_country: Dict[str, int],
    syn_mean_rate: float,
    syn_var_factor: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily enrollment data for the selected countries.

    For each country:
    - Use a user-controlled synthetic per-site mean rate (syn_mean_rate, patients/site/day)
    - Add day-to-day variability controlled by syn_var_factor
    - Multiply by number of sites to get total daily rate
    - Simulate daily counts via Poisson(rate)

    Note: historical priors (mean/var) are **not** used here, only in the Bayesian prior.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for country in countries:
        n_sites = n_sites_per_country.get(country, 0)
        if n_sites <= 0:
            continue

        for day in range(1, days + 1):
            # Per-site rate today: Normal(mean, syn_var_factor * mean), truncated at 0
            per_site_rate_today = rng.normal(
                loc=syn_mean_rate,
                scale=syn_var_factor * syn_mean_rate,
            )
            per_site_rate_today = max(0.0, per_site_rate_today)

            total_rate_today = per_site_rate_today * n_sites

            enrolled = rng.poisson(total_rate_today)
            if enrolled > 0:
                rows.append(
                    {
                        "day": day,
                        "country_name": country,
                        "patients_enrolled_today": enrolled,
                    }
                )

    return pd.DataFrame(rows)


# ============================
# 2. Hierarchical Bayesian Model
# ============================

class HierarchicalBayesianModel:
    """
    Hierarchical Poisson–Gamma model for patient recruitment.

    Priors per country are built from historical enroll_mean and enroll_var.
    prior_exposure_time is a global scale factor controlling prior strength.
    """

    def _get_country_level_params(
        self,
        site_data: pd.DataFrame,
        actual_data: pd.DataFrame | None,
        prior_exposure_time: float = 1.0,
    ) -> pd.DataFrame:
        """
        Calculates posterior Gamma parameters (alpha, beta) for each country.

        Historical inputs per country (via site_data):
            - enroll_mean: historical mean rate (patients/site/day)
            - enroll_var:  historical variance of that rate

        Base Gamma prior per country:
            alpha_base = (mean^2) / var
            beta_base  =  mean    / var

        Then scale both by `prior_exposure_time` to tune prior strength.
        """
        country_params = (
            site_data
            .groupby("country_name")
            .agg(
                num_sites=("site_name", "count"),
                enroll_mean=("enroll_mean", "first"),
                enroll_var=("enroll_var", "first"),
            )
            .reset_index()
        )

        # Avoid zero variance (would make the prior infinitely strong)
        country_params["enroll_var"] = country_params["enroll_var"].replace(0, 1e-6)

        m = country_params["enroll_mean"]
        v = country_params["enroll_var"]

        # Base Gamma parameters from mean & variance:
        alpha_base = (m ** 2) / v
        beta_base = m / v

        # Scale prior strength globally
        country_params["alpha_prior"] = prior_exposure_time * alpha_base
        country_params["beta_prior"] = prior_exposure_time * beta_base

        # Initialize posteriors as priors
        country_params["alpha_posterior"] = country_params["alpha_prior"].copy()
        country_params["beta_posterior"] = country_params["beta_prior"].copy()

        # Bayesian update with actual trial data
        if actual_data is not None and not actual_data.empty:
            for _, row in actual_data.iterrows():
                mask = country_params["country_name"] == row["country_name"]
                if mask.sum() == 0:
                    continue

                total_site_exposure = (
                    country_params.loc[mask, "num_sites"].iloc[0] * row["days_observed"]
                )

                country_params.loc[mask, "alpha_posterior"] += row["patients_enrolled"]
                country_params.loc[mask, "beta_posterior"] += total_site_exposure

        return country_params.drop(columns=["enroll_mean", "enroll_var"]).set_index("country_name")

    def _process_outputs(
        self,
        results_df: pd.DataFrame,
        actual_data: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Aggregates simulation runs and combines them with actuals into a final DataFrame.

        Returns columns:
            time, p10, p25, p50, p75, p90
        """
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        # Actuals part: deterministic trajectory up to the last observed day
        if actual_data is not None and not actual_data.empty:
            start_day = int(actual_data["days_observed"].max())
            patients_so_far = int(actual_data["patients_enrolled"].sum())

            actual_curve = pd.DataFrame({"time": np.arange(1, start_day + 1)})
            for q in quantiles:
                actual_curve[f"p{int(q * 100)}"] = np.linspace(
                    0, patients_so_far, start_day
                )
        else:
            actual_curve = pd.DataFrame()

        # If no simulated futures, return actuals only
        if results_df is None or results_df.empty:
            return actual_curve

        # Aggregate simulations into percentile bands
        grouped = results_df.groupby("time")["total_enrolled"]
        forecast_df = grouped.quantile(quantiles).unstack()
        forecast_df.columns = [f"p{int(q * 100)}" for q in quantiles]
        forecast_df = forecast_df.reset_index()

        # Combine actual + forecast
        combined_df = pd.concat([actual_curve, forecast_df], ignore_index=True)
        combined_df = (
            combined_df.drop_duplicates(subset="time", keep="first")
            .sort_values("time")
            .reset_index(drop=True)
        )

        return combined_df

    def predict(
        self,
        site_data: pd.DataFrame,
        actual_data: pd.DataFrame | None,
        total_target: int,
        prior_exposure_time: float,
        num_sim: int,
        max_days: int,
    ) -> pd.DataFrame:
        """
        Runs the full posterior predictive simulation to generate the forecast.
        """
        country_params = self._get_country_level_params(
            site_data, actual_data, prior_exposure_time
        )
        sim_site_data = site_data.join(country_params, on="country_name")

        all_sim_results = []

        if actual_data is not None and not actual_data.empty:
            start_day = int(actual_data["days_observed"].max())
            patients_so_far = int(actual_data["patients_enrolled"].sum())
        else:
            start_day = 0
            patients_so_far = 0

        for sim_id in range(num_sim):
            cumulative_patients = patients_so_far

            # 1) Parameter uncertainty: draw a rate for each site
            sim_site_data["sim_rate"] = np.random.gamma(
                shape=sim_site_data["alpha_posterior"].values,
                scale=1.0 / sim_site_data["beta_posterior"].values,
            )
            total_study_rate = sim_site_data["sim_rate"].sum()

            # 2) Process uncertainty: Poisson arrivals over future days
            for day in range(start_day + 1, max_days + 1):
                if cumulative_patients >= total_target:
                    break

                daily_enrolled_count = np.random.poisson(total_study_rate)
                cumulative_patients += daily_enrolled_count

                all_sim_results.append(
                    {
                        "sim_id": sim_id,
                        "time": day,
                        "total_enrolled": cumulative_patients,
                    }
                )

        results_df = pd.DataFrame(all_sim_results)
        return self._process_outputs(results_df, actual_data)


# ============================
# 3. Daily-updating simulator
# ============================

class DailyUpdatingSimulator:
    """
    Runs the Bayesian forecast repeatedly over time as new daily data arrive.
    """

    def __init__(self, model: HierarchicalBayesianModel):
        self.model = model

    def run_simulation_over_time(
        self,
        site_data: pd.DataFrame,
        daily_log: pd.DataFrame,
        total_target: int,
        prior_exposure_time: float,
        num_sim: int,
        max_days: int,
    ) -> pd.DataFrame:
        if daily_log.empty:
            return pd.DataFrame()

        all_projections = []
        last_observed_day = int(daily_log["day"].max())

        for day in range(1, last_observed_day + 1):
            current_log = daily_log[daily_log["day"] <= day]

            batch_summary = (
                current_log.groupby("country_name")
                .agg(patients_enrolled=("patients_enrolled_today", "sum"))
                .reset_index()
            )
            batch_summary["days_observed"] = day

            projection = self.model.predict(
                site_data=site_data,
                actual_data=batch_summary,
                total_target=total_target,
                prior_exposure_time=prior_exposure_time,
                num_sim=num_sim,
                max_days=max_days,
            )
            projection["forecast_as_of_day"] = day
            all_projections.append(projection)

        return pd.concat(all_projections, ignore_index=True)


# ============================
# 4. Cached simulation wrapper
# ============================

@st.cache_data(show_spinner=True)
def run_full_simulation(
    n_days: int,
    target_sample_size: int,
    prior_strength: float,
    num_sim: int,
    max_days: int,
    historical_df: pd.DataFrame,
    selected_countries: List[str],
    sites_per_selected: int,
    syn_mean_rate: float,
    syn_var_factor: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the full simulation pipeline and caches results for given parameters.

    Returns:
        site_data, daily_enrollment_log, all_daily_projections
    """
    # Build site-level dataframe: same number of sites for all selected countries
    n_sites_per_country = {c: sites_per_selected for c in selected_countries}

    site_list = [
        {"country_name": c, "site_name": f"{c}-{i}"}
        for c in selected_countries
        for i in range(1, n_sites_per_country[c] + 1)
    ]
    site_data = pd.DataFrame(site_list)

    # Merge historical priors into site_data (for Gamma prior only)
    site_data = site_data.merge(historical_df, on="country_name", how="left")

    # Fill any missing historical numbers with conservative defaults
    site_data["enroll_mean"] = site_data["enroll_mean"].fillna(0.10)
    site_data["enroll_var"] = site_data["enroll_var"].fillna(0.05**2)

    # Generate synthetic daily enrollment log using user-controlled synthetic rates
    daily_enrollment_log = generate_daily_log(
        days=n_days,
        countries=selected_countries,
        n_sites_per_country=n_sites_per_country,
        syn_mean_rate=syn_mean_rate,
        syn_var_factor=syn_var_factor,
    )

    # Run the day-by-day simulation
    model_engine = HierarchicalBayesianModel()
    daily_simulator = DailyUpdatingSimulator(model=model_engine)

    all_daily_projections = daily_simulator.run_simulation_over_time(
        site_data=site_data,
        daily_log=daily_enrollment_log,
        total_target=target_sample_size,
        prior_exposure_time=prior_strength,
        num_sim=num_sim,
        max_days=max_days,
    )

    return site_data, daily_enrollment_log, all_daily_projections


# ============================
# 5. Plotting helpers (Matplotlib)
# ============================

def plot_final_forecast(
    daily_enrollment_log: pd.DataFrame,
    all_daily_projections: pd.DataFrame,
    target_sample_size: int,
):
    if all_daily_projections.empty or daily_enrollment_log.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No enrollment data to plot yet.", ha="center", va="center")
        return fig

    last_day_of_log = int(daily_enrollment_log["day"].max())

    final_projections = all_daily_projections[
        all_daily_projections["forecast_as_of_day"] == last_day_of_log
    ].copy()

    scatter_data = (
        daily_enrollment_log
        .groupby("day")["patients_enrolled_today"]
        .sum()
        .reset_index()
        .sort_values("day")
    )
    scatter_data["cumulative_enrolled"] = scatter_data["patients_enrolled_today"].cumsum()

    fig, ax = plt.subplots(figsize=(9, 5))

    observed_mask = final_projections["time"] <= last_day_of_log
    forecast_mask = final_projections["time"] >= last_day_of_log

    ax.plot(
        final_projections["time"],
        final_projections["p50"],
        label="Final Median Forecast (P50)",
        zorder=5,
    )

    ax.fill_between(
        final_projections.loc[forecast_mask, "time"],
        final_projections.loc[forecast_mask, "p25"],
        final_projections.loc[forecast_mask, "p75"],
        alpha=0.3,
        label="Final IQR (P25–P75)",
        zorder=4,
    )

    ax.plot(
        final_projections.loc[observed_mask, "time"],
        final_projections.loc[observed_mask, "p50"],
        label="Actual Enrollment Trajectory",
        linewidth=2,
        zorder=6,
    )

    ax.scatter(
        scatter_data["day"],
        scatter_data["cumulative_enrolled"],
        s=30,
        label="Cumulative Observed Enrollments",
        zorder=7,
    )

    ax.set_title(
        f"Final Bayesian Forecast (After {last_day_of_log} Days of Observation)",
        fontsize=12,
    )
    ax.set_xlabel("Study Day")
    ax.set_ylabel("Cumulative Patients Enrolled")

    ax.axhline(
        y=target_sample_size,
        linestyle="--",
        label=f"Target ({target_sample_size})",
    )
    ax.axvline(
        x=last_day_of_log,
        linestyle=":",
        linewidth=2,
        label="Last Observation Day",
    )

    ax.set_xlim(0, final_projections["time"].max())
    ymax = max(
        target_sample_size * 1.1,
        final_projections["p90"].max() * 1.05,
    )
    ax.set_ylim(0, ymax)

    ax.legend(loc="upper left")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    return fig


def plot_country_drilldown(
    country: str,
    daily_enrollment_log: pd.DataFrame,
    sites_per_selected: int,
):
    """Country-level drilldown: number of sites + cumulative observed enrollments."""
    st.write(f"**Country:** {country}")
    st.metric("Number of sites (per selected country)", f"{sites_per_selected}")

    country_log = daily_enrollment_log[daily_enrollment_log["country_name"] == country].copy()

    fig, ax = plt.subplots(figsize=(6, 4))

    if not country_log.empty:
        cum = (
            country_log.groupby("day")["patients_enrolled_today"]
            .sum()
            .reset_index()
            .sort_values("day")
        )
        cum["cumulative_enrolled"] = cum["patients_enrolled_today"].cumsum()
        ax.plot(cum["day"], cum["cumulative_enrolled"], marker="o")
        ax.set_ylim(bottom=0)
    else:
        ax.text(0.5, 0.5, "No observed enrollments yet", ha="center", va="center")

    ax.set_title(f"Observed cumulative enrollments – {country}")
    ax.set_xlabel("Study Day")
    ax.set_ylabel("Cumulative patients")
    ax.grid(True, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    st.pyplot(fig)


# ============================
# 6. Plotly animated evolution helper
# ============================

def build_plotly_evolution_figure(
    daily_enrollment_log: pd.DataFrame,
    all_daily_projections: pd.DataFrame,
    target_sample_size: int,
):
    """
    Build a Plotly animated chart showing forecast evolution over time.
    Includes:
        - Median forecast
        - IQR (P25–P75) shaded band
        - Actual cumulative observed (orange)
        - Play/Pause buttons moved below the plot
    """
    if all_daily_projections.empty or daily_enrollment_log.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No forecast available yet.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Compute cumulative observed enrollments
    cum_obs = (
        daily_enrollment_log
        .groupby("day")["patients_enrolled_today"]
        .sum()
        .reset_index()
        .sort_values("day")
    )
    cum_obs["cumulative_enrolled"] = cum_obs["patients_enrolled_today"].cumsum()

    max_time = int(all_daily_projections["time"].max())

    # Global y-axis range: max of p90/p75, observed, and target
    y_candidates = []
    if "p90" in all_daily_projections.columns:
        y_candidates.append(all_daily_projections["p90"].max())
    elif "p75" in all_daily_projections.columns:
        y_candidates.append(all_daily_projections["p75"].max())

    y_candidates.append(cum_obs["cumulative_enrolled"].max())
    y_candidates.append(target_sample_size)
    global_ymax = max(y_candidates) * 1.10

    # Unique "as of" days for which we have forecasts
    as_of_days = sorted(all_daily_projections["forecast_as_of_day"].unique())

    frames = []

    for day in as_of_days:
        proj = all_daily_projections[
            all_daily_projections["forecast_as_of_day"] == day
        ].copy()

        # Forecast quantiles
        x_forecast = proj["time"].values
        p50 = proj["p50"].values
        p25 = proj["p25"].values
        p75 = proj["p75"].values

        # Observed cumulative enrollments up to this day
        obs = cum_obs[cum_obs["day"] <= day]
        x_obs = obs["day"].values
        y_obs = obs["cumulative_enrolled"].values

        frames.append(
            go.Frame(
                name=str(day),
                data=[
                    # LOWER IQR BOUND
                    go.Scatter(
                        x=x_forecast,
                        y=p25,
                        mode="lines",
                        line=dict(width=0),
                        name="P25",
                        showlegend=False,
                    ),
                    # UPPER IQR BOUND + FILL
                    go.Scatter(
                        x=x_forecast,
                        y=p75,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(100, 149, 237, 0.30)",  # cornflowerblue
                        name="IQR (P25–P75)",
                        showlegend=False,
                    ),
                    # MEDIAN FORECAST P50
                    go.Scatter(
                        x=x_forecast,
                        y=p50,
                        mode="lines",
                        line=dict(color="white", width=2),
                        name="Median Forecast (P50)",
                    ),
                    # Actuals (orange)
                    go.Scatter(
                        x=x_obs,
                        y=y_obs,
                        mode="markers+lines",
                        line=dict(color="#FFA500", width=2),
                        marker=dict(color="#FFA500", size=8),
                        name="Cumulative Observed",
                    ),
                    # Target line
                    go.Scatter(
                        x=[0, max_time],
                        y=[target_sample_size, target_sample_size],
                        mode="lines",
                        line=dict(dash="dash", color="red"),
                        name=f"Target {target_sample_size}",
                    ),
                ],
            )
        )

    # Use the first frame as initial data
    init_frame = frames[0].data

    fig = go.Figure(
        data=init_frame,
        frames=frames,
        layout=go.Layout(
            title="Forecast Evolution Over Time",
            xaxis=dict(title="Study Day", range=[0, max_time]),
            yaxis=dict(title="Cumulative Patients Enrolled", range=[0, global_ymax]),
            legend=dict(x=0.01, y=0.99),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    x=0.5,
                    y=-0.15,            # BELOW the plot
                    xanchor="center",
                    yanchor="top",
                    showactive=False,
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 90, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
        ),
    )

    return fig


# ============================
# 7. Streamlit App
# ============================

def main():
    st.set_page_config(
        page_title="Bayesian Recruitment Forecast Demo",
        layout="wide",
    )

    st.title("Bayesian Adaptive Patient Recruitment Demo")

    st.markdown(
        """
This app demonstrates a **hierarchical Bayesian Gamma–Poisson model** for patient recruitment with:

- Country-level priors based on historical mean & variance (used internally)
- User-controlled synthetic enrollment rates for demos
- Daily Bayesian updating as new enrollment data arrive
- Posterior predictive simulation to generate forecast bands
"""
    )

    # Init session_state container for simulation results
    if "sim_results" not in st.session_state:
        st.session_state.sim_results = None

    # ------------------------
    # Historical priors source (used only for priors, not synthetic data)
    # 🔧 Replace this block with your real historical_data (30 countries)
    # ------------------------
    historical_data = {
        "country_name": ["United States", "United Kingdom", "Spain"],
        "enroll_mean": [0.25, 0.15, 0.10],
        "enroll_var": [0.03**2, 0.02**2, 0.02**2],
    }
    historical_df = pd.DataFrame(historical_data)

    # ------------------------
    # Sidebar controls
    # ------------------------
    st.sidebar.header("Simulation Controls")

    # Multi-select countries (EMPTY by default)
    selected_countries = st.sidebar.multiselect(
        "Select countries to include",
        options=historical_df["country_name"].tolist(),
        default=[],
    )

    # Single number of sites for all selected countries
    sites_per_selected = st.sidebar.number_input(
        "Number of sites per selected country",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

    # Synthetic data generation controls
    st.sidebar.markdown("### Synthetic Enrollment Settings")
    syn_mean_rate = st.sidebar.number_input(
        "Synthetic mean enrollment rate per site (patients/day)",
        min_value=0.01,
        max_value=2.0,
        value=0.20,
        step=0.01,
    )
    syn_var_factor = st.sidebar.slider(
        "Synthetic day-to-day variability (fraction of mean)",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="0 = constant rate each day; higher values = more noisy daily rates",
    )

    # Model controls
    n_days = st.sidebar.slider("Observed days in current trial", 30, 180, 90, step=10)
    target_sample_size = st.sidebar.number_input(
        "Target sample size",
        min_value=100,
        max_value=2000,
        value=800,
        step=50,
    )
    prior_strength = st.sidebar.slider(
        "Prior strength (global scale)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="1.0 = use historical variance as-is; >1 = stronger priors, <1 = weaker",
    )
    num_sim = st.sidebar.slider("Number of simulations", 100, 2000, 500, step=100)
    max_days = st.sidebar.slider("Forecast horizon (days)", 365, 1460, 730, step=365)

    st.sidebar.markdown(
        "Countries selected: " + (", ".join(selected_countries) if selected_countries else "None")
    )

    # Run simulation button
    run_clicked = st.sidebar.button("Run simulation")

    if run_clicked:
        if not selected_countries:
            st.sidebar.warning("Please select at least one country before running the simulation.")
        else:
            with st.spinner("Running Bayesian simulation..."):
                sim_results = run_full_simulation(
                    n_days=n_days,
                    target_sample_size=target_sample_size,
                    prior_strength=prior_strength,
                    num_sim=num_sim,
                    max_days=max_days,
                    historical_df=historical_df,
                    selected_countries=selected_countries,
                    sites_per_selected=sites_per_selected,
                    syn_mean_rate=syn_mean_rate,
                    syn_var_factor=syn_var_factor,
                )
            st.session_state.sim_results = sim_results

    # If no simulation yet, show instructions and stop
    if st.session_state.sim_results is None:
        st.info("Select countries and parameters on the left, then click **Run simulation**.")
        return

    # Unpack results from last run
    site_data, daily_enrollment_log, all_daily_projections = st.session_state.sim_results

    # ------------------------
    # KPIs
    # ------------------------
    col1, col2, col3 = st.columns(3)

    total_enrolled = (
        daily_enrollment_log["patients_enrolled_today"].sum()
        if not daily_enrollment_log.empty
        else 0
    )
    last_day_of_log = (
        int(daily_enrollment_log["day"].max())
        if not daily_enrollment_log.empty
        else n_days
    )

    if not all_daily_projections.empty:
        final_proj = all_daily_projections[
            all_daily_projections["forecast_as_of_day"] == last_day_of_log
        ].copy()
        p50_completion_day = final_proj.loc[final_proj["p50"] >= target_sample_size, "time"]
        p90_completion_day = final_proj.loc[final_proj["p90"] >= target_sample_size, "time"]

        p50_completion_str = (
            f"Day {int(p50_completion_day.iloc[0])}" if not p50_completion_day.empty else "Not reached"
        )
        p90_completion_str = (
            f"Day {int(p90_completion_day.iloc[0])}" if not p90_completion_day.empty else "Not reached"
        )
    else:
        p50_completion_str = "Not available"
        p90_completion_str = "Not available"

    with col1:
        st.metric("Enrolled so far", f"{int(total_enrolled)} / {target_sample_size}")
    with col2:
        st.metric("P50 completion", p50_completion_str)
    with col3:
        st.metric("P90 completion", p90_completion_str)

    st.markdown("---")

    # ------------------------
    # Final forecast plot (Matplotlib)
    # ------------------------
    st.subheader("Final Forecast Using All Available Data")
    fig_final = plot_final_forecast(
        daily_enrollment_log, all_daily_projections, target_sample_size
    )
    st.pyplot(fig_final)

    st.markdown("---")

    # ------------------------
    # Forecast evolution (Plotly animation)
    # ------------------------
    st.subheader("Forecast Evolution Over Time")

    fig_evol = build_plotly_evolution_figure(
        daily_enrollment_log,
        all_daily_projections,
        target_sample_size,
    )
    st.plotly_chart(fig_evol, use_container_width=True)

    # ------------------------
    # Country-level drilldown
    # ------------------------
    st.markdown("---")
    st.subheader("Country-level Drilldown (Observed Data)")

    if selected_countries:
        drill_country = st.selectbox(
            "Select a country to inspect:",
            options=selected_countries,
        )

        plot_country_drilldown(
            country=drill_country,
            daily_enrollment_log=daily_enrollment_log,
            sites_per_selected=sites_per_selected,
        )

    st.markdown(
        """
**Notes:**

- You must select countries and press **Run simulation** before any modeling happens.
- A single "Number of sites" setting is applied to **all** selected countries.
- Historical priors (mean and variance) are used internally to define the Gamma prior,
  but synthetic enrollment curves are controlled by the **Synthetic Enrollment Settings**.
- The Forecast Evolution plot is now a **smooth Plotly animation** with IQR shading and Play/Pause controls below the chart.
"""
    )


if __name__ == "__main__":
    main()
