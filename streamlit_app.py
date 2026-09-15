from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import func, select

from collector import collect_for_query
from database import (
    EngagementEvent,
    SessionLocal,
    TrackedQuery,
    initialize_database,
)
from metrics import calculate_profile


def get_api_key() -> str | None:
    """YouTube API key from Streamlit secrets or the environment."""
    try:
        if "YOUTUBE_API_KEY" in st.secrets:
            return st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        pass
    return os.getenv("YOUTUBE_API_KEY")


def collect_query_now(query_id: int) -> None:
    """Run a YouTube collection pass for one query, with UI feedback."""
    api_key = get_api_key()
    if not api_key:
        st.error(
            "No YouTube API key found. Set YOUTUBE_API_KEY in the environment "
            "or in .streamlit/secrets.toml, then try again."
        )
        return
    with st.spinner("Collecting from YouTube…"):
        try:
            result = collect_for_query(query_id, api_key=api_key)
        except Exception as error:  # surface API/quota errors in the UI
            st.error(f"Collection failed: {error}")
            return
    st.success(f"Collected {result['videos']} videos, {result['events']} new events.")


initialize_database()

st.set_page_config(
    page_title="TH-GEE Temporal Engagement Analyzer",
    page_icon="📈",
    layout="wide",
)

st.title("TH-GEE Temporal Engagement Analyzer")

st.caption(
    "Measure persistence, recency, and re-emergence in public "
    "YouTube engagement. These measures describe temporal behavior; "
    "they do not establish malicious intent, coordination, accuracy, "
    "synthetic origin, or persuasion."
)


def load_queries() -> list[TrackedQuery]:
    with SessionLocal() as session:
        statement = select(TrackedQuery).order_by(TrackedQuery.created_at.desc())
        return list(session.scalars(statement))


def add_query(query_text: str, query_type: str) -> int:
    with SessionLocal() as session:
        query = TrackedQuery(
            query=query_text.strip(),
            query_type=query_type,
            active=True,
        )
        session.add(query)
        session.commit()
        return query.id


def set_query_status(query_id: int, active: bool) -> None:
    with SessionLocal() as session:
        query = session.get(TrackedQuery, query_id)

        if query is not None:
            query.active = active
            session.commit()


def load_events(
    query_id: int,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    with SessionLocal() as session:
        statement = (
            select(EngagementEvent)
            .where(EngagementEvent.query_id == query_id)
            .where(EngagementEvent.event_time >= start_time)
            .where(EngagementEvent.event_time <= end_time)
            .order_by(EngagementEvent.event_time)
        )

        events = list(session.scalars(statement))

    rows = [
        {
            "event_time": event.event_time,
            "event_type": event.event_type,
            "target_uri": event.target_uri,
            "actor_did": event.actor_did,
            "magnitude": event.magnitude,
        }
        for event in events
    ]

    return pd.DataFrame(rows)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def event_bounds(query_id: int) -> tuple[datetime | None, datetime | None]:
    """Earliest and latest event time for a query, so the observation horizon
    can span the data that actually exists (YouTube traces can cover years)."""
    with SessionLocal() as session:
        earliest, latest = session.execute(
            select(
                func.min(EngagementEvent.event_time),
                func.max(EngagementEvent.event_time),
            ).where(EngagementEvent.query_id == query_id)
        ).one()
    return _as_utc(earliest), _as_utc(latest)


with st.sidebar:
    st.header("Track a YouTube search")

    new_query = st.text_input(
        "Search term",
        placeholder="climate change",
    )

    if st.button("Add & collect", type="primary"):
        if new_query.strip():
            new_id = add_query(new_query, "keyword")
            collect_query_now(new_id)
            st.rerun()
        else:
            st.error("Enter a search term.")

    if get_api_key() is None:
        st.caption(
            "No YouTube API key detected. Set `YOUTUBE_API_KEY` in the "
            "environment or `.streamlit/secrets.toml` to collect from the app."
        )

queries = load_queries()

if not queries:
    st.info("Add a YouTube search term in the sidebar.")
    st.stop()

query_labels = {
    f"{query.query} — {'active' if query.active else 'paused'}": query.id
    for query in queries
}

selected_label = st.selectbox(
    "Tracked query",
    options=list(query_labels.keys()),
)

selected_query_id = query_labels[selected_label]
selected_query = next(
    query for query in queries if query.id == selected_query_id
)

status_col1, status_col2, status_col3 = st.columns([1, 1, 4])

with status_col1:
    if st.button("Collect from YouTube", type="primary"):
        collect_query_now(selected_query.id)
        st.rerun()

with status_col2:
    if selected_query.active:
        if st.button("Pause collection"):
            set_query_status(selected_query.id, False)
            st.rerun()
    else:
        if st.button("Resume collection"):
            set_query_status(selected_query.id, True)
            st.rerun()

with status_col3:
    st.write(
        f"Collection status: "
        f"**{'Active' if selected_query.active else 'Paused'}**"
    )

st.subheader("Analysis settings")

data_start, data_end = event_bounds(selected_query_id)

if data_start is None:
    st.warning(
        "No events collected for this query yet. Click "
        "**Collect from YouTube** above (needs YOUTUBE_API_KEY), or run "
        "`python collector.py`."
    )
    st.stop()

# The horizon spans the selected query's actual data, so multi-year YouTube
# histories are visible instead of being clipped to a fixed recent window.
available_days = max(1, (data_end - data_start).days + 1)

settings_col1, settings_col2, settings_col3, settings_col4 = st.columns(4)

with settings_col1:
    days = st.number_input(
        "Observation period (days)",
        min_value=1,
        max_value=int(available_days),
        value=int(available_days),
        help=(
            f"Collected data spans {data_start.date()} to {data_end.date()} "
            f"({available_days} days). The window ends at the most recent event."
        ),
    )

with settings_col2:
    window_choice = st.selectbox(
        "Window size",
        ["1 hour", "6 hours", "12 hours", "1 day", "7 days", "30 days"],
        index=4 if available_days > 400 else 3,
    )

with settings_col3:
    gamma = st.number_input(
        "Recency coefficient γ",
        min_value=0.0,
        max_value=5.0,
        value=0.10,
        step=0.05,
    )

with settings_col4:
    dormancy_windows = st.number_input(
        "Dormancy windows d",
        min_value=1,
        max_value=30,
        value=3,
    )

window_lengths = {
    "1 hour": timedelta(hours=1),
    "6 hours": timedelta(hours=6),
    "12 hours": timedelta(hours=12),
    "1 day": timedelta(days=1),
    "7 days": timedelta(days=7),
    "30 days": timedelta(days=30),
}

st.subheader("Interaction weights")

weight_col1, weight_col2, weight_col3 = st.columns(3)

with weight_col1:
    publication_weight = st.number_input(
        "Publication (video)",
        min_value=0.0,
        value=1.0,
        step=0.1,
        help="Weight per video published in the window (magnitude 1 each).",
    )

with weight_col2:
    like_weight = st.number_input(
        "Like",
        min_value=0.0,
        value=0.3,
        step=0.1,
        help="Weight applied to each video's like count.",
    )

with weight_col3:
    comment_weight = st.number_input(
        "Comment",
        min_value=0.0,
        value=0.7,
        step=0.1,
        help="Weight applied to each video's comment count.",
    )

end_time = data_end
start_time = end_time - timedelta(days=int(days))

events = load_events(
    query_id=selected_query_id,
    start_time=start_time,
    end_time=end_time,
)

if events.empty:
    st.warning(
        "No events fall inside this observation period. Widen the "
        "observation period, or click **Collect from YouTube** to fetch more."
    )
    st.stop()

events["event_time"] = pd.to_datetime(events["event_time"], utc=True)

weights = {
    "publication": publication_weight,
    "like": like_weight,
    "comment": comment_weight,
}

events["weighted_value"] = (
    events["event_type"].map(weights).fillna(0.0) * events["magnitude"]
)

window_length = window_lengths[window_choice]

# Tile the observation horizon [start_time, end_time] into T consecutive
# fixed-width windows (matching the framework's definition) and bin events by
# integer offset from start_time. This anchors the window grid and the
# engagement values to the same origin, so no windows are dropped by the
# label misalignment that a calendar resample + reindex can introduce.
start_ts = pd.Timestamp(start_time)
span = pd.Timestamp(end_time) - start_ts
window_count = max(1, int(np.ceil(span / window_length)))

window_starts = pd.to_datetime(
    [start_ts + index * window_length for index in range(window_count)],
    utc=True,
)

window_offsets = (
    ((events["event_time"] - start_ts) // window_length)
    .astype("int64")
    .clip(0, window_count - 1)
)

engagement_values = np.zeros(window_count, dtype=float)
np.add.at(
    engagement_values,
    window_offsets.to_numpy(),
    events["weighted_value"].to_numpy(),
)

windowed = pd.DataFrame(
    {
        "window_start": window_starts,
        "engagement": engagement_values,
    }
)

trace = windowed["engagement"].to_numpy(dtype=float)

default_threshold = float(
    np.quantile(trace[trace > 0], 0.25)
) if np.any(trace > 0) else 1.0

activity_threshold = st.number_input(
    "Active-window threshold q",
    min_value=0.0,
    value=max(default_threshold, 0.01),
    step=0.1,
)

profile = calculate_profile(
    trace,
    gamma=float(gamma),
    activity_threshold=float(activity_threshold),
    dormancy_windows=int(dormancy_windows),
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "Persistence",
    f"{profile.persistence:.3f}",
)

metric_col2.metric(
    "TH-GEE recency",
    f"{profile.thgee:.3f}",
)

metric_col3.metric(
    "Re-emergence",
    f"{profile.reemergence:.3f}",
)

metric_col4.metric(
    "Observed events",
    f"{len(events):,}",
)

st.subheader("Engagement timeline")

figure = go.Figure()

figure.add_trace(
    go.Bar(
        x=windowed["window_start"],
        y=windowed["engagement"],
        name="Weighted engagement",
    )
)

figure.add_hline(
    y=activity_threshold,
    line_dash="dash",
    annotation_text="Activity threshold",
)

for position in profile.reactivation_windows:
    if 0 <= position < len(windowed):
        figure.add_vline(
            x=windowed.iloc[position]["window_start"].timestamp() * 1000,
            line_dash="dot",
            annotation_text="Reactivation",
        )

figure.update_layout(
    xaxis_title="Time window",
    yaxis_title="Weighted engagement",
    hovermode="x unified",
)

st.plotly_chart(figure, use_container_width=True)

st.subheader("Event counts")

counts = (
    events["event_type"]
    .value_counts()
    .rename_axis("event_type")
    .reset_index(name="count")
)

st.dataframe(counts, use_container_width=True, hide_index=True)

st.subheader("Interpretation")

if profile.thgee > profile.persistence:
    recency_text = "Engagement is weighted toward the recent windows."
elif profile.thgee < profile.persistence:
    recency_text = "More engagement occurred earlier in the observation period."
else:
    recency_text = "Recent and earlier windows receive similar effective weight."

if profile.reactivation_windows:
    reemergence_text = (
        f"{len(profile.reactivation_windows)} qualifying reactivation "
        f"point(s) followed the selected dormancy period."
    )
else:
    reemergence_text = (
        "No qualifying return after dormancy was found using the "
        "selected threshold and dormancy duration."
    )

st.write(
    f"""
    The persistence score describes how broadly engagement is distributed
    across the selected period. {recency_text} {reemergence_text}

    These findings describe temporal engagement only. They do not establish
    whether the narrative is coordinated, malicious, accurate, synthetic,
    or persuasive.
    """
)

csv_data = windowed.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download windowed trace",
    data=csv_data,
    file_name=f"thgee_{selected_query.query.strip('#')}.csv",
    mime="text/csv",
)