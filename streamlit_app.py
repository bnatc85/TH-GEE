from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import select

from database import (
    EngagementEvent,
    SessionLocal,
    TrackedQuery,
    initialize_database,
)
from metrics import calculate_profile, recency_weights


initialize_database()

st.set_page_config(
    page_title="TH-GEE Temporal Engagement Analyzer",
    page_icon="📈",
    layout="wide",
)

st.title("TH-GEE Temporal Engagement Analyzer")

st.caption(
    "Measure persistence, recency, and re-emergence in public "
    "Bluesky engagement. These measures describe temporal behavior; "
    "they do not establish malicious intent, coordination, accuracy, "
    "synthetic origin, or persuasion."
)


def load_queries() -> list[TrackedQuery]:
    with SessionLocal() as session:
        statement = select(TrackedQuery).order_by(TrackedQuery.created_at.desc())
        return list(session.scalars(statement))


def add_query(query_text: str, query_type: str) -> None:
    with SessionLocal() as session:
        session.add(
            TrackedQuery(
                query=query_text.strip(),
                query_type=query_type,
                active=True,
            )
        )
        session.commit()


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
        }
        for event in events
    ]

    return pd.DataFrame(rows)


with st.sidebar:
    st.header("Track a Bluesky narrative")

    new_query = st.text_input(
        "Hashtag or keyword",
        placeholder="#cognitivewarfare",
    )

    query_type = st.selectbox(
        "Match type",
        ["hashtag", "keyword"],
    )

    if st.button("Add query", type="primary"):
        if new_query.strip():
            add_query(new_query, query_type)
            st.success("Query added. Start collector.py to begin collection.")
            st.rerun()
        else:
            st.error("Enter a hashtag or keyword.")

queries = load_queries()

if not queries:
    st.info("Add a hashtag or keyword in the sidebar.")
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

status_col1, status_col2 = st.columns([1, 5])

with status_col1:
    if selected_query.active:
        if st.button("Pause collection"):
            set_query_status(selected_query.id, False)
            st.rerun()
    else:
        if st.button("Resume collection"):
            set_query_status(selected_query.id, True)
            st.rerun()

with status_col2:
    st.write(
        f"Collection status: "
        f"**{'Active' if selected_query.active else 'Paused'}**"
    )

st.subheader("Analysis settings")

settings_col1, settings_col2, settings_col3, settings_col4 = st.columns(4)

with settings_col1:
    days = st.number_input(
        "Observation period (days)",
        min_value=1,
        max_value=365,
        value=30,
    )

with settings_col2:
    window_choice = st.selectbox(
        "Window size",
        ["1 hour", "6 hours", "12 hours", "1 day", "7 days"],
        index=3,
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

window_rules = {
    "1 hour": "1h",
    "6 hours": "6h",
    "12 hours": "12h",
    "1 day": "1D",
    "7 days": "7D",
}

st.subheader("Interaction weights")

weight_col1, weight_col2, weight_col3, weight_col4, weight_col5 = st.columns(5)

with weight_col1:
    publication_weight = st.number_input(
        "Publication",
        min_value=0.0,
        value=1.0,
        step=0.1,
    )

with weight_col2:
    like_weight = st.number_input(
        "Like",
        min_value=0.0,
        value=0.3,
        step=0.1,
    )

with weight_col3:
    reply_weight = st.number_input(
        "Reply",
        min_value=0.0,
        value=0.7,
        step=0.1,
    )

with weight_col4:
    repost_weight = st.number_input(
        "Repost",
        min_value=0.0,
        value=1.0,
        step=0.1,
    )

with weight_col5:
    quote_weight = st.number_input(
        "Quote",
        min_value=0.0,
        value=1.0,
        step=0.1,
    )

end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=int(days))

events = load_events(
    query_id=selected_query_id,
    start_time=start_time,
    end_time=end_time,
)

if events.empty:
    st.warning(
        "No matching events have been collected in this period. "
        "Make sure collector.py is running."
    )
    st.stop()

events["event_time"] = pd.to_datetime(events["event_time"], utc=True)

weights = {
    "publication": publication_weight,
    "like": like_weight,
    "reply": reply_weight,
    "repost": repost_weight,
    "quote": quote_weight,
}

events["weighted_value"] = (
    events["event_type"].map(weights).fillna(0.0)
)

window_rule = window_rules[window_choice]

windowed = (
    events.set_index("event_time")
    .resample(window_rule)["weighted_value"]
    .sum()
    .rename("engagement")
    .to_frame()
)

full_index = pd.date_range(
    start=start_time,
    end=end_time,
    freq=window_rule,
    tz="UTC",
)

windowed = (
    windowed.reindex(full_index, fill_value=0.0)
    .rename_axis("window_start")
    .reset_index()
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

st.subheader("Recency weights")

weights_array = recency_weights(len(trace), float(gamma))

weights_frame = pd.DataFrame(
    {
        "window_start": windowed["window_start"],
        "recency_weight": weights_array,
    }
)

st.line_chart(
    weights_frame.set_index("window_start")["recency_weight"]
)

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