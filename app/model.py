"""
Model visualization logic for Streamlit app.

Displays anomaly detection results, user profiles, and model performance.
"""

import json

import pandas as pd
import streamlit as st

from db import (
    DatabaseConnection,
    get_latest_predictions,
    get_prediction_runs,
    get_user_prediction_history,
    get_user_comments,
    get_anomaly_trends,
)


@st.cache_resource
def get_db_connection() -> DatabaseConnection:
    """Initialize and cache database connection."""
    return DatabaseConnection()


def display_model_page():
    """Main model visualization page."""
    st.title("Bot Detection Results")
    st.write("Anomaly detection results from the latest model run")

    try:
        db = get_db_connection()

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Latest Predictions", "User Details", "Run History", "Trends"]
        )

        with tab1:
            display_latest_predictions(db)

        with tab2:
            display_user_details(db)

        with tab3:
            display_run_history(db)

        with tab4:
            display_trends(db)

    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        st.info(
            "Make sure your database is running and environment variables are set:\n"
            "- POSTGRES_HOST\n"
            "- POSTGRES_PORT\n"
            "- POSTGRES_USER\n"
            "- POSTGRES_PASSWORD\n"
            "- POSTGRES_DB"
        )


def display_latest_predictions(db: DatabaseConnection):
    """Display latest prediction results."""
    st.header("Top Anomalous Users")

    # Get predictions
    predictions_df = get_latest_predictions(db, limit=100)

    if predictions_df.empty:
        st.warning("No predictions found. Run the pipeline first.")
        return

    # Display run info
    latest_run = predictions_df.iloc[0]
    st.info(
        f"**Model:** {latest_run['modelname']} ({latest_run['modelversion']})\n\n"
        f"**Run Time:** {latest_run['runtimestamp']}\n\n"
        f"**Anomalies Detected:** {len(predictions_df)}"
    )

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        limit = st.slider("Number of users to display", 10, 100, 50)
    with col2:
        sort_by = st.selectbox(
            "Sort by", ["Anomaly Score (Low to High)", "Account Age", "Comment Velocity"]
        )

    # Filter and sort
    display_df = predictions_df.head(limit).copy()

    # Parse feature values for display
    display_df["features"] = display_df["featurevalues"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    # Create display columns
    display_df["account_age_days"] = display_df["features"].apply(
        lambda x: x.get("account_age", 0)
    )
    display_df["comment_velocity"] = display_df["features"].apply(
        lambda x: x.get("comment_velocity", 0)
    )

    # Sort based on selection
    if sort_by == "Account Age":
        display_df = display_df.sort_values("account_age_days")
    elif sort_by == "Comment Velocity":
        display_df = display_df.sort_values("comment_velocity", ascending=False)

    # Display table
    st.dataframe(
        display_df[
            [
                "username",
                "anomalyscore",
                "accountcreatedate",
                "subcount",
                "videocount",
            ]
        ].rename(
            columns={
                "username": "Username",
                "anomalyscore": "Anomaly Score",
                "accountcreatedate": "Account Created",
                "subcount": "Subscribers",
                "videocount": "Videos",
            }
        ),
        use_container_width=True,
    )

    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name=f"anomalous_users_{latest_run['runtimestamp']}.csv",
        mime="text/csv",
    )


def display_user_details(db: DatabaseConnection):
    """Display detailed information for a specific user."""
    st.header("User Profile")

    # Get latest predictions for user selection
    predictions_df = get_latest_predictions(db, limit=100)

    if predictions_df.empty:
        st.warning("No predictions available.")
        return

    # User selection
    user_options = {
        f"{row['username']} ({row['userid']})": row["userid"]
        for _, row in predictions_df.iterrows()
    }

    selected_user_key = st.selectbox("Select a user", list(user_options.keys()))
    selected_user_id = user_options[selected_user_key]

    # Get user data
    user_row = predictions_df[predictions_df["userid"] == selected_user_id].iloc[0]
    user_history = get_user_prediction_history(db, selected_user_id)
    user_comments = get_user_comments(db, selected_user_id, limit=50)

    # Display user profile
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Anomaly Score", f"{user_row['anomalyscore']:.4f}")
    with col2:
        st.metric("Subscribers", user_row["subcount"] or "Unknown")
    with col3:
        st.metric("Videos", user_row["videocount"] or "Unknown")

    st.write(f"**Account Created:** {user_row['accountcreatedate']}")

    # Feature values
    st.subheader("Feature Values")
    features = json.loads(user_row["featurevalues"])
    feature_df = pd.DataFrame(
        [{"Feature": k, "Value": f"{v:.4f}"} for k, v in features.items()]
    )
    st.dataframe(feature_df, use_container_width=True)

    # Prediction history
    if not user_history.empty:
        st.subheader("Prediction History")
        st.line_chart(
            user_history.set_index("runtimestamp")["anomalyscore"],
            use_container_width=True,
        )

    # Recent comments
    if not user_comments.empty:
        st.subheader("Recent Comments")
        for _, comment in user_comments.head(10).iterrows():
            with st.expander(
                f"On '{comment['videotitle'][:50]}...' - {comment['publishdate']}"
            ):
                st.write(comment["commenttext"])
                st.caption(
                    f"Channel: {comment['channelname']} | Likes: {comment['likecount']}"
                )
    else:
        st.info("No comments found for this user.")


def display_run_history(db: DatabaseConnection):
    """Display history of prediction runs."""
    st.header("Prediction Run History")

    runs_df = get_prediction_runs(db, limit=20)

    if runs_df.empty:
        st.warning("No prediction runs found.")
        return

    # Display runs table
    st.dataframe(
        runs_df.rename(
            columns={
                "runtimestamp": "Run Time",
                "modelname": "Model",
                "modelversion": "Version",
                "anomalycount": "Anomalies",
                "avganomalyscore": "Avg Score",
                "minanomalyscore": "Min Score",
                "maxanomalyscore": "Max Score",
            }
        ),
        use_container_width=True,
    )

    # Chart of anomaly counts over time
    st.subheader("Anomalies Detected Over Time")
    st.bar_chart(
        runs_df.set_index("runtimestamp")["anomalycount"], use_container_width=True
    )


def display_trends(db: DatabaseConnection):
    """Display trends in anomaly detection."""
    st.header("Detection Trends")

    days = st.slider("Days to show", 7, 90, 30)

    trends_df = get_anomaly_trends(db, days=days)

    if trends_df.empty:
        st.warning("No trend data available.")
        return

    # Daily anomaly count
    st.subheader("Daily Anomaly Detections")
    st.line_chart(
        trends_df.set_index("date")["totalanomalies"], use_container_width=True
    )

    # Average anomaly score
    st.subheader("Average Anomaly Score")
    st.line_chart(trends_df.set_index("date")["avgscore"], use_container_width=True)

    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Runs", trends_df["runs"].sum())
    with col2:
        st.metric("Total Anomalies", trends_df["totalanomalies"].sum())
    with col3:
        st.metric("Avg Score", f"{trends_df['avgscore'].mean():.4f}")