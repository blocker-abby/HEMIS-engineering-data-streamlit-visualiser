import sqlite3
import os
import hashlib
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

app_start_time = time.perf_counter()

st.set_page_config(
    page_title="Student Data Explorer",
    page_icon="📊",
    layout="wide",
)

DIMENSIONS = ["year", "cesm", "institute", "degree", "race", "gender"]
MEASURE_NAMES = {
    "students", "student", "number of students", "student count",
    "enrolments", "enrolment", "enrollment", "enrollments",
    "number of enrolments", "number of enrollments", "count",
}
VALUE_COLUMN = "_students_numeric"
SOURCE_COLUMN = "_source_table"


# ---------------------------------------------------------
# DATABASE HELPERS
# ---------------------------------------------------------

def quote_identifier(name):
    return '"' + str(name).replace('"', '""') + '"'


def normalize_name(name):
    return " ".join(
        str(name).strip().casefold().replace("_", " ").split()
    )


def find_actual_column(columns, requested_name):
    target = normalize_name(requested_name)
    for column in columns:
        if normalize_name(column) == target:
            return column
    return None


@st.cache_data(show_spinner="Reading database tables...")
def list_tables(database_path, modified_time):
    del modified_time
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()
    return [row[0] for row in rows]


@st.cache_data(show_spinner="Loading database table...")
def load_table(database_path, table_name, modified_time):
    del modified_time
    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(
            f"SELECT * FROM {quote_identifier(table_name)}",
            connection,
        )


def is_aggregate_value(series):
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .isin(["all", "total"])
    )


def sort_value(value):
    if pd.isna(value):
        return (2, "")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    return (1, str(value).casefold())


def display_value(value):
    return "(Missing)" if pd.isna(value) else str(value)


def set_filter_checkbox_values(keys, checked):
    for key in keys:
        st.session_state[key] = checked


def find_measure_column(data):
    candidates = [
        column for column in data.columns
        if normalize_name(column) in MEASURE_NAMES
    ]
    if candidates:
        return candidates[0]

    numeric_columns = [
        column for column in data.columns
        if pd.api.types.is_numeric_dtype(data[column])
    ]
    return numeric_columns[0] if numeric_columns else None


@st.cache_data(show_spinner="Preparing student data...", max_entries=16)
def prepare_selected_tables(database_path, table_names, modified_time, measure_choices):
    frames = []
    table_stats = {}
    common_dimensions = set(DIMENSIONS)

    for table_name in table_names:
        data = load_table(database_path, table_name, modified_time).copy()
        if data.empty:
            table_stats[table_name] = {
                "original": 0, "removed": 0, "invalid": 0,
                "measure": None, "rows": 0,
            }
            continue

        dimension_map = {
            dimension: find_actual_column(data.columns, dimension)
            for dimension in DIMENSIONS
        }
        common_dimensions &= {
            dimension for dimension, actual in dimension_map.items()
            if actual is not None
        }

        aggregate_mask = pd.Series(False, index=data.index)
        for actual_column in dimension_map.values():
            if actual_column is not None:
                aggregate_mask |= is_aggregate_value(data[actual_column])

        original_count = len(data)
        data = data.loc[~aggregate_mask].copy()
        removed_count = original_count - len(data)

        measure_column = dict(measure_choices).get(table_name)
        if measure_column not in data.columns:
            raise ValueError(
                f"Could not identify a numeric student-count column in table "
                f"'{table_name}'. Columns found: {list(data.columns)}"
            )

        data[VALUE_COLUMN] = pd.to_numeric(
            data[measure_column]
            .astype("string")
            .str.replace(",", "", regex=False),
            errors="coerce",
        )

        # Standardise dimension names so the selected tables can be combined.
        standardised = pd.DataFrame(index=data.index)
        for dimension in DIMENSIONS:
            actual = dimension_map[dimension]
            if actual is None:
                standardised[dimension] = "(Not available)"
            else:
                standardised[dimension] = data[actual]

        standardised[VALUE_COLUMN] = data[VALUE_COLUMN]
        standardised[SOURCE_COLUMN] = table_name
        frames.append(standardised)

        table_stats[table_name] = {
            "original": original_count,
            "removed": removed_count,
            "invalid": int(data[VALUE_COLUMN].isna().sum()),
            "measure": measure_column,
            "rows": len(data),
        }

    if not frames:
        return pd.DataFrame(columns=DIMENSIONS + [VALUE_COLUMN, SOURCE_COLUMN]), table_stats, []

    combined = pd.concat(frames, ignore_index=True)
    return combined, table_stats, sorted(common_dimensions)


@st.cache_data(show_spinner="Applying filters and aggregating...", max_entries=32)
def filter_and_aggregate(
    combined_data,
    selected_values_by_dimension,
    group_columns,
):
    data = combined_data
    combined_mask = pd.Series(True, index=data.index)

    for dimension, selected_values in selected_values_by_dimension.items():
        non_missing = [value for value in selected_values if not pd.isna(value)]
        include_missing = any(pd.isna(value) for value in selected_values)

        dimension_mask = data[dimension].isin(non_missing)
        if include_missing:
            dimension_mask |= data[dimension].isna()
        combined_mask &= dimension_mask

    filtered = data.loc[combined_mask]
    invalid_count = int(filtered[VALUE_COLUMN].isna().sum())
    valid = filtered.dropna(subset=[VALUE_COLUMN])

    if valid.empty:
        return pd.DataFrame(columns=list(group_columns) + ["students"]), 0, 0.0, invalid_count

    aggregated = (
        valid.groupby(list(group_columns), dropna=False, as_index=False)[VALUE_COLUMN]
        .sum()
        .rename(columns={VALUE_COLUMN: "students"})
    )

    for column in group_columns:
        aggregated[column] = (
            aggregated[column]
            .astype("object")
            .where(aggregated[column].notna(), "(Missing)")
        )

    return (
        aggregated,
        int(len(valid)),
        float(valid[VALUE_COLUMN].sum()),
        invalid_count,
    )


# ---------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------

st.title("Student Data Explorer")


# ---------------------------------------------------------
# DATABASE SELECTION
# ---------------------------------------------------------

with st.sidebar:

    if "database_path" not in st.session_state:
        st.session_state["database_path"] = "Sqlite3 - Copy.db"

    database_path = "Sqlite3 - Copy.db"

    try:
        resolved_path = str(path.resolve())
        modified_time = path.stat().st_mtime
        tables = list_tables(resolved_path, modified_time)
    except (sqlite3.Error, OSError) as error:
        st.error(f"Could not open the database: {error}")
        st.stop()

    if not tables:
        st.error("No tables were found in this database.")
        st.stop()

    selected_tables = st.multiselect(
        "Database tables",
        tables,
        default=tables[:2] if len(tables) >= 2 else tables,
    )

    if st.button("Refresh database", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if not selected_tables:
        st.info("Select at least one database table.")
        st.stop()

    measure_choices = []
    for selected_table in selected_tables:
        table_data = load_table(
            resolved_path, selected_table, path.stat().st_mtime
        )
        detected_measure = find_measure_column(table_data)
        if detected_measure is None:
            st.error(
                f"Could not identify a numeric student-count column in "
                f"'{selected_table}'."
            )
            st.stop()

        measure_options = list(table_data.columns)
        default_index = measure_options.index(detected_measure)
        chosen_measure = st.selectbox(
            f"Student-count column — {selected_table}",
            measure_options,
            index=default_index,
            key=f"measure_{hashlib.md5(selected_table.encode()).hexdigest()[:8]}",
        )
        measure_choices.append((selected_table, chosen_measure))

# ---------------------------------------------------------
# LOAD AND PREPARE SELECTED TABLES
# ---------------------------------------------------------

database_load_start = time.perf_counter()
try:
    combined_df, table_stats, common_dimensions = prepare_selected_tables(
        resolved_path,
        tuple(selected_tables),
        path.stat().st_mtime,
        tuple(measure_choices),
    )
except (sqlite3.Error, OSError, pd.errors.DatabaseError, KeyError, ValueError, TypeError) as error:
    st.error(f"Could not prepare the selected tables: {error}")
    st.stop()

database_load_seconds = time.perf_counter() - database_load_start

if combined_df.empty:
    st.warning("The selected tables contain no rows.")
    st.stop()

if not common_dimensions:
    st.error(
        "The selected tables do not share any of the expected dimensions "
        "(year, cesm, institute, degree, race, gender)."
    )
    st.stop()


# ---------------------------------------------------------
# CHART SETTINGS
# ---------------------------------------------------------

with st.sidebar:
    st.header("Chart settings")

    x_dimension = st.selectbox(
        "X-axis",
        common_dimensions,
        index=0,
    )

    chart_type = st.selectbox(
        "Chart type",
        ["Bar chart", "Line chart", "Area chart", "Pie chart"],
    )

    series_options = ["None", "Source table"] + [
        dimension for dimension in common_dimensions
        if dimension != x_dimension
    ]

    series_dimension = st.selectbox(
        "Break down by (optional)",
        series_options,
    )

    sort_option = st.selectbox(
        "Sort categories by",
        [
            "Dimension value",
            "Students (descending)",
            "Students (ascending)",
        ],
    )

    max_categories = 0
    if chart_type in ["Bar chart", "Line chart", "Area chart"]:
        max_categories = st.number_input(
            "Maximum X-axis categories (0 = all)",
            min_value=0,
            max_value=5000,
            value=50,
            step=10,
        )

    max_series = 0
    if series_dimension != "None" and chart_type != "Pie chart":
        max_series = st.number_input(
            "Maximum series (0 = all)",
            min_value=0,
            max_value=1000,
            value=20,
            step=5,
        )

    show_performance = st.checkbox("Show performance timings", value=False)


# ---------------------------------------------------------
# FILTER INTERFACE: BATCHED FORM SUBMISSION
# ---------------------------------------------------------

st.subheader("Filters")

filter_scope = hashlib.md5(
    f"{resolved_path}::{','.join(selected_tables)}".encode("utf-8")
).hexdigest()[:10]
applied_filters_key = f"applied_filters_{filter_scope}"

selected_values_by_dimension = {}

with st.form(key=f"filters_form_{filter_scope}"):
    pending_values_by_dimension = {}
    filter_tabs = st.tabs(common_dimensions)

    for dimension, tab in zip(common_dimensions, filter_tabs):
        distinct_values = combined_df[dimension].drop_duplicates().tolist()
        distinct_values = sorted(distinct_values, key=sort_value)

        checkbox_keys = [
            f"include_{filter_scope}_{dimension}_{i}"
            for i in range(len(distinct_values))
        ]

        with tab:
            st.write(f"**{dimension.title()}**")

            select_all_col, select_none_col = st.columns(2)
            with select_all_col:
                st.form_submit_button(
                    "Select all",
                    key=f"select_all_{filter_scope}_{dimension}",
                    on_click=set_filter_checkbox_values,
                    args=(checkbox_keys, True),
                    use_container_width=True,
                )
            with select_none_col:
                st.form_submit_button(
                    "Select none",
                    key=f"select_none_{filter_scope}_{dimension}",
                    on_click=set_filter_checkbox_values,
                    args=(checkbox_keys, False),
                    use_container_width=True,
                )

            selected_values = []
            number_of_columns = 3
            rows_per_column = (
                len(distinct_values) + number_of_columns - 1
            ) // number_of_columns
            checkbox_columns = st.columns(number_of_columns)

            for column_index, checkbox_column in enumerate(checkbox_columns):
                start = column_index * rows_per_column
                end = min(start + rows_per_column, len(distinct_values))

                with checkbox_column:
                    for i in range(start, end):
                        value = distinct_values[i]
                        key = checkbox_keys[i]

                        if key not in st.session_state:
                            st.session_state[key] = False

                        checked = st.checkbox(display_value(value), key=key)
                        if checked:
                            selected_values.append(value)

            pending_values_by_dimension[dimension] = selected_values

    submitted = st.form_submit_button(
        "Apply filters and update chart",
        type="primary",
        use_container_width=True,
    )

if submitted:
    st.session_state[applied_filters_key] = pending_values_by_dimension

if applied_filters_key not in st.session_state:
    st.info(
        "Choose your filter values above and click **Apply filters and "
        "update chart** to generate the chart."
    )
    st.stop()

selected_values_by_dimension = st.session_state[applied_filters_key]

empty_dimensions = [
    dimension for dimension, selected_values in selected_values_by_dimension.items()
    if not selected_values
]
if empty_dimensions:
    st.info(
        "Select at least one value in each dimension to display results. "
        "No values are selected for: " + ", ".join(empty_dimensions)
    )
    st.stop()


# ---------------------------------------------------------
# FILTER AND AGGREGATE
# ---------------------------------------------------------

x_column = x_dimension
if series_dimension == "Source table":
    series_column = SOURCE_COLUMN
elif series_dimension != "None":
    series_column = series_dimension
else:
    series_column = None

group_columns = [x_column]
if series_column and series_column != x_column:
    group_columns.append(series_column)

group_start = time.perf_counter()
aggregated, filtered_row_count, total_students, invalid_measure_rows = filter_and_aggregate(
    combined_df,
    selected_values_by_dimension,
    tuple(group_columns),
)
filter_aggregation_seconds = time.perf_counter() - group_start

if aggregated.empty:
    st.warning("No numeric student values match the selected filter values.")
    st.stop()


# ---------------------------------------------------------
# SORT CATEGORIES
# ---------------------------------------------------------

if series_column and series_column != x_column:
    category_totals = aggregated.groupby(
        x_column, dropna=False
    )["students"].sum()
else:
    category_totals = aggregated.set_index(x_column)["students"]

if sort_option == "Students (descending)":
    category_order = category_totals.sort_values(ascending=False).index.tolist()
elif sort_option == "Students (ascending)":
    category_order = category_totals.sort_values(ascending=True).index.tolist()
else:
    category_order = sorted(category_totals.index.tolist(), key=sort_value)

if max_categories and len(category_order) > max_categories:
    category_order = category_order[:int(max_categories)]
    chart_df = aggregated[aggregated[x_column].isin(category_order)].copy()
else:
    chart_df = aggregated.copy()

series_order = None
if (
    max_series
    and series_column
    and series_column != x_column
    and chart_type != "Pie chart"
):
    series_totals = (
        chart_df.groupby(series_column, dropna=False)["students"]
        .sum()
        .sort_values(ascending=False)
    )
    series_order = series_totals.head(int(max_series)).index.tolist()
    chart_df = chart_df[chart_df[series_column].isin(series_order)].copy()

chart_df["_x_order"] = chart_df[x_column].map(
    {value: index for index, value in enumerate(category_order)}
)
chart_df = chart_df.sort_values("_x_order")


# ---------------------------------------------------------
# SUMMARY METRICS
# ---------------------------------------------------------

st.subheader("Results")

metric1, metric2, metric3 = st.columns(3)
metric1.metric("Rows after filters", f"{filtered_row_count:,}")
metric2.metric("Total students", f"{total_students:,.0f}")
metric3.metric("Categories shown", f"{chart_df[x_column].nunique():,}")

removed_rows = sum(stats["removed"] for stats in table_stats.values())
if removed_rows:
    st.caption(
        f"Excluded {removed_rows:,} source rows labelled All or Total "
        "in at least one dimension to prevent aggregate double-counting."
    )

if invalid_measure_rows:
    st.caption(
        f"Excluded {invalid_measure_rows:,} filtered rows with missing or "
        "non-numeric student values."
    )


# ---------------------------------------------------------
# CREATE CHART
# ---------------------------------------------------------

chart_build_start = time.perf_counter()
plot_labels = {
    x_column: x_dimension.title(),
    "students": "Students",
}
if series_column == SOURCE_COLUMN:
    plot_labels[SOURCE_COLUMN] = "Source table"

color_column = (
    series_column if series_column and series_column != x_column else None
)

if chart_type == "Bar chart":
    fig = px.bar(
        chart_df,
        x=x_column,
        y="students",
        color=color_column,
        barmode="group",
        category_orders={x_column: category_order},
        title=f"Students by {x_dimension}",
        labels=plot_labels,
    )
elif chart_type == "Line chart":
    fig = px.line(
        chart_df,
        x=x_column,
        y="students",
        color=color_column,
        markers=True,
        category_orders={x_column: category_order},
        title=f"Students by {x_dimension}",
        labels=plot_labels,
    )
elif chart_type == "Area chart":
    fig = px.area(
        chart_df,
        x=x_column,
        y="students",
        color=color_column,
        category_orders={x_column: category_order},
        title=f"Students by {x_dimension}",
        labels=plot_labels,
    )
else:
    pie_df = (
        chart_df.groupby(x_column, as_index=False, dropna=False)["students"]
        .sum()
    )
    fig = px.pie(
        pie_df,
        names=x_column,
        values="students",
        title=f"Share of students by {x_dimension}",
    )

fig.update_layout(
    height=560,
    margin=dict(l=20, r=20, t=70, b=80),
)
fig.update_yaxes(title_text="Students")
fig.update_xaxes(tickangle=-35)

chart_build_seconds = time.perf_counter() - chart_build_start
render_start = time.perf_counter()
st.plotly_chart(fig, use_container_width=True)
render_seconds = time.perf_counter() - render_start


# ---------------------------------------------------------
# AGGREGATED DATA TABLE AND CSV EXPORT
# ---------------------------------------------------------

with st.expander("View aggregated data"):
    if max_categories or max_series:
        st.caption(
            "The chart may be limited to the selected number of categories "
            "or series. The table and CSV contain the full aggregation."
        )

    st.dataframe(
        aggregated,
        use_container_width=True,
        hide_index=True,
    )

    csv = aggregated.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download aggregated data as CSV",
        data=csv,
        file_name="student_summary.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# PERFORMANCE DIAGNOSTICS
# ---------------------------------------------------------

if show_performance:
    with st.expander("Performance timings", expanded=True):
        st.write({
            "SQLite table load and preparation": round(database_load_seconds, 3),
            "Filtering and aggregation": round(filter_aggregation_seconds, 3),
            "Plotly figure construction": round(chart_build_seconds, 3),
            "Chart rendering call (server-side)": round(render_seconds, 3),
            "Total app execution": round(time.perf_counter() - app_start_time, 3),
        })


# ---------------------------------------------------------
# DATABASE DETAILS
# ---------------------------------------------------------

with st.expander("Database details"):
    st.write(f"**Database:** {resolved_path}")
    st.write(f"**Tables:** {', '.join(selected_tables)}")

    for table_name in selected_tables:
        stats = table_stats[table_name]
        st.write(f"**{table_name}**")
        st.write(f"Rows in original table: {stats['original']:,}")
        st.write(f"Rows after aggregate-row removal: {stats['rows']:,}")
        st.write(f"Student-count column: {stats['measure']}")

    st.write(f"**Common dimensions:** {', '.join(common_dimensions)}")
