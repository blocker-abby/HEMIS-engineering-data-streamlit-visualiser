import sqlite3
import os
import hashlib
import time
from contextlib import closing
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
SOURCE_COLUMN = "_source_table"


# ---------------------------------------------------------
# DATABASE HELPERS
# ---------------------------------------------------------

def quote_identifier(name):
    """Safely quote a SQLite table or column identifier."""
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


def database_connection(database_path):
    """
    Open the deployed database read-only. mode=ro prevents SQLite from
    silently creating an empty database if the path is wrong.
    """
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -10000")  # approximately 10 MB
    return connection


@st.cache_data(show_spinner="Reading database tables...", max_entries=4)
def list_tables(database_path, modified_time):
    del modified_time
    with closing(database_connection(database_path)) as connection:
        rows = connection.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """).fetchall()
    return [row[0] for row in rows]


@st.cache_data(show_spinner="Reading database structure...")
def get_table_columns(database_path, table_name, modified_time):
    del modified_time
    with closing(database_connection(database_path)) as connection:
        rows = connection.execute(
            f"PRAGMA table_info({quote_identifier(table_name)})"
        ).fetchall()
    # PRAGMA table_info: cid, name, type, notnull, default, pk
    return [(row[1], row[2] or "") for row in rows]


def find_measure_column_from_schema(database_path, table_name, columns):
    """
    Prefer recognised measure names. Otherwise inspect SQLite's stored type
    for the first non-null value, avoiding a full-table Pandas load.
    """
    names = [name for name, _declared_type in columns]
    candidates = [
        name for name in names if normalize_name(name) in MEASURE_NAMES
    ]
    if candidates:
        return candidates[0]

    with closing(database_connection(database_path)) as connection:
        for name in names:
            row = connection.execute(
                f"""
                SELECT typeof({quote_identifier(name)})
                FROM {quote_identifier(table_name)}
                WHERE {quote_identifier(name)} IS NOT NULL
                LIMIT 1
                """
            ).fetchone()
            if row and row[0] in ("integer", "real"):
                return name

    return None


def is_aggregate_sql(column_names):
    """
    SQL equivalent of the original pandas All/Total exclusion.
    Returns a predicate covering every available dimension in a table.
    """
    checks = [
        f"COALESCE(lower(trim(CAST({quote_identifier(name)} AS TEXT))), '') IN ('all', 'total')"
        for name in column_names
    ]
    return "(" + " OR ".join(checks) + ")" if checks else "0"


@st.cache_data(show_spinner="Inspecting database tables...", max_entries=8)
def build_table_metadata(database_path, table_names, modified_time):
    """
    Inspect schemas and count rows without transferring source tables into
    Pandas. Only compact metadata is returned to the Streamlit process.
    """
    metadata = {}
    common_dimensions = set(DIMENSIONS)

    with closing(database_connection(database_path)) as connection:
        for table_name in table_names:
            column_info = get_table_columns(
                database_path, table_name, modified_time
            )
            columns = [name for name, _declared_type in column_info]
            dimension_map = {
                dimension: find_actual_column(columns, dimension)
                for dimension in DIMENSIONS
            }

            common_dimensions &= {
                dimension for dimension, actual in dimension_map.items()
                if actual is not None
            }

            aggregate_columns = [
                actual for actual in dimension_map.values()
                if actual is not None
            ]
            aggregate_predicate = is_aggregate_sql(aggregate_columns)
            original, removed = connection.execute(
                f"""
                SELECT
                    COUNT(*),
                    COALESCE(SUM(CASE WHEN {aggregate_predicate}
                                      THEN 1 ELSE 0 END), 0)
                FROM {quote_identifier(table_name)}
                """
            ).fetchone()

            metadata[table_name] = {
                "columns": columns,
                "dimension_map": dimension_map,
                "measure": find_measure_column_from_schema(
                    database_path, table_name, column_info
                ),
                "original": int(original),
                "removed": int(removed),
                "rows": int(original - removed),
            }

    return metadata, sorted(common_dimensions)


def build_union_sql(table_names, metadata, measure_choices):
    """
    Build a UNION ALL over the selected source tables. Only the requested
    dimensions, selected measure and source-table label are projected.
    Aggregate All/Total rows are excluded inside SQLite.
    """
    selects = []
    params = []

    chosen_measures = dict(measure_choices)

    for table_name in table_names:
        info = metadata[table_name]
        dimension_map = info["dimension_map"]

        select_columns = []
        for dimension in DIMENSIONS:
            actual = dimension_map[dimension]
            if actual is None:
                select_columns.append(f"'{ '(Not available)' }' AS {quote_identifier(dimension)}")
            else:
                select_columns.append(
                    f"{quote_identifier(actual)} AS {quote_identifier(dimension)}"
                )

        measure_column = chosen_measures[table_name]
        select_columns.append(
            f"{quote_identifier(measure_column)} AS _raw_measure"
        )
        select_columns.append("? AS {source}".format(source=quote_identifier(SOURCE_COLUMN)))

        aggregate_columns = [
            actual for actual in dimension_map.values() if actual is not None
        ]
        aggregate_predicate = is_aggregate_sql(aggregate_columns)

        selects.append(
            f"""
            SELECT {", ".join(select_columns)}
            FROM {quote_identifier(table_name)}
            WHERE NOT {aggregate_predicate}
            """
        )
        params.append(table_name)

    return " UNION ALL ".join(selects), params


def missing_scalar(value):
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def sort_value(value):
    if missing_scalar(value):
        return (2, "")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    return (1, str(value).casefold())


def display_value(value):
    return "(Missing)" if missing_scalar(value) else str(value)


@st.cache_data(show_spinner="Loading filter options...", max_entries=12)
@st.cache_data(show_spinner="Loading filter options...", max_entries=24)
def get_distinct_filter_values(
    database_path, table_names, measure_choices, modified_time, dimension
):
    """
    Fetch only distinct filter values, not full table contents.
    NULL values are retained so users can explicitly include '(Missing)'.
    """
    metadata, _ = build_table_metadata(
        database_path, table_names, modified_time
    )
    selects = []
    for table_name in table_names:
        info = metadata[table_name]
        actual = info["dimension_map"][dimension]
        aggregate_columns = [
            name for name in info["dimension_map"].values() if name is not None
        ]
        predicate = is_aggregate_sql(aggregate_columns)
        selects.append(
            f"""
            SELECT {quote_identifier(actual)} AS value
            FROM {quote_identifier(table_name)}
            WHERE NOT {predicate}
            """
        )

    query = "SELECT DISTINCT value FROM (" + " UNION ALL ".join(selects) + ")"
    with closing(database_connection(database_path)) as connection:
        values = [row[0] for row in connection.execute(query).fetchall()]

    # Deduplicate values across SQLite tables while preserving distinctions
    # between strings and numeric values.
    unique = []
    seen = set()
    for value in values:
        key = ("__NULL__",) if value is None else (type(value).__name__, value)
        try:
            if key not in seen:
                seen.add(key)
                unique.append(value)
        except TypeError:
            if value not in unique:
                unique.append(value)

    return sorted(unique, key=sort_value)


def register_numeric_function(connection):
    """Convert values such as '1,234' to numbers; invalid values become NULL."""
    def try_numeric(value):
        if value is None:
            return None
        try:
            text = str(value).strip().replace(",", "")
            if not text:
                return None
            number = float(text)
            if number != number or number in (float("inf"), float("-inf")):
                return None
            return number
        except (TypeError, ValueError, OverflowError):
            return None

    connection.create_function("TRY_NUMERIC", 1, try_numeric, deterministic=True)


def make_filter_predicate(selected_values_by_dimension):
    conditions = []
    params = []

    for dimension, selected_values in selected_values_by_dimension.items():
        non_missing = [value for value in selected_values if not missing_scalar(value)]
        include_missing = any(missing_scalar(value) for value in selected_values)

        parts = []
        if non_missing:
            placeholders = ",".join("?" for _ in non_missing)
            parts.append(f"{quote_identifier(dimension)} IN ({placeholders})")
            params.extend(non_missing)
        if include_missing:
            parts.append(f"{quote_identifier(dimension)} IS NULL")

        if not parts:
            conditions.append("0")
        else:
            conditions.append("(" + " OR ".join(parts) + ")")

    return (" AND ".join(conditions) if conditions else "1"), params


@st.cache_data(show_spinner="Applying filters and aggregating...", max_entries=24)
def filter_and_aggregate_sql(
    database_path,
    table_names,
    measure_choices,
    selected_values_by_dimension,
    group_columns,
    modified_time,
):
    """
    Perform filtering, numeric conversion, row counting and GROUP BY in
    SQLite. Pandas receives only the aggregated result.
    """
    metadata, _ = build_table_metadata(
        database_path, table_names, modified_time
    )
    union_sql, union_params = build_union_sql(
        table_names, metadata, measure_choices
    )
    filter_sql, filter_params = make_filter_predicate(
        selected_values_by_dimension
    )

    group_sql = ", ".join(quote_identifier(column) for column in group_columns)

    # MATERIALIZED lets the filtered rows be reused by the metrics and
    # aggregation subqueries without loading them into Python memory.
    query = f"""
        WITH combined AS (
            {union_sql}
        ),
        filtered AS MATERIALIZED (
            SELECT *,
                   TRY_NUMERIC(_raw_measure) AS _numeric_measure
            FROM combined
            WHERE {filter_sql}
        ),
        stats AS (
            SELECT
                COUNT(_numeric_measure) AS valid_rows,
                COALESCE(SUM(_numeric_measure), 0) AS total_students,
                COUNT(*) - COUNT(_numeric_measure) AS invalid_rows
            FROM filtered
        ),
        aggregated AS (
            SELECT
                {group_sql},
                SUM(_numeric_measure) AS students
            FROM filtered
            WHERE _numeric_measure IS NOT NULL
            GROUP BY {group_sql}
        )
        SELECT aggregated.*, stats.valid_rows, stats.total_students,
               stats.invalid_rows
        FROM stats
        LEFT JOIN aggregated ON 1 = 1
    """

    with closing(database_connection(database_path)) as connection:
        register_numeric_function(connection)
        rows = connection.execute(
            query, union_params + filter_params
        ).fetchall()

    # The result schema is predictable: group columns, students, then metrics.
    result_columns = (
        list(group_columns)
        + ["students", "_valid_rows", "_total_students", "_invalid_rows"]
    )
    result = pd.DataFrame(rows, columns=result_columns)

    if result.empty:
        return (
            pd.DataFrame(columns=list(group_columns) + ["students"]),
            0, 0.0, 0
        )

    valid_rows = int(result["_valid_rows"].iloc[0] or 0)
    total_students = float(result["_total_students"].iloc[0] or 0)
    invalid_rows = int(result["_invalid_rows"].iloc[0] or 0)

    if valid_rows == 0:
        aggregated = pd.DataFrame(columns=list(group_columns) + ["students"])
    else:
        aggregated = result.loc[
            result["students"].notna(), list(group_columns) + ["students"]
        ].copy()

    for column in group_columns:
        if column in aggregated.columns:
            aggregated[column] = aggregated[column].where(
                aggregated[column].notna(), "(Missing)"
            )

    return aggregated, valid_rows, total_students, invalid_rows


def set_filter_checkbox_values(keys, checked):
    for key in keys:
        st.session_state[key] = checked


# ---------------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------------

st.title("Student Data Explorer")


# ---------------------------------------------------------
# DATABASE CONFIGURATION
# ---------------------------------------------------------

with st.sidebar:
    if "database_path" not in st.session_state:
        st.session_state["database_path"] = "Sqlite3 - Copy.db"

    database_path = "Sqlite3 - Copy.db"
    path = Path(database_path).expanduser()

    try:
        resolved_path = str(path.resolve())
        if not path.is_file():
            st.error(f"Database file not found: {resolved_path}")
            st.stop()

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

    try:
        table_metadata, common_dimensions = build_table_metadata(
            resolved_path, tuple(selected_tables), modified_time
        )
    except (sqlite3.Error, OSError, KeyError, ValueError) as error:
        st.error(f"Could not inspect database tables: {error}")
        st.stop()

    measure_choices = []
    for selected_table in selected_tables:
        measure_options = table_metadata[selected_table]["columns"]
        detected_measure = table_metadata[selected_table]["measure"]

        if not measure_options or detected_measure is None:
            st.error(
                f"Could not identify a numeric student-count column in "
                f"'{selected_table}'."
            )
            st.stop()

        default_index = measure_options.index(detected_measure)
        chosen_measure = st.selectbox(
            f"Student-count column — {selected_table}",
            measure_options,
            index=default_index,
            key=f"measure_{hashlib.md5(selected_table.encode()).hexdigest()[:8]}",
        )
        measure_choices.append((selected_table, chosen_measure))


# ---------------------------------------------------------
# VALIDATE SELECTED TABLES AND DIMENSIONS
# ---------------------------------------------------------

if not common_dimensions:
    st.error(
        "The selected tables do not share any of the expected dimensions "
        "(year, cesm, institute, degree, race, gender)."
    )
    st.stop()

table_stats = {
    table_name: {
        "original": table_metadata[table_name]["original"],
        "removed": table_metadata[table_name]["removed"],
        "rows": table_metadata[table_name]["rows"],
        "measure": dict(measure_choices)[table_name],
    }
    for table_name in selected_tables
}


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

distinct_values_by_dimension = {}
distinct_start = time.perf_counter()

try:
    for dimension in common_dimensions:
        distinct_values_by_dimension[dimension] = get_distinct_filter_values(
            resolved_path,
            tuple(selected_tables),
            tuple(measure_choices),
            modified_time,
            dimension,
        )
except (sqlite3.Error, OSError, KeyError, ValueError) as error:
    st.error(f"Could not retrieve filter values from SQLite: {error}")
    st.stop()

distinct_values_seconds = time.perf_counter() - distinct_start

with st.form(key=f"filters_form_{filter_scope}"):
    pending_values_by_dimension = {}
    filter_tabs = st.tabs(common_dimensions)

    for dimension, tab in zip(common_dimensions, filter_tabs):
        distinct_values = distinct_values_by_dimension[dimension]

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
# FILTER AND AGGREGATE IN SQLITE
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
try:
    aggregated, filtered_row_count, total_students, invalid_measure_rows = (
        filter_and_aggregate_sql(
            resolved_path,
            tuple(selected_tables),
            tuple(measure_choices),
            selected_values_by_dimension,
            tuple(group_columns),
            modified_time,
        )
    )
except (sqlite3.Error, OSError, KeyError, ValueError, TypeError) as error:
    st.error(f"Could not filter and aggregate the database: {error}")
    st.stop()

filter_aggregation_seconds = time.perf_counter() - group_start

if aggregated.empty:
    if invalid_measure_rows:
        st.warning(
            "The selected filters match rows, but their student-count values "
            "are all missing or non-numeric."
        )
    else:
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
            "SQLite schema inspection and row counts": "cached/metadata only",
            "Distinct filter values": round(distinct_values_seconds, 3),
            "SQLite filtering and aggregation": round(filter_aggregation_seconds, 3),
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
