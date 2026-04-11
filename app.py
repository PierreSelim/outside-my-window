from __future__ import annotations

from dash import Dash, Input, Output, dcc, html

from src.pages import map_page, station_page

# suppress_callback_exceptions: callbacks reference component IDs that only exist
# in page layouts rendered dynamically — Dash would otherwise raise on startup.
app = Dash(__name__, title="Outside My Window", suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id="url"),
    html.Nav(
        className="app-navbar",
        children=[
            html.Span("Outside My Window", className="app-navbar-title"),
            html.Span("Météo France · Données climatologiques quotidiennes", className="app-navbar-subtitle"),
        ],
    ),
    dcc.Loading(type="circle", children=html.Div(id="page-content")),
])

map_page.register_callbacks(app)
station_page.register_callbacks(app)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("url", "search"),
)
def render_page(pathname: str, search: str) -> html.Div:
    if pathname == "/station":
        return station_page.layout(search or "")
    return map_page.layout()


if __name__ == "__main__":
    app.run(debug=True)
