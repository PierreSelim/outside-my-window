from __future__ import annotations

from dash import Dash, Input, Output, dcc, html

from src.data_loader import ASSETS_DIR, clear_cache
from src.pages import components, map_page, station_page

# suppress_callback_exceptions: callbacks reference component IDs that only exist
# in page layouts rendered dynamically — Dash would otherwise raise on startup.
# assets_folder is explicit so the stylesheet is still found from inside a PyInstaller bundle,
# where the module path Dash would infer does not exist on disk.
app = Dash(
    __name__,
    title="Outside My Window",
    assets_folder=str(ASSETS_DIR),
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        html.Nav(
            className="app-navbar",
            children=[
                html.Span("Outside My Window", className="app-navbar-title"),
                html.Span("Météo-France daily climate records", className="app-navbar-subtitle"),
            ],
        ),
        dcc.Loading(type="circle", children=html.Div(id="page-content")),
    ]
)

components.register_callbacks(app)
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


@app.server.route("/api/refresh", methods=["POST"])
def refresh_data() -> tuple[str, int]:
    """Force the next page load to re-fetch LATEST-period data instead of serving cached data."""
    clear_cache()
    return "", 204


if __name__ == "__main__":
    app.run(debug=True)
