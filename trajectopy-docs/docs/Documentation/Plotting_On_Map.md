# Plotting Trajectories on a Map

To plot trajectories on a map, several requirements must be met:

- The trajectory must have valid EPSG information.
- The plotting backend must be set to `plotly`.
- The `scatter_plot_on_map` option must be enabled.
- For `scatter_mapbox_style` other than `open-street-map`, a Mapbox access token (`scatter_mapbox_token`) must be provided. The mapbox token can be obtained after free registration at [Mapbox](https://www.mapbox.com/).