library(sf)
geojson_df <- readRDS("geojson_df.rds")

geojson_df_2d <- st_zm(geojson_df, drop = TRUE, what = "ZM")
st_write(geojson_df_2d, "geojson_df.geojson", driver = "GeoJSON", delete_dsn = TRUE)

