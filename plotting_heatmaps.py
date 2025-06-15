
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.affinity import rotate
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data"
FIGURE_PATH = BASE_DIR / "figs"


LIE_MAPPING = {
    'green': 'Green',
    'fairway': 'Fairway',
    'bunker': 'Sand',
    'tee': 'Tee',
    'water': 'Recovery',
    'hazard': 'Recovery'
}


def get_sg_df():
    return pd.read_csv(DATA_PATH / "sg_baseline.csv").set_index("Dist")


def get_get_gdf():
    return gpd.read_file(DATA_PATH / "geojson_df.geojson")


def preprocess_gdf(gdf, course):
    gdf = gdf[gdf['course_name_raw'] == course]
    gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid & ~
              gdf.geometry.is_empty].copy()
    gdf = gdf.set_crs(None, allow_override=True)
    gdf['hole_num'] = gdf['hole_num'].astype(str)
    return gdf


def get_zones(hole_gdf):
    zones = {}
    for key, label in LIE_MAPPING.items():
        polys = hole_gdf[hole_gdf['course_element'] == key]
        if not polys.empty:
            unioned = polys.geometry.union_all()
            if not unioned.is_empty:
                zones[label] = unioned
    return zones


def rotate_hole(hole_gdf):
    tee = hole_gdf[hole_gdf['course_element']
                   == 'tee'].geometry.union_all().centroid
    green = hole_gdf[hole_gdf['course_element']
                     == 'green'].geometry.union_all().centroid
    angle = -np.degrees(np.arctan2(green.y - tee.y, green.x - tee.x)) + 90
    hole_gdf = hole_gdf.copy()
    hole_gdf['geometry'] = hole_gdf.geometry.apply(
        lambda geom: rotate(geom, angle, origin=tee))
    return hole_gdf


def plot_course_elements(ax, hole_gdf):
    face_colours = hole_gdf['color'].fillna('#cccccc')
    hole_gdf.plot(ax=ax, facecolor=face_colours,
                  edgecolor='black', linewidth=0.5, alpha=0.6)
    return get_zones(hole_gdf)


def get_custom_viridis_with_adjusted_integers(vmin, vmax, adjust_factor=0.5, mode='lighten', n_continuous=256):
    base = plt.get_cmap('viridis')
    continuous = base(np.linspace(0, 1, n_continuous))
    for val in range(int(vmin), int(vmax) + 1):
        idx = int((val - vmin) / (vmax - vmin) * (n_continuous - 1))
        rgb = continuous[idx, :3]
        adjusted = rgb + \
            (1 - rgb) * adjust_factor if mode == 'lighten' else rgb * \
            (1 - adjust_factor)
        continuous[idx, :3] = np.clip(adjusted, 0, 1)
    cmap = ListedColormap(continuous)
    norm = BoundaryNorm(np.linspace(vmin, vmax, n_continuous), ncolors=cmap.N)
    return cmap, norm


def generate_sg_grid_filtered(hole_gdf, sg_df, sg_step, res):
    minx, miny, maxx, maxy = hole_gdf.total_bounds
    aspect = (maxy - miny) / (maxx - minx)
    x_coords = np.linspace(minx, maxx, res)
    y_coords = np.linspace(miny, maxy, int(res * aspect))
    grid = gpd.GeoDataFrame(geometry=[Point(x, y)
                            for y in y_coords for x in x_coords], crs=None)

    zones = get_zones(hole_gdf)
    if not zones:
        return None
    green_poly = zones.get('Green')
    if green_poly is None or green_poly.is_empty:
        return None
    hole_point = green_poly.centroid

    grid['lie'] = grid['geometry'].apply(lambda pt: next(
        (lie for lie, poly in zones.items() if poly.contains(pt) and lie in sg_df.columns), None))
    grid = grid[grid['lie'].notna()]
    if grid.empty:
        return None

    grid['dist'] = grid.geometry.distance(hole_point) / 0.9144
    grid['expected'] = grid.apply(lambda r: sg_df.at[int(round(r['dist'])), r['lie']] if int(
        round(r['dist'])) in sg_df.index else np.nan, axis=1)
    grid = grid.dropna(subset=['expected'])
    if grid.empty:
        return None

    vmin, vmax = grid['expected'].min(), grid['expected'].max()
    bounds = np.arange(np.floor(vmin / sg_step) * sg_step,
                       np.ceil(vmax / sg_step) * sg_step + sg_step, sg_step)
    norm = BoundaryNorm(bounds, ncolors=256)
    return grid, norm


def plot_sg_overlay(ax, grid, sg_step, adjust_factor=0.5, mode='lighten'):
    vmin_global, vmax_global = 1, 6
    vmax_local = np.ceil(grid['expected'].max())

    cmap, _ = get_custom_viridis_with_adjusted_integers(
        vmin_global, vmax_global, adjust_factor, mode)
    norm = plt.Normalize(vmin_global, vmax_global)

    grid.plot(ax=ax, column='expected', cmap=cmap,
              markersize=1, alpha=0.5, norm=norm)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Expected Strokes')
    # show integers only
    cbar.set_ticks(np.arange(vmin_global, vmax_local + 1))
    cbar.ax.invert_yaxis()
    # limit display to [1, local_max]
    cbar.ax.set_ylim(vmax_local, vmin_global)


def plot_single_hole(hole, course='erin_hills', plot_sg=False, sg_step=0.1, res=300, gdf=None, adjust_factor=0.5, mode='lighten', show=True):
    if gdf is None:
        gdf = get_get_gdf()
    gdf = preprocess_gdf(gdf, course)
    sg_df = get_sg_df() if plot_sg else None

    hole = str(hole)
    hole_gdf = gdf[gdf['hole_num'] == hole]
    if hole_gdf.empty:
        print(f"Hole {hole} not found in course {course}")
        return

    hole_gdf = rotate_hole(hole_gdf)
    minx, miny, maxx, maxy = hole_gdf.total_bounds
    aspect = (maxy - miny) / (maxx - minx)
    fig, ax = plt.subplots(figsize=(6, 6 * aspect))

    plot_course_elements(ax, hole_gdf)

    if plot_sg and sg_df is not None:
        result = generate_sg_grid_filtered(hole_gdf, sg_df, sg_step, res)
        if result:
            grid, _ = result
            plot_sg_overlay(ax, grid, sg_step, adjust_factor, mode)

    ax.set_title(f"{course.replace('_',' ').title()} – Hole {hole}")
    ax.axis("off")
    ax.set_aspect('equal')
    plt.tight_layout()

    if show:
        plt.show()
    else:
        save_dir = FIGURE_PATH / course
        save_dir.mkdir(parents=True, exist_ok=True)
        sg_flag = "sg_true" if plot_sg else "sg_false"
        filename = f"hole_{hole}_{sg_flag}_step_{sg_step}.png"
        plt.savefig(save_dir / filename, dpi=300)
        plt.close()


def plot_full_course(course='erin_hills', plot_sg=False, sg_step=0.1, res=300, gdf=None, adjust_factor=0.5, mode='lighten'):
    if gdf is None:
        gdf = get_get_gdf()
    gdf = preprocess_gdf(gdf, course)
    sg_df = get_sg_df() if plot_sg else None

    minx, miny, maxx, maxy = gdf.total_bounds
    aspect = (maxy - miny) / (maxx - minx)
    fig, ax = plt.subplots(figsize=(12, 12 * aspect))
    plot_course_elements(ax, gdf)

    if plot_sg and sg_df is not None:
        vmin_global, vmax_global = 1, 7
        cmap, _ = get_custom_viridis_with_adjusted_integers(
            vmin_global, vmax_global, adjust_factor, mode)
        norm = plt.Normalize(vmin_global, vmax_global)

        for hole in sorted(gdf['hole_num'].unique(), key=int):
            hole_gdf = gdf[gdf['hole_num'] == hole]
            result = generate_sg_grid_filtered(hole_gdf, sg_df, sg_step, res)
            if result:
                grid, _ = result
                grid.plot(ax=ax, column='expected', cmap=cmap,
                          markersize=1, alpha=0.5, norm=norm)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Expected Strokes')
        cbar.set_ticks(np.arange(vmin_global, vmax_global + 1))
        cbar.ax.invert_yaxis()

    ax.set_title("Full Course Layout" +
                 (" with Strokes Gained Overlay" if plot_sg else ""))
    ax.axis("off")
    ax.set_aspect('equal')
    plt.tight_layout()

    save_dir = FIGURE_PATH / course
    save_dir.mkdir(parents=True, exist_ok=True)
    sg_flag = "sg_true" if plot_sg else "sg_false"
    filename = f"course_layout_{sg_flag}_step_{sg_step}.png"
    plt.savefig(save_dir / filename, dpi=300)
    plt.close()


def plot_course_subplots(course='erin_hills', plot_sg=False, sg_step=0.1, res=300, gdf=None, adjust_factor=0.5, mode='lighten'):
    if gdf is None:
        gdf = get_get_gdf()
    gdf = preprocess_gdf(gdf, course)
    sg_df = get_sg_df() if plot_sg else None

    holes = sorted(gdf['hole_num'].unique(), key=int)
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 25))
    axes = axes.flatten()

    for idx, hole in enumerate(holes):
        ax = axes[idx]
        hole_gdf = gdf[gdf['hole_num'] == hole]
        if hole_gdf.empty:
            ax.axis("off")
            continue

        hole_gdf = rotate_hole(hole_gdf)
        plot_course_elements(ax, hole_gdf)

        if plot_sg and sg_df is not None:
            result = generate_sg_grid_filtered(hole_gdf, sg_df, sg_step, res)
            if result:
                grid, _ = result
                plot_sg_overlay(ax, grid, sg_step, adjust_factor, mode)

        ax.set_title(f"Hole {hole}")
        ax.axis("off")
        ax.set_aspect('equal')

    for ax in axes[len(holes):]:
        ax.axis("off")

    plt.tight_layout()
    save_dir = FIGURE_PATH / course
    save_dir.mkdir(parents=True, exist_ok=True)
    sg_flag = "sg_true" if plot_sg else "sg_false"
    filename = f"course_holes_{sg_flag}_step_{sg_step}.png"
    plt.savefig(save_dir / filename, dpi=300)
    plt.close()
