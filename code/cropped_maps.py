import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib import patheffects
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# Path to the current file: Aravali/Code/script.py
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

# ============= LOAD SAVED DATA =============
print("📁 Loading saved data...")
final_mask = np.load(DATA_DIR / 'final_mask.npy')
elevation = np.load(DATA_DIR / 'elevation.npy')

# Load DEM metadata
dem_path = DATA_DIR / "Aravali_data.tif"
shapefile_path = DATA_DIR / "in_shp/in.shp"

with rasterio.open(dem_path) as src:
    bounds = src.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    transform = src.transform
    pixel_size_deg_x = transform[0]
    pixel_size_deg_y = -transform[4]

# Load India state borders
india_states = gpd.read_file(shapefile_path)

# ============= DEFINE DATA-ONLY EXTENT =============
valid_mask = ~np.isnan(elevation)
rows = np.any(valid_mask, axis=1)
cols = np.any(valid_mask, axis=0)
row_min, row_max = np.where(rows)[0][[0, -1]]
col_min, col_max = np.where(cols)[0][[0, -1]]

data_lon_min = bounds.left + col_min * pixel_size_deg_x
data_lon_max = bounds.left + col_max * pixel_size_deg_x
data_lat_max = bounds.top - row_min * pixel_size_deg_y
data_lat_min = bounds.top - row_max * pixel_size_deg_y

extent_data = [data_lon_min, data_lon_max, data_lat_min, data_lat_max]

# ============= GET STATES IN DATA BOX =============
from shapely.geometry import box
data_box = box(data_lon_min, data_lat_min, data_lon_max, data_lat_max)
states_in_data = india_states[india_states.intersects(data_box)].copy()

# ============= DEFINE OLD MASK =============
old_threshold = 200
old_mask = elevation > old_threshold

# ============= HELPER FUNCTION FOR STATE LABELS =============
def add_state_labels(ax, states_gdf, extent):
    """Add state labels"""
    lon_min, lon_max, lat_min, lat_max = extent
    
    for idx, state in states_gdf.iterrows():
        centroid = state.geometry.centroid
        label_x = np.clip(centroid.x, lon_min + 0.2, lon_max - 0.2)
        label_y = np.clip(centroid.y, lat_min + 0.2, lat_max - 0.2)
        
        text = ax.text(label_x, label_y, state['name'], 
                      fontsize=12, fontweight='bold',
                      ha='center', va='center',
                      color='black', zorder=1000)
        
        text.set_path_effects([
            patheffects.Stroke(linewidth=3, foreground='white'),
            patheffects.Normal()
        ])

# ============= IMAGE 1: OLD DEFINITION (BLUE ONLY) =============
print("\n🎨 Creating Image 1: Old Definition (Blue)...")

fig1, ax1 = plt.subplots(figsize=(16, 14))
ax1.axis('off')

# Light gray background
ax1.imshow(elevation, cmap='gray', extent=extent, alpha=0.15,
           vmin=np.nanpercentile(elevation, 2),
           vmax=np.nanpercentile(elevation, 98))

# OLD definition in BLUE
old_display = elevation.copy()
old_display[~old_mask] = np.nan
ax1.imshow(old_display, cmap='Blues', extent=extent, 
           alpha=0.65, vmin=200, vmax=1000)

# State borders
states_in_data.boundary.plot(ax=ax1, color='black', linewidth=1, alpha=0.8)

# State names
add_state_labels(ax1, states_in_data, extent_data)

ax1.set_xlim(extent_data[0], extent_data[1])
ax1.set_ylim(extent_data[2], extent_data[3])

plt.tight_layout(pad=0)
plt.savefig('slider_image1_old.png', dpi=300, bbox_inches='tight', 
            facecolor='white', pad_inches=0)
print("✓ Saved: slider_image1_old.png")
plt.close()

# ============= IMAGE 2: OLD (BLUE) + NEW DEFINITION (PINK) =============
print("🎨 Creating Image 2: Old (Blue) + New Definition (Pink)...")

fig2, ax2 = plt.subplots(figsize=(16, 14))
ax2.axis('off')

# Light gray background
ax2.imshow(elevation, cmap='gray', extent=extent, alpha=0.15,
           vmin=np.nanpercentile(elevation, 2),
           vmax=np.nanpercentile(elevation, 98))

# OLD definition in BLUE (same as image 1)
old_display = elevation.copy()
old_display[~old_mask] = np.nan
ax2.imshow(old_display, cmap='Blues', extent=extent, 
           alpha=0.65, vmin=200, vmax=1000)

# NEW definition in PINK/PURPLE (ALL of final_mask)
new_display = elevation.copy()
new_display[~final_mask] = np.nan
ax2.imshow(new_display, cmap='RdPu', extent=extent, 
           alpha=0.75, vmin=100, vmax=1500)

# State borders
states_in_data.boundary.plot(ax=ax2, color='black', linewidth=1, alpha=0.8)

# State names
add_state_labels(ax2, states_in_data, extent_data)

ax2.set_xlim(extent_data[0], extent_data[1])
ax2.set_ylim(extent_data[2], extent_data[3])

plt.tight_layout(pad=0)
plt.savefig('slider_image2_new.png', dpi=300, bbox_inches='tight', 
            facecolor='white', pad_inches=0)
print("✓ Saved: slider_image2_new.png")
plt.close()

print("\n" + "="*70)
print("✅ SLIDER IMAGES COMPLETE!")
print("  ✓ Image 1: Old definition (blue)")
print("  ✓ Image 2: Old (blue) + New definition (pink/purple overlay)")
print("  ✓ Pink will appear over blue in slider!")
print("="*70)