import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import ndimage
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patheffects
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path


# Path to the current file: Aravali/Code/script.py
BASE_DIR = Path(__file__).resolve().parent.parent
# BASE_DIR -> Aravali/

DATA_DIR = BASE_DIR / "Data"

# ============= LOAD SAVED DATA =============
print("📁 Loading saved data...")
final_mask = np.load(DATA_DIR / 'final_mask.npy')
hill_mask = np.load(DATA_DIR / 'hill_mask.npy')
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
    pixel_size_m_x = pixel_size_deg_x * 111111
    pixel_area_km2 = (pixel_size_m_x ** 2) / 1_000_000

# Load India state borders
india_states = gpd.read_file(shapefile_path)

print("✓ Data loaded")
print(f"  Elevation range: {np.nanmin(elevation):.0f}m to {np.nanmax(elevation):.0f}m")

# ============= DEFINE VIEWING EXTENT (INCLUDE DELHI + CONTEXT) =============
print("\n🗺️  Setting viewing extent to include Delhi...")

# Find data bounds but add LARGE buffer to show Delhi and context
valid_mask = ~np.isnan(elevation)
rows = np.any(valid_mask, axis=1)
cols = np.any(valid_mask, axis=0)
row_min, row_max = np.where(rows)[0][[0, -1]]
col_min, col_max = np.where(cols)[0][[0, -1]]

# Calculate geographic bounds of data
data_lon_min = bounds.left + col_min * pixel_size_deg_x
data_lon_max = bounds.left + col_max * pixel_size_deg_x
data_lat_max = bounds.top - row_min * pixel_size_deg_y
data_lat_min = bounds.top - row_max * pixel_size_deg_y

# Extend to ensure Delhi is fully visible
# Delhi is roughly at 76.8-77.3°E, 28.4-28.9°N
view_lon_min = min(data_lon_min - 1.5, 71.5)
view_lon_max = max(data_lon_max + 1.5, 77.5)
view_lat_min = min(data_lat_min - 1.5, 23.5)
view_lat_max = max(data_lat_max + 1.5, 29.5)

extent_view = [view_lon_min, view_lon_max, view_lat_min, view_lat_max]

print(f"  Viewing extent: {view_lon_min:.2f}° to {view_lon_max:.2f}°E")
print(f"                  {view_lat_min:.2f}° to {view_lat_max:.2f}°N")

# ============= GET STATES IN VIEW =============
from shapely.geometry import box
view_box = box(view_lon_min, view_lat_min, view_lon_max, view_lat_max)
states_in_view = india_states[india_states.intersects(view_box)].copy()

print(f"  States in view: {', '.join(states_in_view['name'].values)}")

# ============= CALCULATE STATISTICS =============
old_threshold = 200
old_mask = elevation > old_threshold

old_area = np.sum(old_mask) * pixel_area_km2
new_area = np.sum(final_mask) * pixel_area_km2
hill_area = np.sum(hill_mask) * pixel_area_km2
overlap_area = np.sum(old_mask & final_mask) * pixel_area_km2
only_old_area = np.sum(old_mask & ~final_mask) * pixel_area_km2
only_new_area = np.sum(final_mask & ~old_mask) * pixel_area_km2

print(f"\n📊 STATISTICS:")
print(f"  Old definition (>200m): {old_area:,.0f} km²")
print(f"  New definition: {new_area:,.0f} km²")

# ============= COLORMAP =============
terrain_cmap = LinearSegmentedColormap.from_list('terrain', 
    ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#fefebe', 
     '#fdb863', '#e08214', '#b35806', '#8c510a'])

# ============= HELPER FUNCTION FOR STATE LABELS =============
def add_state_labels(ax, states_gdf, extent):
    """Add state labels, ensuring they're within the viewing extent"""
    lon_min, lon_max, lat_min, lat_max = extent
    
    for idx, state in states_gdf.iterrows():
        centroid = state.geometry.centroid
        
        # Get label position
        label_x = centroid.x
        label_y = centroid.y
        
        # Clip to viewing extent with margin
        margin = 0.2
        label_x = np.clip(label_x, lon_min + margin, lon_max - margin)
        label_y = np.clip(label_y, lat_min + margin, lat_max - margin)
        
        # Add text with outline for visibility
        text = ax.text(label_x, label_y, state['name'], 
                      fontsize=12, fontweight='bold',
                      ha='center', va='center',
                      color='black', zorder=1000)
        
        # Add white outline
        text.set_path_effects([
            patheffects.Stroke(linewidth=3, foreground='white'),
            patheffects.Normal()
        ])

# ============= MAP 1: TOPOGRAPHIC BASE WITH CONTOURS =============
print("\n🎨 Creating Map 1: Topographic Base...")

fig1, ax1 = plt.subplots(figsize=(16, 14))

# Better terrain colormap
terrain_better = LinearSegmentedColormap.from_list('terrain_soft', 
    ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',  # Light blues
     '#fefebe', '#fee391', '#fec44f',              # Yellows
     '#fe9929', '#ec7014', '#cc4c02', '#8c2d04'])  # Oranges to browns

im1 = ax1.imshow(elevation, cmap=terrain_better, extent=extent,
                 vmin=np.nanmin(elevation),
                 vmax=np.nanpercentile(elevation, 98))

# MINOR contours (unlabeled, very subtle) - every 50m
minor_levels = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450, 1550]
cs_minor = ax1.contour(elevation, levels=minor_levels, 
                       colors='gray', linewidths=0.3, alpha=0.3, 
                       extent=extent, linestyles='--')

# MAJOR contours (labeled, more visible) - every 200m
major_levels = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600]
cs_major = ax1.contour(elevation, levels=major_levels, 
                       colors='black', linewidths=0.6, alpha=0.5, 
                       extent=extent)

# Label only major contours, with controlled spacing
ax1.clabel(cs_major, inline=True, fontsize=5, fmt='%dm', 
           inline_spacing=35,  # More spacing between labels
           manual=False)        # Automatic placement

# State borders
states_in_view.boundary.plot(ax=ax1, color='black', linewidth=1, alpha=0.8)

# State names
add_state_labels(ax1, states_in_view, extent_view)

# Colorbar
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.035, pad=0.04)
cbar1.set_label('Elevation (meters)', fontsize=13, fontweight='bold')

ax1.set_title('Aravalli Region: Topographic Map with Contours', 
              fontsize=18, fontweight='bold', pad=20)
ax1.set_xlabel('Longitude', fontsize=12)
ax1.set_ylabel('Latitude', fontsize=12)
ax1.set_xlim(extent_view[0], extent_view[1])
ax1.set_ylim(extent_view[2], extent_view[3])
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('Output/map1_topographic.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: map1_topographic.png")

# ============= MAP 2: INDIVIDUAL HILLS =============
print("\n🎨 Creating Map 2: Individual Hills...")

fig2, ax2 = plt.subplots(figsize=(16, 14))

# Light terrain background
ax2.imshow(elevation, cmap='gray', extent=extent, alpha=0.25,
           vmin=np.nanpercentile(elevation, 2),
           vmax=np.nanpercentile(elevation, 98))

# Bright orange/red for hills
hills_display = elevation.copy()
hills_display[~hill_mask] = np.nan

im2 = ax2.imshow(hills_display, cmap='YlOrRd', extent=extent, alpha=0.85,
                 vmin=np.nanmin(hills_display), vmax=np.nanmax(hills_display))

# State borders
states_in_view.boundary.plot(ax=ax2, color='black', linewidth=1, alpha=0.8)

# State names
add_state_labels(ax2, states_in_view, extent_view)

# Colorbar
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.035, pad=0.04)
cbar2.set_label('Elevation (meters)', fontsize=13, fontweight='bold')

# Statistics box
stats_text = f"""Individual Hills
≥100m prominence

Total area: {hill_area:,.0f} km²"""

ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=12,
         bbox=dict(boxstyle="round,pad=0.6", facecolor='white', 
                   edgecolor='darkorange', linewidth=2, alpha=0.95),
         verticalalignment='top', fontweight='bold')

ax2.set_title('Step 1: Hills with ≥100m Prominence', 
              fontsize=18, fontweight='bold', pad=20)
ax2.set_xlabel('Longitude', fontsize=12)
ax2.set_ylabel('Latitude', fontsize=12)
ax2.set_xlim(extent_view[0], extent_view[1])
ax2.set_ylim(extent_view[2], extent_view[3])
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('Output/map2_individual_hills.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: map2_individual_hills.png")

# ============= MAP 3: OLD vs NEW COMPARISON (TWO COLORBARS) =============
print("\n🎨 Creating Map 3: Old vs New Comparison...")

fig3, ax3 = plt.subplots(figsize=(18, 14))

# Very light terrain background
ax3.imshow(elevation, cmap='gray', extent=extent, alpha=0.15,
           vmin=np.nanpercentile(elevation, 2),
           vmax=np.nanpercentile(elevation, 98))

# LAYER 1: Only in OLD definition (blue)
only_old_display = elevation.copy()
only_old_display[~(old_mask & ~final_mask)] = np.nan
if np.any(~np.isnan(only_old_display)):
    im_old = ax3.imshow(only_old_display, cmap='Blues', extent=extent, 
                        alpha=0.65, vmin=200, vmax=1000)

# LAYER 2: Only in NEW definition (orange)
only_new_display = elevation.copy()
only_new_display[~(final_mask & ~old_mask)] = np.nan
if np.any(~np.isnan(only_new_display)):
    im_new = ax3.imshow(only_new_display, cmap='Oranges', extent=extent, 
                        alpha=0.75, vmin=100, vmax=600)

# LAYER 3: OVERLAP (magenta/purple)
overlap_display = elevation.copy()
overlap_display[~(old_mask & final_mask)] = np.nan
if np.any(~np.isnan(overlap_display)):
    im_overlap = ax3.imshow(overlap_display, cmap='RdPu', extent=extent, 
                            alpha=0.85, vmin=200, vmax=1500)

# State borders
states_in_view.boundary.plot(ax=ax3, color='black', linewidth=1, alpha=0.9)

# State names
add_state_labels(ax3, states_in_view, extent_view)

# Statistics box
if old_area > 0:
    change_pct = ((new_area - old_area) / old_area) * 100
else:
    change_pct = 0

stats_text = f"""OLD vs NEW DEFINITION

OLD (>200m elevation):
  Area: {old_area:,.0f} km²

NEW (prominence + clustering):
  Area: {new_area:,.0f} km²

BREAKDOWN:
  Both definitions: {overlap_area:,.0f} km²
  Only OLD (lost): {only_old_area:,.0f} km²
  Only NEW (gained): {only_new_area:,.0f} km²

NET CHANGE: {new_area - old_area:,.0f} km²
({change_pct:+.1f}%)"""

ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=11,
         bbox=dict(boxstyle="round,pad=0.6", facecolor='white', 
                   edgecolor='black', linewidth=2, alpha=0.95),
         verticalalignment='top', fontfamily='monospace')

# TWO COLORBARS - Blue (OLD) and Magenta (OVERLAP)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Blue colorbar for OLD areas (left side)
if np.any(~np.isnan(only_old_display)):
    cax_blue = inset_axes(ax3, width="2.5%", height="25%", loc='center left',
                         bbox_to_anchor=(0.02, 0, 1, 1), bbox_transform=ax3.transAxes)
    cbar_blue = plt.colorbar(im_old, cax=cax_blue, orientation='vertical')
    cbar_blue.set_label('Only OLD\nElevation (m)', fontsize=10, fontweight='bold')

# Magenta colorbar for OVERLAP (right side)
if np.any(~np.isnan(overlap_display)):
    cax_magenta = inset_axes(ax3, width="2.5%", height="25%", loc='center right',
                            bbox_to_anchor=(0.12, 0, 1, 1), bbox_transform=ax3.transAxes)
    cbar_magenta = plt.colorbar(im_overlap, cax=cax_magenta, orientation='vertical')
    cbar_magenta.set_label('Overlap\nElevation (m)', fontsize=10, fontweight='bold')

# Legend
legend_elements = [
    Patch(facecolor='#8e44ad', alpha=0.85, 
          label=f'Both (overlap): {overlap_area:,.0f} km²'),
    Patch(facecolor='#3498db', alpha=0.65, 
          label=f'Only OLD (lost): {only_old_area:,.0f} km²'),
    Patch(facecolor='#e67e22', alpha=0.75, 
          label=f'Only NEW (gained): {only_new_area:,.0f} km²'),
]
ax3.legend(handles=legend_elements, loc='lower right', 
           fontsize=11, framealpha=0.95, edgecolor='black', 
           title='Legend', title_fontsize=12,
           fancybox=True, shadow=True)

ax3.set_title('Aravalli Range: Old Definition vs New Court Ruling', 
              fontsize=20, fontweight='bold', pad=25)
ax3.set_xlabel('Longitude', fontsize=13)
ax3.set_ylabel('Latitude', fontsize=13)
ax3.set_xlim(extent_view[0], extent_view[1])
ax3.set_ylim(extent_view[2], extent_view[3])
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('Output/map3_comparison_main.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: map3_comparison_main.png")

plt.show()

print("\n" + "="*70)
print("✅ ALL MAPS COMPLETE!")
print("  ✓ Zoomed to include Delhi + context")
print("  ✓ White areas visible (no aggressive cropping)")
print("  ✓ State names in frame (no boxes)")
print("  ✓ State borders width=1")
print("  ✓ Map 3: TWO colorbars (blue + magenta)")
print("="*70)

