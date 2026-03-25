import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import ndimage
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ============= LOAD SAVED DATA =============
print("📁 Loading saved data...")
final_mask = np.load('final_mask.npy')
hill_mask = np.load('hill_mask.npy')
elevation = np.load('elevation.npy')

# Load DEM metadata
dem_path = "Aravali_data.tif"
shapefile_path = "in_shp/in.shp"

with rasterio.open(dem_path) as src:
    bounds = src.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    transform = src.transform
    pixel_size_deg_x = transform[0]
    pixel_size_deg_y = -transform[4]
    pixel_size_m_x = pixel_size_deg_x * 111111
    pixel_area_km2 = (pixel_size_m_x ** 2) / 1_000_000

# Load state borders
india_states = gpd.read_file(shapefile_path)

print("✓ Data loaded")
print(f"  Elevation range: {np.nanmin(elevation):.0f}m to {np.nanmax(elevation):.0f}m")

# ============= CROP TO ARAVALLI REGION =============
print("\n🗺️  Cropping to Aravalli region...")

# Find where we have actual elevation data
valid_mask = ~np.isnan(elevation)
rows = np.any(valid_mask, axis=1)
cols = np.any(valid_mask, axis=0)
row_min, row_max = np.where(rows)[0][[0, -1]]
col_min, col_max = np.where(cols)[0][[0, -1]]

# Add buffer (10% on each side)
buffer = int(0.1 * (row_max - row_min))
row_min = max(0, row_min - buffer)
row_max = min(elevation.shape[0], row_max + buffer)
col_min = max(0, col_min - buffer)
col_max = min(elevation.shape[1], col_max + buffer)

# Crop all arrays
elevation_crop = elevation[row_min:row_max, col_min:col_max]
final_mask_crop = final_mask[row_min:row_max, col_min:col_max]
hill_mask_crop = hill_mask[row_min:row_max, col_min:col_max]

# Calculate cropped extent
lon_min = bounds.left + col_min * pixel_size_deg_x
lon_max = bounds.left + col_max * pixel_size_deg_x
lat_max = bounds.top - row_min * pixel_size_deg_y
lat_min = bounds.top - row_max * pixel_size_deg_y
extent_crop = [lon_min, lon_max, lat_min, lat_max]

print(f"  Cropped from {elevation.shape} to {elevation_crop.shape}")
print(f"  New extent: {lon_min:.2f}° to {lon_max:.2f}°E, {lat_min:.2f}° to {lat_max:.2f}°N")

# ============= CALCULATE STATISTICS =============
old_threshold = 200
old_mask = elevation_crop > old_threshold

old_area = np.sum(old_mask) * pixel_area_km2
new_area = np.sum(final_mask_crop) * pixel_area_km2
hill_area = np.sum(hill_mask_crop) * pixel_area_km2
overlap_area = np.sum(old_mask & final_mask_crop) * pixel_area_km2
only_old_area = np.sum(old_mask & ~final_mask_crop) * pixel_area_km2
only_new_area = np.sum(final_mask_crop & ~old_mask) * pixel_area_km2

print(f"\n📊 STATISTICS:")
print(f"  Old definition (>200m): {old_area:,.0f} km²")
print(f"  New definition: {new_area:,.0f} km²")
print(f"  Overlap: {overlap_area:,.0f} km²")
print(f"  Lost from old: {only_old_area:,.0f} km²")
print(f"  Gained in new: {only_new_area:,.0f} km²")

# ============= COLORMAP =============
terrain_cmap = LinearSegmentedColormap.from_list('terrain', 
    ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#fefebe', 
     '#fdb863', '#e08214', '#b35806', '#8c510a'])

# ============= MAP 1: TOPOGRAPHIC BASE WITH CONTOURS =============
print("\n🎨 Creating Map 1: Topographic Base with Contours...")

fig1, ax1 = plt.subplots(figsize=(12, 10))

# Display elevation
im1 = ax1.imshow(elevation_crop, cmap=terrain_cmap, extent=extent_crop,
                 vmin=np.nanpercentile(elevation_crop, 2),
                 vmax=np.nanpercentile(elevation_crop, 98))

# Add contour lines
contour_levels = np.arange(200, np.nanmax(elevation_crop), 200)  # Every 200m
cs1 = ax1.contour(elevation_crop, levels=contour_levels, 
                  colors='black', linewidths=0.5, alpha=0.4, 
                  extent=extent_crop)
ax1.clabel(cs1, inline=True, fontsize=8, fmt='%dm')

# Add state borders
india_states.boundary.plot(ax=ax1, color='black', linewidth=1.5, alpha=0.8)

# Colorbar
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Elevation (meters)', fontsize=12, fontweight='bold')

ax1.set_title('Aravalli Region: Topographic Map with Contours', 
              fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Longitude', fontsize=11)
ax1.set_ylabel('Latitude', fontsize=11)
ax1.set_xlim(extent_crop[0], extent_crop[1])
ax1.set_ylim(extent_crop[2], extent_crop[3])

plt.tight_layout()
plt.savefig('map1_topographic_contours.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: map1_topographic_contours.png")

# ============= MAP 2: INDIVIDUAL HILLS (BRIGHT COLORS) =============
print("\n🎨 Creating Map 2: Individual Hills (≥100m prominence)...")

fig2, ax2 = plt.subplots(figsize=(12, 10))

# Light terrain background
ax2.imshow(elevation_crop, cmap='gray', extent=extent_crop, alpha=0.25,
           vmin=np.nanpercentile(elevation_crop, 2),
           vmax=np.nanpercentile(elevation_crop, 98))

# Bright orange/red for hills
hills_display = elevation_crop.copy()
hills_display[~hill_mask_crop] = np.nan

im2 = ax2.imshow(hills_display, cmap='YlOrRd', extent=extent_crop, alpha=0.85,
                 vmin=np.nanmin(hills_display), vmax=np.nanmax(hills_display))

# State borders
india_states.boundary.plot(ax=ax2, color='black', linewidth=1.5, alpha=0.8)

# Colorbar
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Elevation (meters)', fontsize=12, fontweight='bold')

# Statistics box
stats_text = f"""Individual Hills
≥100m prominence

Total area: {hill_area:,.0f} km²"""

ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=12,
         bbox=dict(boxstyle="round,pad=0.6", facecolor='white', 
                   edgecolor='darkorange', linewidth=2, alpha=0.95),
         verticalalignment='top', fontweight='bold')

ax2.set_title('Step 1: Hills with ≥100m Prominence', 
              fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('Longitude', fontsize=11)
ax2.set_ylabel('Latitude', fontsize=11)
ax2.set_xlim(extent_crop[0], extent_crop[1])
ax2.set_ylim(extent_crop[2], extent_crop[3])

plt.tight_layout()
plt.savefig('map2_individual_hills.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: map2_individual_hills.png")

# ============= MAP 3: OLD vs NEW COMPARISON (MAIN FINDING) =============
print("\n🎨 Creating Map 3: Old vs New Comparison (MAIN MAP)...")

fig3, ax3 = plt.subplots(figsize=(14, 12))

# Very light terrain background
ax3.imshow(elevation_crop, cmap='gray', extent=extent_crop, alpha=0.15,
           vmin=np.nanpercentile(elevation_crop, 2),
           vmax=np.nanpercentile(elevation_crop, 98))

# LAYER 1: Only in OLD definition (light blue) - what we're LOSING
only_old_display = elevation_crop.copy()
only_old_display[~(old_mask & ~final_mask_crop)] = np.nan
if np.any(~np.isnan(only_old_display)):
    ax3.imshow(only_old_display, cmap='Blues', extent=extent_crop, 
               alpha=0.6, vmin=200, vmax=800)

# LAYER 2: Only in NEW definition (orange) - newly included
only_new_display = elevation_crop.copy()
only_new_display[~(final_mask_crop & ~old_mask)] = np.nan
if np.any(~np.isnan(only_new_display)):
    ax3.imshow(only_new_display, cmap='Oranges', extent=extent_crop, 
               alpha=0.7, vmin=100, vmax=600)

# LAYER 3: OVERLAP (dark red/purple) - still protected
overlap_display = elevation_crop.copy()
overlap_display[~(old_mask & final_mask_crop)] = np.nan
if np.any(~np.isnan(overlap_display)):
    ax3.imshow(overlap_display, cmap='RdPu', extent=extent_crop, 
               alpha=0.8, vmin=200, vmax=1200)

# Add contour lines for reference
contour_levels = [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600]
cs3 = ax3.contour(elevation_crop, levels=contour_levels, 
                  colors='black', linewidths=0.3, alpha=0.3, 
                  extent=extent_crop, linestyles='dashed')

# State borders
india_states.boundary.plot(ax=ax3, color='black', linewidth=1.5, alpha=0.9)

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

NET CHANGE: {new_area - old_area:,.0f} km² ({change_pct:+.1f}%)"""

ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=11,
         bbox=dict(boxstyle="round,pad=0.6", facecolor='white', 
                   edgecolor='black', linewidth=2, alpha=0.95),
         verticalalignment='top', fontfamily='monospace')

# Legend
legend_elements = [
    Patch(facecolor='darkred', alpha=0.8, 
          label=f'Both (overlap): {overlap_area:,.0f} km²'),
    Patch(facecolor='lightblue', alpha=0.6, 
          label=f'Only OLD (lost): {only_old_area:,.0f} km²'),
    Patch(facecolor='orange', alpha=0.7, 
          label=f'Only NEW (gained): {only_new_area:,.0f} km²'),
]
ax3.legend(handles=legend_elements, loc='lower right', 
           fontsize=11, framealpha=0.95, edgecolor='black', 
           title='Legend', title_fontsize=12)

ax3.set_title('Aravalli Range: Old Definition vs New Court Ruling', 
              fontsize=18, fontweight='bold', pad=20)
ax3.set_xlabel('Longitude', fontsize=12)
ax3.set_ylabel('Latitude', fontsize=12)
ax3.set_xlim(extent_crop[0], extent_crop[1])
ax3.set_ylim(extent_crop[2], extent_crop[3])

plt.tight_layout()
plt.savefig('map3_comparison_main.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: map3_comparison_main.png")

# ============= FINAL SUMMARY =============
print("\n" + "="*70)
print("FINAL SUMMARY - 3 MAPS CREATED")
print("="*70)
print("\n📍 MAP 1: Topographic base with contour lines")
print("   Shows the full Aravalli terrain and elevation structure")
print(f"   File: map1_topographic_contours.png")

print("\n📍 MAP 2: Individual hills (≥100m prominence)")
print("   Shows all hills meeting the first criterion")
print(f"   Area: {hill_area:,.0f} km²")
print(f"   File: map2_individual_hills.png")

print("\n📍 MAP 3: OLD vs NEW comparison (MAIN FINDING)")
print("   Shows what changes under the new court definition")
print(f"   OLD: {old_area:,.0f} km² | NEW: {new_area:,.0f} km²")
print(f"   Change: {new_area - old_area:,.0f} km² ({change_pct:+.1f}%)")
print(f"   File: map3_comparison_main.png")
print("="*70)

plt.show()
print("\n✅ All maps complete!")
##########################################################################################################################################


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

# ============= LOAD SAVED DATA =============
print("📁 Loading saved data...")
final_mask = np.load('final_mask.npy')
hill_mask = np.load('hill_mask.npy')
elevation = np.load('elevation.npy')

# Load DEM metadata
dem_path = "Aravali_data.tif"
shapefile_path = "in_shp/in.shp"

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

# ============= MAP 1: TOPOGRAPHIC BASE =============
print("\n🎨 Creating Map 1: Topographic Base...")

fig1, ax1 = plt.subplots(figsize=(16, 14))

# Display elevation
im1 = ax1.imshow(elevation, cmap=terrain_cmap, extent=extent,
                 vmin=np.nanpercentile(elevation, 2),
                 vmax=np.nanpercentile(elevation, 98))

# Add state borders
states_in_view.boundary.plot(ax=ax1, color='black', linewidth=1, alpha=0.8)

# Add state names (in frame)
add_state_labels(ax1, states_in_view, extent_view)

# Colorbar
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.035, pad=0.04)
cbar1.set_label('Elevation (meters)', fontsize=13, fontweight='bold')

ax1.set_title('Aravalli Region: Topographic Map', 
              fontsize=18, fontweight='bold', pad=20)
ax1.set_xlabel('Longitude', fontsize=12)
ax1.set_ylabel('Latitude', fontsize=12)
ax1.set_xlim(extent_view[0], extent_view[1])
ax1.set_ylim(extent_view[2], extent_view[3])
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('map1_topographic.png', dpi=300, bbox_inches='tight', facecolor='white')
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
plt.savefig('map2_individual_hills.png', dpi=300, bbox_inches='tight', facecolor='white')
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
plt.savefig('map3_comparison_main.png', dpi=300, bbox_inches='tight', facecolor='white')
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