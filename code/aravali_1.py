# pip install rasterio matplotlib numpy scipy geopandas
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import ndimage
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')

# ============= LOAD DATA =============
dem_path = "Aravali_data.tif"
shapefile_path = "in_shp/in.shp"

print("📁 Loading data...")

# Load DEM at full resolution for accurate analysis
with rasterio.open(dem_path) as src:
    # Get original dimensions
    print(f"  Original dimensions: {src.width} x {src.height}")
    print(f"  Original resolution: {src.res}")
    
    # Read the data
    elevation = src.read(1).astype(float)
    elevation[elevation == src.nodata] = np.nan
    
    bounds = src.bounds
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    transform = src.transform
    
    # Calculate pixel size in degrees and meters
    # Assuming approximately 111,111 meters per degree at equator
    pixel_size_deg_x = transform[0]
    pixel_size_deg_y = -transform[4]
    pixel_size_m_x = pixel_size_deg_x * 111111  # Approximate conversion
    pixel_size_m_y = pixel_size_deg_y * 111111
    
    print(f"  Pixel size: ~{pixel_size_m_x:.1f} x {pixel_size_m_y:.1f} meters")
    print(f"  Elevation range: {np.nanmin(elevation):.0f}m to {np.nanmax(elevation):.0f}m")

# Load India state borders
india_states = gpd.read_file(shapefile_path)
print("✓ Data loaded")

# ============= ACCURATE PROMINENCE-BASED ANALYSIS =============
def find_hills_by_prominence(elevation_data, prominence_threshold=100, 
                            search_radius_m=2000, pixel_size_m=30):
    """
    Find hills based on prominence (height above surrounding terrain)
    Using efficient windowed approach
    """
    print(f"\n📊 IDENTIFYING HILLS BY PROMINENCE...")
    print(f"  Prominence threshold: {prominence_threshold}m")
    print(f"  Search radius: {search_radius_m}m")
    
    # Convert search radius to pixels
    search_radius_px = int(search_radius_m / pixel_size_m)
    print(f"  Search radius in pixels: {search_radius_px}")
    
    # If still too large, use a smaller radius
    if search_radius_px > 50:
        search_radius_px = 50  # 1.5km radius
        print(f"  Reduced to {search_radius_px} pixels (~{search_radius_px * pixel_size_m}m)")
    
    # Create a sample grid to avoid memory issues
    # Sample every 3rd pixel in each direction
    sample_step = 3
    rows = np.arange(0, elevation_data.shape[0], sample_step)
    cols = np.arange(0, elevation_data.shape[1], sample_step)
    
    print(f"  Sampling grid: {len(rows)} x {len(cols)} points")
    
    # Initialize arrays
    hill_mask = np.zeros_like(elevation_data, dtype=bool)
    prominence_map = np.full_like(elevation_data, 0.0)
    
    # Process each sample point
    total_points = len(rows) * len(cols)
    processed = 0
    
    for i in rows:
        for j in cols:
            if np.isnan(elevation_data[i, j]):
                continue
                
            current_elev = elevation_data[i, j]
            
            # Define search window
            i_min = max(0, i - search_radius_px)
            i_max = min(elevation_data.shape[0], i + search_radius_px + 1)
            j_min = max(0, j - search_radius_px)
            j_max = min(elevation_data.shape[1], j + search_radius_px + 1)
            
            # Get window and find minimum
            window = elevation_data[i_min:i_max, j_min:j_max]
            if np.any(~np.isnan(window)):
                window_min = np.nanmin(window)
                prominence = current_elev - window_min
                prominence_map[i, j] = prominence
                
                if prominence >= prominence_threshold:
                    # Mark this pixel and its immediate neighbors as hill
                    r_min = max(0, i-1)
                    r_max = min(elevation_data.shape[0], i+2)
                    c_min = max(0, j-1)
                    c_max = min(elevation_data.shape[1], j+2)
                    hill_mask[r_min:r_max, c_min:c_max] = True
            
            processed += 1
            if processed % 10000 == 0:
                print(f"  Processed {processed}/{total_points} points...")
    
    print(f"  Found {np.sum(hill_mask)} hill pixels")
    return hill_mask, prominence_map

def apply_clustering_rule(hill_mask, cluster_distance_m=500, pixel_size_m=30):
    """
    Apply 500m clustering rule using morphological operations
    """
    print(f"\n📊 APPLYING 500m CLUSTERING RULE...")
    
    # Convert distance to dilation radius in pixels
    cluster_radius_px = int(cluster_distance_m / pixel_size_m)
    print(f"  Cluster radius: {cluster_radius_px} pixels")
    
    # Create structure for dilation
    structure = np.ones((cluster_radius_px*2+1, cluster_radius_px*2+1))
    
    # Dilate hill mask
    print("  Dilating hill mask...")
    dilated_mask = ndimage.binary_dilation(hill_mask, structure=structure)
    
    # Find connected components in dilated mask
    print("  Finding connected components...")
    labeled_dilated, num_clusters = ndimage.label(dilated_mask)
    
    # Only keep original hill pixels that are in clusters
    final_mask = np.zeros_like(hill_mask, dtype=bool)
    
    # For each cluster
    for cluster_id in range(1, num_clusters + 1):
        cluster_mask = labeled_dilated == cluster_id
        
        # Count original hill pixels in this cluster
        original_in_cluster = np.sum(hill_mask & cluster_mask)
        
        # If cluster has at least 2 original hill pixels (or 1 with neighbors), keep it
        if original_in_cluster >= 2:
            final_mask = final_mask | (hill_mask & cluster_mask)
        elif original_in_cluster == 1:
            # Check if this single hill has neighbors within original mask
            # (already handled by dilation)
            final_mask = final_mask | (hill_mask & cluster_mask)
    
    print(f"  Original hill pixels: {np.sum(hill_mask)}")
    print(f"  After clustering: {np.sum(final_mask)}")
    print(f"  Number of qualifying clusters: {num_clusters}")
    
    return final_mask

# ============= MAIN ANALYSIS =============
print("\n" + "="*60)
print("ANALYZING ARAVALLI DEFINITION PER SUPREME COURT RULING")
print("="*60)
print("Definition: Landform rising ≥100m above surrounding terrain")
print("            Two or more such hills within 500m = Aravalli range")
print("="*60)

# Step 1: Find hills by prominence
hill_mask, prominence = find_hills_by_prominence(
    elevation, 
    prominence_threshold=100,
    search_radius_m=2000,  # 2km search radius
    pixel_size_m=pixel_size_m_x
)

# Step 2: Apply clustering rule
final_mask = apply_clustering_rule(
    hill_mask,
    cluster_distance_m=500,
    pixel_size_m=pixel_size_m_x
)

# Calculate areas
pixel_area_m2 = pixel_size_m_x * pixel_size_m_y
pixel_area_km2 = pixel_area_m2 / 1_000_000

initial_hill_area = np.sum(hill_mask) * pixel_area_km2
final_area = np.sum(final_mask) * pixel_area_km2

print(f"\n📊 AREA CALCULATIONS:")
print(f"  Pixels identified as hills (≥100m prominence): {np.sum(hill_mask):,}")
print(f"  Area of individual hills: {initial_hill_area:,.0f} km²")
print(f"  Pixels after 500m clustering: {np.sum(final_mask):,}")
print(f"  Final qualifying area: {final_area:,.0f} km²")

# For comparison with old definition
old_threshold = 200
old_mask = elevation > old_threshold
old_area = np.sum(old_mask) * pixel_area_km2

print(f"\n📊 COMPARISON WITH OLD DEFINITION:")
print(f"  Old definition (>200m elevation): {old_area:,.0f} km²")
print(f"  New definition (prominence+clustering): {final_area:,.0f} km²")

if old_area > 0:
    percentage = (final_area / old_area) * 100
    print(f"  New as % of old: {percentage:.1f}%")
    
    if final_area < old_area:
        loss = old_area - final_area
        print(f"  Area potentially lost: {loss:,.0f} km²")
    else:
        print(f"  Area gain: {final_area - old_area:,.0f} km²")

# Add this RIGHT AFTER the analysis completes (before Map 1):
np.save('final_mask.npy', final_mask)
np.save('hill_mask.npy', hill_mask)
np.save('elevation.npy', elevation)

# ============= CREATE VISUALIZATIONS =============
# Create terrain colormap
terrain_cmap = LinearSegmentedColormap.from_list('terrain', 
['#4575b4', '#91bfdb', '#e0f3f8', '#ffffbf', '#fee090', '#fc8d59', '#d73027'])

# ============= MAP 1: TOPOGRAPHIC BASE =============
print("\n🎨 CREATING MAP 1: Topographic Base...")

fig1, ax1 = plt.subplots(figsize=(14, 12))

# Display elevation
elev_display = elevation.copy()
elev_display[np.isnan(elev_display)] = np.nanmin(elevation)

im1 = ax1.imshow(elev_display, cmap=terrain_cmap, extent=extent,
            vmin=np.nanpercentile(elevation, 5),
            vmax=np.nanpercentile(elevation, 95))

# Add state borders
india_states.boundary.plot(ax=ax1, color='black', linewidth=1, alpha=0.7)

# Add colorbar
cbar1 = plt.colorbar(im1, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
cbar1.set_label('Elevation (meters)', fontsize=12)

ax1.set_title('Aravalli Region: Topographic Base Map', 
             fontsize=18, fontweight='bold', pad=20)
ax1.set_xlabel('Longitude', fontsize=12)
ax1.set_ylabel('Latitude', fontsize=12)

plt.tight_layout()
plt.savefig('map1_topographic_base.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: map1_topographic_base.png")

# ============= MAP 2: INDIVIDUAL HILLS (STEP 1) =============
print("\n🎨 CREATING MAP 2: Individual Hills (≥100m prominence)...")

fig2, ax2 = plt.subplots(figsize=(14, 12))

# Base terrain in light colors
ax2.imshow(elev_display, cmap='gray', extent=extent, alpha=0.4,
          vmin=np.nanpercentile(elevation, 5),
          vmax=np.nanpercentile(elevation, 95))

# Overlay individual hills
hills_display = np.zeros_like(elevation)
hills_display[hill_mask] = 1
hills_display[~hill_mask] = np.nan

ax2.imshow(hills_display, cmap='YlOrBr', extent=extent, alpha=0.8)

# Add state borders
india_states.boundary.plot(ax=ax2, color='black', linewidth=1, alpha=0.7)

# Statistics box
stats_text = f"""INDIVIDUAL HILLS (Step 1)
Definition: ≥100m prominence above surrounding terrain

Results:
• Hill pixels identified: {np.sum(hill_mask):,}
• Total hill area: {initial_hill_area:,.0f} km²
• Search radius used: 2 km"""

ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.95,
                 edgecolor='black', linewidth=1),
        verticalalignment='top', fontfamily='monospace')

ax2.set_title('Step 1: Individual Hills with ≥100m Prominence', 
             fontsize=18, fontweight='bold', pad=20)
ax2.set_xlabel('Longitude', fontsize=12)
ax2.set_ylabel('Latitude', fontsize=12)

plt.tight_layout()
plt.savefig('map2_individual_hills.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: map2_individual_hills.png")

# ============= MAP 3: FINAL QUALIFYING AREAS (STEP 2) =============
print("\n🎨 CREATING MAP 3: Final Qualifying Areas...")

fig3, ax3 = plt.subplots(figsize=(14, 12))

# Base terrain
ax3.imshow(elev_display, cmap='gray', extent=extent, alpha=0.3,
          vmin=np.nanpercentile(elevation, 5),
          vmax=np.nanpercentile(elevation, 95))

# Overlay final qualifying areas
final_display = np.zeros_like(elevation)
final_display[final_mask] = 1
final_display[~final_mask] = np.nan

ax3.imshow(final_display, cmap='RdYlGn_r', extent=extent, alpha=0.8)

# Add state borders
india_states.boundary.plot(ax=ax3, color='black', linewidth=1, alpha=0.7)

# Highlight clusters
labeled_clusters, num_clusters = ndimage.label(final_mask)
for cluster_id in range(1, min(num_clusters+1, 20)):  # Show first 20 clusters
    cluster_mask = labeled_clusters == cluster_id
    if np.sum(cluster_mask) > 10:  # Only label significant clusters
        # Find centroid
        y_coords, x_coords = np.where(cluster_mask)
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)
        
        # Convert to geographic coordinates
        lon = bounds.left + center_x * pixel_size_deg_x
        lat = bounds.top - center_y * pixel_size_deg_y
        
        # Add small label for large clusters
        if np.sum(cluster_mask) > 100:
            ax3.text(lon, lat, f'C{cluster_id}', fontsize=8,
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

# Final statistics
if old_area > 0:
    comparison = f"""Comparison with old definition:
    • Old (>200m): {old_area:,.0f} km²
    • New (this map): {final_area:,.0f} km²
    • Change: {final_area - old_area:,.0f} km² ({((final_area/old_area)-1)*100:.1f}%)"""
else:
    comparison = ""

stats_text = f"""FINAL QUALIFYING AREAS (Step 2)
Definition: Hills within 500m of each other

Results:
• Qualifying pixels: {np.sum(final_mask):,}
• Total area: {final_area:,.0f} km²
• Number of hill clusters: {num_clusters}

{comparison}"""

ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.95,
                 edgecolor='black', linewidth=1),
        verticalalignment='top', fontfamily='monospace')

ax3.set_title('Step 2: Final Aravalli Range (after 500m clustering)', 
             fontsize=18, fontweight='bold', pad=20)
ax3.set_xlabel('Longitude', fontsize=12)
ax3.set_ylabel('Latitude', fontsize=12)

plt.tight_layout()
plt.savefig('map3_final_aravalli.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: map3_final_aravalli.png")

# ============= MAP 4: SIMPLIFIED COMPARISON (ONE MAP) =============
print("\n🎨 CREATING MAP 4: Simplified Comparison...")

fig4, ax4 = plt.subplots(figsize=(14, 12))

# Base terrain
ax4.imshow(elev_display, cmap='gray', extent=extent, alpha=0.3,
          vmin=np.nanpercentile(elevation, 5),
          vmax=np.nanpercentile(elevation, 95))

# Overlay both old and new for comparison
# Old definition in blue (semi-transparent)
old_display = elevation.copy()
old_display[~old_mask] = np.nan
ax4.imshow(old_display, cmap='Blues', extent=extent, alpha=0.4)

# New definition in red (semi-transparent)
new_display = np.zeros_like(elevation)
new_display[final_mask] = 1
new_display[~final_mask] = np.nan
ax4.imshow(new_display, cmap='Reds', extent=extent, alpha=0.6)

# Areas that overlap (both old and new)
overlap_mask = old_mask & final_mask
if np.any(overlap_mask):
    overlap_display = np.zeros_like(elevation)
    overlap_display[overlap_mask] = 1
    overlap_display[~overlap_mask] = np.nan
    ax4.imshow(overlap_display, cmap='Purples', extent=extent, alpha=0.7)

# Add state borders
india_states.boundary.plot(ax=ax4, color='black', linewidth=1, alpha=0.7)

# Calculate overlap statistics
overlap_area = np.sum(overlap_mask) * pixel_area_km2
only_old_area = np.sum(old_mask & ~final_mask) * pixel_area_km2
only_new_area = np.sum(final_mask & ~old_mask) * pixel_area_km2

# Add statistics box
stats_text = f"""COMPARISON: OLD vs NEW DEFINITION

OLD (>200m elevation):
• Total area: {old_area:,.0f} km²

NEW (prominence + clustering):
• Total area: {final_area:,.0f} km²

OVERLAP ANALYSIS:
• Both definitions: {overlap_area:,.0f} km²
• Only in old: {only_old_area:,.0f} km²
• Only in new: {only_new_area:,.0f} km²

CHANGE: {final_area - old_area:,.0f} km² ({((final_area/old_area)-1)*100:.1f}%)"""

ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.95,
                 edgecolor='black', linewidth=1),
        verticalalignment='top', fontfamily='monospace')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.4, label=f'Old only: {only_old_area:,.0f} km²'),
    Patch(facecolor='red', alpha=0.6, label=f'New only: {only_new_area:,.0f} km²'),
    Patch(facecolor='purple', alpha=0.7, label=f'Both: {overlap_area:,.0f} km²'),
    Patch(facecolor='gray', alpha=0.3, label='Base terrain')
]
ax4.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.9)

ax4.set_title('Aravalli Range: Old vs New Definition Comparison', 
             fontsize=18, fontweight='bold', pad=20)
ax4.set_xlabel('Longitude', fontsize=12)
ax4.set_ylabel('Latitude', fontsize=12)

plt.tight_layout()
plt.savefig('map4_simplified_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✓ Saved: map4_simplified_comparison.png")

# ============= SUMMARY STATISTICS =============
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"1. OLD DEFINITION (elevation > 200m):")
print(f"   • Area: {old_area:,.0f} km²")
print(f"   • Pixels: {np.sum(old_mask):,}")
print()
print(f"2. NEW DEFINITION ANALYSIS:")
print(f"   Step 1 - Individual hills (≥100m prominence):")
print(f"   • Area: {initial_hill_area:,.0f} km²")
print(f"   • Pixels: {np.sum(hill_mask):,}")
print()
print(f"   Step 2 - After 500m clustering rule:")
print(f"   • Area: {final_area:,.0f} km²")
print(f"   • Pixels: {np.sum(final_mask):,}")
print(f"   • Number of hill clusters: {num_clusters}")
print()
print(f"3. COMPARISON:")
print(f"   • Change in area: {final_area - old_area:,.0f} km²")
if old_area > 0:
    print(f"   • Percentage change: {((final_area/old_area)-1)*100:.1f}%")
print(f"   • Overlap area: {overlap_area:,.0f} km²")
print(f"   • Only in old: {only_old_area:,.0f} km²")
print(f"   • Only in new: {only_new_area:,.0f} km²")
print("="*60)

plt.show()
print("\n✅ All analyses and visualizations complete!")

final_mask = np.load('final_mask.npy')
hill_mask = np.load('hill_mask.npy')
elevation = np.load('elevation.npy')