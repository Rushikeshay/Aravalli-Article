from pathlib import Path
import numpy as np
import json
import rasterio

# Path to the current file: Aravali/Code/script.py
BASE_DIR = Path(__file__).resolve().parent.parent
# BASE_DIR -> Aravali/

DATA_DIR = BASE_DIR / "Data"

print("🔄 Pre-computing longitude-based statistics...")

# ============= LOAD DATA =============
final_mask = np.load(DATA_DIR / "final_mask.npy")
elevation = np.load(DATA_DIR / "elevation.npy")

dem_path = DATA_DIR / "Aravali_data.tif"

with rasterio.open(dem_path) as src:
    bounds = src.bounds
    transform = src.transform
    pixel_size_deg_x = transform[0]
    pixel_size_deg_y = -transform[4]
    pixel_size_m_x = pixel_size_deg_x * 111111
    pixel_area_km2 = (pixel_size_m_x ** 2) / 1_000_000

# Calculate old mask
old_threshold = 200
old_mask = elevation > old_threshold

print(f"✓ Data loaded")
print(f"  Array shape: {elevation.shape}")
print(f"  Pixel area: {pixel_area_km2:.6f} km²")

# ============= COMPUTE CUMULATIVE STATISTICS =============
num_cols = elevation.shape[1]
num_slices = 100  # Calculate for every 1% of longitude

# Storage for results
statistics = []

print(f"\n📊 Computing statistics for {num_slices} longitude slices...")

for i in range(num_slices + 1):  # 0% to 100%
    # Calculate which column this percentage corresponds to
    col_index = int((i / 100) * num_cols)
    col_index = min(col_index, num_cols)  # Ensure we don't exceed bounds
    
    # Get slice from left edge to this column
    elevation_slice = elevation[:, :col_index]
    old_mask_slice = old_mask[:, :col_index]
    final_mask_slice = final_mask[:, :col_index]
    
    # Calculate areas
    old_area = np.sum(old_mask_slice) * pixel_area_km2
    new_area = np.sum(final_mask_slice) * pixel_area_km2
    overlap_area = np.sum(old_mask_slice & final_mask_slice) * pixel_area_km2
    only_old_area = np.sum(old_mask_slice & ~final_mask_slice) * pixel_area_km2
    only_new_area = np.sum(final_mask_slice & ~old_mask_slice) * pixel_area_km2
    
    # Calculate longitude at this position
    lon = bounds.left + col_index * pixel_size_deg_x
    
    statistics.append({
        'percentage': i,
        'longitude': float(lon),
        'oldArea': float(old_area),
        'newArea': float(new_area),
        'overlapArea': float(overlap_area),
        'onlyOldArea': float(only_old_area),
        'onlyNewArea': float(only_new_area)
    })
    
    if i % 10 == 0:
        print(f"  {i}% complete... (lon: {lon:.2f}°, old: {old_area:,.0f} km², new: {new_area:,.0f} km²)")

print("\n✓ Computation complete!")

# ============= SAVE TO JSON =============
output_data = {
    'metadata': {
        'description': 'Cumulative area statistics from west to east',
        'pixelAreaKm2': float(pixel_area_km2),
        'bounds': {
            'left': float(bounds.left),
            'right': float(bounds.right),
            'bottom': float(bounds.bottom),
            'top': float(bounds.top)
        },
        'arrayShape': elevation.shape,
        'numSlices': num_slices
    },
    'statistics': statistics
}

with open(DATA_DIR / 'aravalli_stats.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n💾 Saved to: aravalli_stats.json")
print(f"   File size: {len(json.dumps(output_data)) / 1024:.1f} KB")

# ============= DISPLAY SUMMARY =============
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total statistics calculated: {len(statistics)}")
print(f"Longitude range: {statistics[0]['longitude']:.2f}° to {statistics[-1]['longitude']:.2f}°")
print(f"\nFinal totals (100% of map):")
print(f"  OLD definition: {statistics[-1]['oldArea']:,.0f} km²")
print(f"  NEW definition: {statistics[-1]['newArea']:,.0f} km²")
print(f"  Overlap: {statistics[-1]['overlapArea']:,.0f} km²")
print(f"  Only OLD (lost): {statistics[-1]['onlyOldArea']:,.0f} km²")
print(f"  Only NEW (gained): {statistics[-1]['onlyNewArea']:,.0f} km²")
print("="*70)
print("\n✅ Ready to use in React component!")