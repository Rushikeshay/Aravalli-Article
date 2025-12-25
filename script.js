const container = d3.select("#map");
const rightMap = d3.select("#right-map");
const slider = d3.select("#slider");

const oldEl = d3.select("#old");
const newEl = d3.select("#new");
const lostEl = d3.select("#lost");

const width = 600; // Fixed square size

// ---------- helpers ----------
function format(value) {
  return Math.round(value).toLocaleString();
}

// Linear interpolation between two points
function interpolate(stats, pct) {
  const i = Math.floor(pct);
  const j = Math.min(i + 1, stats.length - 1);
  const t = pct - i;

  const a = stats[i];
  const b = stats[j];

  return {
    oldArea: a.oldArea + (b.oldArea - a.oldArea) * t,
    newArea: a.newArea + (b.newArea - a.newArea) * t,
    onlyOldArea: a.onlyOldArea + (b.onlyOldArea - a.onlyOldArea) * t,
    onlyNewArea: a.onlyNewArea + (b.onlyNewArea - a.onlyNewArea) * t,
    longitude: a.longitude + (b.longitude - a.longitude) * t
  };
}

// ---------- main ----------
d3.json("data/aravalli_stats.json").then(data => {
  const stats = data.statistics;

  function updateFromX(x) {
    const clampedX = Math.max(0, Math.min(width, x));
    const pct = (clampedX / width) * 100;

    // Visual slider + clip
    slider.style("left", `${pct}%`);
    rightMap.style("clip-path", `inset(0 0 0 ${pct}%)`);

    // Data lookup
    const stat = interpolate(stats, pct);

    // Update numbers with animation
    oldEl.transition().duration(300).text(format(stat.oldArea));
    newEl.transition().duration(300).text(format(stat.newArea));
    lostEl.transition().duration(300).text(format(stat.onlyOldArea));
  }

  // Initial state (middle)
  updateFromX(width / 2);

  // Drag behavior
  let isDragging = false;
  
  slider.call(
    d3.drag()
      .on("start", () => {
        isDragging = true;
        slider.style("cursor", "grabbing");
        slider.style("width", "8px");
      })
      .on("drag", (event) => {
        const [x] = d3.pointer(event, container.node());
        updateFromX(x);
      })
      .on("end", () => {
        isDragging = false;
        slider.style("cursor", "ew-resize");
        slider.style("width", "4px");
      })
  );

  // Click anywhere on map
  container.on("click", (event) => {
    if (!isDragging) {
      const [x] = d3.pointer(event);
      updateFromX(x);
    }
  });

}).catch(err => {
  console.error("Failed to load aravalli_stats.json", err);
  // Display error message to user
  d3.select("#stats").html(
    '<div style="color: #e74c3c; text-align: center; padding: 20px; font-weight: bold;">Error loading data. Please check if aravalli_stats.json exists.</div>'
  );
});