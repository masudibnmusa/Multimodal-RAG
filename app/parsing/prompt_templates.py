LAYOUT_PROMPT = """You are a document layout analyzer. Find the visual regions on this page.
Types: "table", "chart" (plots with axes: bar, line, pie, scatter), "diagram" (flowcharts, architecture, network, UML), "figure" (photos or other images).
Skip normal body text, headings and page numbers.
Give each region a bounding box [x0, y0, x1, y1] as fractions of page width/height (0 to 1, origin top-left).
Return ONLY JSON: {"regions": [{"type": "...", "bbox": [x0, y0, x1, y1]}]}
If there are none, return {"regions": []}."""

TABLE_PROMPT = """Convert this table image into a GitHub-flavored markdown table.
Keep headers, units and every cell value exactly as shown. Flatten multi-level headers into single header cells.
Return ONLY the markdown table, no commentary."""

CHART_PROMPT = """Analyze this chart. Return ONLY JSON:
{"chart_type": "...", "title": "...", "description": "2-3 sentences on what it shows and the main trend",
 "axes": {"x": "label and unit", "y": "label and unit"},
 "data_points": [{"series": "...", "label": "x value or category", "value": "number as shown", "estimated": false}]}
Use printed data labels when present. If you must estimate from the axis, set "estimated": true."""

DIAGRAM_PROMPT = """Analyze this diagram. Return ONLY JSON:
{"caption": "1-2 sentence summary", "entities": ["every labeled box/node/component"],
 "relationships": [{"source": "...", "target": "...", "label": "arrow label or relation, if any"}]}"""

OCR_PROMPT = "Transcribe all text on this page in reading order. Return only the text."