import json
import re

html_path = "Segmentation Studio Wireframes.html"

with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

template_match = re.search(r'<script type="__bundler/template">\s*([\s\S]*?)\s*</script>', html_content)
template_str = template_match.group(1).strip()
template_html = json.loads(template_str)

# Find instances of bell or notification
for match in re.finditer(r'(?:bell|notification)', template_html, re.IGNORECASE):
    start = max(0, match.start() - 150)
    end = min(len(template_html), match.end() + 150)
    print("--- MATCH ---")
    print(template_html[start:end])
