import re

html_path = "Segmentation Studio Wireframes.html"

with open(html_path, "r", encoding="utf-8") as f:
    content = f.read()

# Locate the template script content
template_match = re.search(r'<script type="__bundler/template">\s*([\s\S]*?)\s*</script>', content)
if not template_match:
    print("Error: template not found")
    exit(1)

template_str = template_match.group(1).strip()

# Replace any occurrence of </script> (case-insensitive or normal) with <\/script>
# We must escape the backslash in python: "<\\/script>"
fixed_template_str = template_str.replace("</script>", "<\\/script>")
fixed_template_str = fixed_template_str.replace("</Script>", "<\\/Script>")

# Replace in content
new_content = content.replace(template_str, fixed_template_str)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Successfully escaped script tags in template JSON!")
