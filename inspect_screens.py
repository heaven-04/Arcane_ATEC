import pathlib
import sys
from html.parser import HTMLParser

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

class ScreenParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.in_title = False
        self.headers = []
        self.current_tag = None
        self.current_text = []
        self.buttons = []
        self.inputs = []
        self.ids = []
        self.scripts = []
        self.in_script = False
        self.script_src = None
        self.script_content = []

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        if "id" in attr_dict:
            self.ids.append(attr_dict["id"])
        if tag == "title":
            self.in_title = True
        elif tag in ["h1", "h2", "h3", "h4"]:
            self.current_tag = tag
            self.current_text = []
        elif tag == "button":
            self.current_tag = "button"
            self.current_text = []
        elif tag in ["input", "select"]:
            t = attr_dict.get("type", tag)
            ident = attr_dict.get("id") or attr_dict.get("name") or "unnamed"
            self.inputs.append(f"{t}: {ident}")
        elif tag == "script":
            self.in_script = True
            self.script_src = attr_dict.get("src")
            self.script_content = []

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        elif tag in ["h1", "h2", "h3", "h4"]:
            text = "".join(self.current_text).strip()
            if text:
                self.headers.append(f"<{tag}> {text}")
            self.current_tag = None
        elif tag == "button":
            text = "".join(self.current_text).strip()
            if text:
                self.buttons.append(text)
            self.current_tag = None
        elif tag == "script":
            self.in_script = False
            content = "".join(self.script_content).strip()
            self.scripts.append((self.script_src, content))
            self.script_src = None

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif self.current_tag:
            self.current_text.append(data)
        elif self.in_script:
            self.script_content.append(data)


dash = pathlib.Path("dashboard")
for name in ["screen 1.html", "screen 2.html", "screen 3.html"]:
    p = dash / name
    if not p.exists():
        continue
    parser = ScreenParser()
    parser.feed(p.read_text(encoding="utf-8", errors="ignore"))
    print(f"=== {name} ===")
    print(f"Title: {parser.title.strip()}")
    print(f"Headers ({len(parser.headers)}): {parser.headers[:10]}")
    print(f"Buttons ({len(parser.buttons)}): {parser.buttons[:10]}")
    print(f"Inputs/Selects ({len(parser.inputs)}): {parser.inputs[:10]}")
    print(f"All IDs ({len(parser.ids)}): {parser.ids}")
    print(f"Scripts count: {len(parser.scripts)}")
    for i, (src, cont) in enumerate(parser.scripts):
        prev = cont[:80].replace("\n", " ") if cont else "(empty)"
        print(f"  Script {i}: src={src}, preview={prev}")
    print("=" * 60)
