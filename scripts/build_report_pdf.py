"""Render docs/report.md to docs/report.pdf via headless Chrome.

Chrome is used because it renders MathJax before printing, so the LaTeX in the
report survives into the PDF. Pandoc/LaTeX are not required.

Usage:  python scripts/build_report_pdf.py
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import markdown

CHROME_CANDIDATES = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
]

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script>
  window.MathJax = {{
    tex: {{inlineMath: [['$', '$']], displayMath: [['$$', '$$']]}},
    options: {{skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']}}
  }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script"></script>
<style>
  @page {{ size: letter; margin: 22mm 20mm; }}
  html {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  body {{
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 10.5pt; line-height: 1.55; color: #16161a;
    max-width: 100%; margin: 0;
  }}
  h1 {{ font-size: 19pt; line-height: 1.25; margin: 0 0 0.6rem; }}
  h2 {{ font-size: 13pt; margin: 1.6rem 0 0.5rem; padding-bottom: 0.2rem;
       border-bottom: 1px solid #d9d8d2; page-break-after: avoid; }}
  h3 {{ font-size: 11.5pt; margin: 1.1rem 0 0.4rem; page-break-after: avoid; }}
  p {{ margin: 0 0 0.7rem; }}
  a {{ color: #1c5cab; text-decoration: none; }}
  code {{ font-family: 'Consolas', 'SF Mono', monospace; font-size: 9pt;
         background: #f2f1ec; padding: 0.1em 0.3em; border-radius: 3px; }}
  pre {{ background: #f7f6f2; border: 1px solid #e3e2dc; border-radius: 4px;
        padding: 0.7rem 0.9rem; overflow-x: auto; page-break-inside: avoid; }}
  pre code {{ background: none; padding: 0; font-size: 8.8pt; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.9rem 0;
          font-size: 9.5pt; page-break-inside: avoid; }}
  th, td {{ border: 1px solid #d9d8d2; padding: 0.4rem 0.55rem; text-align: left; }}
  th {{ background: #f2f1ec; font-weight: 700; }}
  td:nth-child(n+2), th:nth-child(n+2) {{ text-align: right; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 1rem auto;
        page-break-inside: avoid; }}
  .byline {{ font-size: 10.5pt; line-height: 1.5; color: #45443f;
            margin: 0 0 1.6rem; padding-bottom: 1rem;
            border-bottom: 2px solid #16161a; }}
  .byline strong {{ color: #16161a; font-size: 11.5pt; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def find_chrome(explicit=None):
    if explicit:
        return explicit
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    found = shutil.which('chrome') or shutil.which('google-chrome')
    if found:
        return found
    sys.exit('Chrome not found; pass --chrome PATH')


def build_html(md_path):
    text = open(md_path, encoding='utf-8').read()
    html = markdown.markdown(
        text, extensions=['tables', 'fenced_code', 'attr_list', 'md_in_html']
    )
    # The byline block sits between the <h1> and the Abstract heading. Markdown
    # joins its lines into one paragraph, so restore the line breaks.
    start = html.find('<p><strong>Jeorge D. Anderson II</strong>')
    if start != -1:
        end = html.find('</p>', start) + len('</p>')
        byline = html[start:end]
        html = html[:start] + (
            byline.replace('<p>', '<p class="byline">', 1).replace('\n', '<br>\n')
        ) + html[end:]
    title = text.split('\n', 1)[0].lstrip('# ').strip()
    return PAGE_TEMPLATE.format(title=title, body=html)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--md', default=os.path.join(repo, 'docs', 'report.md'))
    parser.add_argument('--output', default=os.path.join(repo, 'docs', 'report.pdf'))
    parser.add_argument('--chrome', default=None)
    args = parser.parse_args()

    chrome = find_chrome(args.chrome)
    docs_dir = os.path.dirname(os.path.abspath(args.md))

    # Write the intermediate HTML beside the markdown so relative image paths resolve.
    fd, tmp_html = tempfile.mkstemp(suffix='.html', dir=docs_dir)
    os.close(fd)
    try:
        with open(tmp_html, 'w', encoding='utf-8') as f:
            f.write(build_html(args.md))

        cmd = [
            chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
            '--no-pdf-header-footer',
            '--virtual-time-budget=20000',      # let MathJax typeset first
            f'--print-to-pdf={args.output}',
            'file:///' + tmp_html.replace('\\', '/'),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if not os.path.exists(args.output):
            sys.exit(f'Chrome did not produce a PDF:\n{result.stderr[:800]}')
    finally:
        os.remove(tmp_html)

    size = os.path.getsize(args.output)
    print(f'Wrote {args.output} ({size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
