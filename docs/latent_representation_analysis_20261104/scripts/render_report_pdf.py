#!/usr/bin/env python3
"""Render this report to PDF; requires Markdown, Matplotlib, Pillow and WeasyPrint.

CJK font must be supplied explicitly. All temporary images remain in a temporary
folder, and output is created exclusively so existing files cannot be overwritten.
"""
import argparse
import base64
import html
import io
import re
import tempfile
from pathlib import Path

import markdown
import matplotlib
matplotlib.use('Agg')
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from PIL import Image
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
MATH = re.compile(r'\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)', re.S)
IMAGES = re.compile(r'!\[([^]]*)\]\(([^)]+)\)')


def math_svg(source):
    display = source.startswith('$$')
    tex = source[2:-2] if display else source[1:-1]
    tex = (tex.replace(r'\tfrac32', r'\frac{3}{2}')
           .replace(r'\mathbb E', r'\mathbb{E}')
           .replace(r'\lVert', r'\Vert').replace(r'\rVert', r'\Vert')
           .replace(r'\le ', r'\leq '))
    buffer = io.BytesIO()
    mathtext.math_to_image('$' + tex + '$', buffer,
                           prop=FontProperties(size=11), format='svg')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    element = f'<img class="math" alt="{html.escape(tex, quote=True)}" src="data:image/svg+xml;base64,{encoded}">'
    return f'<div class="display-math">{element}</div>' if display else element


def image_html(alt, image_path, temporary):
    path = (ROOT / image_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        width, height = image.size
        if height / width <= 1.5:
            return f'<figure><img src="{path.as_uri()}"><figcaption>{html.escape(alt)}</figcaption></figure>'
        # The two multi-task grids are tall; split at nearly-white row gaps so
        # task titles, axes, and labels stay on the same page and readable.
        gray = image.convert('L')
        import numpy as np
        pixels = np.asarray(gray)
        cuts = [0]
        count = 4
        for n in range(1, count):
            target = round(height * n / count)
            a, b = max(cuts[-1] + 1, target - height // 30), min(height, target + height // 30)
            nonwhite = (pixels[a:b] < 248).sum(axis=1)
            gaps = np.diff(np.r_[False, nonwhite == 0, False].astype(int))
            starts = np.flatnonzero(gaps == 1) + a
            ends = np.flatnonzero(gaps == -1) + a
            # Prefer a wide gap just below the nominal cut: this lies after
            # the current row's x-axis label, before the next row's title.
            centers = np.array([(lo + hi) // 2 for lo, hi in zip(starts, ends) if hi - lo >= 10])
            if not len(centers):
                raise ValueError(f'No safe whitespace gap near {target} in {path.name}')
            desired = target + height // 75
            cuts.append(int(centers[np.argmin(abs(centers - desired))]))
        cuts.append(height)
        parts = []
        for i, (top, bottom) in enumerate(zip(cuts, cuts[1:])):
            out = temporary / f'{path.stem}_part_{i+1}.png'
            image.crop((0, top, width, bottom)).save(out)
            caption = f'{alt}（分图 {i+1}/{count}）'
            parts.append(f'<figure><img src="{out.as_uri()}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', required=True, type=Path, help='CJK font file, e.g. NotoSansCJKsc-Regular.otf')
    parser.add_argument('--output', required=True, type=Path, help='PDF path, must not exist')
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error('font file does not exist')
    if args.output.exists():
        parser.error('output already exists')
    source = (ROOT / 'README.md').read_text(encoding='utf-8')
    tokens = {}
    with tempfile.TemporaryDirectory(prefix='ewm-report-render-') as tmp:
        tmp = Path(tmp)
        def replace_math(match):
            key = f'EWMMATHTOKEN{len(tokens)}END'
            tokens[key] = math_svg(match.group(0))
            return key
        def replace_image(match):
            key = f'EWMIMAGETOKEN{len(tokens)}END'
            tokens[key] = image_html(match.group(1), match.group(2), tmp)
            return key
        text = MATH.sub(replace_math, source)
        text = IMAGES.sub(replace_image, text)
        body = markdown.markdown(text, extensions=['tables', 'fenced_code'])
        for key, value in tokens.items():
            # Standalone images/equations were placed into an enclosing paragraph.
            body = body.replace(f'<p>{key}</p>', value).replace(key, value)
        css = f'''<style>
@font-face {{ font-family: NotoCJK; src: url("{args.font.resolve().as_uri()}"); }}
@page {{ size: A4; margin: 18mm 17mm 17mm 17mm;
  @bottom-center {{ content: "第 " counter(page) " 页 / 共 " counter(pages) " 页"; font-family: NotoCJK; color: #64748b; font-size: 8pt; }} }}
body {{ font-family: NotoCJK, sans-serif; font-size: 9.6pt; line-height: 1.65; color: #17263a; }}
h1 {{ font-size: 20pt; line-height: 1.32; color: #152b48; margin: 0 0 10pt; }}
h2 {{ font-size: 14pt; color: #183a5f; margin: 14pt 0 5pt; border-bottom: .7pt solid #a8bcd0; padding-bottom: 3pt; break-after: avoid; }}
h3 {{ font-size: 11pt; color: #24476b; margin: 10pt 0 4pt; break-after: avoid; }}
p {{ margin: 4pt 0 7pt; orphans: 3; widows: 3; }}
ul {{ padding-left: 17pt; margin: 4pt 0 8pt; }} li {{ margin: 3pt 0; }}
a {{ color: #205a88; text-decoration: none; }} code {{ font-family: monospace; font-size: 8pt; overflow-wrap: anywhere; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.1pt; margin: 7pt 0 10pt; }}
thead {{ display: table-header-group; }} tr {{ break-inside: avoid; }}
th, td {{ border: .45pt solid #cbd5df; padding: 4pt 5pt; line-height: 1.35; overflow-wrap: anywhere; }}
th {{ background: #e8eff6; color: #183a5f; }} tbody tr:nth-child(even) {{ background: #f6f8fb; }}
figure {{ margin: 8pt 0 12pt; text-align: center; break-inside: avoid; }}
figure img {{ max-width: 100%; max-height: 195mm; width: auto; height: auto; }}
figcaption {{ font-size: 8pt; color: #52657a; margin-top: 4pt; }}
.math {{ display: inline; width: auto; height: auto; max-width: 100%; vertical-align: middle; }}
.display-math {{ text-align: center; margin: 8pt 0; break-inside: avoid; }}
.display-math .math {{ max-width: 100%; }}
</style>'''
        document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">' + css + '</head><body>' + body + '</body></html>'
        rendered = HTML(string=document, base_url=str(ROOT)).write_pdf()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as f:
            f.write(rendered)
        print(f'PDF: {args.output} | formulae: {len(MATH.findall(source))} | original figures: {len(IMAGES.findall(source))}')


if __name__ == '__main__':
    main()
