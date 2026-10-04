"""Wrap AI-generated animation frames in a self-contained, animated SVG."""
import base64
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'asset/images/pai-chet-ng-ai-avatar-v4-frames.png'
OUTPUT = ROOT / 'asset/images/pai-chet-ng-ai-avatar-animated-v4.svg'


def main():
    data = SOURCE.read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Animation frames must be a PNG image.')
    width, height = struct.unpack('>II', data[16:24])
    if width % 3 or height % 2 or width // 3 != height // 2:
        raise ValueError('Expected six square frames in a 3 by 2 grid.')
    size = width // 3
    image = base64.b64encode(data).decode('ascii')
    frames = []
    for index in range(6):
        delay = -((6 - index) % 6) * 2
        x, y = -(index % 3) * size, -(index // 3) * size
        frames.append(f'<g class="frame frame-{index}" style="animation-delay:{delay}s"><use href="#frames" x="{x}" y="{y}"/></g>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" aria-labelledby="title description">
<title id="title">Pai Chet Ng · animated avatar</title>
<desc id="description">A cartoon portrait with gentle hair movement and a subtle professional smile.</desc>
<style>
.frame{{opacity:0;animation:avatar-frame 12s linear infinite}}
.frame-0{{opacity:1}}
@keyframes avatar-frame{{0%,11.6667%{{opacity:1}}16.6667%,95%{{opacity:0}}100%{{opacity:1}}}}
@media(prefers-reduced-motion:reduce){{.frame{{animation:none;opacity:0}}.frame-0{{opacity:1}}}}
</style>
<defs><image id="frames" width="{width}" height="{height}" href="data:image/png;base64,{image}"/><clipPath id="portrait"><rect width="{size}" height="{size}"/></clipPath></defs>
<g clip-path="url(#portrait)">{''.join(frames)}</g>
</svg>'''
    OUTPUT.write_text(svg + '\n')
    print(f'Built animated avatar: six {size}px frames, 12-second loop.')


if __name__ == '__main__':
    main()
