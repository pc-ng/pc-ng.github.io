"""Animate one portrait using continuous, local SVG displacement fields."""
import base64
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'asset/images/pai-chet-ng-ai-avatar-cartoon-v3.png'
OUTPUT = ROOT / 'asset/images/pai-chet-ng-ai-avatar-smooth-v5.svg'


def svg_data(svg):
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg.encode()).decode()


def main():
    portrait = SOURCE.read_bytes()
    size, height = struct.unpack('>II', portrait[16:24])
    if size != height or size != 1254:
        raise ValueError('The movement fields are authored for the 1254px portrait.')
    wind = '''<svg xmlns="http://www.w3.org/2000/svg" width="1254" height="1254" viewBox="0 0 1254 1254">
<defs><linearGradient id="bend" x1="0" y1="530" x2="0" y2="1150" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="rgb(128,128,128)"/><stop offset=".3" stop-color="rgb(146,128,128)"/><stop offset=".65" stop-color="rgb(187,128,128)"/><stop offset="1" stop-color="rgb(246,128,128)"/></linearGradient><filter id="soft" x="-25%" y="-25%" width="150%" height="150%" color-interpolation-filters="sRGB"><feGaussianBlur stdDeviation="17"/></filter></defs>
<rect width="1254" height="1254" fill="rgb(128,128,128)"/>
<g filter="url(#soft)"><path d="M330 535 C245 575 165 632 112 738 C75 835 104 1040 277 1145 L505 1140 C455 1020 455 864 421 733 L371 598 Z" fill="url(#bend)"/><ellipse cx="688" cy="75" rx="175" ry="48" fill="rgb(192,128,128)"/></g>
</svg>'''
    smile = '''<svg xmlns="http://www.w3.org/2000/svg" width="1254" height="1254" viewBox="0 0 1254 1254">
<defs><filter id="soft" x="-70%" y="-90%" width="240%" height="280%" color-interpolation-filters="sRGB"><feGaussianBlur stdDeviation="12"/></filter></defs>
<rect width="1254" height="1254" fill="rgb(128,128,128)"/>
<g filter="url(#soft)" fill="rgb(128,225,128)"><ellipse cx="593" cy="651" rx="27" ry="22"/><ellipse cx="769" cy="651" rx="27" ry="22"/></g>
</svg>'''
    timing = 'dur="7s" repeatCount="indefinite" calcMode="spline" keyTimes="0;.25;.5;.75;1" keySplines=".33 .52 .66 1;.34 0 .67 .48;.33 .52 .66 1;.34 0 .67 .48"'
    image = base64.b64encode(portrait).decode()
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}" preserveAspectRatio="xMidYMid slice" role="img" aria-labelledby="avatar-title avatar-description">
<title id="avatar-title">Pai Chet Ng · gently animated avatar</title>
<desc id="avatar-description">One stable cartoon portrait with continuous, light hair movement and a subtle closed-mouth smile.</desc>
<style>.avatar-art{{filter:url(#avatar-motion)}}@media(prefers-reduced-motion:reduce){{.avatar-art{{filter:none}}}}</style>
<defs><filter id="avatar-motion" x="0" y="0" width="{size}" height="{size}" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">
<feImage href="{svg_data(wind)}" x="0" y="0" width="{size}" height="{size}" preserveAspectRatio="none" result="wind-field"/>
<feDisplacementMap id="hair-motion" in="SourceGraphic" in2="wind-field" scale="0" xChannelSelector="R" yChannelSelector="G" result="wind-portrait"><animate attributeName="scale" values="0;42;0;-42;0" {timing}/></feDisplacementMap>
<feImage href="{svg_data(smile)}" x="0" y="0" width="{size}" height="{size}" preserveAspectRatio="none" result="smile-field"/>
<feDisplacementMap id="smile-motion" in="wind-portrait" in2="smile-field" scale="0" xChannelSelector="R" yChannelSelector="G"><animate attributeName="scale" values="0;12;0;12;0" {timing}/></feDisplacementMap>
</filter></defs>
<image class="avatar-art" width="{size}" height="{size}" href="data:image/png;base64,{image}"/>
</svg>'''
    OUTPUT.write_text(svg + '\n')
    print('Built a continuous avatar from one unchanged portrait; seven-second loop.')


if __name__ == '__main__':
    main()
