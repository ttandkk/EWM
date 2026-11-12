"""Deterministic, original packaging/pattern atlases (no external artwork).

Six tiles: front/right/back across the top, left/top/bottom across the bottom.
Pillow is a build-time dependency; Isaac only loads the generated PNG files.
"""
from pathlib import Path
import math
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

SIZE = 512


def font(size, bold=False):
    # Pillow's bundled font keeps the generator portable across deployment nodes.
    return ImageFont.load_default(size=size)


def text(draw, xy, value, size, fill, anchor='mm'):
    draw.text(xy, value, font=font(size), fill=fill, anchor=anchor, align='center')


def flower(draw, x, y, radius, ink, center):
    for a in np.linspace(0, 2*math.pi, 8, endpoint=False):
        cx, cy = x+radius*.65*math.cos(a), y+radius*.65*math.sin(a)
        r=radius*.43
        draw.ellipse((cx-r,cy-r,cx+r,cy+r), fill=ink)
    r=radius*.22
    draw.ellipse((x-r,y-r,x+r,y+r), fill=center)


def floral(face, ceramic=False):
    img=Image.new('RGB',(SIZE,SIZE),(243,242,227) if ceramic else (62,155,206))
    d=ImageDraw.Draw(img)
    ink=(32,67,131) if ceramic else (229,242,242)
    for y in range(-40,560,128):
        for x in range(-40,560,128):
            xx=x+(64 if (y//128)%2 else 0)
            r=40 if ceramic else 26
            flower(d,xx,y,r,ink,(226,231,228) if ceramic else (91,176,219))
            d.arc((xx-58,y-62,xx+58,y+62),35,285,fill=ink,width=4)
            for dx,dy in ((48,38),(-40,-40)):
                d.ellipse((xx+dx-8,y+dy-17,xx+dx+8,y+dy+17),fill=ink)
    if ceramic:
        d.rectangle((7,7,504,504),outline=ink,width=5)
        d.rectangle((18,18,493,493),outline=ink,width=2)
    return img


def wood(face):
    rng=np.random.default_rng(782+face)
    y,x=np.mgrid[0:SIZE,0:SIZE].astype(float)
    if face>=4:
        r=np.sqrt(((x-55)/1.3)**2+((y+160)/.8)**2)
        grain=np.sin(r*.13+2*np.sin(r*.015))+.35*np.sin(r*.55)
    else:
        warp=x+13*np.sin(y*.009)+7*np.sin(y*.025+x*.007)
        grain=np.sin(warp*.22+3*np.sin(warp*.018))+.28*np.sin(warp*.8)
    value=grain*7+rng.normal(0,1.3,(SIZE,SIZE))
    rgb=np.array([190,146,94])[None,None,:]+value[:,:,None]*[1,.85,.65]
    return Image.fromarray(np.uint8(np.clip(rgb,0,255)))


def tile(kind, face):
    if kind=='wooden_block': return wood(face)
    if kind in ('tissue_box','ceramic_box'):return floral(face,kind=='ceramic_box')
    img=Image.new('RGB',(SIZE,SIZE),(239,235,218));d=ImageDraw.Draw(img)
    if kind=='rubik_cube':
        d.rectangle((0,0,511,511),fill=(18,20,20))
        colors=[(223,33,21),(255,179,12),(245,239,215),(27,133,70),(13,90,190),(244,89,10)]
        ids=((3,5,2,4,1,4,0,2,1),(5,2,4,1,0,3,2,4,0),(2,3,0,5,4,1,3,0,2),
             (4,0,1,2,3,5,0,5,3),(3,5,2,4,1,2,0,4,1),(1,3,0,5,2,4,0,1,2))[face]
        for i,c in enumerate(ids):
            x,y=(i%3)*170,(i//3)*170
            d.rounded_rectangle((x+11,y+11,x+159,y+159),radius=8,fill=colors[c])
    elif kind=='checker_cube':
        for y in range(4):
            for x in range(4):
                d.rectangle((x*128,y*128,(x+1)*128,(y+1)*128),fill=(24,29,27) if (x+y)%2 else (235,237,231))
    elif kind=='marker_cube':
        d.rectangle((0,0,511,511),fill=(238,239,229))
        d.rectangle((26,26,485,485),fill=(20,22,22))
        d.rectangle((56,56,455,455),fill=(242,242,234))
        # Decorative fiducial-like motifs, deliberately not a detector dictionary.
        rng=np.random.default_rng(149+face)
        bits=rng.integers(0,2,(5,5));bits[0,:]=1;bits[:,0]=1
        for y in range(5):
            for x in range(5):
                if bits[y,x]:d.rectangle((91+x*66,91+y*66,91+(x+1)*66,91+(y+1)*66),fill=(16,19,20))
    elif kind=='dice':
        d.rectangle((0,0,511,511),fill=(205,22,17))
        n=(5,1,2,6,3,4)[face]
        positions={1:[(1,1)],2:[(0,0),(2,2)],3:[(0,0),(1,1),(2,2)],
                   4:[(0,0),(2,0),(0,2),(2,2)],5:[(0,0),(2,0),(1,1),(0,2),(2,2)],
                   6:[(0,0),(0,1),(0,2),(2,0),(2,1),(2,2)]}[n]
        for x,y in positions:
            cx,cy=124+x*132,124+y*132
            d.ellipse((cx-36,cy-36,cx+36,cy+36),fill=(248,243,220))
    elif kind=='cardboard_box':
        rng=np.random.default_rng(200+face)
        a=np.array([166,124,77])+rng.normal(0,2.0,(512,512,1))
        img=Image.fromarray(np.uint8(np.clip(a,0,255)));d=ImageDraw.Draw(img)
        if face>=4:
            d.line((256,0,256,511),fill=(61,46,29),width=5)
            d.rectangle((221,0,291,511),fill=(187,154,110))
            d.line((0,256,511,256),fill=(71,51,30),width=2)
            d.rectangle((0,223,511,290),fill=(190,157,115))
        else:
            d.rectangle((221,0,291,95),fill=(187,154,110))
            for x in (44,187,330):d.rectangle((x,350,x+120,477),outline=(40,33,25),width=6)
            # Up arrows, fragile glass, umbrella (handling symbols).
            for x in (76,132):
                d.line((x,445,x,382),fill=(40,33,25),width=9)
                d.polygon([(x,369),(x-17,394),(x+17,394)],fill=(40,33,25))
            d.polygon([(220,375),(278,375),(268,411),(230,411)],fill=(40,33,25))
            d.line((249,409,249,450),fill=(40,33,25),width=7);d.line((227,450,271,450),fill=(40,33,25),width=7)
            d.pieslice((351,376,433,442),180,360,fill=(40,33,25));d.line((392,402,392,451),fill=(40,33,25),width=6)
    elif kind=='milk_box':
        d.rectangle((0,0,511,511),fill=(234,239,227))
        poly=[(0,512),(0,405)]+[(x,390+int(26*math.sin(x*.014))) for x in range(0,513,8)]+[(512,512)]
        d.polygon(poly,fill=(21,100,165))
        d.rectangle((0,0,511,60),fill=(24,112,181))
        text(d,(256,190),'MILK',94,(20,86,145));text(d,(256,272),'FRESH DAILY',29,(42,99,142))
        text(d,(256,463),'WHOLE MILK',27,(241,244,230))
    elif kind=='cereal_box':
        d.rectangle((0,0,511,511),fill=(243,234,206))
        text(d,(256,65),'Morning',57,(176,40,29))
        text(d,(256,175),'CORN\nFLAKES',76,(48,43,29))
        text(d,(256,280),'CRISP & GOLDEN',26,(141,88,25))
        d.ellipse((48,337,465,580),fill=(249,248,230),outline=(201,183,145),width=6)
        rng=np.random.default_rng(72+face)
        for _ in range(100):
            x,y=rng.uniform(82,430),rng.uniform(351,505)
            if ((x-256)/178)**2+((y-465)/116)**2<1:
                r=rng.uniform(9,18)
                d.ellipse((x-r,y-r*.65,x+r,y+r*.65),fill=(int(rng.integers(205,241)),int(rng.integers(134,183)),34),outline=(186,114,24),width=2)
    elif kind=='juice_box':
        d.rectangle((0,0,511,511),fill=(248,165,24))
        d.rectangle((20,20,491,238),fill=(248,239,205))
        text(d,(256,96),'ORANGE',68,(199,104,13));text(d,(256,177),'JUICE',70,(199,104,13))
        d.ellipse((105,265,408,568),fill=(253,230,151),outline=(255,251,219),width=8)
        for i in range(10):
            a0=i*36+3;a1=(i+1)*36-3
            d.pieslice((116,276,397,557),a0,a1,fill=(247,156,15),outline=(255,233,162),width=3)
        d.ellipse((361,250,463,288),fill=(69,133,39))
    elif kind=='gift_box':
        d.rectangle((0,0,511,511),fill=(149,24,26))
        for y in range(38,512,100):
            for x in range(38,512,100):
                flower(d,x,y,14,(179,75,33),(191,104,42))
    else: raise ValueError(kind)
    return img


def generate_textures(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    kinds=('rubik_cube','wooden_block','checker_cube','marker_cube','dice','cardboard_box',
           'tissue_box','milk_box','cereal_box','juice_box','gift_box','ceramic_box')
    for kind in kinds:
        atlas=Image.new('RGB',(SIZE*3,SIZE*2))
        for face in range(6):atlas.paste(tile(kind,face),((face%3)*SIZE,(face//3)*SIZE))
        temporary=output/f'.{kind}.tmp.png'
        atlas.save(temporary,optimize=True)
        temporary.replace(output/f'{kind}.png')
    return kinds
