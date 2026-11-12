"""Build five original procedural ball skins and portable visual-only USD assets."""
import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from pxr import Sdf
from pxr import UsdShade
from piper_bench.scenes.build_tabletop_props import Builder
from piper_bench.scenes.ball_props import ASSET_DIR
from piper_bench.scenes.ball_props import ASSET_RADIUS
from piper_bench.scenes.ball_props import VERSION
from piper_bench.tasks.tennis.appearance import sphere_mesh
from piper_bench.tasks.tennis.appearance import seam_curve

SPECS = {
    'baseball': ('ivory leather / red double-curved stitching', .58),
    'globe_ball': ('blue ocean / stylized green continents', .32),
    'mini_basketball': ('orange leather / black panel channels', .62),
    'soccer_ball': ('twelve black pentagons / twenty white hexagons', .48),
    'billiard_ball': ('polished black resin / white circles and numeral eight', .22),
}


def sphere_uv(normals):
    return np.column_stack((np.arctan2(normals[:,1], normals[:,0])/(2*np.pi)+.5,
                            np.arcsin(np.clip(normals[:,2],-1,1))/np.pi+.5))


def texture_directions(width, height):
    lon=(np.arange(width)+.5)/width*2*np.pi-np.pi
    lat=np.pi/2-(np.arange(height)+.5)/height*np.pi
    return np.stack(np.broadcast_arrays(np.cos(lat)[:,None]*np.cos(lon)[None,:],
        np.cos(lat)[:,None]*np.sin(lon)[None,:],np.sin(lat)[:,None]),axis=-1).astype(np.float32)


def soccer_colors(directions):
    phi=(1+math.sqrt(5))/2
    vertices=np.array([(0,a,b*phi) for a in (-1,1) for b in (-1,1)]+
                      [(a,b*phi,0) for a in (-1,1) for b in (-1,1)]+
                      [(b*phi,0,a) for a in (-1,1) for b in (-1,1)],dtype=float)
    vertices/=np.linalg.norm(vertices,axis=1)[:,None]
    distances=np.linalg.norm(vertices[:,None]-vertices[None,:],axis=-1)
    edge=distances[distances>1e-6].min()
    normals=[];planes=[]
    for i,v in enumerate(vertices):
        neighbor=vertices[np.flatnonzero(np.isclose(distances[i],edge))[0]]
        normals.append(v);planes.append(v@((2*v+neighbor)/3))
    for i,j,k in combinations(range(12),3):
        if not all(np.isclose(distances[a,b],edge) for a,b in ((i,j),(j,k),(k,i))):continue
        n=vertices[[i,j,k]].mean(axis=0);n/=np.linalg.norm(n)
        normals.append(n);planes.append(n@vertices[i])
    assert len(normals)==32
    scores=directions@np.asarray(np.array(normals)/np.array(planes)[:,None],dtype=np.float32).T
    panel=scores.argmax(axis=-1);best=np.partition(scores,-2,axis=-1)[...,-2:]
    color=np.where((panel<12)[...,None],[19,22,23],[235,235,228]).astype(np.uint8)
    color[best[...,1]-best[...,0]<.008]=[57,58,54]
    return color


def draw_spherical_line(draw, points, size, fill, width):
    uv=sphere_uv(points/np.linalg.norm(points,axis=1)[:,None])
    u=np.unwrap(uv[:,0]*2*np.pi)/(2*np.pi)
    xy=np.column_stack((u*size[0],(1-uv[:,1])*size[1]))
    for shift in (-size[0],0,size[0]):
        draw.line([(float(x+shift),float(y)) for x,y in xy],fill=fill,width=width)


def baseball_texture(size):
    image=Image.new('RGB',size,(239,237,224));draw=ImageDraw.Draw(image)
    curve=seam_curve(np.linspace(0,2*np.pi,1801))
    draw_spherical_line(draw,curve,size,(181,148,133),2)
    for t in np.linspace(0,2*np.pi,100,endpoint=False):
        center=seam_curve(np.array([t]))[0]
        tangent=(seam_curve(np.array([t+.001]))-seam_curve(np.array([t-.001])))[0]
        tangent/=np.linalg.norm(tangent);side=np.cross(center,tangent)
        # Paired diagonal stitches bridge the inset panel seam.
        for sign in (-1,1):
            points=np.array([center+sign*.057*side-.023*tangent,center+sign*.008*side+.018*tangent])
            draw_spherical_line(draw,points,size,(183,31,28),3)
    return image


# Hand-drawn geographic silhouettes for a decorative globe, not a map dataset.
CONTINENTS = [
 [(-168,69),(-145,71),(-130,58),(-124,49),(-124,39),(-115,29),(-106,23),(-98,17),(-86,20),(-82,10),(-77,8),(-82,25),(-80,32),(-67,45),(-53,51),(-64,61),(-83,63),(-96,74),(-123,72)],
 [(-80,10),(-67,11),(-54,4),(-49,-3),(-35,-8),(-41,-20),(-50,-30),(-54,-40),(-67,-56),(-74,-44),(-72,-29),(-78,-13),(-81,0)],
 [(-53,60),(-42,60),(-20,72),(-25,82),(-48,84),(-63,76)],
 [(-17,36),(6,37),(13,33),(30,31),(34,26),(43,12),(51,11),(44,0),(40,-12),(32,-27),(20,-35),(11,-26),(10,-6),(-5,5),(-17,15)],
 [(-10,36),(-10,44),(0,49),(8,55),(5,60),(18,70),(29,70),(31,59),(45,55),(55,50),(55,40),(40,37),(31,31),(24,36),(13,45),(5,42)],
 [(28,70),(60,76),(99,76),(124,70),(159,65),(177,66),(165,52),(144,50),(140,40),(126,37),(121,22),(111,20),(108,9),(101,1),(96,8),(91,22),(80,8),(72,19),(66,25),(57,24),(50,14),(44,13),(35,31),(45,42),(31,55)],
 [(112,-11),(130,-12),(137,-10),(145,-18),(154,-28),(147,-39),(130,-33),(115,-35),(113,-24)],
 [(47,-13),(51,-17),(49,-25),(44,-24)],[(130,31),(137,35),(141,42),(146,44),(143,35)],
 [(95,5),(105,-6),(115,-8),(118,-3),(108,0)],[(119,-1),(124,1),(130,-4),(135,-5),(139,-8),(128,-9)],
 [(166,-35),(174,-40),(178,-44),(170,-47),(166,-42)],
]


def globe_texture(size):
    image=Image.new('RGB',size,(15,124,196));draw=ImageDraw.Draw(image)
    for polygon in CONTINENTS:
        draw.polygon([((lon+180)/360*size[0],(90-lat)/180*size[1]) for lon,lat in polygon],fill=(108,162,65))
    return image


def make_texture(kind, size=(1024,512)):
    directions=texture_directions(*size)
    if kind=='soccer_ball':return Image.fromarray(soccer_colors(directions))
    if kind=='baseball':return baseball_texture(size)
    if kind=='globe_ball':return globe_texture(size)
    if kind=='mini_basketball':
        seams=np.min(np.abs(directions@np.array([[1,0,0],[0,1,0],[0,0,1],[1/math.sqrt(2),1/math.sqrt(2),0]],dtype=np.float32).T),axis=-1)
        rng=np.random.default_rng(7103)
        color=np.array([222,91,19])[None,None,:]*rng.uniform(.975,1.025,(*seams.shape,1))
        color[seams<.018]=[25,23,20]
        return Image.fromarray(np.uint8(np.clip(color,0,255)))
    image=Image.new('RGB',size,(16,18,18));draw=ImageDraw.Draw(image)
    mask=abs(directions[...,0])>math.cos(.48)
    rgb=np.array(image);rgb[mask]=[241,240,226];image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',116)
    for u in (0,size[0]/2,size[0]):draw.text((u,size[1]/2),'8',font=font,anchor='mm',fill=(10,11,11),stroke_width=1)
    return image


def build(output):
    output.mkdir(parents=True,exist_ok=True);(output/'textures').mkdir(exist_ok=True)
    props={}
    for kind,(description,roughness) in SPECS.items():
        make_texture(kind).save(output/'textures'/f'{kind}.png')
        b=Builder(output,kind);material=b.material('Ball',roughness=roughness,texture=kind)
        UsdShade.Shader.Get(b.stage,'/Prop/Looks/Ball/Texture').GetInput('wrapS').Set('repeat')
        points,faces,normals=sphere_mesh(ASSET_RADIUS,64,128)
        uv=sphere_uv(normals)[faces].copy()
        crossing=np.ptp(uv[:,:,0],axis=1)>.5
        uv[crossing,:,0]+=uv[crossing,:,0]<.5
        for i in range(3):
            polar=np.abs(normals[faces[:,i],2])>.999999
            uv[polar,i,0]=uv[polar][:,[j for j in range(3) if j!=i],0].mean(axis=1)
        mesh=b.mesh('Ball',points+[0,0,ASSET_RADIUS],faces.ravel().tolist(),[3]*len(faces),material,normals)
        from pxr import UsdGeom
        from pxr import Vt
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying').Set(
            Vt.Vec2fArray.FromNumpy(uv.reshape(-1,2).astype(np.float32)))
        b.stage.GetDefaultPrim().CreateAttribute('piper:ball_skin',Sdf.ValueTypeNames.String).Set(kind)
        props[kind]=b.finish(description)
    (output/'catalog.json').write_text(json.dumps(dict(version=VERSION,units='metres',origin='bottom_center',
        visual_only=True,collision=False,artwork='Original procedural patterns; decorative globe silhouettes',props=props),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=ASSET_DIR)
    build(parser.parse_args().output_dir)
