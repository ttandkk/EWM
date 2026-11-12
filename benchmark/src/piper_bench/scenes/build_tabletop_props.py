"""Build the portable tabletop USD asset pack. Run with the Isaac Python env.

No SimulationApp or GPU is needed for authoring. All artwork is procedural and
original; marker_cube is decorative, not a calibrated ArUco/AprilTag target.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdShade
from pxr import Vt

from piper_bench.scenes.prop_textures import generate_textures
from piper_bench.scenes.tabletop_props import ASSET_DIR

# Body dimensions in metres; detail geometry may extend above these dimensions.
SPECS = {
    'rubik_cube': ((.06,.06,.06), 'multicolour tiles / black plastic', .38),
    'lego_block': ((.08,.08,.07), 'orange plastic / four raised studs', .32),
    'wooden_block': ((.075,.075,.075), 'light wood / side grain and end grain', .78),
    'checker_cube': ((.07,.07,.07), 'black and ivory checker pattern', .42),
    'marker_cube': ((.07,.07,.07), 'decorative black and white fiducial motifs', .65),
    'dice': ((.065,.065,.065), 'red resin / white pips / opposite faces sum to seven', .28),
    'cardboard_box': ((.10,.10,.10), 'kraft paper / packing tape / handling symbols', .90),
    'tissue_box': ((.105,.09,.09), 'blue floral paper / folded white tissue', .80),
    'milk_box': ((.075,.075,.105), 'blue and ivory printed carton / gable top', .65),
    'cereal_box': ((.085,.065,.12), 'cream cereal packaging / original artwork', .72),
    'juice_box': ((.08,.07,.10), 'orange juice packaging / original artwork', .65),
    'gift_box': ((.09,.09,.09), 'red patterned paper / gold ribbon and bow', .62),
    'plastic_box': ((.10,.09,.075), 'translucent plastic walls / blue lid and clips', .23),
    'ceramic_box': ((.085,.085,.078), 'blue and white floral glaze / separate lid', .22),
}


def rounded_box_mesh(size, radius):
    """Bevelled box with analytic smooth normals and per-face atlas coordinates."""
    h=np.asarray(size,dtype=float)/2
    if not 0 < radius < min(h):raise ValueError('invalid bevel radius')
    points=[];normals=[];uv=[];indices=[]
    # face normal, image-right, image-up; cross(u,v) = outward normal
    bases=[((0,-1,0),(1,0,0),(0,0,1)),((1,0,0),(0,1,0),(0,0,1)),
           ((0,1,0),(-1,0,0),(0,0,1)),((-1,0,0),(0,-1,0),(0,0,1)),
           ((0,0,1),(1,0,0),(0,1,0)),((0,0,-1),(1,0,0),(0,-1,0))]
    for face,(normal,u,v) in enumerate(bases):
        n,u,v=map(lambda a:np.asarray(a,dtype=float),(normal,u,v))
        hu,hv=np.dot(abs(u),h),np.dot(abs(v),h)
        def coords(a):
            return np.r_[-a+radius*np.array([0.,.12,.4,1.]), a-radius*np.array([1.,.4,.12,0.])]
        us,vs=coords(hu),coords(hv);offset=len(points)
        for yy in vs:
            for xx in us:
                p=n*h+u*xx+v*yy
                c=np.clip(p,-h+radius,h-radius);delta=p-c
                direction=delta/np.linalg.norm(delta)
                points.append(c+radius*direction);normals.append(direction)
                # 2 px inset avoids filtering from adjacent atlas faces.
                tu=.004+.992*(xx/hu+1)/2;tv=.004+.992*(yy/hv+1)/2
                uv.append(((face%3+tu)/3, (1-face//3+tv)/2))
        for y in range(len(vs)-1):
            for x in range(len(us)-1):
                i=offset+y*len(us)+x;indices.extend((i,i+1,i+1+len(us),i+len(us)))
    return np.asarray(points),np.asarray(normals),np.asarray(uv),indices


class Builder:
    def __init__(self, output, kind):
        self.kind=kind
        self.stage=Usd.Stage.CreateNew(str(output/f'{kind}.usdc'))
        root=UsdGeom.Xform.Define(self.stage,'/Prop').GetPrim()
        self.stage.SetDefaultPrim(root)
        UsdGeom.SetStageUpAxis(self.stage,UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage,1.)
        root.SetAssetInfoByKey('name',kind)
        root.CreateAttribute('piper:prop_kind',Sdf.ValueTypeNames.String,custom=True).Set(kind)
        root.CreateAttribute('piper:visual_only',Sdf.ValueTypeNames.Bool,custom=True).Set(True)
        self.bounds=[]

    def material(self,name,color=(1.,1.,1.),roughness=.5,texture=None,opacity=1.,metallic=0.,ior=1.5):
        path='/Prop/Looks/'+name
        material=UsdShade.Material.Define(self.stage,path)
        shader=UsdShade.Shader.Define(self.stage,path+'/Surface');shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(ior)
        shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(opacity)
        diffuse=shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f)
        if texture:
            reader=UsdShade.Shader.Define(self.stage,path+'/ST');reader.CreateIdAttr('UsdPrimvarReader_float2')
            reader.CreateInput('varname',Sdf.ValueTypeNames.String).Set('st')
            reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
            tex=UsdShade.Shader.Define(self.stage,path+'/Texture');tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('textures/'+texture+'.png'))
            tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
            tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('clamp')
            tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('clamp')
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
            diffuse.ConnectToSource(tex.ConnectableAPI(),'rgb')
        else:
            # Artist-facing solid colours use sRGB, just like the PNG atlases.
            linear=[c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4 for c in color]
            diffuse.Set(Gf.Vec3f(*linear))
        shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        return material

    def mesh(self,name,points,indices,counts,material,normals=None,uv=None,double_sided=False):
        points=np.asarray(points,dtype=np.float32)
        mesh=UsdGeom.Mesh.Define(self.stage,'/Prop/Geometry/'+name)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexIndicesAttr(indices);mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(double_sided)
        mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0).tolist()),Gf.Vec3f(*points.max(axis=0).tolist())])
        if normals is not None:
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals,dtype=np.float32)))
            mesh.SetNormalsInterpolation('vertex')
        if uv is not None:
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(
                Vt.Vec2fArray.FromNumpy(np.asarray(uv,dtype=np.float32)))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        self.bounds.extend((points.min(axis=0),points.max(axis=0)))
        return mesh

    def box(self,name,size,center,material,radius=.001):
        p,n,uv,idx=rounded_box_mesh(size,min(radius,min(size)*.2))
        return self.mesh(name,p+center,idx,[4]*(len(idx)//4),material,n,uv)

    def cylinder(self,name,radius,height,center,material):
        # Rounded profile, closed caps and smoothly shaded cylindrical body.
        bevel=min(.001,height*.2,radius*.1)
        profile=[(0.,-height/2),(radius-bevel,-height/2),(radius-bevel*.3,-height/2+bevel*.3),
                 (radius,-height/2+bevel),(radius,height/2-bevel),
                 (radius-bevel*.3,height/2-bevel*.3),(radius-bevel,height/2),(0.,height/2)]
        points=[];indices=[]
        for r,z in profile:
            points.extend([(r*math.cos(a)+center[0],r*math.sin(a)+center[1],z+center[2])
                           for a in np.linspace(0,2*math.pi,48,endpoint=False)])
        for j in range(len(profile)-1):
            for i in range(48):indices.extend((j*48+i,j*48+(i+1)%48,(j+1)*48+(i+1)%48,(j+1)*48+i))
        # Give each profile ring its area-independent radial/vertical normal.
        normals=[]
        for j,(r,z) in enumerate(profile):
            if j<2:nr,nz=0.,-1.
            elif j>5:nr,nz=0.,1.
            else:
                nr,nz=((.707,-.707) if j==2 else (1.,0.) if j in (3,4) else (.707,.707))
            normals.extend([(nr*math.cos(a),nr*math.sin(a),nz) for a in np.linspace(0,2*math.pi,48,endpoint=False)])
        return self.mesh(name,points,indices,[4]*(len(indices)//4),material,normals)

    def finish(self,description):
        points=np.asarray(self.bounds)
        lo,hi=points.min(axis=0),points.max(axis=0)
        if abs(lo[2])>1e-6:raise ValueError(f'{self.kind} does not rest at z=0: {lo}')
        self.stage.GetRootLayer().Save()
        return dict(usd=self.kind+'.usdc',dimensions_m=[round(float(x),6) for x in hi-lo],
                    bounds_m=[lo.tolist(),hi.tolist()],appearance=description)


def build_prop(output,kind):
    size,description,roughness=SPECS[kind];sx,sy,sz=size
    b=Builder(output,kind)
    body=b.material('Body',roughness=roughness,texture=kind) if kind not in ('lego_block','plastic_box') else None
    if kind not in ('plastic_box',):
        if kind=='lego_block':body=b.material('Orange',(.96,.28,.012),roughness)
        radius=.005 if kind=='dice' else .002 if kind in ('rubik_cube','lego_block','gift_box','ceramic_box') else .001
        b.box('Body',size,(0,0,sz/2),body,radius)
    if kind=='lego_block':
        for i,(x,y) in enumerate(((-.02,-.02),(.02,-.02),(-.02,.02),(.02,.02))):
            b.cylinder(f'Stud_{i}',.0125,.012,(x,y,sz+.006),body)
    elif kind=='tissue_box':
        dark=b.material('Opening',(.025,.055,.065),.9)
        b.box('Opening',(.051,.017,.001),(0,0,sz+.0005),dark,.004)
        paper=b.material('Tissue',(.93,.93,.89),.9)
        points=[];idx=[];nu,nv=20,18
        for j in range(nv):
            v=j/(nv-1)
            for i in range(nu):
                u=i/(nu-1)*2-1
                points.append((u*(.012+.02*v)+.006*math.sin(v*4),
                               .004*math.sin(u*7+v*3)+.014*v*v,
                               sz+.001+.052*v+.004*math.sin(u*5)*v))
        for j in range(nv-1):
            for i in range(nu-1):
                k=j*nu+i;idx.extend((k,k+1,k+1+nu,k+nu))
        b.mesh('FoldedTissue',points,idx,[4]*(len(idx)//4),paper,double_sided=True)
    elif kind=='milk_box':
        blue=b.material('GableBlue',(.012,.28,.59),.55)
        ivory=b.material('FoldedPaper',(.91,.92,.87),.72)
        ridge=sz+.033
        p=[(-sx/2,-sy/2,sz),(sx/2,-sy/2,sz),(sx/2,0,ridge),(-sx/2,0,ridge),
           (-sx/2,sy/2,sz),(sx/2,sy/2,sz)]
        b.mesh('GablePanels',p,[0,1,2,3,3,2,5,4],[4,4],blue,double_sided=True)
        b.mesh('GableFolds',p,[0,3,4,1,5,2],[3,3],ivory,double_sided=True)
        b.box('TopSeal',(sx,.003,.009),(0,0,ridge+.0045),ivory,.001)
    elif kind in ('cardboard_box','cereal_box','juice_box'):
        # A restrained folded seam gives the printed boxes real thickness cues.
        seam=b.material('FoldSeam',(.48,.34,.20) if kind=='cardboard_box' else (.73,.67,.48),.85)
        b.box('TopFold',(sx*.96,.0005,.0004),(0,0,sz+.0002),seam,.0001)
    elif kind=='gift_box':
        gold=b.material('Ribbon',(.72,.42,.09),.32,metallic=.28)
        b.box('Lid',(sx+.003,sy+.003,.009),(0,0,sz-.002),body,.0015)
        # Wraps lie above the paper surface with visible, restrained thickness.
        b.box('RibbonTopX',(sx+.004,.015,.0012),(0,0,sz+.0031),gold,.0004)
        b.box('RibbonTopY',(.015,sy+.004,.0012),(0,0,sz+.0032),gold,.0004)
        for i,x in enumerate((-sx/2-.0005,sx/2+.0005)):
            b.box(f'RibbonSideX{i}',(.001,.015,sz),(x,0,sz/2),gold,.0003)
        for i,y in enumerate((-sy/2-.0005,sy/2+.0005)):
            b.box(f'RibbonSideY{i}',(.015,.001,sz),(0,y,sz/2),gold,.0003)
        # Four open ribbon loops with two edges, bent in 3D.
        for k in range(4):
            angle=k*math.pi/2+math.pi/4;points=[];idx=[]
            for j,t in enumerate(np.linspace(0,2*math.pi,41)):
                r=.013*(1-math.cos(t));z=sz+.006+.007*math.sin(t)+.003*(1-math.cos(t))
                for side in (-1,1):
                    points.append((r*math.cos(angle)-side*.004*math.sin(angle),
                                   r*math.sin(angle)+side*.004*math.cos(angle),z))
                if j:idx.extend((2*j-2,2*j-1,2*j+1,2*j))
            b.mesh(f'BowLoop{k}',points,idx,[4]*(len(idx)//4),gold,double_sided=True)
        b.box('BowKnot',(.012,.012,.008),(0,0,sz+.009),gold,.002)
    elif kind=='plastic_box':
        clear=b.material('FrostedPlastic',(.77,.86,.88),.18,opacity=.22,ior=1.47)
        blue=b.material('BlueLid',(.012,.27,.65),.26)
        # Hollow, thin walls instead of a solid transparent cube.
        b.box('Base',(sx,sy,.0025),(0,0,.00125),clear,.0005)
        for i,x in enumerate((-sx/2+.001,sx/2-.001)):
            b.box(f'WallX{i}',(.002,sy,sz),(x,0,sz/2),clear,.0004)
        for i,y in enumerate((-sy/2+.001,sy/2-.001)):
            b.box(f'WallY{i}',(sx-.004,.002,sz),(0,y,sz/2),clear,.0004)
        b.box('Lid',(sx+.008,sy+.008,.006),(0,0,sz+.003),blue,.002)
        b.box('LidInset',(sx-.010,sy-.010,.002),(0,0,sz+.007),blue,.0008)
        for i,x in enumerate((-sx/2-.003,sx/2+.003)):
            b.box(f'ClipX{i}',(.004,.029,.013),(x,0,sz-.003),blue,.001)
        for i,y in enumerate((-sy/2-.003,sy/2+.003)):
            b.box(f'ClipY{i}',(.029,.004,.013),(0,y,sz-.003),blue,.001)
    elif kind=='ceramic_box':
        seam=b.material('LidSeam',(.14,.22,.32),.55)
        b.box('LidSeam',(sx*.98,sy*.98,.0012),(0,0,sz+.0006),seam,.001)
        b.box('Lid',(sx+.002,sy+.002,.006),(0,0,sz+.0042),body,.002)
    return b.finish(description)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ASSET_DIR)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    generate_textures(out/'textures')
    props={kind:build_prop(out,kind) for kind in SPECS}
    data=dict(version='piper_tabletop_props_v1',units='metres',up_axis='Z',origin='bottom_center',
              visual_only=True,collision=False,artwork='Original deterministic procedural artwork',props=props)
    (out/'catalog.json').write_text(json.dumps(data,indent=2)+'\n')
    print(f'Built {len(props)} tabletop props: {out}',flush=True)


if __name__=='__main__':main()
