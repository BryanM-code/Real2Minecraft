"""Regressions for mesh-backed roof lattices and conservative structural details."""
import json
from pathlib import Path
import unittest

import numpy as np

from mcvoxel import roofs, vocab
from mcvoxel import block_shapes as bs
from mcvoxel.assign import assign_blocks
from mcvoxel.blockstates import finish_states, rotate_y, mirror_x
from mcvoxel.grid import VoxelModel, shell_of, pad_model, voxelize_scene, occluded_interior
from scene_fixtures import (_box, mesh_to_scene, gable_house, hip_house, lean_to,
                            compound_house, WALL, ROOF, KIND_SURFACE)


def plane(pitch=1., width=12, facing=0, offset=.2, perturb=0., opening=False):
    X, Y, Z, sub = width+4, 40, width+4, 4
    xx, zz = np.meshgrid((np.arange(X*sub)+.5)/sub, (np.arange(Z*sub)+.5)/sub, indexing='ij')
    u = (xx if facing < 2 else zz)
    if facing in (1,2):
        u = X-u
    slope = pitch+perturb
    h = 4. + offset + slope*(u-2)
    mask = (xx >= 2) & (xx < X-2) & (zz >= 2) & (zz < Z-2)
    if opening:
        mask &= ~((xx >= 6) & (xx < 8) & (zz >= 6) & (zz < 8))
    h = np.where(mask, h, -np.inf)
    grad = np.zeros(h.shape+(2,), np.float32)
    grad[...,0 if facing<2 else 1] = slope * (-1 if facing in (1,2) else 1)
    centre = np.max(h.reshape(X,sub,Z,sub), axis=(1,3))
    solid = np.arange(Y)[None,:,None] < centre[:,None,:]
    shapes = np.where(solid, bs.SHAPE_FULL, -1).astype(np.int8)
    rgb = np.full(solid.shape+(3,), (120,100,80), np.uint8)
    kind = np.zeros(solid.shape+(3,), np.float16);kind[...,0]=1
    vm = VoxelModel(solid,solid.copy(),shell_of(solid),np.zeros(3,int),1.,shapes,rgb,kind)
    return vm,h,grad


def build_plane(**kw):
    vm,h,g = plane(**kw)
    plan = roofs.plan_roofs(vm.solid,vm.kind_frac,h,g)
    return vm,roofs.apply_roofs(vm,plan),plan


def labelled(vm):
    out=np.zeros(vm.solid.shape,np.uint8)
    for code,label in vocab.FROM_BLOCK_SHAPES.items(): out[vm.shapes==code]=label
    return out


class RoofRegions(unittest.TestCase):
    def test_canonical_and_near_pitches_all_directions_offsets_widths(self):
        for pitch,pattern in ((1.,'1:1'),(.5,'2:1'),(1/3,'3:1'),(2.,'1:2')):
            for direction in range(4):
                for offset in (.02,.29,.51,.88):
                    for width in (7,12):
                        for perturb in (0.,.012,-.012):
                            with self.subTest(pitch=pitch,direction=direction,offset=offset,width=width,perturb=perturb):
                                vm,vb,p=build_plane(pitch=pitch,width=width,facing=direction,offset=offset,perturb=perturb)
                                self.assertEqual(p.info['regions'],1)
                                self.assertEqual(p.info['plane_decisions'][0]['pattern'],pattern)
                                self.assertEqual(int((p.level>=0).sum()),width*width)
                                for x,z in np.argwhere(p.level>=0):
                                    self.assertTrue(vb.solid[x,p.level[x,z],z])
                                    for a,b in ((x+1,z),(x,z+1)):
                                        if p.level[a,b] < 0: continue
                                        lo,hi=sorted([int(p.level[x,z]),int(p.level[a,b])])
                                        high=(x,z) if p.level[x,z]>=p.level[a,b] else (a,b)
                                        self.assertTrue(vb.solid[high[0],lo:hi+1,high[1]].all())
                                        self.assertTrue(vb.shell[high[0],lo,high[1]])

    def test_missing_course_restored_but_mesh_opening_preserved(self):
        vm,h,g=plane(pitch=.5,opening=True)
        vm.solid[4,4:,:]=False
        vm.shell=shell_of(vm.solid)
        p=roofs.plan_roofs(vm.solid,vm.kind_frac,h,g)
        vb=roofs.apply_roofs(vm,p)
        self.assertTrue((p.level[4,2:14]>=0).all())
        self.assertTrue((p.level[6:8,6:8]<0).all())
        self.assertFalse(vb.solid[6:8,:,6:8].any())

    def test_rotation_mirror_and_repeat_are_exact(self):
        for pitch in (1/3,.5,1.,2.):
            vm,vb,p=build_plane(pitch=pitch,facing=0)
            for direction,k in ((0,0),(3,1),(1,2),(2,3)):
                _,other,q=build_plane(pitch=pitch,facing=direction)
                np.testing.assert_array_equal(rotate_y(labelled(vb),k)[0],labelled(other))
            _,other,_=build_plane(pitch=pitch,facing=1)
            np.testing.assert_array_equal(mirror_x(labelled(vb))[0],labelled(other))
            _,repeat,r=build_plane(pitch=pitch,facing=0)
            np.testing.assert_array_equal(vb.shapes,repeat.shapes)
            self.assertEqual(p.info,r.info)

    def test_ridge_hip_valley_and_compound_seams(self):
        # Actual triangle meshes exercise top_surface, including opposite gradients
        # inside a ridge cell, cardinal hip seams, and adjoining gables forming valleys.
        for scene in (gable_house(window=False,overhang=1.), hip_house(),
                      lean_to(), compound_house()):
            vm=pad_model(voxelize_scene(scene,1.,(.21,0,.37),drop_ground=False,clean=False),((2,2),(0,3),(2,2)))
            vb,p=roofs.builder_roofs(vm,scene.verts+(.21,0,.37)-vm.origin,scene.faces)
            for x,z in np.argwhere(p.level>=0):
                self.assertTrue(vb.solid[x,p.level[x,z],z])
                for a,b in ((x+1,z),(x,z+1)):
                    if a >= p.level.shape[0] or b >= p.level.shape[1] or p.level[a,b] < 0:
                        continue
                    lo,hi=sorted([int(p.level[x,z]),int(p.level[a,b])])
                    if hi-lo <= 2 and p.group[x,z] and p.group[a,b]:
                        high=(x,z) if p.level[x,z]>=p.level[a,b] else (a,b)
                        self.assertTrue(vb.solid[high[0],lo:hi+1,high[1]].all())
        vm,h,g=plane(pitch=.5)
        # V-shaped valley: two planes meet continuously without an invented opening.
        mid=h.shape[0]//2
        u=(np.arange(h.shape[0])+.5)/4
        h=np.where(np.isfinite(h),4+np.abs(u[:,None]-8)*.5,-np.inf)
        g[:mid,...,0]*=-1
        centre=np.max(h.reshape(16,4,16,4),axis=(1,3))
        vm.solid=np.arange(40)[None,:,None]<centre[:,None,:]
        p=roofs.plan_roofs(vm.solid,vm.kind_frac,h,g)
        self.assertTrue((p.level[2:14,2:14]>=0).all())

    def test_raised_flat_parapet_does_not_blend_into_lower_roof(self):
        vm,h,g=plane(pitch=0.)
        h[np.isfinite(h)]=6.
        # A quarter-block-wide parapet shares its column with the broad lower roof.
        h[8,8:56]=6.8
        vm.solid[:]=np.arange(40)[None,:,None]<6
        p=roofs.plan_roofs(vm.solid,vm.kind_frac,h,g)
        self.assertTrue((p.level[2,2:14]<0).all())
        self.assertTrue((p.level[4:12,4:12]==5).all())

    def test_roof_material_boundaries_are_separate_groups(self):
        vm,h,g=plane(pitch=.5)
        materials=np.zeros((16,16),int);materials[:,8:]=1
        p=roofs.plan_roofs(vm.solid,vm.kind_frac,h,g,surface_material=materials)
        self.assertEqual(p.info['regions'],1)
        self.assertEqual(p.info['roof_groups'],2)
        self.assertFalse(set(p.group[2:14,2:8].ravel()) & set(p.group[2:14,8:14].ravel()))

    def test_lower_roof_rays_cannot_cut_down_a_thin_vertical_wall(self):
        vm,h,g=plane(pitch=0.)
        vm.solid[2,:15,2:14]=True
        p=roofs.plan_roofs(vm.solid,vm.kind_frac,h,g)
        self.assertTrue((p.level[2,2:14]<0).all())

    def test_eaves_and_shell_survive_padding(self):
        scene=lean_to()
        vm=voxelize_scene(scene,1.,(.21,0,.37),drop_ground=False,clean=False)
        padded=pad_model(vm,((2,2),(0,3),(2,2)))
        np.testing.assert_array_equal(padded.shell[2:-2,:-3,2:-2],vm.shell)
        vb,p=roofs.builder_roofs(padded,scene.verts+(.21,0,.37)-padded.origin,scene.faces)
        x=int(5.21-padded.origin[0]);z=int(.37-padded.origin[2])
        self.assertGreaterEqual(p.level[x,z],0)
        self.assertTrue(vb.solid[x,p.level[x,z],z])
        self.assertFalse(vb.solid[x,:2,z].any(), 'Eave must not become a floor-to-roof pillar')

    def test_partial_caps_retain_exported_backing(self):
        _,vb,p=build_plane(pitch=.5)
        t=assign_blocks(vb)
        for x,z in np.argwhere(p.group>0):
            b=int(p.level[x,z])
            if p.shape[x,z]==bs.SHAPE_SLAB_BOTTOM and vb.solid[x,b-1,z]:
                self.assertNotEqual(int(t.shape[x,b-1,z]),vocab.IGNORE_SHAPE)


def porch(material='wood', fragments=False, wall=False, floating=False):
    pieces=[(_box((10,.4,8),(5,.2,4)),WALL,KIND_SURFACE),
            (_box((10,.25,8),(5,4.9,4)),ROOF,KIND_SURFACE)]
    if wall:
        pieces.append((_box((8,4,.25),(5,2.5,4.5)),WALL,KIND_SURFACE))
    else:
        for x in (2.4,5.4,8.4):
            height= .4 if fragments else (2. if floating else 4.4)
            pieces.append((_box((.2,height,.2),(x,2.6,4.4)),WALL,KIND_SURFACE))
    scene=mesh_to_scene(pieces)
    for m in scene.materials[2:]: m.name=material+' support'
    return scene


class Details(unittest.TestCase):
    def test_wood_fences_and_masonry_walls(self):
        for mat,kind in (('wood','fence'),('concrete','wall')):
            scene=porch(mat)
            vm=voxelize_scene(scene,1.,drop_ground=False,clean=False)
            target=assign_blocks(vm)
            self.assertGreaterEqual(target.info['context_details'][kind],6)
            self.assertTrue(any('_'+kind+'[' in s for s in target.states))

    def test_facades_noise_floaters_and_metal_do_not_become_posts(self):
        for kw in ({'wall':True},{'fragments':True},{'floating':True},{'material':'metal'}):
            vm=voxelize_scene(porch(**kw),1.,drop_ground=False,clean=False)
            t=assign_blocks(vm)
            self.assertFalse(any('_fence[' in s or '_wall[' in s for s in t.states),kw)

    def test_colonnade_air_is_not_filled_as_a_room(self):
        s=np.zeros((12,7,12),bool)
        s[:,0,:]=True;s[:,6,:]=True
        for x in (1,4,7,10):
            for z in (1,4,7,10):s[x,1:6,z]=True
        filled=occluded_interior(s)
        self.assertFalse(filled[:,1:6,:].any())
        s[0,:,:]=s[-1,:,:]=s[:,:,0]=s[:,:,-1]=True
        self.assertTrue(occluded_interior(s)[5,3,5])

    def test_thick_detached_columns_are_not_room_walls(self):
        s=np.zeros((40,12,60),bool)
        s[:,0,:]=s[:,11,:]=True
        for x in (3,32):
            for z in range(3,55,10):s[x:x+4,1:11,z:z+4]=True
        self.assertFalse(occluded_interior(s)[:,1:11,:].any())

    def test_named_parapet_has_connected_wall_blocks(self):
        parts=[(_box((10,4,8),(5,2,4)),WALL,KIND_SURFACE),
               (_box((.2,1.7,6),(1.4,4.85,4)),WALL,KIND_SURFACE)]
        scene=mesh_to_scene(parts);scene.materials[1].name='stone parapet'
        vm=voxelize_scene(scene,1.,drop_ground=False,clean=False)
        target=assign_blocks(vm)
        self.assertGreaterEqual(target.info['context_details']['wall'],4)
        grid,names=finish_states(*target.sponge_grid(vm.solid.shape))
        self.assertTrue(any('_wall[' in n and 'north=low' in n and 'south=low' in n for n in names))

    def test_thin_canopy_keeps_slab_shapes(self):
        vm=voxelize_scene(porch(),1.,drop_ground=False,clean=False)
        self.assertGreater(int(np.isin(vm.shapes,[bs.SHAPE_SLAB_BOTTOM,bs.SHAPE_SLAB_TOP]).sum()),20)


class V2States(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema=json.loads((Path(__file__).parent/'fixtures/target_blocks_1_20_1.json').read_text())['blocks']

    def assert_valid(self,state):
        name=state.split('[')[0].split(':')[-1]
        self.assertIn(name,self.schema)
        for prop,value in vocab._props(state).items():
            self.assertIn(prop,self.schema[name],state)
            self.assertIn(value,self.schema[name][prop],state)

    def test_every_v2_pair_still_decodes_to_a_valid_1_20_1_state(self):
        # The v2 contract: every v2 (shape, material) pair still decodes to the 1.20.1 state
        # it always did -- except the pairs v3 makes valid with their real 1.21 block (tuff's
        # slab/stairs/wall, the nether brick fence), which v2 could only colour-substitute.
        for sh in range(vocab.V2_SHAPE_COUNT):
            for mat in range(vocab.V2_MATERIAL_COUNT):
                state=vocab.decode(sh,mat)
                if vocab.VALID[sh,mat] and state!=vocab._decode_v2(sh,mat):
                    vocab.validate_state(state,vocab.TARGET_DATA_VERSION)
                else:
                    self.assert_valid(state)

    def test_new_families_encode_decode_and_symmetry(self):
        for sh,family in ((8,bs.FENCE_VARIANTS),(9,bs.WALL_VARIANTS)):
            for base in family:
                mat=vocab.MATERIAL_INDEX[base]
                state=vocab.decode(sh,mat)
                if vocab.VALID[sh,mat]:
                    self.assertEqual(vocab.encode_state(state),(sh,mat))
                else:   # v2's deepslate shortcut: cobbled deepslate is its own material in v3
                    self.assertEqual(vocab.encode_state(state),
                                     (sh,vocab.MATERIAL_INDEX['minecraft:cobbled_'+base.split(':')[1]]))
        arr=np.array([8,9,1,4,7,255],np.uint8).reshape(3,1,2)
        np.testing.assert_array_equal(rotate_y(arr,4)[0],arr)
        np.testing.assert_array_equal(mirror_x(mirror_x(arr)[0])[0],arr)

    def test_connections_are_derived_across_a_window_boundary(self):
        for sh,base,connected in ((8,'oak_planks','true'),(9,'stone_bricks','low')):
            names=[vocab.AIR,vocab.decode(sh,vocab.MATERIAL_INDEX['minecraft:'+base]),'minecraft:stone']
            grid=np.zeros((3,3,130),np.uint16)
            grid[0,1,62:67]=1
            out,ns=finish_states(grid,names)
            state=ns[out[0,1,64]]
            self.assertIn('east='+connected,state);self.assertIn('west='+connected,state)
            if sh==9:self.assertIn('up=false',state)
            for s in ns:self.assert_valid(s)
            grid[1,1,64]=2
            out,ns=finish_states(grid,names)
            if sh==9:
                self.assertIn('east=tall',ns[out[0,1,64]])
                self.assertIn('up=true',ns[out[0,1,64]])


if __name__=='__main__':unittest.main()
