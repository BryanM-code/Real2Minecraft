"""Behavioural fixtures for the opt-in architectural representation."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcvoxel.architecture import align_grid, enclosed_cross_sections, material_families, coherent_wall_shapes
from mcvoxel.architectural_roofs import surface_evidence, constrain_plan
from mcvoxel.grid import occluded_interior, voxelize_scene, pad_model
from mcvoxel.roofs import top_surface, plan_roofs, RoofParams, builder_roofs
from mcvoxel import block_shapes as bs
from scene_fixtures import gable_house, mesh_to_scene, _box, WALL
from mcvoxel.scene import KIND_SURFACE


class ArchitecturalEvidence(unittest.TestCase):
    def test_alignment_is_translation_and_triangle_order_invariant(self):
        sm = gable_house(window=False)
        a, _ = align_grid(sm.verts, sm.faces, 1.7)
        shift = np.array([17.321, -3.42, 20.05])
        b, _ = align_grid(sm.verts+shift, sm.faces[::-1], 1.7)
        np.testing.assert_allclose(sm.verts*1.7+a, (sm.verts+shift)*1.7+b, atol=1e-9)

    def test_three_walls_and_floor_are_not_permission_to_fill_open_bay(self):
        s = np.zeros((12,8,12), bool)
        s[1:11,0,1:11] = True; s[1:11,6,1:11] = True
        s[1,1:6,1:11] = True; s[10,1:6,1:11] = True; s[1:11,1:6,1] = True
        self.assertTrue(occluded_interior(s)[5,3,5])
        self.assertFalse(enclosed_cross_sections(s).any())
        s[1:11,1:6,10] = True
        self.assertTrue(enclosed_cross_sections(s)[5,3,5])

    def test_source_vertical_wall_corrects_shapes_without_filling_windows(self):
        s = np.ones((3,8,8),bool); s[1,3:5,3:5]=False
        v = np.array([[1,0,0],[1,8,0],[1,8,8],[1,0,8]],float)
        f = np.array([[0,1,2],[0,2,3]])
        sh=np.full(s.shape,bs.SHAPE_SLAB_BOTTOM,np.int8); sh[~s]=-1
        out, info = coherent_wall_shapes(s, sh, v, f, np.zeros(3))
        self.assertGreater(info['vertical_surface_full_cells'],0)
        self.assertTrue((out[~s]==-1).all())
        self.assertEqual(info['occupancy_changed'],0)

    def test_tile_normals_do_not_control_a_broad_planar_roof(self):
        X,Z,sub=18,18,4
        x,z=np.meshgrid((np.arange(X*sub)+.5)/sub,(np.arange(Z*sub)+.5)/sub,indexing='ij')
        h=6+.5*x+.05*np.sin(x*12)
        g=np.stack([.5+.6*np.cos(x*12),np.zeros_like(x)],-1)
        _, fitted, ev=surface_evidence(h,g,(X,Z),sub)
        self.assertGreater(ev['confidence'].mean(),.9)
        self.assertLess(float(np.abs(fitted[...,0]-.5).mean()),.03)

    def test_tiny_fragment_is_preserved_instead_of_independently_rebuilt(self):
        solid=np.zeros((5,8,5),bool); solid[2,:5,2]=True
        height=np.full((20,20),-np.inf); height[8:12,8:12]=5.3
        grad=np.zeros((20,20,2)); grad[8:12,8:12,0]=1.
        plan=plan_roofs(solid,None,height,grad)
        ev={'confidence':np.ones((5,5),bool),'height':np.ones((5,5))*5.3,'gradient':np.ones((5,5,2))}
        constrain_plan(plan,ev)
        self.assertFalse((plan.level>=0).any())

    def test_pitch_bin_noise_does_not_split_one_plane_into_independent_courses(self):
        X,Z,sub=16,16,4
        x,z=np.meshgrid((np.arange(X*sub)+.5)/sub,(np.arange(Z*sub)+.5)/sub,indexing='ij')
        h=5+.7*x
        g=np.stack([.7+.08*np.sin(z*2),np.zeros_like(x)],-1)
        solid=np.arange(20)[None,:,None] < (5+.7*np.arange(X))[:,None,None]
        solid=np.broadcast_to(solid,(X,20,Z)).copy()
        plan=plan_roofs(solid,None,h,g,RoofParams(coherent=True))
        self.assertEqual(len([r for r in plan.info['plane_decisions'] if r.get('planned',True)]),1)

    def test_ceiling_selects_lower_source_surface(self):
        v=np.array([[0,3,0],[8,3,0],[0,3,8],[0,9,0],[8,9,0],[0,9,8]],float)
        f=np.array([[0,1,2],[3,4,5]])
        high,_=top_surface(v,f,(8,8))
        low,_=top_surface(v,f,(8,8),ceiling=np.full((32,32),5.))
        self.assertTrue((high[np.isfinite(high)]==9).all())
        self.assertTrue((low[np.isfinite(low)]==3).all())

    def test_lower_awning_does_not_erase_upper_roof_or_open_air(self):
        parts=[(_box((12,1.4,12),(6,3.9,6)),WALL,KIND_SURFACE),
               (_box((12,.4,12),(6,10.2,6)),WALL,KIND_SURFACE),
               (_box((1,10,1),(.5,5,.5)),WALL,KIND_SURFACE)]
        sm=mesh_to_scene(parts)
        vm=voxelize_scene(sm,1,offset=(.15,0,.15),drop_ground=False,clean=False,architecture=True,with_colors=False)
        vm=pad_model(vm,((2,2),(0,3),(2,2)))
        world_y = np.arange(vm.solid.shape[1]) + vm.origin[1]
        high=vm.solid & (world_y[None,:,None] >= 9)
        vb,plan=builder_roofs(vm,sm.verts+(.15,0,.15)-vm.origin,sm.faces,RoofParams(coherent=True))
        self.assertTrue((vb.solid[high]).all())
        self.assertGreater(len(plan.layers),0)
        # A lower roof reconstruction must not invent connecting floor-to-roof columns.
        self.assertFalse(vb.solid[4:10,:,4:10][:,(world_y>=6)&(world_y<8),:].any())

    def test_connected_material_family_does_not_cross_trim(self):
        occ=np.array([[x,0,z] for x in range(4) for z in range(4)])
        ids=['minecraft:white_concrete']*8+['minecraft:red_terracotta']*8
        colors=np.array([(207,213,214)]*8+[(144,61,47)]*8)
        shapes=np.array([bs.SHAPE_FULL,bs.SHAPE_SLAB_BOTTOM]*8)
        out,info=material_families(occ,ids,colors,shapes,(4,1,4),np.zeros(16,bool))
        self.assertEqual(len(set(out[:8])),1); self.assertEqual(len(set(out[8:])),1)
        self.assertNotEqual(out[0],out[-1])

    def test_large_family_colour_shift_is_rejected(self):
        occ=np.array([[x,0,z] for x in range(4) for z in range(4)])
        ids=['minecraft:yellow_concrete']*16; colors=np.tile([241,175,21],(16,1))
        shapes=np.array([bs.SHAPE_FULL,bs.SHAPE_STAIR_E]*8)
        out,info=material_families(occ,ids,colors,shapes,(4,1,4),np.zeros(16,bool))
        self.assertEqual(out,ids); self.assertGreater(info['colour_shift_rejected'],0)


if __name__=='__main__': unittest.main()
