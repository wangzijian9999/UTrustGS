"""可见行、跨相机重复ID和非法输入的增密归属回归。"""
import unittest
import torch
from gaussian_parameter_contract import accumulate_density_gradients


class DensityGradientIDTests(unittest.TestCase):
    def run_case(self, grads, radii, ids=None):
        means = torch.zeros_like(grads, requires_grad=True)
        means.grad = grads
        info = dict(means2d=means, radii=radii, width=2, height=2, n_cameras=1)
        if ids is not None:
            info['gaussian_ids'] = ids
        accum, count = torch.zeros(4), torch.zeros(4, dtype=torch.int32)
        accumulate_density_gradients(info, accum, count, 2, 2)
        return accum.tolist(), count.tolist()

    def test_packed_noncontiguous_and_repeated_ids(self):
        self.assertEqual(self.run_case(torch.tensor([[3.,4.],[0.,2.],[0.,1.]]),
                                      torch.ones(3,2), torch.tensor([1,3,1])),
                         ([0.,6.,0.,2.], [0,2,0,1]))

    def test_dense_camera_visibility(self):
        grads=torch.ones(2,4,2);grads[...,1]=0
        radii=torch.zeros_like(grads);radii[0,1]=1;radii[1,1]=1;radii[1,3]=1
        self.assertEqual(self.run_case(grads,radii), ([0.,2.,0.,1.],[0,2,0,1]))

    def test_empty_is_noop(self):
        self.assertEqual(self.run_case(torch.empty(0,2),torch.empty(0,2),torch.empty(0,dtype=torch.int64)),
                         ([0.,0.,0.,0.],[0,0,0,0]))

    def test_invalid_mapping_fails(self):
        for ids in [None,torch.tensor([4]),torch.tensor([-1]),torch.tensor([1.,2.]),torch.tensor([1,2])]:
            with self.assertRaises(ValueError):
                self.run_case(torch.ones(1,2),torch.ones(1,2),ids)
        with self.assertRaises(ValueError):
            self.run_case(torch.ones(1,2),torch.ones(1),torch.tensor([1]))


if __name__ == '__main__':
    unittest.main()
