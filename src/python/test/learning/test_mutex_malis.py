import unittest
import numpy as np


# TODO check correctness for toy data
# TODO check ignore data
class TesMutextMalis(unittest.TestCase):

    def test_mutex_malis_2d(self):
        from affogato.affinities import compute_affinities
        from affogato.learning import mutex_malis
        shape = (100, 100)
        labels = np.random.randint(0, 100, size=shape)
        offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3]]
        affs, _ = compute_affinities(labels, offsets)
        affs += 0.1 * np.random.randn(*affs.shape)
        affs -= affs.min()
        affs /= affs.max()
        loss, grads, _, _ = mutex_malis(affs, labels, offsets, 2)
        self.assertEqual(grads.shape, affs.shape)
        self.assertNotEqual(loss, 0)
        self.assertFalse(np.allclose(grads, 0))

    def seg2edges(self, segmentation):
        from scipy.ndimage import convolve
        gx = convolve(segmentation + 1, np.array([-1., 1.]).reshape(1, 2))
        gy = convolve(segmentation + 1, np.array([-1., 1.]).reshape(2, 1))
        return ((gx ** 2 + gy ** 2) > 0)

    def test_mutex_malis_2d_gradient_descent(self):
        from affogato.segmentation import compute_mws_segmentation
        from affogato.learning import mutex_malis
        shape = (100, 100)
        labels = np.zeros(shape, dtype=int)
        for i in range(10):
            for j in range(10):
                idx = (10 * i + j + 1)
                if idx % 7 == 0:
                    idx = 0
                labels[10 * i:10 * (i + 1), 10 * j:10 * (j + 1)] = idx

        offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3]]
        affs = 0.5 * np.ones((len(offsets), 100, 100))
        affs = 0.5 * np.random.rand(len(offsets), 100, 100)

        for epoch in range(300):
            loss, grads, seg1, seg2 = mutex_malis(affs, labels, offsets, 2, learn_in_ignore_label=True)
            affs -= grads
            affs = np.clip(affs, 0, 1)
            print(loss)


        number_of_attractive_channels = 2
        labels1 = compute_mws_segmentation(affs, offsets,
                                           number_of_attractive_channels,
                                           algorithm='kruskal')

        from scipy.misc import imsave
        number_of_colors = labels1.max() + 1
        cmap = np.random.randint(255, size=(3, int(number_of_colors)))
        cmap[:, 0] = 0
        rgb_images = cmap[:, labels1].transpose(1,2,0)
        imsave("mws_labels.png", rgb_images)

        number_of_colors = seg1.max() + 1
        cmap = np.random.randint(255, size=(3, int(number_of_colors)))
        cmap[:, 0] = 0
        rgb_images = cmap[:, seg1].transpose(1,2,0)
        imsave("seg1.png", rgb_images)

        number_of_colors = seg2.max() + 1
        cmap = np.random.randint(255, size=(3, int(number_of_colors)))
        cmap[:, 0] = 0
        rgb_images = cmap[:, seg2].transpose(1,2,0)
        imsave("seg2.png", rgb_images)


        number_of_colors = labels.max() + 1
        cmap = np.random.randint(255, size=(3, int(number_of_colors)))
        cmap[:, 0] = 0
        rgb_images = cmap[:, labels].transpose(1,2,0)
        imsave("gt_labels.png", rgb_images)

        import h5py
        with h5py.File("debug.h5", "w") as h5file:
            h5file.create_dataset("affs", data=affs)
            h5file.create_dataset("labels1", data=labels1)
            h5file.create_dataset("labels", data=labels)
            h5file.create_dataset("grads", data=grads)

        self.assertEqual(grads.shape, affs.shape)
        self.assertEqual(loss, 0)

        edges1 = self.seg2edges(labels1)
        edges2 = self.seg2edges(labels)
        self.assertTrue(np.allclose(edges1, edges2))

    def test_mutex_malis_3d(self):
        from affogato.affinities import compute_affinities
        from affogato.learning import mutex_malis
        shape = (32, 64, 64)
        labels = np.random.randint(0, 1000, size=shape)
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                   [-3, 0, 0], [0, -3, 0], [0, 0, -3]]
        affs, _ = compute_affinities(labels, offsets)
        affs += 0.1 * np.random.randn(*affs.shape)
        affs -= affs.min()
        affs /= affs.max()
        loss, grads, _, _ = mutex_malis(affs, labels, offsets, 3)
        self.assertEqual(grads.shape, affs.shape)
        # FIXME this fails
        self.assertNotEqual(loss, 0)
        self.assertFalse(np.allclose(grads, 0))


if __name__ == '__main__':
    TesMutextMalis().test_malis_2d_gradient_descent()
    # unittest.main()
