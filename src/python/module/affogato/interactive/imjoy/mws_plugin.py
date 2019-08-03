import asyncio
import base64
import concurrent

import numpy as np
import h5py
from skimage.io import imread, imsave
from imjoy import api

from .util import parse_geojson
from ..segmentation import InteractiveMWS


# TODO model loading and prediction should be moved to imjoy tiktorch plugin
# TODO allow selecting user image
class ImJoyPlugin():

    def setup(self):
        api.log('initialized')

    def load_image(self, file):
        return imread(file)

    def load_model(self, model_file):
        import torch
        return torch.load(model_file)

    def predict(self, model, img):

        # with torch.no_grad():
        #     batch = img[None].astype(np.float32)

        #     # whiten
        #     batch -= batch.mean()
        #     batch /= batch.std()
        #     batch = torch.from_numpy(batch)

        #     affs = model(batch)[0].cpu().numpy()

        # return affs

        # TODO: use the actual model

        # aff_path = "/home/swolf/local/data/hackathon2019/slice0.h5"
        aff_path = "/home/pape/Work/data/ilastik/mulastik/data/slice0.h5"
        with h5py.File(aff_path, "r") as h5file:
            prediction = h5file["data"][:]

        return prediction

    def update_segmentation(self, interactive, coords):

        print('update seeds')
        parsed_coords = parse_geojson(coords, interactive.shape)
        print("parsed coords", parsed_coords)
        interactive.clear_seeds()
        interactive.update_seeds(parsed_coords)
        print('run mws')
        segmentation = interactive()
        print('mws done')

        return segmentation

    async def blend(self, img, seg, seed_color_map):

        seg_color = np.stack(((seg % 255).astype(np.uint8),
                             ((seg / 255) % 255).astype(np.uint8),
                             ((seg / 255**2) % 255).astype(np.uint8)), axis=-1)

        for l in seed_color_map:
            mask = seg == l
            for c in range(3):
                seg_color[..., c][mask] = seed_color_map[l][c]

        img = np.repeat(img[:512, :512, None], 3, axis=-1)
        blend = 0.6 * seg_color + 0.4 * img
        return blend

    async def displayimage(self, ann_window, img):
        # name_plot = "/home/swolf/pictures/tmp.png"
        name_plot = "/home/pape/Pictures/tmp.png"
        imsave(name_plot, img)

        with open(name_plot, 'rb') as f:
            result = base64.b64encode(f.read()).decode('ascii')
        imgurl = 'data:image/png;base64,' + result
        ann_window.displayimage({'url': imgurl,
                                 'w': img.shape[0],
                                 'h': img.shape[1]})

    def init_mws(self):
        model = None
        img = None
        affs = self.predict(model, img)[..., :512, :512]
        eps = 0.01
        affs *= (1 - eps)
        affs += eps * np.random.rand(*affs.shape)

        offsets = [[-1, 0],
                   [0, -1],
                   [-3, 0],
                   [0, -3],
                   [-9, 0],
                   [0, -9],
                   [-27, 0],
                   [0, -27]]

        return InteractiveMWS(affs,
                              offsets,
                              n_attractive_channels=2,
                              strides=[8, 8],
                              randomize_strides=True)

    def init_raw(self):
        # raw_path = "/home/swolf/local/data/hackathon2019/slice0.png"
        raw_path = "/home/pape/Work/data/ilastik/mulastik/data/slice0.png"
        img = imread(raw_path)
        return img

    async def run(self, ctx):
        ann_window = await api.createWindow(type="MWSAnnotator",
                                            name="thiswindow",
                                            sandbox="allow-same-origin allow-scripts",
                                            data={})
        loop = asyncio.get_event_loop()

        seed_name_map = {}
        seed_color_map = {}
        max_label = 1
        centered = False

        with concurrent.futures.ThreadPoolExecutor() as pool:

            interactive = await loop.run_in_executor(pool, self.init_mws)
            img = await loop.run_in_executor(pool, self.init_raw)

            counter = 0
            while True:
                counter += 1
                dirty = await ann_window.is_dirty()
                if dirty:
                    print("found dirty")
                    coords = await ann_window.getAnnotation()

                    # create names for coord objects
                    for c_dict in coords['features']:
                        label = c_dict['properties']['label']
                        if label not in seed_name_map:
                            seed_name_map[label] = max_label
                            seed_color_map[max_label] = 128 + 128 * np.random.rand(3)
                            max_label += 1

                        c_dict['properties']["name"] = str(seed_name_map[label])

                    print("coords ", coords)

                    segmentation = await loop.run_in_executor(
                        pool, self.update_segmentation, interactive, coords)

                    # alpha blending
                    blend = await self.blend(img, segmentation, seed_color_map)
                    await self.displayimage(ann_window, blend)

                    if not centered:
                        await ann_window.centeronimage({'w': blend.shape[0],
                                                        'h': blend.shape[1]})
                        centered = True

                    await ann_window.set_clean()
                await asyncio.sleep(1)
