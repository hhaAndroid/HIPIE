import torch
from PIL import Image
from matplotlib import pyplot as plt
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import convert_PIL_to_numpy

from projects.HIPIE.demo_lib.part_segm_demo import PartSegmDemo
from projects.HIPIE.hipie.data.coco_dataset_mapper_uni import get_openseg_labels

import os

os.chdir('../')

config_file = 'projects/HIPIE/configs/image_joint_r50_pan_maskdino_parts.yaml'
ckpt = 'weights/r50_parts.pth'

device = 'cpu'
uninext_demo = PartSegmDemo(config_file=config_file, weight=ckpt, device=device)

uninext_demo.demo.predictor.model.device = device

# url = 'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png'
# url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/405px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg'
# url = 'https://encrypted-tbn0.gstatic.com/licensed-image?q=tbn:ANd9GcSh82Dm7OEK7SU7Rnv3Fa-9bi6BZrMM0NJvtm94eGajmxl7mObM7Jp9h3z5UfxaY5IsXGBoB9IX1QMVMlE'
# url = 'https://hips.hearstapps.com/hmg-prod/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=0.752xw:1.00xh;0.175xw,0&resize=1200:*'
# image = Image.open(requests.get(url,stream=True).raw)
image = Image.open('assets/demo_hipie.jpg')
image_np_d2 = convert_PIL_to_numpy(image, format="BGR")

COCO_OPENSEG_LABELS = get_openseg_labels('coco_panoptic')
coco_labels = dict(
    things_labels=[x['name'] for x in get_openseg_labels('coco_panoptic')[:80]],
    stuff_labels=[x['name'] for x in get_openseg_labels('coco_panoptic')[80:]]
)
custom_labels = dict(
    things_labels=['cats', 'books', 'mouse', 'bottle', 'clock', 'cellphone'],
    stuff_labels=['sky', 'shelf', 'table']
)

selected_labels = coco_labels

mask = uninext_demo.foward_panoptic(image_np_d2, do_part=True, instance_thres=0.49, **selected_labels)

visualizer = Visualizer(image, metadata=mask['meta_data'])


def show_panoptic():
    vis_output = visualizer.draw_panoptic_seg(mask['panoptic_seg'][0].cpu(), mask['panoptic_seg'][1])
    return vis_output


def show_instance():
    vis_output = visualizer.overlay_instances(masks=torch.stack(mask['output_refined'][0]),
                                              labels=mask['output_refined'][1])
    return vis_output


def show_seg():
    vis_output = visualizer.draw_instance_predictions(mask['instances'].to('cpu'))
    return vis_output


def show_ref():
    ref_str = "the mother behind table"
    mask = uninext_demo.foward_reference(image_np_d2, ref_str, '')

    vis = Visualizer(image)
    # vis.overlay_instances(masks=mask['instance_mask'])
    # vis.overlay_instances(masks=part_mask)
    vis_output = vis.overlay_instances(masks=[mask['final_mask'][0].numpy()], labels=[ref_str])
    return vis_output


if __name__ == '__main__':
    visualized_output = show_ref()
    visualized_output.save('ref_results.jpg')
    # cv2.namedWindow('COCO detections', cv2.WINDOW_NORMAL)
    # cv2.imshow('COCO detections', visualized_output.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
