
"""
--------------------------------------------------------
Mhub / GC - Run Module for DeepSpA Classifier
--------------------------------------------------------

--------------------------------------------------------
Author: Felix Dorfner
Email:  felix.dorfner@charite.de
--------------------------------------------------------
"""
import torch.cuda
from mhubio.core import Instance, InstanceData, IO, Module, ValueOutput, Meta
from typing import Dict
import os
import torch
import Model_Class
import Model_Seg
import SimpleITK as sitk
import utils
from numpy import uint8, rot90, fliplr
from monai.transforms import Rotate90
from segdb.classes.Segment import Segment, Triplet
import matplotlib.pyplot as plt
import numpy as np


@ValueOutput.Name('raxSpAprob')
@ValueOutput.Meta(Meta(min=0.0, max=1.0, type="probability"))
@ValueOutput.Label('radiographic axSpA probability score')
@ValueOutput.Type(float)
@ValueOutput.Description('The predicted probability score for the patient having radiographic axSpA by the algorithm')
class raxSpAprob(ValueOutput):
   pass

@ValueOutput.Name('nraxSpAprob')
@ValueOutput.Meta(Meta(min=0.0, max=1.0, type="probability"))
@ValueOutput.Label('no radiographic axSpA probability score')
@ValueOutput.Type(float)
@ValueOutput.Description('The predicted probability score for the patient not having r-axSpA by the algorithm')
class nraxSpAprob(ValueOutput):
   pass

@ValueOutput.Name('binarized_pred')
@ValueOutput.Meta(Meta(min=0.0, max=1.0, type="probability"))
@ValueOutput.Label('Binarized prediction for the presence of r-axSpA, using the cut-off 0.59')
@ValueOutput.Type(float)
@ValueOutput.Description('Binarized prediction for the presence of r-axSpA, using the cut-off 0.59')
class binarized_pred(ValueOutput):
   pass


# register custom segmentation before class definition
Triplet.register("PELVIS", code="12921003", meaning="Pelvis", scheme_designator="SCT", overwrite=True)
Segment.register("LEFT_PELVIS", name="Left Pelvis", category="C_BODY_STRUCTURE", type="PELVIS", modifier="M_LEFT")
Segment.register("RIGHT_PELVIS", name="Right Pelvis", category="C_BODY_STRUCTURE", type="PELVIS", modifier="M_RIGHT")

class DeepSpARunner(Module):

    @IO.Instance()
    @IO.Input('in_data', 'dicom', the='input xray scan')
    @IO.Output('out_data', 'segmentation_overlay.png', 'png:mod=seg:model=DeepSpA:roi=SACRUM,LEFT_PELVIS,RIGHT_PELVIS', the="Output segmentation mask containing all labels, overlayed on the input image")
    @IO.Output('out_data_box', 'boxed_image.png', 'png:mod=box:model=DeepSpA:roi=SACRUM,LEFT_PELVIS,RIGHT_PELVIS', the="The input image, with the bounding box applied. This image was used as the input to the classification model.")
    @IO.Output('out_data_gradcam', 'gradcam_image.png', 'png:mod=gradcam:model=DeepSpA:roi=SACRUM,LEFT_PELVIS,RIGHT_PELVIS', the="Grad-CAM visualization of the classification model, which runs its inference on the bounding box cropped image.")
    @IO.OutputData('raxSpAprob', raxSpAprob)
    @IO.OutputData('nraxSpAprob', nraxSpAprob)
    @IO.OutputData('binarized_pred', binarized_pred)
    def task(self, instance: Instance, in_data: InstanceData, out_data: InstanceData, raxSpAprob: raxSpAprob, nraxSpAprob: nraxSpAprob, binarized_pred: binarized_pred, out_data_box: InstanceData, out_data_gradcam: InstanceData) -> None:

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
      im_series = os.listdir(in_data.abspath)[0]
      in_path = os.path.join(in_data.abspath, im_series)

      image_mask = Model_Seg.load_and_segment_image(in_path, device)



      overlay_image_np, original_image_np = utils.overlay_mask(in_path, image_mask)
      overlay_image_np = rot90(overlay_image_np, k=3)
      overlay_image_np = fliplr(overlay_image_np)
      plt.imsave(out_data.abspath, np.ascontiguousarray(overlay_image_np), cmap='gray')

      image_mask_im = sitk.GetImageFromArray(image_mask[None, :, :].astype(uint8))
      image_im = sitk.GetImageFromArray(original_image_np[None, :, :].astype(uint8))
      cropped_boxed_im, _ = utils.mask_and_crop(image_im, image_mask_im)

      cropped_boxed_array = sitk.GetArrayFromImage(cropped_boxed_im)
      cropped_boxed_tensor = torch.Tensor(cropped_boxed_array)
      rotate = Rotate90(spatial_axes=(0, 1), k=3)

      cropped_boxed_tensor = rotate(cropped_boxed_tensor)
      cropped_boxed_array_disp = cropped_boxed_tensor.numpy().squeeze().astype(uint8)
      prediction, image_transformed = Model_Class.load_and_classify_image(cropped_boxed_tensor, device)
      plt.imsave(out_data_box.abspath, cropped_boxed_array_disp, cmap='gray')


      gradcam = Model_Class.make_GradCAM(image_transformed, device)
      plt.imsave(out_data_gradcam.abspath, gradcam, cmap='jet')

      nr_axSpA_prob = float(prediction[0].item())
      r_axSpA_prob = float(prediction[1].item())


      raxSpAprob.value = r_axSpA_prob
      nraxSpAprob.value = nr_axSpA_prob

      # Prediction gets binarized based on the predetermined threshold
      binarized_pred.value = float(r_axSpA_prob > 0.59)
