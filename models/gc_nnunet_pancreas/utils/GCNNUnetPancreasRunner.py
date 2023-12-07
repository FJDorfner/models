"""
-----------------------------------------------------------
GC / MHub - Run Module for the GC NNUnet Pancreas Algorithm
-----------------------------------------------------------

-----------------------------------------------------------
Author: Sil van de Leemput
Email:  sil.vandeleemput@radboudumc.nl
-----------------------------------------------------------
"""

from mhubio.core import Module, Instance, InstanceData, DataType, Meta, IO

from pathlib import Path
import SimpleITK
import numpy as np
import sys


CLI_PATH = Path(__file__).parent / "cli.py"


class GCNNUnetPancreasRunner(Module):
    @IO.Instance()
    @IO.Input('in_data', 'mha:mod=ct', the="input data")
    @IO.Output('heatmap', 'heatmap.mha', 'mha:mod=heatmap:model=GCNNUnetPancreas', data="in_data",
               the="heatmap of the pancreatic tumor likelihood")
    @IO.Output('segmentation', 'segmentation.mha', 'mha:mod=seg:type=original:model=GCNNUnetPancreas', data="in_data",
               the="original segmentation of the pancreas, with the following classes: "
                   "0-background, 1-veins, 2-arteries, 3-pancreas, 4-pancreatic duct, 5-bile duct, 6-cysts, 7-renal vein")
    @IO.Output('segmentation_remapped', 'segmentation_remapped.mha', 'mha:mod=seg:type=remapped:model=GCNNUnetPancreas:roi=PANCREAS,PANCREATIC_DUCT,BILE_DUCT,PANCREAS+CYST,RENAL_VEIN', data="in_data",
               the="remapped segmentation of the pancreas (without the veins and arteries), with the following classes: "
                   "0-background, 1-pancreas, 2-pancreatic duct, 3-bile duct, 4-cysts, 5-renal vein")
    def task(self, instance: Instance, in_data: InstanceData, heatmap: InstanceData, segmentation: InstanceData, segmentation_remapped: InstanceData, **kwargs) -> None:
        # Call the PDAC CLI
        cmd = [
            sys.executable,
            str(CLI_PATH),
            in_data.abspath,
            heatmap.abspath,
            segmentation.abspath
        ]
        self.subprocess(cmd, text=True)

        # Generate remapped segmentation
        self.remap_segementation(
            segmentation=segmentation,
            segmentation_remapped=segmentation_remapped
        )

    def remap_segementation(self, segmentation: InstanceData, segmentation_remapped: InstanceData):
        mapping = {0:0, 1:0, 2:0, 3:1, 4:2, 5:3, 6:4, 7:5}
        mapping_numpy = np.array(list(mapping.values()), dtype=np.uint8)
        self.log("Creating remapped segmentation", level="NOTICE")
        seg_sitk = SimpleITK.ReadImage(segmentation.abspath)
        seg_numpy = SimpleITK.GetArrayFromImage(seg_sitk)
        remapped_numpy = mapping_numpy[seg_numpy]
        remapped_sitk = SimpleITK.GetImageFromArray(remapped_numpy)
        remapped_sitk.CopyInformation(seg_sitk)
        SimpleITK.WriteImage(remapped_sitk, segmentation_remapped.abspath, True)
