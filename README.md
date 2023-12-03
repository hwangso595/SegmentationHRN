> **Improvement of 3D Face Reconstruction with a Segmentation Approach
Performance Improvement Track** </br>
> Sung Ha Hwang

This project aims to enhance 3D facial reconstruction by integrating  accurate occlusion segmentation and inpainting, improving the Hierarchical Representation Network's (HRN) ability to handle occluded facial areas and produce high-fidelity 3D models.

## Getting Started
Clone the repo:
  ```bash
  git clone https://github.com/hwangso595/SegmentationHRN.git
  cd SegmentationHRN
  ```

### Requirements
**This implementation is only tested under Ubuntu/CentOS environment with Nvidia GPUs and CUDA installed.**

* Python >= 3.8
* PyTorch >= 1.6
* [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
  ```bash
  conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
  conda install -c fvcore -c iopath -c conda-forge fvcore iopath
  conda install -c bottler nvidiacub
  conda install pytorch3d -c pytorch3d
  ```
* Basic requirements, you can run
  ```bash
  conda create -n SegmentationHRN python=3.8
  source activate SegmentationHRN
  pip install -r requirements.txt
  ```
* [nvdiffrast](https://nvlabs.github.io/nvdiffrast/#installation)
  ```bash
  cd ..
  git clone https://github.com/NVlabs/nvdiffrast.git
  cd nvdiffrast
  pip install .

  apt-get install freeglut3-dev
  apt-get install binutils-gold g++ cmake libglew-dev mesa-common-dev build-essential libglew1.5-dev libglm-dev
  apt-get install mesa-utils
  apt-get install libegl1-mesa-dev
  apt-get install libgles2-mesa-dev
  apt-get install libnvidia-gl-525
  ```
  If there is a "[F glutil.cpp:338] eglInitialize() failed" error, you can try to change all the "dr.RasterizeGLContext" in util/nv_diffrast.py into "dr.RasterizeCudaContext".

### Testing with pre-trained network of base model
1. Prepare assets and pretrained models

    Download the assets from [Google Drive](https://drive.google.com/drive/folders/1aziPI4j6jfRabYDkmYe6upmRpV8eCInP?usp=sharing) and unzip in SegmentationHRN/assets.


2. Run demos

    a. single-view face reconstruction
    ```bash
    CUDA_VISIBLE_DEVICES=0 python demo.py --input_type single_view --input_root ./assets/examples/single_view_image --output_root ./assets/examples/single_view_image_results
    ```

    b. multi-view face reconstruction
    ```bash
    CUDA_VISIBLE_DEVICES=0 python demo.py --input_type multi_view --input_root ./assets/examples/multi_view_images --output_root ./assets/examples/multi_view_image_results
    ```
    where the "input_root" saves the multi-view images of the same subject.


3. inference time

    The pure inference time of HRN for single view reconstruction is less than 1 second. We added some visualization codes to the pipeline, resulting in an overall time of about 5 to 10 seconds. The multi-view reconstruction of MV-HRN involves the fitting process and the overall time is about 1 minute.

## Results
Work in progress

## Contact
Sung Ha Hwang (hwansung595@kaist.ac.kr)

## References
```
@misc{lei2023hierarchical,
      title={A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images},
      author={Biwen Lei and Jianqiang Ren and Mengyang Feng and Miaomiao Cui and Xuansong Xie},
      year={2023},
      eprint={2302.14434},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{chen2017rethinking,
      title={Rethinking Atrous Convolution for Semantic Image Segmentation},
      author={Liang-Chieh Chen and George Papandreou and Florian Schroff and Hartwig Adam},
      year={2017},
      eprint={1706.05587},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{lugmayr2022repaint,
      title={RePaint: Inpainting using Denoising Diffusion Probabilistic Models},
      author={Andreas Lugmayr and Martin Danelljan and Andres Romero and Fisher Yu and Radu Timofte and Luc Van Gool},
      year={2022},
      eprint={2201.09865},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
There are some functions or scripts in this implementation that are based on external sources. We thank the authors for their excellent works.
Here are some great resources we benefit:
- [HRN](https://github.com/youngLBW/HRN) for the base model of SegmentHRN.
- [DeeplabV3](https://github.com/tensorflow/models/tree/master/research/deeplab) for the segmentation model
- [RePaint](https://github.com/andreas128/RePaint) for the inpainting model

We would also like to thank these great datasets and benchmarks used in our testing
- [facescape](https://github.com/zhuhao-nju/facescape)
- [ESRC](http://pics.stir.ac.uk/ESRC/)