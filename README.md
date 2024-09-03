# robot_depth
A simple app that makes a robot detect how far a person is using a normal camera and make specific actions depending on the distance of the person from the robot

### Setup 

1) Download the model weights and place them in the `weights` folder:

    [dpt_hybrid-midas-501f0c75.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt), [Mirror](https://drive.google.com/file/d/1dgcJEYYw1F8qirXhZxgNK8dWWz_8gZBD/view?usp=sharing)

2) Set up dependencies: 

    ```shell
    pip install -r requirements.txt
    ```

   The code was tested with Python 3.11, PyTorch 2.4.0, OpenCV 4.10.0.84, and timm 1.0.9

### Usage 

Run a monocular depth estimation model:
	
    ```shell
    python run_monodepth_webcam.py
    ```

### Citation & Acknowledgements

This project builds on the amazing work done by the Intel Labs team in the DPT repository:
    [DPT](https://github.com/isl-org/DPT/)
Many thanks to the authors of PyTorch and Timm for making this possible and making their work available or everyone:
    [timm](https://github.com/rwightman/pytorch-image-models)
    [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

Papers:
```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}
```

```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```
