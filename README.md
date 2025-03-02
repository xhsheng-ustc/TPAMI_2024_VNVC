# Introduction
* Official Pytorch implementation for [VNVC: A Versatile Neural Video Coding Framework for Efficient Human-Machine Vision](https://ieeexplore.ieee.org/document/10411051), in IEEE Transactions on Pattern Analysis and Machine Intelligence.

# Prerequisites
* Python 3.6 and conda, get [Conda](https://www.anaconda.com/)
* CUDA if want to use GPU
* pytorch==1.10.0
* torchvision==0.11.0
* cudatoolkit=11.1
* Other tools
    ```
    pip install -r requirements.txt
    ```
# Test dataset
We support arbitrary original resolution. The input video resolution will be padded to 64x automatically. The reconstructed video will be cropped back to the original size. The distortion (PSNR/MS-SSIM) is calculated at original resolution.

The dataset format can be seen in dataset_config_example.json.

For example, one video of HEVC Class B can be prepared as:
* Make the video path:
    ```
    mkdir BasketballDrive_1920x1080_50
    ```
* Convert YUV to PNG:
    ```
    ffmpeg -pix_fmt yuv420p -s 1920x1080 -i BasketballDrive_1920x1080_50.yuv -f image2 BasketballDrive_1920x1080_50/im%05d.png
    ```
At last, the folder structure of dataset is like:

    /media/data/HEVC_B/
        * BQTerrace_1920x1080_60/
            - im00001.png
            - im00002.png
            - im00003.png
            - ...
        * BasketballDrive_1920x1080_50/
            - im00001.png
            - im00002.png
            - im00003.png
            - ...
        * ...
    /media/data/HEVC_D
    /media/data/HEVC_C/
    ...

# Build the project
Please build the C++ code if want to test with actual bitstream writing. There is minor difference about the bits for calculating the bits using entropy (the method used in the paper to report numbers) and actual bitstreaming writing. There is overhead when writing the bitstream into the file and the difference percentage depends on the bitstream size. Usually, the overhead for 1080p content is less than 0.5%.
## On Windows
```bash
cd src
mkdir build
cd build
conda activate $YOUR_PY38_ENV_NAME
cmake ../cpp -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

## On Linux
```bash
sudo apt-get install cmake g++
cd src
mkdir build
cd build
conda activate $YOUR_PY36_ENV_NAME
cmake ../cpp -DCMAKE_BUILD_TYPE=Release
make -j
```

# Pretrained Models

* Download our [pretrained models](https://drive.google.com/drive/folders/1uCV9g1wfjywH1Bn-XSqfCL0sv02Ujafr?usp=drive_link).

# Test
```bash
./run_test_psnr.sh
```

# Acknowledgement
The implementation is based on [DCVC-TCM](https://github.com/microsoft/DCVC/tree/main/DCVC-TCM).
# Citation
If you find this work useful for your research, please cite:

```
@article{sheng2024vnvc,
  title={Vnvc: A versatile neural video coding framework for efficient human-machine vision},
  author={Sheng, Xihua and Li, Li and Liu, Dong and Li, Houqiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={46},
  number={7},
  pages={4579--4596},
  year={2024},
  publisher={IEEE}
}
```
