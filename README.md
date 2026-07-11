# ShadowRemovalColorTransfer

![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M2-white) ![License](https://img.shields.io/badge/License-MIT-purple) ![Blender](https://img.shields.io/badge/Blender-5.1.2-orange?logo=blender) ![Sun Position Add-on](https://img.shields.io/badge/Sun%20Position%20Add--on-4.4.0-orange?logo=blender) ![Python](https://img.shields.io/badge/Python-3.12.6-blue?logo=python)

This repository contains the implementation of the shadow removal method based on color transfer presented in the paper *Enhanced Illumination Adjustment in 3D Outdoor Reconstructions via Shadow Removal through Color Transfer*.

<img width="2608" height="1228" alt="MMSP_Pipeline-Visual_Pipeline" src="https://github.com/user-attachments/assets/b9cb68d1-2d62-4596-b2ef-224ccdc18c81" />


## Workflow

The method consists of the following steps:

1. Manual orientation adjustment of the reconstruction
2. Illumination configuration in Blender
3. Texture baking in Blender
4. Shadow removal and color transfer in Python

## Usage

### Step 1: Orientation Adjustment

This step must be performed manually in Blender.

Using the known geographic location of the reconstructed object, determine its orientation with Google Maps or another map service. Then rotate the reconstruction in Blender so that it matches its real-world orientation.

### Steps 2 and 3: Illumination Configuration and Texture Baking

These steps are executed in Blender using `blender.py`.

#### Prerequisites

Install:

- [Blender](https://www.blender.org/download/)
- [Sun Position add-on](https://extensions.blender.org/add-ons/sun-position/)

After installing the Sun Position add-on, enable it in Blender under:

```text
File > Preferences > Add-ons
```

#### Configuration

Update the following values in `options.json`:

- `directory`: Directory containing the input OBJ file
- `export_directory`: Directory in which the generated files are stored
- Time parameters: Date and time at which the reconstruction was captured
- Location parameters: Geographic location of the reconstructed object

The time and location parameters are used by the Sun Position add-on to reproduce the corresponding illumination conditions.

#### Run

Clone the repository:

```bash
git clone git@github.com:hpotechius/ShadowRemovalColorTransfer.git
cd ShadowRemovalColorTransfer
```

Run `blender.py` using Blender in background mode:

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
  --background \
  --python blender.py
```

The path shown above is the default Blender application path on macOS. Adjust it when using another installation path or operating system.

### Step 4: Shadow Removal and Color Transfer

The remaining processing steps are executed with Python.

#### Configuration

Adjust the following parameters in `options.json` if necessary:

- `clusters`: Number of clusters used to divide the texture into regions with different illumination conditions
- `color_transfer_method`: Color transfer method to apply; supported values are `reinhard` and `fuzzy`
- `blending`: Determines whether the boundaries between clusters are blended or directly stitched together

#### Installation and Execution

Create and activate a Python virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the shadow removal pipeline:

```bash
python main.py
```

## Citation

When using this code in academic work, please cite the following publication:

```bibtex
@inproceedings{potechius2024,
    author = {Potechius, Herbert and Essaky, Selvam and Raja, Gunasekaran and Sikora, Thomas and Knorr, Sebastian},
    title = {Enhanced Illumination Adjustment in 3D Outdoor Reconstructions via Shadow Removal through Color Transfer},
    year = {2024},
    isbn = {9798400712814},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3697294.3697308},
    doi = {10.1145/3697294.3697308},
    booktitle = {Proceedings of the 21st ACM SIGGRAPH Conference on Visual Media Production},
    articleno = {2},
    numpages = {10},
    location = {London, United Kingdom},
    series = {CVMP '24}
}
```

## License

This project is licensed under the MIT License.
