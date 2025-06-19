# motion-Enhanced-mvcnn

A deep learning framework for multi-view object recognition using motion-enhanced Multi-View Convolutional Neural Networks.

## Overview

This project implements a motion-enhanced Multi-View Convolutional Neural Network (MVCNN) approach for 3D object recognition. By leveraging multiple 2D views of 3D objects, our framework achieves improved feature extraction and classification performance.

## Features

- Multi-view 3D object classification
- Motion-enhanced feature aggregation
- Support for standard 3D model datasets (ModelNet40)
- Custom dataset generation tools

## Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA (for GPU acceleration)
- Blender 2.9+ (for dataset generation)

## MVCNN Dataset Generator

The `mvcnn_dataset_generator.py` is a Blender-based script designed to create multi-view rendered datasets from 3D models. It captures multiple views of 3D models from different camera angles and elevations, generating standardized images for training Multi-View CNNs.

### Features

- Renders 3D models from multiple viewpoints with configurable camera positions
- Supports multiple elevation angles for comprehensive object capture
- Automatically imports and processes OFF file format common in ModelNet40 dataset
- Generates organized dataset structure with metadata
- Creates visual previews of camera setup
- Includes demo animation mode for visualizing camera trajectories

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--modelnet-dir` | Path to ModelNet40 directory | `ModelNet40` |
| `--output-dir` | Output directory for rendered images | `mvcnn_dataset` |
| `--num-views` | Number of camera views per elevation | `12` |
| `--radius` | Camera distance from object | `3.0` |
| `--image-size` | Output image size (pixels) | `224` |
| `--max-models` | Maximum models per class (optional) | All models |
| `--render-quality` | Rendering quality (DRAFT, MEDIUM, HIGH) | `DRAFT` |
| `--elevation-angles` | Elevation angles in degrees | `[0, 30]` |
| `--no-class-previews` | Disable creation of class preview images | Enabled by default |
| `--only-class` | Only process a specific class | Process all classes |
| `--list-classes` | List all available classes and exit | |

#### Demo Mode Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--demo` | Run in demonstration mode (animation) | |
| `--demo-class` | Class to use for demo | `airplane` |
| `--demo-speed` | Seconds per camera position in demo | `2.0` |
| `--demo-pause` | Pause time at each position in demo | `1.0` |
| `--demo-output` | Output video file for demo | `mvcnn_camera_demo.mp4` |

### Usage Examples

#### Basic Dataset Generation

```bash
blender --background --python mvcnn_dataset_generator.py -- --modelnet-dir /path/to/ModelNet40 --output-dir /path/to/output --num-views 20 --elevation-angles 0 15 30 45
```
