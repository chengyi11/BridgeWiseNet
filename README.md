# BridgeWiseNet: Interference-Aware Bridge Detection in Remote Sensing Images with Vision Foundation Models

<p align="center">
  <img src="images/BridgeWise.png" alt="BridgeWiseNet framework" width="95%">
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#highlights">Highlights</a> •
  <a href="#method">Method</a> •
  <a href="#results">Results</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

Bridge detection in high-resolution remote sensing images (HRSIs) is important for infrastructure inventory, disaster assessment, traffic monitoring, and emergency response. However, under the frozen-transfer setting of Vision Foundation Models (VFMs), bridge detection remains challenging because two recurrent interference patterns degrade performance in different ways:

- **Texture-dominated high-frequency (HF) interference**, which corrupts feature-level evidence competition.
- **Contextual contrast interference**, which disrupts query-level span reasoning and often leads to premature endpoint truncation.

To address this problem, we propose **BridgeWiseNet**, an interference-aware bridge detection framework for remote sensing images. BridgeWiseNet introduces:

- **WSS (Wavelet Subband Suppression)**: a frequency-domain feature rectification module that selectively suppresses nuisance-dominated HF responses while preserving bridge-supportive structural details.
- **PRISM (Profile Reasoning and Inlier Span Module)**: a training-only query-guided span reasoning module that regularizes continuity, endpoint integrity, and center-context separation under complex backgrounds.

In addition, we build **MBDDv2**, an expanded bridge detection benchmark with broader scene coverage and more diverse bridge instances.

---

## Highlights

- Reformulates frozen-transfer bridge detection as a **mechanism-aware interference modeling** problem rather than a generic clutter suppression task.
- Explicitly decomposes bridge detection failures into **two distinct interference modes** and addresses them at different stages.
- Introduces **WSS** for structure-preserving feature rectification in the wavelet domain.
- Introduces **PRISM** for training-only span-aware query supervision without inference-time overhead.
- Achieves strong reproduced results on **MBDDv2** and **GLH-Bridge** under a shared benchmark protocol.

---

## Method

### 1. Wavelet Subband Suppression (WSS)

WSS is inserted into later blocks of the frozen VFM backbone. It performs a fixed two-level Haar wavelet decomposition on token-grid features and uses subband statistics to distinguish nuisance-dominated responses from bridge-supportive structure.

Key properties:

- fixed wavelet decomposition in feature space
- statistics-guided suppression-preservation routing
- explicit spatial cue map generation
- lightweight deployment overhead

### 2. Profile Reasoning and Inlier Span Module (PRISM)

PRISM is a **training-only** auxiliary branch. It uses the matched query, GT box, and the cue map exported by WSS to build a structured span-field teacher for query-level supervision.

Key objectives:

- span continuity
- endpoint integrity
- center-context separation

PRISM is **removed during inference**, so it does not introduce extra test-time cost.

### 3. Unified Design Philosophy

BridgeWiseNet is not a loose stacking of wavelet processing and distillation. Instead, it is a unified interference-aware framework:

- **WSS** rectifies feature-level HF interference.
- **PRISM** stabilizes query-level span reasoning.
- The **cue map from WSS** bridges feature rectification and query supervision.

---

## Results

### Main benchmark results

| Method | Backbone | MBDDv2 AP50 | MBDDv2 AP | GLH-Bridge AP50 | GLH-Bridge AP |
|---|---|---:|---:|---:|---:|
| BridgeWiseNet | DINOv3-ViT-L/16 | **56.7** | **27.8** | **73.9** | **35.5** |

### Efficiency

| Variant | Params | FLOPs | FPS |
|---|---:|---:|---:|
| BridgeWiseNet (deployed) | **313.5M** | **4.12T** | **4.0** |

### Key observations

- On **MBDDv2**, BridgeWiseNet improves AP50 by **2.7 points** over the vanilla frozen-VFM baseline.
- On **GLH-Bridge**, BridgeWiseNet improves AP50 by **1.8 points** over the vanilla frozen-VFM baseline.
- WSS improves the feature substrate by reallocating HF responses toward bridge-supportive regions.
- PRISM significantly reduces span truncation, especially under strong contextual contrast.
- Because PRISM is removed during inference, the final detector gains better span reasoning **without extra deployment overhead**.

---

## Datasets

### MBDDv2

MBDDv2 is an expanded bridge detection benchmark built upon MBDD.

- **30,914** optical HRSIs
- **42,318** manually annotated bridge instances
- image size: **1024 × 1024**
- supports both **HBB** and **OBB** annotations

MBDDv2 broadens bridge scale distribution and scene diversity, covering vegetation, cropland, dry riverbeds, sandbars, roads, and built-up areas.

### GLH-Bridge

GLH-Bridge is a large-scale benchmark for bridge detection in large-size very-high-resolution images.

- **6,000** optical images
- **59,737** bridge instances
- spatial resolution: **0.3 m to 1.0 m**
- image size: **2048 × 2048** to **16384 × 16384**

> In this project, HBB annotations are used for training and primary benchmarking, while rotated annotations are used for bridge-aligned geometric analysis.

---

## Repository Structure

A recommended project structure is shown below:

```bash
BridgeWiseNet/
├── README.md
├── configs/
│   ├── mbddv2/
│   └── glh_bridge/
├── datasets/
│   ├── MBDDv2/
│   └── GLH-Bridge/
├── images/
│   ├── BridgeWise.png
│   ├── Phase.png
│   ├── WSS_demo.png
│   └── PRISM_demo.png
├── models/
│   ├── backbone/
│   ├── wss/
│   ├── prism/
│   └── detector/
├── tools/
│   ├── train.py
│   ├── test.py
│   ├── inference.py
│   └── eval.py
├── checkpoints/
├── outputs/
└── requirements.txt
