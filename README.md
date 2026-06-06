# 🌕 Lunar: Hybrid AI System for Lunar Landslide & Boulder Detection

<div align="center">

### Physics-Aware Artificial Intelligence for Automated Lunar Surface Analysis

A research-oriented hybrid AI system that combines deep learning, terrain analysis, and explainable artificial intelligence to detect lunar landslides and boulders using Chandrayaan mission datasets.

**Status:** 🚧 Active Development & Testing

</div>

---

## Overview

Lunar is a hybrid AI framework designed to automatically identify, analyze, and explain geological features on the Moon using remote sensing data from Chandrayaan missions.

The system integrates:

* Deep Learning (CNN-based classification)
* Terrain-aware anomaly detection
* Physics-informed reasoning
* Explainable AI techniques
* Automated geological reporting

The objective is to assist planetary scientists in identifying lunar landslides, boulder fields, and potentially active geological regions.

---

## Key Features

### Boulder Detection

* CNN-based boulder classification
* Automated boulder mask generation
* Boulder statistics extraction
* Spatial distribution analysis

### Landslide Detection

* Terrain anomaly identification
* Landslide segmentation
* Source region estimation
* Geometric characterization

### Terrain Analysis

* Slope computation
* Curvature extraction
* Multi-layer terrain representation
* Physics-aware feature engineering

### Explainable AI

* Detection reasoning
* Terrain feature interpretation
* Geological explanation generation
* Automated analysis reports

---

## Dataset Sources

This project is designed to process:

* Chandrayaan-1 TMC imagery
* Chandrayaan-2 TMC imagery
* Chandrayaan-2 OHRC imagery
* Digital Terrain Models (DTM)

---

## Project Structure

```text
lunar/
│
├── data/
│   └── stack.tif
│
├── models/
│
├── modules/
│   ├── cnn/
│   ├── boulder_detection.py
│   ├── landslide_detection.py
│   ├── feature_extraction.py
│   ├── feedback_trainer.py
│   └── utils.py
│
├── results/
│   ├── annotated_output.png
│   ├── analysis_report.txt
│   ├── boulder_mask.tif
│   ├── landslide_mask.tif
│   └── landslide_sources.tif
│
├── generate_patches.py
├── train_boulder_classifier.py
├── train_landslide_classifier.py
├── main.py
└── requirements.txt
```

---

## Current Development Status

This project is currently under active research, development, and testing.

Implemented:

* Data preprocessing pipeline
* Terrain feature extraction
* Boulder detection workflow
* Landslide detection workflow
* Result visualization
* Automated report generation

In Progress:

* Model optimization
* Accuracy improvements
* Cross-region validation
* Performance benchmarking
* Explainability enhancements

Planned:

* Web dashboard
* Interactive visualization tools
* Multi-region analysis
* Enhanced geological reasoning

---

## Installation

```bash
git clone https://github.com/SanjibBayen/lunar.git

cd lunar

pip install -r requirements.txt
```

Run:

```bash
python main.py
```

---

## Research Disclaimer

This project is currently experimental and intended for research and educational purposes.

Outputs should not be considered scientifically validated without further verification and expert review.

---

## Author

**Sanjib Bayen**

B.Tech Computer Science Engineering

Research Interests:

* Artificial Intelligence
* Computer Vision
* Planetary Science
* Explainable AI
* Remote Sensing

---

## License

This project is licensed under the MIT License.
