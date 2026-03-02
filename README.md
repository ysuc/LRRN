# LRRN: Lightweight Remote Sensing Image Recognition Network

This repository contains the official PyTorch implementation of the paper  
**"LRRN: A Lightweight Remote Sensing Image Recognition Network with Parallel Multi-Scale Feature Extraction Paths"** by Chen et al. (2025).  
The code includes the proposed LRRN model (composed of **FIN** and **CNH** modules), baseline networks, training/testing scripts, and the **HBUA-NR5** dataset used in transfer learning experiments.

**Paper URL:** [https://github.com/ysuc/LRRN](https://github.com/ysuc/LRRN) (publicly available as stated in the paper)

---

## 📌 Overview

LRRN is a lightweight convolutional neural network designed for efficient remote sensing image recognition. It consists of two main modules:

- **Feature Integration Network (FIN):** Extracts multi‑scale features using parallel convolutional paths and fuses them through skip connections (implemented in `modul_zoo.py` as class `FIN`).
- **Classification Network Head (CNH):** Aggregates sparse discriminative features via a **Tandem Pooling** mechanism and performs final classification (implemented in `modul_zoo.py` as class `CNH`).

The model achieves competitive accuracy on multiple public datasets while having an extremely low parameter count (only 2.07M) and fast inference speed, making it suitable for deployment on resource‑constrained devices.

---

## 📁 Repository Structure
