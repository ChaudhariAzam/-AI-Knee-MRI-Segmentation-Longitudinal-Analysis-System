# 🦵 AI Knee MRI Segmentation & Longitudinal Analysis System

## 📌 Project Overview

This project is a full-stack AI-powered Knee MRI segmentation and
clinical tracking system.

It integrates:

-   Orthanc PACS (DICOM server)
-   dcm2niix (DICOM → NIfTI conversion)
-   nnUNet v2 (3D segmentation)
-   Flask (Web Interface)
-   SQLite (Patient history database)
-   NiBabel, NumPy, SciPy (Medical image processing)

------------------------------------------------------------------------

## 🏗 System Workflow

Orthanc PACS\
↓\
Async DICOM Download\
↓\
dcm2niix Conversion\
↓\
Image Reorientation (RAS)\
↓\
Resampling (256×256×144)\
↓\
nnUNet v2 Segmentation\
↓\
Volume Calculation\
↓\
Database Storage\
↓\
3D Visualization & Trend Analysis

------------------------------------------------------------------------

## 🎯 Features

### ✅ Automatic Protocol Detection

Processes only: `t2_de3d_we_sag_iso`

### ✅ AI Segmentation

Uses nnUNet v2 (3d_fullres configuration)

Segmented Structures: - Femur - Tibia - Fibula - Patella - Cartilage

### ✅ Volume Calculation

-   Computes voxel counts
-   Converts mm³ → cm³
-   Stores structured metrics

### ✅ Longitudinal Tracking

-   Study comparison
-   \% volume change
-   Trend detection
-   Time interval calculation

### ✅ 3D Multi-View Visualization

-   Axial View
-   Coronal View
-   Sagittal View
-   Mouse & keyboard navigation
-   Touch support

### ✅ LVEF Data Management

-   Stores cardiac LVEF values
-   Calculates improvement percentage
-   Historical tracking

------------------------------------------------------------------------

## 🗄 Database Tables

-   patients
-   studies
-   volume_measurements
-   lvef_measurements

------------------------------------------------------------------------

## ⚙️ Requirements

### System Tools

-   Python 3.9+
-   nnUNet v2 installed
-   dcm2niix installed
-   Orthanc PACS running

### Python Libraries

-   flask
-   aiohttp
-   nibabel
-   numpy
-   scipy
-   pandas
-   pydicom
-   matplotlib
-   scikit-image

Install with:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🚀 How to Run

1️⃣ Verify tools:

``` bash
dcm2niix --version
nnUNetv2_predict --help
```

2️⃣ Start server:

``` bash
python app.py
```

Server runs at:

http://0.0.0.0:7050

------------------------------------------------------------------------

## 📂 Folder Structure

    dicom_data/
    nifti_output/
    temp_input/
    temp_output/
    knee_segmentation.db
    app.py

------------------------------------------------------------------------

## 📊 Web Routes

  Route              Description
  ------------------ ----------------------
  /                  Patient search
  /select_knee       Protocol filtering
  /process           Processing animation
  /success           Segmentation results
  /patient_history   Historical tracking
  /lvef_data         LVEF management
  /export_report     Export reports

------------------------------------------------------------------------

## 🔐 Validation & Safety

-   DICOM validation
-   File integrity checks
-   Segmentation label verification
-   Empty mask detection
-   Retry logic for downloads

------------------------------------------------------------------------

## 📈 Future Enhancements

-   PDF export
-   Excel export
-   Authentication system
-   Cloud deployment
-   REST API version
-   Multi-organ segmentation

------------------------------------------------------------------------

## 🧠 Clinical Applications

-   Osteoarthritis monitoring
-   Cartilage degeneration tracking
-   Post-operative comparison
-   Research dataset generation
-   PACS-AI hospital integration

------------------------------------------------------------------------

## 👨‍⚕️ Built For

Radiology departments\
Orthopedic researchers\
Medical AI startups\
Clinical research environments

------------------------------------------------------------------------

© 2026 Knee Segmentation AI System
