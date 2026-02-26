# -AI-Knee-MRI-Segmentation-Longitudinal-Analysis-System
📌 Project Overview

This project is a full-stack AI-powered Knee MRI segmentation and clinical tracking system built using:

🧠 nnUNet v2 for automated 3D segmentation

🏥 Orthanc PACS integration for DICOM retrieval

🔄 dcm2niix for DICOM → NIfTI conversion

🧪 NiBabel & NumPy for medical image processing

🌐 Flask for web-based workflow

🗄 SQLite for structured patient history & volume tracking

The system automatically:

Retrieves MRI studies from Orthanc

Filters for the target protocol (t2_de3d_we_sag_iso)

Converts DICOM → NIfTI

Preprocesses images (reorientation + resampling)

Runs nnUNet segmentation

Computes anatomical volumes

Stores longitudinal data

Provides interactive 3D visualization

Tracks volume trends & LVEF data over time

🏗 System Architecture
4

Pipeline Flow:

Orthanc PACS
     ↓
DICOM Download (Async)
     ↓
dcm2niix Conversion
     ↓
Image Reorientation (RAS)
     ↓
Resampling (256×256×144)
     ↓
nnUNet v2 Segmentation
     ↓
Volume Calculation
     ↓
SQLite Storage
     ↓
Web Visualization & Trend Analysis
🎯 Key Features
🔍 Automatic Protocol Detection

Only processes MRI series containing:

t2_de3d_we_sag_iso

With automatic:

Laterality detection (Left / Right)

Metadata validation

Fallback intelligent selection

🤖 AI Segmentation

Uses nnUNet v2 (3d_fullres configuration)

Disables TTA for faster inference

Validates output labels

Ensures non-empty segmentation masks

Segmented Structures:

Label	Structure
1	Femur
2	Tibia
3	Fibula
4	Patella
5	Cartilage
📊 Volume Computation

Voxel-based calculation

Converts mm³ → cm³

Stores:

Voxel count

Volume (mm³)

Volume (cm³)

📈 Longitudinal Analysis

The system automatically:

Tracks multiple studies per patient

Computes % volume change

Detects increasing / decreasing trends

Calculates total improvement

Tracks time intervals

🖥 Interactive 3D Multi-View Viewer
4

Includes:

Axial View

Coronal View

Sagittal View

Scroll-wheel slice navigation

Arrow-key navigation

Touch support

Slice synchronization

Real-time coordinate display

🗄 Database Schema

SQLite Tables:

patients

studies

volume_measurements

lvef_measurements

Supports:

Historical comparisons

Volume trend reports

LVEF improvement tracking

Structured patient reports

🔌 Orthanc Integration

Async downloads via aiohttp

Supports retry logic

Parallel instance downloading

Metadata parsing for:

ProtocolName

SeriesDescription

Laterality

BodyPartExamined
