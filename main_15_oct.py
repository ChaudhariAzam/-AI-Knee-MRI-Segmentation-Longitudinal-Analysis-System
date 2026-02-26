from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation, aff2axcodes
import scipy.ndimage
import os
import io
import subprocess
import nibabel as nib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, Response
import json
import pydicom
import nrrd
import tempfile
import shutil
import asyncio
import aiohttp
import logging
import re
import glob
import zipfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage
from skimage import morphology
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd

# Set environment variable for matplotlib
os.environ['MPLBACKEND'] = 'Agg'

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Configurable Paths ---
ORTHANC_URL = "http://172.16.61.10:8042/"
ORTHANC_USERNAME = "orthanc"
ORTHANC_PASSWORD = "orthanc"
INPUT_FOLDER = "/mnt/e/before format pc/azam/knee/azam_pipeline/temp_input"
OUTPUT_FOLDER = "/mnt/e/before format pc/azam/knee/azam_pipeline/temp_output"
DICOM_DIR = "dicom_data"
NIFTI_OUTPUT_DIR = "nifti_output"
DB_PATH = "knee_segmentation.db"
MAX_CONCURRENT_DOWNLOADS = 50
DOWNLOAD_CHUNK_SIZE = 8192

# Target protocol name
TARGET_PROTOCOL = 't2_de3d_we_sag_iso'

# Create necessary directories
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DICOM_DIR, exist_ok=True)
os.makedirs(NIFTI_OUTPUT_DIR, exist_ok=True)

# Use ThreadPoolExecutor for parallel execution
executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# --- Database Functions ---
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            name TEXT,
            age INTEGER,
            gender TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(patient_id)
        )
    ''')
    
    # Studies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS studies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            study_date DATE NOT NULL,
            side TEXT NOT NULL,
            series_id TEXT NOT NULL,
            protocol_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id),
            UNIQUE(patient_id, study_date, side)
        )
    ''')
    
    # Volume measurements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS volume_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id INTEGER NOT NULL,
            label INTEGER NOT NULL,
            structure_name TEXT NOT NULL,
            volume_cm3 REAL NOT NULL,
            voxel_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (study_id) REFERENCES studies (id),
            UNIQUE(study_id, label)
        )
    ''')
    
    # LVEF measurements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lvef_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            measurement_date DATE NOT NULL,
            lvef_value REAL NOT NULL,
            improvement_percentage REAL,
            time_interval TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

def save_patient_data(patient_id, name=None, age=None, gender=None):
    """Save or update patient information"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO patients (patient_id, name, age, gender)
        VALUES (?, ?, ?, ?)
    ''', (patient_id, name, age, gender))
    
    conn.commit()
    conn.close()

def save_study_data(patient_id, study_date, side, series_id, protocol_name):
    """Save study information and return study ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First ensure patient exists
    save_patient_data(patient_id)
    
    cursor.execute('''
        INSERT OR REPLACE INTO studies (patient_id, study_date, side, series_id, protocol_name)
        VALUES (?, ?, ?, ?, ?)
    ''', (patient_id, study_date, side, series_id, protocol_name))
    
    study_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    return study_id

def save_volume_measurements(study_id, metrics, label_names):
    """Save volume measurements for a study"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for metric in metrics:
        structure_name = label_names.get(metric["Label"], f"Label {metric['Label']}")
        cursor.execute('''
            INSERT OR REPLACE INTO volume_measurements 
            (study_id, label, structure_name, volume_cm3, voxel_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (study_id, metric["Label"], structure_name, metric["Volume (cm³)"], metric["Voxel Count"]))
    
    conn.commit()
    conn.close()

def get_patient_studies(patient_id):
    """Get all studies for a patient"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT s.id, s.study_date, s.side, s.protocol_name, s.created_at,
               p.name, p.age, p.gender
        FROM studies s
        JOIN patients p ON s.patient_id = p.patient_id
        WHERE s.patient_id = ?
        ORDER BY s.study_date DESC
    ''', (patient_id,))
    
    studies = cursor.fetchall()
    conn.close()
    
    return studies

def get_study_measurements(study_id):
    """Get volume measurements for a specific study"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT structure_name, volume_cm3, voxel_count, created_at
        FROM volume_measurements
        WHERE study_id = ?
        ORDER BY label
    ''', (study_id,))
    
    measurements = cursor.fetchall()
    conn.close()
    
    return measurements

def get_patient_comparison_data(patient_id):
    """Get data for comparing multiple studies of the same patient"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT s.study_date, s.side, vm.structure_name, vm.volume_cm3
        FROM studies s
        JOIN volume_measurements vm ON s.id = vm.study_id
        WHERE s.patient_id = ?
        ORDER BY s.study_date, vm.label
    ''', (patient_id,))
    
    data = cursor.fetchall()
    conn.close()
    
    return data

def save_lvef_measurement(patient_id, measurement_date, lvef_value, notes=''):
    """Save LVEF measurement to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Calculate improvement if previous measurements exist
    cursor.execute('''
        SELECT lvef_value FROM lvef_measurements 
        WHERE patient_id = ? 
        ORDER BY measurement_date DESC 
        LIMIT 1
    ''', (patient_id,))
    
    previous = cursor.fetchone()
    improvement = None
    
    if previous:
        improvement = ((lvef_value - previous[0]) / previous[0]) * 100
    
    cursor.execute('''
        INSERT INTO lvef_measurements 
        (patient_id, measurement_date, lvef_value, improvement_percentage, notes)
        VALUES (?, ?, ?, ?, ?)
    ''', (patient_id, measurement_date, lvef_value, improvement, notes))
    
    conn.commit()
    conn.close()

def get_lvef_measurements(patient_id):
    """Get LVEF measurements for a patient"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT measurement_date, lvef_value, improvement_percentage, notes
        FROM lvef_measurements
        WHERE patient_id = ?
        ORDER BY measurement_date
    ''', (patient_id,))
    
    data = cursor.fetchall()
    conn.close()
    
    return data

def generate_patient_report(patient_id, studies):
    """Generate comprehensive patient report"""
    report = {
        'patient_id': patient_id,
        'total_studies': len(studies),
        'studies_by_side': {},
        'volume_trends': {},
        'improvement_summary': {}
    }
    
    # Group studies by side
    for study in studies:
        side = study[2]
        if side not in report['studies_by_side']:
            report['studies_by_side'][side] = []
        report['studies_by_side'][side].append(study)
    
    # Calculate trends and improvements
    comparison_data = get_patient_comparison_data(patient_id)
    if comparison_data:
        df = pd.DataFrame(comparison_data, columns=['study_date', 'side', 'structure_name', 'volume_cm3'])
        
        # Ensure study_date is in proper format
        try:
            df['study_date'] = pd.to_datetime(df['study_date'])
        except Exception as e:
            logger.warning(f"Error converting study dates: {e}")
        
        report['volume_trends'] = calculate_volume_trends(df)
        report['improvement_summary'] = calculate_improvement_summary(df)
    
    return report
def calculate_volume_trends(df):
    """Calculate volume trends over time"""
    trends = {}
    
    # Ensure study_date is datetime
    df_copy = df.copy()
    df_copy['study_date'] = pd.to_datetime(df_copy['study_date'])
    
    for structure in df_copy['structure_name'].unique():
        structure_data = df_copy[df_copy['structure_name'] == structure]
        structure_data = structure_data.sort_values('study_date')
        
        if len(structure_data) > 1:
            first_volume = structure_data.iloc[0]['volume_cm3']
            last_volume = structure_data.iloc[-1]['volume_cm3']
            
            if first_volume > 0:  # Avoid division by zero
                change = ((last_volume - first_volume) / first_volume) * 100
            else:
                change = 0
            
            trends[structure] = {
                'first_volume': first_volume,
                'last_volume': last_volume,
                'change_percentage': change,
                'trend': 'increasing' if change > 0 else 'decreasing'
            }
    
    return trends
def parse_date_safe(date_str):
    """Safely parse date string in various formats"""
    if isinstance(date_str, str):
        # Try different date formats
        formats = ['%Y-%m-%d', '%d-%m-%Y', '%Y%m%d', '%m/%d/%Y', '%d/%m/%Y']
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        # If none of the formats work, try pandas automatic parsing
        try:
            return pd.to_datetime(date_str)
        except:
            return None
    return date_str
def calculate_improvement_summary(df):
    """Calculate improvement summary"""
    summary = {}
    
    # Convert study_date to datetime if it's string
    df_copy = df.copy()
    df_copy['study_date'] = pd.to_datetime(df_copy['study_date'])
    
    total_volumes = df_copy.groupby('study_date')['volume_cm3'].sum().reset_index()
    total_volumes = total_volumes.sort_values('study_date')
    
    if len(total_volumes) >= 2:
        first_total = total_volumes.iloc[0]['volume_cm3']
        last_total = total_volumes.iloc[-1]['volume_cm3']
        total_improvement = ((last_total - first_total) / first_total) * 100
        
        # Calculate time interval
        first_date = total_volumes.iloc[0]['study_date']
        last_date = total_volumes.iloc[-1]['study_date']
        time_interval_days = (last_date - first_date).days
        
        summary['total_volume_improvement'] = total_improvement
        summary['time_interval'] = f"{time_interval_days} days"
    
    return summary

def calculate_improvement(previous, current):
    """Calculate percentage improvement"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def generate_comparison_html(patient_id, studies, current_date):
    """Generate HTML for comparing multiple studies"""
    try:
        comparison_data = get_patient_comparison_data(patient_id)
        
        if not comparison_data:
            return "<p>No previous studies found for comparison.</p>"
        
        df = pd.DataFrame(comparison_data, columns=['study_date', 'side', 'structure_name', 'volume_cm3'])
        
        pivot_df = df.pivot_table(
            index='structure_name', 
            columns='study_date', 
            values='volume_cm3', 
            aggfunc='first'
        ).reset_index()

        pivot_df = pivot_df.reindex(sorted(pivot_df.columns, key=lambda x: str(x) if x != 'structure_name' else ''), axis=1)
        
        if len(pivot_df.columns) > 2:
            dates = [col for col in pivot_df.columns if col != 'structure_name']
            dates.sort()
            
            for i in range(1, len(dates)):
                prev_col = dates[i-1]
                curr_col = dates[i]
                improvement_col = f'improvement_{i}'
                
                pivot_df[improvement_col] = pivot_df.apply(
                    lambda row: calculate_improvement(row[prev_col], row[curr_col]) if pd.notna(row[prev_col]) and pd.notna(row[curr_col]) else None,
                    axis=1
                )
        
        html = '<div class="table-responsive"><table class="comparison-table">'
        
        html += '<thead><tr>'
        html += '<th>Structure</th>'
        for col in pivot_df.columns:
            if col != 'structure_name' and not col.startswith('improvement_'):
                html += f'<th class="date-header">{col}</th>'
            elif col.startswith('improvement_'):
                html += f'<th>Improvement</th>'
        html += '</tr></thead>'
        
        html += '<tbody>'
        for _, row in pivot_df.iterrows():
            html += '<tr>'
            html += f'<td><strong>{row["structure_name"]}</strong></td>'
            
            for col in pivot_df.columns:
                if col != 'structure_name':
                    if col.startswith('improvement_'):
                        improvement = row[col]
                        if pd.notna(improvement):
                            css_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                            html += f'<td class="{css_class}">{improvement:+.1f}%</td>'
                        else:
                            html += '<td>-</td>'
                    else:
                        value = row[col]
                        if pd.notna(value):
                            html += f'<td>{value:.2f} cm³</td>'
                        else:
                            html += '<td>-</td>'
            
            html += '</tr>'
        html += '</tbody></table></div>'
        
        return html
        
    except Exception as e:
        logger.error(f"Error generating comparison HTML: {e}")
        return f"<p>Error generating comparison: {str(e)}</p>"

# --- Orthanc API Functions ---
async def fetch(session, url, retries=3):
    """Asynchronously fetch data from the Orthanc server with retries."""
    for attempt in range(retries):
        try:
            async with session.get(url, auth=aiohttp.BasicAuth(ORTHANC_USERNAME, ORTHANC_PASSWORD)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Error fetching {url}: HTTP {response.status} (attempt {attempt+1}/{retries})")
                    if attempt == retries - 1:
                        return None
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Exception while fetching {url}: {str(e)} (attempt {attempt+1}/{retries})")
            if attempt == retries - 1:
                return None
            await asyncio.sleep(1)

async def post_request(session, url, data, retries=3):
    """Make an async POST request to the Orthanc server."""
    for attempt in range(retries):
        try:
            async with session.post(
                url,
                json=data,
                auth=aiohttp.BasicAuth(ORTHANC_USERNAME, ORTHANC_PASSWORD)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Error posting to {url}: HTTP {response.status} (attempt {attempt+1}/{retries})")
                    if attempt == retries - 1:
                        return None
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Exception while posting to {url}: {str(e)} (attempt {attempt+1}/{retries})")
            if attempt == retries - 1:
                return None
            await asyncio.sleep(1)

async def find_studies_by_patient_and_date(session, patient_id, date_obj):
    """Find studies directly using the Orthanc /tools/find API with both patient ID and date."""
    date_str = date_obj.strftime("%Y%m%d")
    query = {
        "Level": "Study",
        "Query": {
            "PatientID": patient_id,
            "StudyDate": date_str
        }
    }
    results = await post_request(session, f"{ORTHANC_URL}/tools/find", query)
    if not results:
        logger.warning(f"No studies found or API request failed for patient ID {patient_id} on date {date_str}")
        return []
    return results

async def list_series_with_protocol(session, study_id):
    """List all series in a study and filter for target protocol with side detection."""
    try:
        series_list = await fetch(session, f"{ORTHANC_URL}/studies/{study_id}/series")
        if not series_list:
            logger.warning(f"No series found for study {study_id}")
            return []

        target_series = []
        
        for series in series_list:
            series_id = series['ID'] if isinstance(series, dict) and 'ID' in series else series
            
            series_meta = await fetch(session, f"{ORTHANC_URL}/series/{series_id}")
            if not series_meta:
                continue
                
            tags = series_meta.get('MainDicomTags', {})
            protocol = tags.get('ProtocolName', '').lower()
            description = tags.get('SeriesDescription', '').lower()
            
            if TARGET_PROTOCOL in protocol or TARGET_PROTOCOL in description:
                instances = await fetch(session, f"{ORTHANC_URL}/series/{series_id}/instances")
                instance_laterality = ""
                instance_body_part = ""
                
                if instances and len(instances) > 0:
                    instance_id = instances[0]['ID'] if isinstance(instances[0], dict) else instances[0]
                    instance_meta = await fetch(session, f"{ORTHANC_URL}/instances/{instance_id}")
                    if instance_meta:
                        instance_tags = instance_meta.get('MainDicomTags', {})
                        instance_laterality = instance_tags.get('Laterality', '').upper()
                        instance_body_part = instance_tags.get('BodyPartExamined', '').lower()
                
                side = "Unknown"
                laterality = tags.get('Laterality', '').upper() or instance_laterality
                
                if laterality in ['R', 'RIGHT']:
                    side = "Right"
                elif laterality in ['L', 'LEFT']:
                    side = "Left"
                elif any(keyword in protocol for keyword in ['right', 'rt', 'r_knee', 'r knee', 'rknee']):
                    side = "Right"
                elif any(keyword in protocol for keyword in ['left', 'lt', 'l_knee', 'l knee', 'lknee']):
                    side = "Left"
                elif any(keyword in description for keyword in ['right', 'rt', 'r_knee', 'r knee', 'rknee']):
                    side = "Right"
                elif any(keyword in description for keyword in ['left', 'lt', 'l_knee', 'l knee', 'lknee']):
                    side = "Left"
                
                series_info = {
                    "series_id": series_id,
                    "side": side,
                    "protocol": protocol,
                    "description": description,
                    "laterality": laterality,
                    "body_part": tags.get('BodyPartExamined', '') or instance_body_part,
                    "instance_count": len(instances) if instances else 0,
                    "study_id": study_id
                }
                target_series.append(series_info)
        
        return target_series
        
    except Exception as e:
        logger.error(f"Error listing series with protocol detection: {str(e)}")
        return []

async def download_dicom_for_series(session, series_id, output_folder):
    """Download DICOM files for a specific series."""
    try:
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Downloading series {series_id} to {output_folder}")
        
        instances = await fetch(session, f"{ORTHANC_URL}/series/{series_id}/instances")
        if not instances:
            logger.warning(f"No instances found for series {series_id}")
            return 0, 0
            
        download_tasks = []
        for instance in instances:
            instance_id = instance['ID'] if isinstance(instance, dict) and 'ID' in instance else instance
            download_tasks.append(
                download_dicom_instances(session, instance_id, output_folder)
            )
        
        if not download_tasks:
            logger.warning(f"No valid download tasks created for series {series_id}")
            return 0, 0
            
        semaphore = asyncio.Semaphore(min(MAX_CONCURRENT_DOWNLOADS, 50))
        
        async def limited_download(task):
            async with semaphore:
                try:
                    return await task
                except Exception as e:
                    logger.error(f"Download failed: {str(e)}")
                    return False
        
        results = await asyncio.gather(*[limited_download(t) for t in download_tasks])
        success_count = sum(1 for r in results if r)
        total_count = len(download_tasks)
        
        logger.info(f"Download completed: {success_count}/{total_count} instances for series {series_id}")
        return success_count, total_count
        
    except Exception as e:
        logger.error(f"Error downloading series {series_id}: {str(e)}")
        return 0, 0

async def download_dicom_instances(session, instance_id, output_folder):
    """Download a single DICOM instance asynchronously."""
    try:
        dicom_filename = f"instance_{instance_id}.dcm"
        dicom_path = os.path.join(output_folder, dicom_filename)
        
        if os.path.exists(dicom_path):
            logger.info(f"File exists, skipping download: {dicom_path}")
            return True
            
        async with session.get(
            f"{ORTHANC_URL}/instances/{instance_id}/file",
            auth=aiohttp.BasicAuth(ORTHANC_USERNAME, ORTHANC_PASSWORD)
        ) as response:
            if response.status == 200:
                with open(dicom_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)
                logger.info(f"Successfully downloaded: {dicom_path}")
                return True
            else:
                logger.error(f"Download failed for instance {instance_id}: HTTP {response.status}")
                return False
                
    except Exception as e:
        logger.error(f"Error downloading instance {instance_id}: {str(e)}")
        return False

def verify_dicom_files(folder):
    """Verify that DICOM files in the folder are valid and readable."""
    try:
        dicom_files = glob.glob(os.path.join(folder, "**/*.dcm"), recursive=True)
        if not dicom_files:
            logger.warning(f"No DICOM files found in {folder}")
            return False
            
        sample_files = dicom_files[:min(5, len(dicom_files))]
        valid_count = 0
        
        for file_path in sample_files:
            try:
                ds = pydicom.dcmread(file_path)
                if hasattr(ds, 'SOPClassUID'):
                    valid_count += 1
                else:
                    logger.warning(f"DICOM file missing SOPClassUID: {file_path}")
            except Exception as e:
                logger.warning(f"Invalid DICOM file {file_path}: {str(e)}")
                
        return valid_count >= len(sample_files) * 0.5
        
    except Exception as e:
        logger.error(f"Error verifying DICOM files in {folder}: {str(e)}")
        return False

def convert_to_nifti(dicom_folder, nifti_output_folder):
    """Convert DICOM to NIfTI using multi-threaded dcm2niix with enhanced settings."""
    os.makedirs(nifti_output_folder, exist_ok=True)
    
    try:
        if not verify_dicom_files(dicom_folder):
            logger.error(f"Folder {dicom_folder} contains invalid or no DICOM files")
            return False
            
        command = [
            'dcm2niix',
            '-p', 'y',
            '-z', 'y',
            '-f', '%p_%s_%t',
            '-v', 'y',
            '-b', 'y',
            '-ba', 'n',
            '-i', 'y',
            '-m', 'y',
            '-s', 'y',
            '-o', nifti_output_folder,
            dicom_folder
        ]
        
        logger.info(f"Running conversion command: {' '.join(command)}")
        
        process = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if process.stdout:
            logger.info(f"dcm2niix stdout:\n{process.stdout}")
        if process.stderr:
            logger.warning(f"dcm2niix stderr:\n{process.stderr}")
            
        if process.returncode != 0:
            logger.error(f"dcm2niix failed with exit code {process.returncode}")
            nifti_files = glob.glob(os.path.join(nifti_output_folder, "*.nii*"))
            if nifti_files:
                logger.info(f"Despite error, {len(nifti_files)} NIfTI files were created")
                return True
            return False
            
        nifti_files = glob.glob(os.path.join(nifti_output_folder, "*.nii*"))
        logger.info(f"Created {len(nifti_files)} NIfTI files in {nifti_output_folder}")
        return len(nifti_files) > 0
        
    except Exception as e:
        logger.error(f"Error during conversion process: {str(e)}")
        return False

def process_selected_protocol(input_path, output_path):
    """Process the selected protocol NIfTI file with reorientation and resampling."""
    try:
        logger.info(f"Processing selected protocol: {input_path} -> {output_path}")
        
        case_img = nib.load(input_path)
        case_data = case_img.get_fdata()
        case_affine = case_img.affine
        
        current_ornt = io_orientation(case_affine)
        target_ornt = axcodes2ornt(('R', 'A', 'S'))
        transform = ornt_transform(current_ornt, target_ornt)
        reoriented_data = apply_orientation(case_data, transform)
        
        reordered_data = np.transpose(reoriented_data, (1, 2, 0))
        
        target_shape = (256, 256, 144)
        if reordered_data.shape != target_shape:
            zoom_factors = np.array(target_shape) / np.array(reordered_data.shape)
            reordered_data = scipy.ndimage.zoom(reordered_data, zoom_factors, order=1)
        
        final_affine = np.eye(4)
        final_img = nib.Nifti1Image(reordered_data, final_affine)
        nib.save(final_img, output_path)
        
        logger.info(f"Processed image saved to: {output_path}")
        logger.info(f"Final shape: {reordered_data.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing protocol: {str(e)}")
        return False

def clear_folders():
    """Clear input and output folders"""
    for folder in [INPUT_FOLDER, OUTPUT_FOLDER]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                file_path = os.path.join(folder, f)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.warning(f"Could not remove {file_path}: {e}")

def calculate_volume(mask_path):
    """Calculate volume metrics from segmentation mask."""
    mask_img = nib.load(mask_path)
    mask_data = np.squeeze(mask_img.get_fdata())
    labels = np.unique(mask_data)
    labels = labels[labels > 0]
    
    voxel_volume_mm3 = 0.2059937082
    
    results = []
    for label_val in labels:
        binary_mask = (mask_data == label_val).astype(np.uint8)
        voxel_count = np.sum(binary_mask)
        volume_mm3 = voxel_count * voxel_volume_mm3
        volume_cm3 = volume_mm3 / 1000
        
        logger.info(f"Label {label_val} voxel count: {voxel_count}")
        
        results.append({
            "Label": int(label_val),
            "Voxel Count": int(voxel_count),
            "Voxel Volume (mm³)": float(voxel_volume_mm3),
            "Volume (mm³)": float(volume_mm3),
            "Volume (cm³)": float(volume_cm3)
        })
    
    return results

def visualize_segmentation(mask_path, original_path=None):
    """Generate visualization of segmentation results with interactive scrolling."""
    try:
        mask_img = nib.load(mask_path)
        mask_data = np.squeeze(mask_img.get_fdata())
        
        original_data = None
        if original_path and os.path.exists(original_path):
            original_img = nib.load(original_path)
            original_data = np.squeeze(original_img.get_fdata())
        
        colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        cmap = ListedColormap(colors[:len(np.unique(mask_data))])
        
        num_slices = mask_data.shape[2]
        slice_images = []
        
        for slice_idx in range(num_slices):
            fig, ax = plt.subplots(figsize=(8, 8))
            
            if original_data is not None:
                ax.imshow(original_data[:, :, slice_idx], cmap='gray')
                ax.imshow(mask_data[:, :, slice_idx], cmap=cmap, alpha=0.5)
            else:
                ax.imshow(mask_data[:, :, slice_idx], cmap=cmap)
            
            ax.set_title(f'Slice {slice_idx + 1}/{num_slices}')
            ax.axis('off')
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            slice_images.append(f"data:image/png;base64,{image_base64}")
        
        return slice_images
        
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return None

def find_specific_protocol_file(nifti_folder, protocol_pattern=TARGET_PROTOCOL):
    """Find the specific protocol file in the NIfTI folder."""
    nifti_files = glob.glob(os.path.join(nifti_folder, "**/*.nii*"), recursive=True)
    
    for file_path in nifti_files:
        if protocol_pattern.lower() in os.path.basename(file_path).lower():
            return file_path
    
    json_files = glob.glob(os.path.join(nifti_folder, "**/*.json"), recursive=True)
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            fields_to_check = ['ProtocolName', 'SequenceName', 'SeriesDescription', 'PulseSequenceName']
            for field in fields_to_check:
                if field in metadata:
                    field_value = str(metadata[field]).lower()
                    if protocol_pattern.lower() in field_value:
                        nifti_file = json_file.replace('.json', '.nii.gz')
                        if not os.path.exists(nifti_file):
                            nifti_file = json_file.replace('.json', '.nii')
                        if os.path.exists(nifti_file):
                            logger.info(f"Found matching file by metadata: {nifti_file}")
                            return nifti_file
        except Exception as e:
            logger.warning(f"Error reading JSON file {json_file}: {e}")
    
    if nifti_files:
        largest_file = max(nifti_files, key=lambda x: os.path.getsize(x))
        logger.warning(f"No specific protocol found, using largest file: {largest_file}")
        return largest_file
    
    logger.error("No NIfTI files found at all")
    return None

def parse_date(date_str):
    """Parse date from various string formats."""
    try:
        return datetime.strptime(date_str, "%d-%m-%Y").date()
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            try:
                return datetime.strptime(date_str, "%Y%m%d").date()
            except ValueError:
                raise ValueError(f"Could not parse date '{date_str}'. Please use format YYYY-MM-DD, DD-MM-YYYY, or YYYYMMDD")

def check_dcm2niix_availability():
    """Check if dcm2niix is available in the system path."""
    try:
        result = subprocess.run(
            ["dcm2niix", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"dcm2niix version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.error("Error: dcm2niix not found in system path. Please install dcm2niix.")
        return False
    except Exception as e:
        logger.error(f"Error checking dcm2niix: {str(e)}")
        return False

def check_nnunet_availability():
    """Check if nnUNet is available in the system path."""
    try:
        result = subprocess.run(
            ["nnUNetv2_predict", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("nnUNetv2 is available")
        return True
    except FileNotFoundError:
        logger.error("Error: nnUNetv2 not found in system path. Please install nnUNet.")
        return False
    except Exception as e:
        logger.error(f"Error checking nnUNet: {str(e)}")
        return False

def generate_3d_slicer_views(mask_path, original_path=None):
    """Generate 3D Slicer-like multi-view visualization with axial, coronal, and sagittal views."""
    try:
        mask_img = nib.load(mask_path)
        mask_data = np.squeeze(mask_img.get_fdata())
        
        original_data = None
        if original_path and os.path.exists(original_path):
            original_img = nib.load(original_path)
            original_data = np.squeeze(original_img.get_fdata())
        
        unique_labels = np.unique(mask_data)
        num_labels = len(unique_labels)
        
        colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        cmap = ListedColormap(colors[:min(num_labels, len(colors))])
        
        views = {
            'axial': [],
            'coronal': [],
            'sagittal': []
        }
        
        def create_slice_image(data_slice, mask_slice, title):
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='#1a1a1a')
            ax.set_facecolor('#1a1a1a')
            
            if data_slice is not None:
                ax.imshow(data_slice, cmap='gray', aspect='auto')
                ax.imshow(mask_slice, cmap=cmap, alpha=0.6, aspect='auto')
            else:
                ax.imshow(mask_slice, cmap=cmap, aspect='auto')
            
            ax.set_title(title, color='white', fontsize=12)
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
        
        for slice_idx in range(min(50, mask_data.shape[2])):
            data_slice = original_data[:, :, slice_idx] if original_data is not None else None
            mask_slice = mask_data[:, :, slice_idx]
            image_data = create_slice_image(data_slice, mask_slice, f'Axial Slice {slice_idx + 1}')
            views['axial'].append(image_data)
        
        for slice_idx in range(min(50, mask_data.shape[1])):
            data_slice = original_data[:, slice_idx, :] if original_data is not None else None
            mask_slice = mask_data[:, slice_idx, :]
            image_data = create_slice_image(data_slice, mask_slice, f'Coronal Slice {slice_idx + 1}')
            views['coronal'].append(image_data)
        
        for slice_idx in range(min(50, mask_data.shape[0])):
            data_slice = original_data[slice_idx, :, :] if original_data is not None else None
            mask_slice = mask_data[slice_idx, :, :]
            image_data = create_slice_image(data_slice, mask_slice, f'Sagittal Slice {slice_idx + 1}')
            views['sagittal'].append(image_data)
        
        return views
        
    except Exception as e:
        logger.error(f"Error generating 3D slicer views: {str(e)}")
        return None

# --- Flask Routes ---

@app.route("/", methods=["GET", "POST"])
def index():
    """Main page for patient ID and date input with history search"""
    if request.method == "POST":
        patient_id = request.form.get("patient_id")
        date_str = request.form.get("date")
        
        if not patient_id:
            return "Patient ID is required"
        
        if date_str:
            return redirect(url_for("select_knee", patient_id=patient_id, date=date_str))
        else:
            return redirect(url_for("patient_history", patient_id=patient_id))
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Knee Segmentation Pipeline</title>
        <style>
            body { 
                font-family: 'Arial', sans-serif; 
                margin: 0; 
                padding: 0; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container { 
                max-width: 500px; 
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
                font-size: 2.2em;
            }
            .form-group { 
                margin-bottom: 25px; 
            }
            label { 
                display: block; 
                margin-bottom: 8px; 
                font-weight: bold;
                color: #555;
            }
            input[type="text"], input[type="date"] { 
                width: 100%; 
                padding: 15px; 
                border: 2px solid #ddd; 
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            input[type="text"]:focus, input[type="date"]:focus {
                outline: none;
                border-color: #667eea;
            }
            input[type="submit"] { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                width: 100%;
                transition: transform 0.2s;
            }
            input[type="submit"]:hover { 
                transform: translateY(-2px);
            }
            .icon {
                font-size: 3em;
                text-align: center;
                margin-bottom: 20px;
            }
            .protocol-info {
                background: #e3f2fd;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border-left: 4px solid #2196f3;
            }
            .history-search {
                background: #e8f5e8;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #4caf50;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon" style="font-size: 64px;">🦵</div>
            <h1>Knee Segmentation Pipeline</h1>
            
            <div class="protocol-info">
                <h3>📋 Auto-Selected Protocol</h3>
                <p><strong>t2_de3d_we_sag_iso</strong></p>
                <p>Only this protocol will be processed automatically.</p>
            </div>
            
            <form method="post">
                <div class="form-group">
                    <label for="patient_id">Patient ID:</label>
                    <input type="text" id="patient_id" name="patient_id" required placeholder="Enter patient ID">
                </div>
                <div class="form-group">
                    <label for="date">Study Date (Optional):</label>
                    <input type="date" id="date" name="date" placeholder="Leave empty for patient history">
                </div>
                
                <div class="history-search">
                    <p><strong>💡 Tip:</strong> Leave date empty to view patient history and previous studies.</p>
                </div>
                
                <input type="submit" value="🔍 Search Studies">
            </form>
        </div>
    </body>
    </html>
    '''

@app.route("/select_knee")
def select_knee():
    """Display only t2_de3d_we_sag_iso protocol series for the selected date."""
    patient_id = request.args.get('patient_id')
    date_str = request.args.get('date')
    
    if not patient_id or not date_str:
        return redirect(url_for('index'))
    
    async def get_protocol_series():
        async with aiohttp.ClientSession() as session:
            try:
                date_obj = parse_date(date_str)
                studies = await find_studies_by_patient_and_date(session, patient_id, date_obj)
                
                if not studies:
                    return []
                
                all_target_series = []
                
                for study in studies:
                    study_id = study if isinstance(study, str) else study.get('ID', '')
                    target_series = await list_series_with_protocol(session, study_id)
                    all_target_series.extend(target_series)
                
                return all_target_series
                
            except ValueError as e:
                logger.error(f"Date parsing error: {e}")
                return []
    
    target_series = asyncio.run(get_protocol_series())
    
    if not target_series:
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>No Protocol Found</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    text-align: center;
                }}
                h2 {{ color: #333; }}
                p {{ color: #666; }}
                a {{ color: #667eea; text-decoration: none; font-weight: bold; }}
                .protocol-highlight {{
                    background: #fff3cd;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                    border-left: 4px solid #ffc107;
                }}
                .info-box {{
                    background: #e3f2fd;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                    border-left: 4px solid #2196f3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>❌ No Target Protocol Found</h2>
                <div class="protocol-highlight">
                    <strong>Target Protocol:</strong> {TARGET_PROTOCOL}
                </div>
                <div class="info-box">
                    <p><strong>Patient ID:</strong> {patient_id}</p>
                    <p><strong>Date:</strong> {date_str}</p>
                </div>
                <p>No studies with the target protocol were found for the specified patient ID and date.</p>
                <p>Please check:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>Patient ID is correct</li>
                    <li>Date is correct</li>
                    <li>Study contains {TARGET_PROTOCOL} protocol</li>
                </ul>
                <a href="/">← Back to search</a>
            </div>
        </body>
        </html>
        '''
    
    left_series = [s for s in target_series if s['side'] == 'Left']
    right_series = [s for s in target_series if s['side'] == 'Right']
    unknown_series = [s for s in target_series if s['side'] == 'Unknown']
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Select Knee Side</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
            }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ 
                text-align: center; 
                margin-bottom: 30px; 
                background: white; 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1); 
            }}
            .header h1 {{ color: #333; margin: 0; }}
            .header p {{ color: #666; margin: 10px 0; }}
            .protocol-banner {{
                background: linear-gradient(135deg, #28a745, #20c997);
                color: white;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 20px;
                font-weight: bold;
            }}
            .info-box {{
                background: #e3f2fd;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
                border-left: 4px solid #2196f3;
            }}
            .knee-section {{ margin-bottom: 30px; }}
            .knee-title {{ 
                font-size: 1.5em; 
                margin-bottom: 15px; 
                padding: 15px; 
                border-radius: 10px; 
                color: white; 
                text-align: center;
                font-weight: bold;
            }}
            .left-title {{ background: linear-gradient(135deg, #28a745, #20c997); }}
            .right-title {{ background: linear-gradient(135deg, #dc3545, #fd7e14); }}
            .unknown-title {{ background: linear-gradient(135deg, #6c757d, #adb5bd); }}
            .series-grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
                gap: 20px; 
            }}
            .series-card {{ 
                background: white; 
                border-radius: 12px; 
                padding: 25px; 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1); 
                transition: all 0.3s ease;
                border-left: 4px solid #007bff;
            }}
            .series-card:hover {{ 
                transform: translateY(-5px); 
                box-shadow: 0 10px 25px rgba(0,0,0,0.15); 
            }}
            .left-card {{ border-left-color: #28a745; }}
            .right-card {{ border-left-color: #dc3545; }}
            .series-header {{
                display: flex;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 2px solid #f0f0f0;
            }}
            .side-icon {{ 
                font-size: 1.5em; 
                margin-right: 10px; 
            }}
            .side-text {{
                font-size: 1.2em;
                font-weight: bold;
                color: #333;
            }}
            .series-info {{ margin: 15px 0; }}
            .info-row {{ 
                display: flex; 
                margin-bottom: 8px; 
                align-items: center;
            }}
            .info-label {{ 
                font-weight: bold; 
                color: #555; 
                min-width: 120px;
                margin-right: 10px;
            }}
            .info-value {{ 
                color: #333; 
                flex: 1;
                word-break: break-word;
            }}
            .btn {{ 
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white; 
                padding: 12px 24px; 
                text-decoration: none; 
                border-radius: 8px; 
                display: block; 
                text-align: center; 
                font-weight: bold; 
                margin-top: 20px;
                transition: all 0.3s ease;
            }}
            .btn:hover {{ 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,123,255,0.4);
            }}
            .btn-left {{ 
                background: linear-gradient(135deg, #28a745, #20c997);
            }}
            .btn-left:hover {{ 
                box-shadow: 0 5px 15px rgba(40,167,69,0.4);
            }}
            .btn-right {{ 
                background: linear-gradient(135deg, #dc3545, #fd7e14);
            }}
            .btn-right:hover {{ 
                box-shadow: 0 5px 15px rgba(220,53,69,0.4);
            }}
            .empty-section {{ 
                text-align: center; 
                padding: 40px; 
                background: white; 
                border-radius: 12px; 
                color: #666; 
                font-style: italic;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .manual-buttons {{
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 10px; 
                margin-top: 20px;
            }}
            .manual-buttons .btn {{
                margin-top: 0;
            }}
            .back-link {{
                text-align: center; 
                margin-top: 40px;
            }}
            .back-link a {{
                color: #007bff; 
                font-size: 1.1em; 
                text-decoration: none;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🦴 Select Knee Side</h1>
                <div class="protocol-banner">
                    📋 Auto-Selected Protocol: <strong>{TARGET_PROTOCOL}</strong>
                </div>
                <div class="info-box">
                    <p><strong>Patient ID:</strong> {patient_id}</p>
                    <p><strong>Date:</strong> {date_str}</p>
                    <p><strong>Found:</strong> {len(target_series)} series with target protocol</p>
                </div>
            </div>
            
            <div class="knee-section">
                <div class="knee-title left-title">🦵 Left Knee ({len(left_series)} series)</div>
                {f'''
                <div class="series-grid">
                    {"".join([f'''
                    <div class="series-card left-card">
                        <div class="series-header">
                            <span class="side-icon">🦵</span>
                            <span class="side-text">Left Knee</span>
                        </div>
                        <div class="series-info">
                            <div class="info-row">
                                <span class="info-label">Series ID:</span>
                                <span class="info-value">{s["series_id"][:12]}...</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Protocol:</span>
                                <span class="info-value">{s["protocol"][:50]}{'...' if len(s["protocol"]) > 50 else ''}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Description:</span>
                                <span class="info-value">{s["description"][:50]}{'...' if len(s["description"]) > 50 else ''}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Instances:</span>
                                <span class="info-value">{s["instance_count"]} images</span>
                            </div>
                            {f'''<div class="info-row">
                                <span class="info-label">Laterality:</span>
                                <span class="info-value">{s["laterality"]}</span>
                            </div>''' if s["laterality"] else ''}
                        </div>
                        <a href="/process?patient_id={patient_id}&date={date_str}&series_id={s["series_id"]}&side=Left" 
                           class="btn btn-left">
                            ✅ Process Left Knee
                        </a>
                    </div>
                    ''' for s in left_series])}
                </div>
                ''' if left_series else '<div class="empty-section">❌ No left knee series detected</div>'}
            </div>
            
            <div class="knee-section">
                <div class="knee-title right-title">🦵 Right Knee ({len(right_series)} series)</div>
                {f'''
                <div class="series-grid">
                    {"".join([f'''
                    <div class="series-card right-card">
                        <div class="series-header">
                            <span class="side-icon">🦵</span>
                            <span class="side-text">Right Knee</span>
                        </div>
                        <div class="series-info">
                            <div class="info-row">
                                <span class="info-label">Series ID:</span>
                                <span class="info-value">{s["series_id"][:12]}...</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Protocol:</span>
                                <span class="info-value">{s["protocol"][:50]}{'...' if len(s["protocol"]) > 50 else ''}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Description:</span>
                                <span class="info-value">{s["description"][:50]}{'...' if len(s["description"]) > 50 else ''}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Instances:</span>
                                <span class="info-value">{s["instance_count"]} images</span>
                            </div>
                            {f'''<div class="info-row">
                                <span class="info-label">Laterality:</span>
                                <span class="info-value">{s["laterality"]}</span>
                            </div>''' if s["laterality"] else ''}
                        </div>
                        <a href="/process?patient_id={patient_id}&date={date_str}&series_id={s["series_id"]}&side=Right" 
                           class="btn btn-right">
                            ✅ Process Right Knee
                        </a>
                    </div>
                    ''' for s in right_series])}
                </div>
                ''' if right_series else '<div class="empty-section">❌ No right knee series detected</div>'}
            </div>
            
            {f'''
            <div class="knee-section">
                <div class="knee-title unknown-title">❓ Manual Selection ({len(unknown_series)} series)</div>
                <div class="series-grid">
                    {"".join([f'''
                    <div class="series-card">
                        <div class="series-header">
                            <span class="side-icon">❓</span>
                            <span class="side-text">Unknown Side</span>
                        </div>
                        <div class="series-info">
                            <div class="info-row">
                                <span class="info-label">Series ID:</span>
                                <span class="info-value">{s["series_id"][:12]}...</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Protocol:</span>
                                <span class="info-value">{s["protocol"][:50]}{'...' if len(s["protocol"]) > 50 else ''}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Description:</span>
                                <span class="info-value">{s["description"][:50]}{'...' if len(s["description"]) > 50 else ''}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Instances:</span>
                                <span class="info-value">{s["instance_count"]} images</span>
                            </div>
                            {f'''<div class="info-row">
                                <span class="info-label">Body Part:</span>
                                <span class="info-value">{s["body_part"]}</span>
                            </div>''' if s["body_part"] else ''}
                        </div>
                        <div class="manual-buttons">
                            <a href="/process?patient_id={patient_id}&date={date_str}&series_id={s["series_id"]}&side=Left" 
                               class="btn btn-left">
                                Left Knee
                            </a>
                            <a href="/process?patient_id={patient_id}&date={date_str}&series_id={s["series_id"]}&side=Right" 
                               class="btn btn-right">
                                Right Knee
                            </a>
                        </div>
                    </div>
                    ''' for s in unknown_series])}
                </div>
            </div>
            ''' if unknown_series else ''}
            
            <div class="back-link">
                <a href="/">← Back to search</a>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route("/process")
def process():
    """Process the selected knee series."""
    patient_id = request.args.get('patient_id')
    date_str = request.args.get('date')
    series_id = request.args.get('series_id')
    side = request.args.get('side', 'Unknown')
    
    if not patient_id or not series_id:
        return redirect(url_for('index'))
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Processing {side} Knee</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .container {{
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                text-align: center;
                max-width: 600px;
            }}
            .spinner {{
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .steps {{
                text-align: left;
                margin: 20px 0;
            }}
            .step {{
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                background: #f8f9fa;
            }}
            .step.active {{
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
            }}
            .step.completed {{
                background: #e8f5e8;
                border-left: 4px solid #4caf50;
            }}
            h1 {{ color: #333; }}
            p {{ color: #666; }}
            .protocol-info {{
                background: #e3f2fd;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #2196f3;
            }}
            .patient-info {{
                background: #fff3cd;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #ffc107;
            }}
        </style>
        <script>
            let currentStep = 0;
            const steps = [
                'Initializing...',
                'Downloading DICOM files from Orthanc...',
                'Converting DICOM to NIfTI...',
                'Preprocessing images...',
                'Running nnUNet segmentation...',
                'Calculating volumes...',
                'Generating results...'
            ];
            
            function updateProgress() {{
                const stepElements = document.querySelectorAll('.step');
                stepElements.forEach((el, index) => {{
                    el.classList.remove('active', 'completed');
                    if (index < currentStep) {{
                        el.classList.add('completed');
                    }} else if (index === currentStep) {{
                        el.classList.add('active');
                    }}
                }});
                
                if (currentStep < steps.length - 1) {{
                    currentStep++;
                    setTimeout(updateProgress, 3000);
                }} else {{
                    setTimeout(() => {{
                        window.location.href = '/process_complete?patient_id={patient_id}&date={date_str}&series_id={series_id}&side={side}';
                    }}, 2000);
                }}
            }}
            
            window.onload = function() {{
                setTimeout(updateProgress, 1000);
            }};
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🔄 Processing {side} Knee</h1>
            
            <div class="patient-info">
                <p><strong>Patient ID:</strong> {patient_id}</p>
                <p><strong>Date:</strong> {date_str}</p>
                <p><strong>Series ID:</strong> {series_id[:12]}...</p>
            </div>
            
            <div class="protocol-info">
                <strong>Auto-Selected Protocol:</strong> {TARGET_PROTOCOL}
            </div>
            
            <div class="spinner"></div>
            
            <div class="steps">
                {"".join([f'<div class="step">📋 {step}</div>' for step in [
                    'Initializing...',
                    'Downloading DICOM files from Orthanc...',
                    'Converting DICOM to NIfTI...',
                    'Preprocessing images...',
                    'Running nnUNet segmentation...',
                    'Calculating volumes...',
                    'Generating results...'
                ]])}
            </div>
            
            <p><em>Please wait while we process your knee segmentation...</em></p>
        </div>
    </body>
    </html>
    '''

@app.route("/process_complete")
def process_complete():
    """Actually process the knee series in the background."""
    patient_id = request.args.get('patient_id')
    date_str = request.args.get('date')
    series_id = request.args.get('series_id')
    side = request.args.get('side', 'Unknown')
    
    if not patient_id or not series_id:
        return redirect(url_for('index'))
    
    clear_folders()
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    async def process_series():
        async with aiohttp.ClientSession() as session:
            dicom_folder = os.path.join(DICOM_DIR, series_id)
            nifti_folder = os.path.join(NIFTI_OUTPUT_DIR, series_id)
            
            logger.info(f"Downloading {side} knee series {series_id}...")
            success_count, total_count = await download_dicom_for_series(session, series_id, dicom_folder)
            
            if success_count == 0:
                return None, f"Failed to download any instances for series {series_id}"
            
            logger.info(f"Successfully downloaded {success_count}/{total_count} instances")
            
            logger.info(f"Converting DICOMs to NIfTI for series {series_id}...")
            loop = asyncio.get_running_loop()
            success = await loop.run_in_executor(executor, convert_to_nifti, dicom_folder, nifti_folder)
            
            if not success:
                return None, f"Failed to convert series {series_id} to NIfTI"
            
            nifti_path = find_specific_protocol_file(nifti_folder)
            if not nifti_path:
                return None, f"No NIfTI files found in {nifti_folder}"
            
            return nifti_path, None
    
    try:
        nifti_path, error = asyncio.run(process_series())
        
        if error:
            return f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Processing Error</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }}
                    .container {{
                        background: white;
                        padding: 40px;
                        border-radius: 15px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                        text-align: center;
                        max-width: 600px;
                    }}
                    h2 {{ color: #dc3545; }}
                    p {{ color: #666; }}
                    a {{ color: #667eea; text-decoration: none; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>❌ Processing Error</h2>
                    <p>{error}</p>
                    <a href="/">← Back to search</a>
                </div>
            </body>
            </html>
            '''
        
        if not nifti_path or not os.path.exists(nifti_path):
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Processing Error</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .container {
                        background: white;
                        padding: 40px;
                        border-radius: 15px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                        text-align: center;
                        max-width: 600px;
                    }
                    h2 { color: #dc3545; }
                    p { color: #666; }
                    a { color: #667eea; text-decoration: none; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>❌ Processing Error</h2>
                    <p>Failed to process DICOM data from Orthanc</p>
                    <a href="/">← Back to search</a>
                </div>
            </body>
            </html>
            '''
        
        processed_path = os.path.join(INPUT_FOLDER, "case_000_0000.nii.gz")
        if not process_selected_protocol(nifti_path, processed_path):
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Processing Error</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .container {
                        background: white;
                        padding: 40px;
                        border-radius: 15px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                        text-align: center;
                        max-width: 600px;
                    }
                    h2 { color: #dc3545; }
                    p { color: #666; }
                    a { color: #667eea; text-decoration: none; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>❌ Processing Error</h2>
                    <p>Failed to process selected protocol</p>
                    <a href="/">← Back to search</a>
                </div>
            </body>
            </html>
            '''
        
        if not os.path.exists(processed_path):
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Processing Error</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .container {
                        background: white;
                        padding: 40px;
                        border-radius: 15px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                        text-align: center;
                        max-width: 600px;
                    }
                    h2 { color: #dc3545; }
                    p { color: #666; }
                    a { color: #667eea; text-decoration: none; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>❌ Processing Error</h2>
                    <p>Preprocessed input file was not created</p>
                    <a href="/">← Back to search</a>
                </div>
            </body>
            </html>
            '''
        
        file_size = os.path.getsize(processed_path)
        if file_size < 1000:
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Processing Error</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .container {
                        background: white;
                        padding: 40px;
                        border-radius: 15px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                        text-align: center;
                        max-width: 600px;
                    }
                    h2 { color: #dc3545; }
                    p { color: #666; }
                    a { color: #667eea; text-decoration: none; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>❌ Processing Error</h2>
                    <p>Preprocessed file appears to be empty or corrupted</p>
                    <a href="/">← Back to search</a>
                </div>
            </body>
            </html>
            '''
        
        logger.info(f"Input file ready for segmentation: {processed_path} ({file_size} bytes)")
        
        try:
            cmd = [
                "nnUNetv2_predict",
                "-i", INPUT_FOLDER,
                "-o", OUTPUT_FOLDER,
                "-d", "012",
                "-c", "3d_fullres",
                "-f", "3",
                "--disable_tta",
                "--verbose"
            ]
            
            logger.info(f"Running nnUNet command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            logger.info(f"nnUNet output:\n{result.stdout}")
            
            pred_path = os.path.join(OUTPUT_FOLDER, "case_000.nii.gz")
            if not os.path.exists(pred_path):
                return '''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Segmentation Error</title>
                    <style>
                        body { 
                            font-family: Arial, sans-serif; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            min-height: 100vh;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        }
                        .container {
                            background: white;
                            padding: 40px;
                            border-radius: 15px;
                            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                            text-align: center;
                            max-width: 600px;
                        }
                        h2 { color: #dc3545; }
                        p { color: #666; }
                        a { color: #667eea; text-decoration: none; font-weight: bold; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h2>❌ Segmentation Error</h2>
                        <p>Segmentation completed but output file not found</p>
                        <a href="/">← Back to search</a>
                    </div>
                </body>
                </html>
                '''
            
            try:
                seg_img = nib.load(pred_path)
                seg_data = seg_img.get_fdata()
                unique_labels = np.unique(seg_data)
                logger.info(f"Segmentation labels found: {unique_labels}")
                
                if len(unique_labels) <= 1:
                    return '''
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Segmentation Warning</title>
                        <style>
                            body { 
                                font-family: Arial, sans-serif; 
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                min-height: 100vh;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                            }
                            .container {
                                background: white;
                                padding: 40px;
                                border-radius: 15px;
                                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                                text-align: center;
                                max-width: 600px;
                            }
                            h2 { color: #ff9800; }
                            p { color: #666; }
                            a { color: #667eea; text-decoration: none; font-weight: bold; }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h2>⚠️ Segmentation Warning</h2>
                            <p>Segmentation completed but no anatomical structures were detected</p>
                            <a href="/">← Back to search</a>
                        </div>
                    </body>
                    </html>
                    '''
            except Exception as e:
                logger.error(f"Error validating segmentation output: {e}")
                return '''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Segmentation Error</title>
                    <style>
                        body { 
                            font-family: Arial, sans-serif; 
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            min-height: 100vh;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        }
                        .container {
                            background: white;
                            padding: 40px;
                            border-radius: 15px;
                            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                            text-align: center;
                            max-width: 600px;
                        }
                        h2 { color: #dc3545; }
                        p { color: #666; }
                        a { color: #667eea; text-decoration: none; font-weight: bold; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h2>❌ Segmentation Error</h2>
                        <p>Segmentation output file is corrupted</p>
                        <a href="/">← Back to search</a>
                    </div>
                </body>
                </html>
                '''
            
            return redirect(url_for("success", side=side, patient_id=patient_id, date=date_str, series_id=series_id))
            
        except subprocess.CalledProcessError as e:
            logger.error(f"nnUNet segmentation failed: {e}")
            return f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Segmentation Error</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }}
                    .container {{
                        background: white;
                        padding: 40px;
                        border-radius: 15px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                        text-align: center;
                        max-width: 600px;
                    }}
                    h2 {{ color: #dc3545; }}
                    p {{ color: #666; }}
                    a {{ color: #667eea; text-decoration: none; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>❌ Segmentation Error</h2>
                    <p>Segmentation failed: {e.stdout if e.stdout else str(e)}</p>
                    <a href="/">← Back to search</a>
                </div>
            </body>
            </html>
            '''
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Processing Error</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }}
                    .container {{
                        background: white;
                        padding: 40px;
                        border-radius: 15px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                        text-align: center;
                        max-width: 600px;
                    }}
                    h2 {{ color: #dc3545; }}
                    p {{ color: #666; }}
                    a {{ color: #667eea; text-decoration: none; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>❌ Processing Error</h2>
                    <p>Processing error: {str(e)}</p>
                    <a href="/">← Back to search</a>
                </div>
            </body>
            </html>
            '''
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Unexpected Error</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .container {{
                    background: white;
                    padding: 40px;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    text-align: center;
                    max-width: 600px;
                }}
                h2 {{ color: #dc3545; }}
                p {{ color: #666; }}
                a {{ color: #667eea; text-decoration: none; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>❌ Unexpected Error</h2>
                <p>An unexpected error occurred: {str(e)}</p>
                <a href="/">← Back to search</a>
            </div>
        </body>
        </html>
        '''

@app.route("/success")
def success():
    """Display segmentation results and save to database"""
    side = request.args.get('side', 'Unknown')
    patient_id = request.args.get('patient_id', 'Unknown')
    date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    series_id = request.args.get('series_id', 'Unknown')
    
    pred_path = os.path.join(OUTPUT_FOLDER, "case_000.nii.gz")
    input_path = os.path.join(INPUT_FOLDER, "case_000_0000.nii.gz")
    
    if not os.path.exists(pred_path):
        return "Segmentation file not found!"
    
    try:
        metrics = calculate_volume(pred_path)
        
        label_names = {
            1: "Femur",
            2: "Tibia", 
            3: "Fibula",
            4: "Patella",
            5: "Cartilage"
        }
        
        study_id = save_study_data(patient_id, date_str, side, series_id, TARGET_PROTOCOL)
        save_volume_measurements(study_id, metrics, label_names)
        
        visualization_data = generate_3d_slicer_views(pred_path, input_path)
        if not visualization_data:
            return "Error generating 3D visualization"
        
        axial_images_js = json.dumps(visualization_data['axial'])
        coronal_images_js = json.dumps(visualization_data['coronal'])
        sagittal_images_js = json.dumps(visualization_data['sagittal'])
        
        previous_studies = get_patient_studies(patient_id)
        
        comparison_html = ""
        if len(previous_studies) > 1:
            comparison_html = generate_comparison_html(patient_id, previous_studies, date_str)
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>3D Segmentation Results</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    min-height: 100vh;
                }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                .results-header {{ 
                    text-align: center; 
                    margin-bottom: 30px; 
                    background: white;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .results-header h1 {{ 
                    color: #333; 
                    margin: 0; 
                    font-size: 2.5em;
                }}
                
                .unified-card {{
                    background: white;
                    padding: 40px; 
                    border-radius: 15px; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                
                .section {{
                    margin-bottom: 40px;
                }}
                
                .section:last-child {{
                    margin-bottom: 0;
                }}
                
                .section h3 {{ 
                    color: #333; 
                    margin-top: 0; 
                    font-size: 1.8em;
                    border-bottom: 3px solid #007bff;
                    padding-bottom: 12px;
                    margin-bottom: 25px;
                }}
                
                .stats-summary {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 25px;
                }}
                
                .stat-item {{
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    border-left: 5px solid #007bff;
                }}
                
                .stat-number {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #007bff;
                    margin-bottom: 5px;
                }}
                
                .stat-label {{
                    font-size: 0.9em;
                    color: #666;
                    font-weight: 600;
                }}
                
                .volume-table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin-top: 15px; 
                }}
                .volume-table th, .volume-table td {{ 
                    border: 1px solid #ddd; 
                    padding: 15px; 
                    text-align: left; 
                }}
                .volume-table th {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    font-weight: bold;
                }}
                .volume-table tr:nth-child(even) {{ 
                    background-color: #f8f9fa; 
                }}
                .volume-table tr:hover {{
                    background-color: #e3f2fd;
                }}
                
                .comparison-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .comparison-table th, .comparison-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: center;
                }}
                .comparison-table th {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }}
                .improvement-positive {{
                    color: #28a745;
                    font-weight: bold;
                }}
                .improvement-negative {{
                    color: #dc3545;
                    font-weight: bold;
                }}
                .date-header {{
                    background: #17a2b8 !important;
                    color: white;
                }}
                
                .slicer-container {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    grid-template-rows: auto auto;
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                
                .view-panel {{
                    background: #1a1a1a;
                    border-radius: 10px;
                    padding: 15px;
                    position: relative;
                }}
                
                .view-panel.large {{
                    grid-column: span 2;
                }}
                
                .view-title {{
                    color: white;
                    font-weight: bold;
                    margin-bottom: 10px;
                    text-align: center;
                    font-size: 1.2em;
                }}
                
                .view-content {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                
                .slice-image {{
                    max-width: 100%;
                    height: auto;
                    border: 2px solid #444;
                    border-radius: 5px;
                    cursor: crosshair;
                }}
                
                .slice-controls {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin: 10px 0;
                    gap: 10px;
                }}
                
                .slice-controls button {{
                    background: #007bff;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                }}
                
                .slice-controls button:hover {{
                    background: #0056b3;
                }}
                
                .slice-controls button:disabled {{
                    background: #6c757d;
                    cursor: not-allowed;
                }}
                
                .slice-info {{
                    color: white;
                    font-weight: bold;
                    min-width: 120px;
                    text-align: center;
                }}
                
                .coordinate-display {{
                    background: #2a2a2a;
                    color: #00ff00;
                    padding: 10px;
                    border-radius: 5px;
                    font-family: monospace;
                    margin-top: 10px;
                    text-align: center;
                }}
                
                .section-divider {{
                    height: 2px;
                    background: linear-gradient(90deg, transparent, #ddd, transparent);
                    margin: 30px 0;
                }}
                
                .btn {{ 
                    background: linear-gradient(135deg, #007bff, #0056b3);
                    color: white; 
                    padding: 12px 24px; 
                    text-decoration: none; 
                    border-radius: 8px; 
                    display: inline-block; 
                    margin: 10px 5px;
                    font-weight: bold;
                    transition: all 0.3s ease;
                }}
                .btn:hover {{ 
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0,123,255,0.4);
                }}
                .btn-success {{ 
                    background: linear-gradient(135deg, #28a745, #20c997);
                }}
                .btn-info {{ 
                    background: linear-gradient(135deg, #17a2b8, #138496);
                }}
                .btn-warning {{ 
                    background: linear-gradient(135deg, #ffc107, #e0a800);
                }}
                .actions {{ 
                    text-align: center; 
                    margin-top: 30px; 
                    background: white;
                    padding: 20px;
                    border-radius: 15px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                
                .instructions {{
                    background: #e3f2fd;
                    padding: 15px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    border-left: 4px solid #2196f3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="results-header">
                    <h1>3D Segmentation Results - {side} Knee</h1>
                    <p>Patient ID: <strong>{patient_id}</strong> | Date: <strong>{date_str}</strong></p>
                </div>
                
                <div class="unified-card">
                    <div class="section">
                        <h3>Volume Measurements</h3>
                        <div class="stats-summary">
                            <div class="stat-item">
                                <div class="stat-number">{len(metrics)}</div>
                                <div class="stat-label">Structures</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-number">{sum(item["Volume (cm³)"] for item in metrics):.1f}</div>
                                <div class="stat-label">Total Volume (cm³)</div>
                            </div>
                        </div>
                        <table class="volume-table">
                            <thead>
                                <tr>
                                    <th>Structure</th>
                                    <th>Volume (cm³)</th>
                                    <th>Voxel Count</th>
                                </tr>
                            </thead>
                            <tbody>
                                {"".join([f'''
                                <tr>
                                    <td><strong>{label_names.get(item["Label"], f"Label {item['Label']}")}</strong></td>
                                    <td>{item["Volume (cm³)"]:.3f}</td>
                                    <td>{item["Voxel Count"]:,}</td>
                                </tr>
                                ''' for item in metrics])}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="section-divider"></div>
                    
                    <div class="section">
                        <h3>3D Multi-View Visualization</h3>
                        
                        <div class="instructions">
                            <h4>Navigation Instructions:</h4>
                            <ul>
                                <li><strong>Mouse Wheel:</strong> Scroll through slices in each view</li>
                                <li><strong>Arrow Keys:</strong> Navigate through slices</li>
                                <li><strong>Click + Drag:</strong> Pan the image (when zoomed)</li>
                                <li><strong>Sync Views:</strong> All views update together</li>
                            </ul>
                        </div>
                        
                        <div class="slicer-container">
                            <div class="view-panel">
                                <div class="view-title">🔄 Axial View (Superior-Inferior)</div>
                                <div class="view-content">
                                    <div class="slice-controls">
                                        <button id="axialPrev">◀</button>
                                        <div class="slice-info" id="axialInfo">Slice 1/{len(visualization_data['axial'])}</div>
                                        <button id="axialNext">▶</button>
                                    </div>
                                    <div class="slice-image-container" id="axialContainer">
                                        <img id="axialImage" class="slice-image" src="{visualization_data['axial'][0]}" alt="Axial View">
                                    </div>
                                    <div class="coordinate-display" id="axialCoords">Z: 0</div>
                                </div>
                            </div>
                            
                            <div class="view-panel">
                                <div class="view-title">📊 Coronal View (Anterior-Posterior)</div>
                                <div class="view-content">
                                    <div class="slice-controls">
                                        <button id="coronalPrev">◀</button>
                                        <div class="slice-info" id="coronalInfo">Slice 1/{len(visualization_data['coronal'])}</div>
                                        <button id="coronalNext">▶</button>
                                    </div>
                                    <div class="slice-image-container" id="coronalContainer">
                                        <img id="coronalImage" class="slice-image" src="{visualization_data['coronal'][0]}" alt="Coronal View">
                                    </div>
                                    <div class="coordinate-display" id="coronalCoords">Y: 0</div>
                                </div>
                            </div>
                            
                            <div class="view-panel large">
                                <div class="view-title">📐 Sagittal View (Left-Right)</div>
                                <div class="view-content">
                                    <div class="slice-controls">
                                        <button id="sagittalPrev">◀</button>
                                        <div class="slice-info" id="sagittalInfo">Slice 1/{len(visualization_data['sagittal'])}</div>
                                        <button id="sagittalNext">▶</button>
                                    </div>
                                    <div class="slice-image-container" id="sagittalContainer">
                                        <img id="sagittalImage" class="slice-image" src="{visualization_data['sagittal'][0]}" alt="Sagittal View">
                                    </div>
                                    <div class="coordinate-display" id="sagittalCoords">X: 0</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    {f'''
                    <div class="section-divider"></div>
                    
                    <div class="section">
                        <h3>📊 Historical Comparison</h3>
                        {comparison_html}
                    </div>
                    ''' if comparison_html else ''}
                </div>
                
                <div class="actions">
                    <a href="/" class="btn">🔄 Process Another Patient</a>
                    <a href="/patient_history?patient_id={patient_id}" class="btn btn-info">📋 Patient History</a>
                    <a href="/lvef_data?patient_id={patient_id}" class="btn btn-warning">📈 LVEF Data</a>
                    <a href="/export_report?patient_id={patient_id}" class="btn btn-success">📄 Export Report</a>
                </div>
            </div>

            <script>
                const axialImages = {axial_images_js};
                const coronalImages = {coronal_images_js};
                const sagittalImages = {sagittal_images_js};
                
                let currentAxial = 0;
                let currentCoronal = 0;
                let currentSagittal = 0;
                
                const axialImage = document.getElementById('axialImage');
                const coronalImage = document.getElementById('coronalImage');
                const sagittalImage = document.getElementById('sagittalImage');
                
                const axialInfo = document.getElementById('axialInfo');
                const coronalInfo = document.getElementById('coronalInfo');
                const sagittalInfo = document.getElementById('sagittalInfo');
                
                const axialCoords = document.getElementById('axialCoords');
                const coronalCoords = document.getElementById('coronalCoords');
                const sagittalCoords = document.getElementById('sagittalCoords');
                
                function updateAllViews() {{
                    axialImage.src = axialImages[currentAxial];
                    coronalImage.src = coronalImages[currentCoronal];
                    sagittalImage.src = sagittalImages[currentSagittal];
                    
                    axialInfo.textContent = `Slice ${{currentAxial + 1}}/${{axialImages.length}}`;
                    coronalInfo.textContent = `Slice ${{currentCoronal + 1}}/${{coronalImages.length}}`;
                    sagittalInfo.textContent = `Slice ${{currentSagittal + 1}}/${{sagittalImages.length}}`;
                    
                    axialCoords.textContent = `Z: ${{currentAxial}}`;
                    coronalCoords.textContent = `Y: ${{currentCoronal}}`;
                    sagittalCoords.textContent = `X: ${{currentSagittal}}`;
                    
                    updateButtonStates();
                }}
                
                function updateButtonStates() {{
                    document.getElementById('axialPrev').disabled = currentAxial === 0;
                    document.getElementById('axialNext').disabled = currentAxial === axialImages.length - 1;
                    document.getElementById('coronalPrev').disabled = currentCoronal === 0;
                    document.getElementById('coronalNext').disabled = currentCoronal === coronalImages.length - 1;
                    document.getElementById('sagittalPrev').disabled = currentSagittal === 0;
                    document.getElementById('sagittalNext').disabled = currentSagittal === sagittalImages.length - 1;
                }}
                
                function navigateAxial(direction) {{
                    const newSlice = currentAxial + direction;
                    if (newSlice >= 0 && newSlice < axialImages.length) {{
                        currentAxial = newSlice;
                        updateAllViews();
                    }}
                }}
                
                function navigateCoronal(direction) {{
                    const newSlice = currentCoronal + direction;
                    if (newSlice >= 0 && newSlice < coronalImages.length) {{
                        currentCoronal = newSlice;
                        updateAllViews();
                    }}
                }}
                
                function navigateSagittal(direction) {{
                    const newSlice = currentSagittal + direction;
                    if (newSlice >= 0 && newSlice < sagittalImages.length) {{
                        currentSagittal = newSlice;
                        updateAllViews();
                    }}
                }}
                
                document.getElementById('axialPrev').addEventListener('click', () => navigateAxial(-1));
                document.getElementById('axialNext').addEventListener('click', () => navigateAxial(1));
                document.getElementById('coronalPrev').addEventListener('click', () => navigateCoronal(-1));
                document.getElementById('coronalNext').addEventListener('click', () => navigateCoronal(1));
                document.getElementById('sagittalPrev').addEventListener('click', () => navigateSagittal(-1));
                document.getElementById('sagittalNext').addEventListener('click', () => navigateSagittal(1));
                
                document.getElementById('axialContainer').addEventListener('wheel', (e) => {{
                    e.preventDefault();
                    navigateAxial(e.deltaY > 0 ? 1 : -1);
                }});
                
                document.getElementById('coronalContainer').addEventListener('wheel', (e) => {{
                    e.preventDefault();
                    navigateCoronal(e.deltaY > 0 ? 1 : -1);
                }});
                
                document.getElementById('sagittalContainer').addEventListener('wheel', (e) => {{
                    e.preventDefault();
                    navigateSagittal(e.deltaY > 0 ? 1 : -1);
                }});
                
                document.addEventListener('keydown', (e) => {{
                    switch(e.key) {{
                        case 'ArrowUp':
                            e.preventDefault();
                            navigateAxial(-1);
                            break;
                        case 'ArrowDown':
                            e.preventDefault();
                            navigateAxial(1);
                            break;
                        case 'ArrowLeft':
                            e.preventDefault();
                            navigateSagittal(-1);
                            break;
                        case 'ArrowRight':
                            e.preventDefault();
                            navigateSagittal(1);
                            break;
                        case 'PageUp':
                            e.preventDefault();
                            navigateCoronal(-1);
                            break;
                        case 'PageDown':
                            e.preventDefault();
                            navigateCoronal(1);
                            break;
                    }}
                }});
                
                function setupTouchNavigation(container, navigateFunc) {{
                    let touchStart = 0;
                    
                    container.addEventListener('touchstart', (e) => {{
                        touchStart = e.touches[0].clientY;
                    }});
                    
                    container.addEventListener('touchmove', (e) => {{
                        e.preventDefault();
                        const touchY = e.touches[0].clientY;
                        const diff = touchStart - touchY;
                        
                        if (Math.abs(diff) > 50) {{
                            if (diff > 0) {{
                                navigateFunc(1);
                            }} else {{
                                navigateFunc(-1);
                            }}
                            touchStart = touchY;
                        }}
                    }});
                }}
                
                setupTouchNavigation(document.getElementById('axialContainer'), navigateAxial);
                setupTouchNavigation(document.getElementById('coronalContainer'), navigateCoronal);
                setupTouchNavigation(document.getElementById('sagittalContainer'), navigateSagittal);
                
                updateAllViews();
            </script>
        </body>
        </html>
        '''
        
    except Exception as e:
        logger.error(f"Error in success route: {e}")
        return f"Error: {str(e)}"
@app.route("/patient_history")
def patient_history():
    """Display patient history and generate reports"""
    patient_id = request.args.get('patient_id')
    
    if not patient_id:
        return redirect(url_for('index'))
    
    studies = get_patient_studies(patient_id)
    
    if not studies:
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Patient History</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 20px; 
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    min-height: 100vh;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .no-data {{ 
                    text-align: center; 
                    padding: 40px; 
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .btn {{
                    background: linear-gradient(135deg, #007bff, #0056b3);
                    color: white;
                    padding: 10px 20px;
                    text-decoration: none;
                    border-radius: 6px;
                    display: inline-block;
                    margin: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Patient History - {patient_id}</h1>
                <div class="no-data">
                    <h3>No studies found for this patient.</h3>
                    <a href="/" class="btn">Back to search</a>
                </div>
            </div>
        </body>
        </html>
        '''
    
    report_data = generate_patient_report(patient_id, studies)
    
    # Generate the volume trends HTML separately to avoid f-string issues
    volume_trends_html = ''
    if report_data and 'volume_trends' in report_data and report_data['volume_trends']:
        trends_rows = []
        for structure, trend in report_data["volume_trends"].items():
            trend_class = "positive" if trend['change_percentage'] > 0 else "negative"
            trend_sign = "+" if trend['change_percentage'] > 0 else ""
            row = f'''
                        <tr>
                            <td><strong>{structure}</strong></td>
                            <td>{"%.2f" % trend["first_volume"]}</td>
                            <td>{"%.2f" % trend["last_volume"]}</td>
                            <td class="{trend_class}">
                                {trend_sign}{trend["change_percentage"]:.1f}%
                            </td>
                            <td>{trend["trend"]}</td>
                        </tr>'''
            trends_rows.append(row)
        
        volume_trends_html = f'''
            <div class="card">
                <h2>📈 Volume Trends</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Structure</th>
                            <th>First Volume (cm³)</th>
                            <th>Last Volume (cm³)</th>
                            <th>Change</th>
                            <th>Trend</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(trends_rows)}
                    </tbody>
                </table>
            </div>'''
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Patient History - {patient_id}</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
            }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .header {{ 
                background: white; 
                padding: 30px; 
                border-radius: 15px; 
                margin-bottom: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .card {{ 
                background: white; 
                padding: 25px; 
                border-radius: 12px; 
                margin-bottom: 20px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            table {{ 
                width: 100%; 
                border-collapse: collapse; 
                margin: 15px 0; 
            }}
            th, td {{ 
                border: 1px solid #ddd; 
                padding: 12px; 
                text-align: left; 
            }}
            th {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .positive {{ color: #28a745; font-weight: bold; }}
            .negative {{ color: #dc3545; font-weight: bold; }}
            .btn {{ 
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white; 
                padding: 10px 20px; 
                text-decoration: none; 
                border-radius: 6px; 
                margin: 5px;
                display: inline-block;
            }}
            .btn-info {{ background: linear-gradient(135deg, #17a2b8, #138496); }}
            .btn-warning {{ background: linear-gradient(135deg, #ffc107, #e0a800); }}
            .btn-success {{ background: linear-gradient(135deg, #28a745, #20c997); }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📋 Patient History - {patient_id}</h1>
                <p>Total Studies: {report_data.get('total_studies', 'N/A')}</p>

            </div>

            <div class="card">
                <h2>📊 Studies Overview</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Side</th>
                            <th>Protocol</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([f'''
                        <tr>
                            <td>{study[1]}</td>
                            <td>{study[2]}</td>
                            <td>{study[3]}</td>
                            <td>
                                <a href="/success?patient_id={patient_id}&side={study[2]}&date={study[1]}" class="btn">
                                    View Details
                                </a>
                            </td>
                        </tr>
                        ''' for study in studies])}
                    </tbody>
                </table>
            </div>

            {volume_trends_html}

            <div class="card">
                <div style="text-align: center;">
                    <a href="/" class="btn">🔍 New Search</a>
                    <a href="/lvef_data?patient_id={patient_id}" class="btn btn-warning">📈 Manage LVEF Data</a>
                    <a href="/export_report?patient_id={patient_id}" class="btn btn-success">📄 Export Report</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''


@app.route("/lvef_data", methods=['GET', 'POST'])
def manage_lvef_data():
    """Manage LVEF measurements"""
    patient_id = request.args.get('patient_id')
    
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        measurement_date = request.form.get('measurement_date')
        lvef_value = float(request.form.get('lvef_value'))
        notes = request.form.get('notes', '')
        
        save_lvef_measurement(patient_id, measurement_date, lvef_value, notes)
        return redirect(url_for('manage_lvef_data', patient_id=patient_id))
    
    lvef_data = get_lvef_measurements(patient_id) if patient_id else []
    
    # Generate LVEF table HTML separately
    lvef_table_html = ''
    if lvef_data:
        lvef_rows = []
        for measurement in lvef_data:
            if measurement[2] is not None and measurement[2] > 0:
                trend_class = "positive"
            else:
                trend_class = "negative"
            
            improvement_text = f"{measurement[2]:+.1f}%" if measurement[2] is not None else "-"
            
            row = f'''
                        <tr>
                            <td>{measurement[0]}</td>
                            <td>{measurement[1]:.1f}%</td>
                            <td class="{trend_class}">
                                {improvement_text}
                            </td>
                            <td>{measurement[3] or "-"}</td>
                        </tr>'''
            lvef_rows.append(row)
        
        lvef_table_html = f'''
            <div class="card">
                <h3>LVEF Measurements History</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>LVEF (%)</th>
                            <th>Improvement</th>
                            <th>Notes</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(lvef_rows)}
                    </tbody>
                </table>
            </div>'''
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>LVEF Data - {patient_id}</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
            }}
            .container {{ max-width: 1000px; margin: 0 auto; }}
            .card {{ 
                background: white; 
                padding: 25px; 
                border-radius: 12px; 
                margin-bottom: 20px; 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1); 
            }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
            th {{ background: #17a2b8; color: white; }}
            .form-group {{ margin-bottom: 15px; }}
            input, button {{ padding: 10px; margin: 5px; }}
            .positive {{ color: #28a745; font-weight: bold; }}
            .negative {{ color: #dc3545; font-weight: bold; }}
            .btn {{
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 6px;
                display: inline-block;
                margin: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>📈 LVEF Data Management - {patient_id}</h1>
                
                <h3>Add New Measurement</h3>
                <form method="post">
                    <input type="hidden" name="patient_id" value="{patient_id}">
                    <div class="form-group">
                        <label>Date:</label>
                        <input type="date" name="measurement_date" required>
                    </div>
                    <div class="form-group">
                        <label>LVEF Value (%):</label>
                        <input type="number" name="lvef_value" step="0.1" min="0" max="100" required>
                    </div>
                    <div class="form-group">
                        <label>Notes:</label>
                        <input type="text" name="notes" placeholder="Optional notes">
                    </div>
                    <button type="submit" style="background: #28a745; color: white; padding: 10px 20px; border: none; border-radius: 6px;">Add Measurement</button>
                </form>
            </div>

            {lvef_table_html}

            <div class="card" style="text-align: center;">
                <a href="/patient_history?patient_id={patient_id}" class="btn">Back to Patient History</a>
                <a href="/" class="btn">New Search</a>
            </div>
        </div>
    </body>
    </html>
    '''


@app.route("/export_report")
def export_report():
    """Export patient report"""
    patient_id = request.args.get('patient_id')
    
    if not patient_id:
        return redirect(url_for('index'))
    
    studies = get_patient_studies(patient_id)
    report_data = generate_patient_report(patient_id, studies)
    lvef_data = get_lvef_measurements(patient_id)
    
    # Fix the conditional part by using a variable
    volume_trends_item = '<li>Volume trends for multiple structures</li>' if report_data.volume_trends else ''
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Export Report</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
            }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .card {{
                background: white;
                padding: 25px;
                border-radius: 12px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .btn {{
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 6px;
                display: inline-block;
                margin: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>Export Report - {patient_id}</h1>
                <p>Export functionality would be implemented here with libraries like ReportLab for PDF or pandas for Excel.</p>
                <p>Available data:</p>
                <ul>
                    <li>{len(studies)} studies</li>
                    <li>{len(lvef_data)} LVEF measurements</li>
                    {volume_trends_item}
                </ul>
                <div style="text-align: center;">
                    <a href="/patient_history?patient_id={patient_id}" class="btn">Back to History</a>
                    <a href="/" class="btn">New Search</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''


if __name__ == "__main__":
    if not check_dcm2niix_availability() or not check_nnunet_availability():
        print("Required tools not found. Please install dcm2niix and nnUNet.")
        exit(1)
    
    app.run(host="0.0.0.0", port=7050, debug=True)