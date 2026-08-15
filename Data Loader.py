import pandas as pd
import numpy as np

import zipfile
import urllib.request
import os

# Create data directory
os.makedirs("cms_data", exist_ok=True)

print("1. Downloading CMS DE-SynPUF Sample 1 Files...")

# Official CMS Public Download URLs for Sample 1
bene_url = "https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/Downloads/DE1_0_2008_Beneficiary_Summary_File_Sample_1.zip"
carrier_url = "https://downloads.cms.gov/files/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.zip"

# Download Beneficiary File
urllib.request.urlretrieve(bene_url, "cms_data/bene_2008.zip")
with zipfile.ZipFile("cms_data/bene_2008.zip", 'r') as zip_ref:
    zip_ref.extractall("cms_data")

# Download Carrier Claims File
urllib.request.urlretrieve(carrier_url, "cms_data/carrier_claims.zip")
with zipfile.ZipFile("cms_data/carrier_claims.zip", 'r') as zip_ref:
    zip_ref.extractall("cms_data")

print("Files downloaded and unzipped successfully!")

# Find extracted CSV file paths
bene_csv = [f for f in os.listdir("cms_data") if "Beneficiary" in f and f.endswith(".csv")][0]
carrier_csv = [f for f in os.listdir("cms_data") if "Carrier" in f and f.endswith(".csv")][0]

print("2. Loading CSVs into DataFrames...")
# Load a subset of 10,000 rows for high-speed prototyping in Colab
bene_df = pd.read_csv(os.path.join("cms_data", bene_csv), nrows=5000)
carrier_df = pd.read_csv(os.path.join("cms_data", carrier_csv), nrows=10000)