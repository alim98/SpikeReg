#!/bin/bash

cd SpikeReg

# Install requirements
pip install -r requirements.txt

# Download datasets
wget https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.v1.0.tar
wget https://cloud.imi.uni-luebeck.de/s/MSkzxTJTrtfZY5e/download/L2R_2021_Task3_validation.zip
wget https://cloud.imi.uni-luebeck.de/s/3QCzQ3PsrFxj8eA/download/pairs_val.csv

# Create directories and extract data
mkdir -p data
mkdir -p data/L2R_2021_Task3_train
tar xvf neurite-oasis.v1.0.tar -C data/L2R_2021_Task3_train 
mkdir -p data/L2R_2021_Task3_val
unzip L2R_2021_Task3_validation.zip -d data/L2R_2021_Task3_val
cp pairs_val.csv data/L2R_2021_Task3_val

# Remove validation subjects from training data to prevent data leakage
echo "Removing validation subjects (438-457) from training data to prevent data leakage..."
cd data/L2R_2021_Task3_train

# Remove overlapping subjects that appear in validation set
for subject_id in {438..457}; do
    subject_dir="OASIS_OAS1_$(printf "%04d" $subject_id)_MR1"
    if [ -d "$subject_dir" ]; then
        echo "Removing training subject: $subject_dir"
        rm -rf "$subject_dir"
    fi
done

cd ../..

# Clean up unnecessary files
rm neurite-oasis.v1.0.tar
rm L2R_2021_Task3_validation.zip
rm pairs_val.csv

echo "Data setup complete! Validation subjects have been excluded from training data."
echo "Training subjects: $(ls data/L2R_2021_Task3_train | wc -l) directories"
echo "Validation subjects: $(ls data/L2R_2021_Task3_val/img*.nii.gz | wc -l) files" 