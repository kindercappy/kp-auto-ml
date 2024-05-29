import shutil
import os

# Define the file paths relative to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

model_files = {
    'model_feature_selector': os.path.join(script_dir, 'auto_ml_kinder', 'model_feature_selector.py'),
    'model_list_helper': os.path.join(script_dir, 'auto_ml_kinder', 'model_list_helper.py'),
    'model_training_data_prep': os.path.join(script_dir, 'auto_ml_kinder', 'model_training_data_prep.py'),
    'model_training_helper': os.path.join(script_dir, 'auto_ml_kinder', 'model_training_helper.py'),
    'neural_network_regression': os.path.join(script_dir, 'auto_ml_kinder', 'neural_network_regression.py'),
    'pre_processing': os.path.join(script_dir, 'auto_ml_kinder', 'pre_processing.py'),
    '__init__': os.path.join(script_dir, 'auto_ml_kinder', '__init__.py'),
    'README': os.path.join(script_dir, 'README.md'),
    'LICENSE': os.path.join(script_dir, 'LICENSE'),
    'pyproject': os.path.join(script_dir, 'pyproject.toml')
}

packaging_src_auto_ml_kinder = os.path.join(script_dir, 'packaging', 'src', 'auto_ml_kinder')
packaging = os.path.join(script_dir, 'packaging')

# Ensure the destination directories exist
os.makedirs(packaging_src_auto_ml_kinder, exist_ok=True)

# Copy the files to the packaging directory
shutil.copyfile(model_files['model_feature_selector'], os.path.join(packaging_src_auto_ml_kinder, 'model_feature_selector.py'))
shutil.copyfile(model_files['model_list_helper'], os.path.join(packaging_src_auto_ml_kinder, 'model_list_helper.py'))
shutil.copyfile(model_files['model_training_data_prep'], os.path.join(packaging_src_auto_ml_kinder, 'model_training_data_prep.py'))
shutil.copyfile(model_files['model_training_helper'], os.path.join(packaging_src_auto_ml_kinder, 'model_training_helper.py'))
shutil.copyfile(model_files['neural_network_regression'], os.path.join(packaging_src_auto_ml_kinder, 'neural_network_regression.py'))
shutil.copyfile(model_files['pre_processing'], os.path.join(packaging_src_auto_ml_kinder, 'pre_processing.py'))
shutil.copyfile(model_files['__init__'], os.path.join(packaging_src_auto_ml_kinder, '__init__.py'))

shutil.copyfile(model_files['README'], os.path.join(packaging, 'README.md'))
shutil.copyfile(model_files['LICENSE'], os.path.join(packaging, 'LICENSE'))
shutil.copyfile(model_files['pyproject'], os.path.join(packaging, 'pyproject.toml'))
