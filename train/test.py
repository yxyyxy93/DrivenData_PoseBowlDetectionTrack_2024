import os
import shutil
from pathlib import Path
import zipfile


def setup_directories():
    """Ensure all required directories exist."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('submission', exist_ok=True)
    os.makedirs('models', exist_ok=True)


def prepare_files():
    """Place necessary files in the correct locations."""
    # Assuming 'main.py' and 'main.sh' are in the current directory or specify their paths
    shutil.copy('main.py', '.')
    shutil.copy('main.sh', '.')

    # If you have model files or additional scripts, copy them similarly
    # shutil.copy('path/to/model.pt', './models/')


def zip_submission():
    """Create a zip file for submission."""
    with zipfile.ZipFile('submission/submission.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write('main.py')
        zipf.write('main.sh')
        # Add other necessary files
        # zipf.write('models/model.pt')

        # Optionally include data files if required for the submission
        # for file in Path('data').rglob('*'):
        #     zipf.write(str(file))


if __name__ == '__main__':
    setup_directories()
    prepare_files()
    zip_submission()
    print("Submission zip file is ready.")
