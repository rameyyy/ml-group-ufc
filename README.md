# UFC ML Group Project

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

Create `setup/.env` by copying what's in `setup/.env.example` and fill in:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME`
- `S3_PREFIX`

### 3. Download Data from S3

```bash
python download_s3.py
```

Downloads all files from the S3 bucket into `data/` maintaining the same folder structure.

## S3 Scripts

### Download All Files

```bash
python download_s3.py
```

Downloads all files from S3 bucket (with prefix) into `data/` directory. Maintains folder structure. Skips files that already exist locally.

### Upload New Files

```bash
python upload_s3.py
```

Scans `data/` directory and uploads any new files that don't already exist in S3. Maintains folder structure.

### Delete Files

```bash
python delete_s3.py
```

Deletes files or folders from S3. Edit the script at the bottom to specify what to delete:

```python
if __name__ == "__main__":
    path_to_delete = "sample_fight.json"  # Change this
    delete_path(path_to_delete)
```

Examples:
- `"sample_fight.json"` - deletes single file
- `"models/model1.pkl"` - deletes file in subfolder
- `"models"` - deletes entire folder

Includes confirmation prompt before deleting.

## Using S3 Functions in Code

### Upload

```python
from setup.s3 import upload_file

upload_file("data/model.pkl", "models/model.pkl")
```

### Download

```python
from setup.s3 import download_file

download_file("models/model.pkl", "data/model.pkl")
```

### List Files

```python
from setup.s3 import list_objects

all_files = list_objects()
models = list_objects(prefix="models/")
```

### Delete

```python
from setup.s3 import delete_file

delete_file("models/old_model.pkl")
```
