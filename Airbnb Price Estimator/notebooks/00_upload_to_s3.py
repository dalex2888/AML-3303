import boto3
from botocore.exceptions import ClientError

# S3 client (automatically uses credentials from ~/.aws/credentials)
s3 = boto3.client('s3')

# Configuration
bucket_name = 'software-tools-ai'
local_file_path = r'C:\Users\diego\OneDrive - Lambton College\AIML\Term 3\Software Tools and Emerging Technologies\GitExcercises\AML-3303\Airbnb Price Estimator\data\AB_NYC_2019.csv'
s3_key = 'raw_data/listings.csv'

# Step 1: Create bucket
try:
    s3.create_bucket(Bucket=bucket_name)
    print(f"✓ Bucket '{bucket_name}' created successfully")
except ClientError as e:
    if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
        print(f"Bucket '{bucket_name}' already exists (owned by you)")
        print(f"Error creating bucket: {e}")

# Step 2: Upload file
try:
    s3.upload_file(local_file_path, bucket_name, s3_key)
    print(f"File uploaded to s3://{bucket_name}/{s3_key}")
except FileNotFoundError:
    print(f"File not found: {local_file_path}")
except ClientError as e:
    print(f"Error uploading file: {e}")

# Step 3: Verify upload
try:
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix='raw_data/')
    if 'Contents' in response:
        print(f"\nFiles in bucket:")
        for obj in response['Contents']:
            print(f"  - {obj['Key']} ({obj['Size']} bytes)")
except ClientError as e:
    print(f"Error listing objects: {e}")