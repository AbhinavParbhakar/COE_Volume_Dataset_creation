from google.cloud import storage

client = storage.Client(project='skilled-box-451205-r3')
bucket = client.get_bucket(bucket_or_name='coe_data_test')

file = bucket.blob('file.tf')
file.upload_from_filename()