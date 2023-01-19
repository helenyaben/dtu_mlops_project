BUCKET_NAME = 'fingers_model'

MODEL_FILE = 'my_trained_model.pt'

client = storage.Client()

bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
#blob.download_to_filename(MODEL_FILE)
blob.upload_from_filename(MODEL_FILE)