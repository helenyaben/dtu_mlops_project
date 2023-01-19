from fastapi import FastAPI
from fastapi import UploadFile, File
from typing import Optional
from http import HTTPStatus
from src.data.make_dataset import load_data, transform_data
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import os
from typing import Dict, List, Tuple
from src.models.model import MyAwesomeModel
from google.cloud import storage

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str):
        
        # 3. Create class attributes
        self.classes = list(range(0, 10))
        self.class_to_idx = {str(idx):idx for idx in self.classes}

        # Import preprocessed data
        data = torch.load(os.path.join(targ_dir, 'data_processed.pt'))
        self.images = data['data']
        self.labels = data['labels']

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:

        return self.images.shape[0]
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.images[index]
        class_name  = str(int(self.labels[index]))
        class_idx  = self.labels[index]

        # Transform if necessary
        return img, class_idx # return data, label (X, y)

app = FastAPI()

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        
        image_paths = [file.filename]

        with open(file.filename, 'wb') as f:
            f.write(contents)

        # Define transform (mean=0 and std=1) to apply to images
        transform = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,))])

        # Import data
        images, labels = load_data(image_paths)

        # Transform data and create tensor
        images, labels = transform_data(images, transform), transform_data(labels, None)

        # Save output
        output = {'data': images,
                    'labels': labels}

        torch.save(output, 'data_processed.pt')

        # Remove image
        os.remove(file.filename)

    except Exception as e:
        return {"message": "An error has occurred while uploading the data."}

    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}

@app.get("/predict")
def predict_model():

    try:
        model = MyAwesomeModel()
        
        # fs = gcsfs.GCSFileSystem(project = 'dtumlops')
        # fs.ls('fingers_model')
        # with fs.open('fingers_model/my_trained_model.pt', 'rb') as file:
        #     state_dict = torch.load(file)

        BUCKET_NAME = 'fingers_model'
        MODEL_FILE = 'my_trained_model.pt'

        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blob = bucket.get_blob(MODEL_FILE)
        blob.download_to_filename(MODEL_FILE)
        
        state_dict = torch.load(MODEL_FILE)
        model.load_state_dict(state_dict)

        # Dataset
        dataset_ = ImageFolderCustom(targ_dir=os.getcwd())
        # DataLoader
        dataloader_ = DataLoader(dataset=dataset_, # use custom created train Dataset
                                            batch_size=1, # how many samples per batch?
                                            shuffle=True) # shuffle the data?

        model.eval()

        for images, labels in dataloader_:
            # Caculate log probabilities
            log_ps = model(images)
            # Calculate probabilities
            ps = torch.exp(log_ps)
            # Obtain top probabilities and corresponding classes
            top_p, top_class = ps.topk(1, dim=1)
            class_index = {'0R': 0, '1R': 1, '2R': 2, '3R': 3, '4R': 4, '5R': 5, '0L': 6, '1L':7, '2L':8, '3L':9, '4L':10, '5L':11}
            class_index_inv = {v: k for k, v in class_index.items()}
            message = f'Prediction: {class_index_inv[top_class.item()]}, True: {class_index_inv[labels[0].item()]}'

    except Exception as e:
        return {"message": "An error has occurred while predicting the label."}

    return {"message": message} 



