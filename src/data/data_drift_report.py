import numpy as np
import pandas as pd
from make_dataset import load_data
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
from pathlib import Path
from PIL import ImageStat, Image
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import click
import warnings

def load_images(path="data/raw"):
    train_paths = list(Path(os.path.join(path, 'train')).glob("*.png")) 
    test_paths = list(Path(os.path.join(path, 'test')).glob("*.png")) 
    images, labels = load_data(train_paths+test_paths)
    return images, labels

def get_stats(im):
    gy, gx = np.gradient(im)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)

    im_pil = Image.fromarray(im.astype(np.uint8))
    return ImageStat.Stat(im_pil).mean[0], ImageStat.Stat(im_pil).var[0], sharpness

def get_stats_from_images(images):
    stats = []
    for image in images:
        brightness, contrast, sharpness = get_stats(image)
        stats.append({"brightness":brightness, "contrast":contrast, "sharpness":sharpness})

    return pd.DataFrame(stats)

def get_drift_report_from_data(images_reference, images_current, output_path):
    print("getting stats for images")
    df_reference = get_stats_from_images(images_reference)
    df_current = get_stats_from_images(images_current)

    print("making report")
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=df_reference, current_data=df_current)
    report.save_html(Path(os.path.join(output_path, "data_drift_report.html")))

@click.command()
@click.argument('filepath_reference', type=click.Path(exists=True))
@click.argument('filepath_current', type=click.Path(exists=True))
def main(filepath_reference="data/raw/", filepath_current="data/raw/", output_path="reports/"):
    """creates a report in the reports folder comparing the reference image data to current image data.
    Stats used for comparison is brightness, contrast and sharpness"""

    warnings.simplefilter(action='ignore', category=FutureWarning)

    project_folder = Path(__file__).resolve().parents[2]

    print("loading images")
    images_reference, _ = load_images(Path(os.path.join(project_folder, filepath_reference)))
    images_current, _ = load_images(Path(os.path.join(project_folder, filepath_current)))
    output_path = Path(os.path.join(project_folder, output_path))

    get_drift_report_from_data(images_reference, images_current, output_path)

if __name__ == "__main__":
    main()
