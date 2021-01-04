import os
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt" # A workaround in case this happens: https://github.com/mapbox/rasterio/issues/1289
import argparse

import numpy as np
import pandas as pd

import rasterio

import utils

parser = argparse.ArgumentParser(description='DFC2021 baseline training script')
parser.add_argument('--output_dir', type=str, default="results/nlcd_only_baseline/output/", help='The path to save the output to.')
args = parser.parse_args()

def main():
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv("data/splits/val_inference_both.csv")
    fns = df["label_fn"].values

    for i, fn in enumerate(fns):
        print("(%d/%d) %s" % (i+1, len(fns), fn))
        output_fn = os.path.join(
            args.output_dir,
            fn.split("/")[-1].replace("nlcd", "predictions")
        )

        if "predictions-2016" in output_fn:
            output_fn = output_fn.replace("predictions-2016", "predictions-2017")

        with rasterio.open(fn) as f:
            data_nlcd_class = f.read(1)
            input_profile = f.profile.copy()

        output_profile = input_profile.copy()
        output_profile["driver"] = "GTiff"

        data_nlcd_idx = utils.NLCD_CLASS_TO_IDX_MAP[data_nlcd_class].astype(np.uint8)

        with rasterio.open(output_fn, "w", **output_profile) as f:
            f.write(data_nlcd_idx, 1)
            f.write_colormap(1, utils.NLCD_IDX_COLORMAP)

if __name__ == "__main__":
    main()