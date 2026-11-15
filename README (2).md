# PoultryScan-UG

**Poultry disease triage from photographs of chicken droppings**
Soroti University | School of Engineering and Technology | Department of Electronics and Computer Engineering
Digital Image and Video Processing | Capstone Deliverable 3 (15 November 2026)

**Team:** Nalwadda Teddy Phionah (2301600095), Mirembe Agnes Costance (2301600104), Namakhanyu Williams (2301600082), Ategeka Alex (2301600077)

## 1\. What it does

The user uploads a photo of a chicken dropping. The application suggests one of four classes, **Healthy**, **Coccidiosis**, **Salmonellosis** or **Newcastle disease (NCD) suspected**, shows how sure it is, and gives simple advice. If it is not sure enough it says **"Not sure: refer to a vet"**, and if the photo is blank, tiny or very blurry it says **"Please retake the photo"**.

It runs **offline on a PC** (no internet after installation, no GPU needed). It is a **triage aid, not a diagnosis**.

## 2\. Results (971 leakage-free test photos)

|Metric|Baseline (D2 re-run)|Proposed (D3)|
|-|-|-|
|Overall accuracy|95.1%|**97.1%**|
|NCD recall|58.5%|**94.3%**|
|Sick birds caught (not called healthy)|95.6%|**98.8%**|
|Sick birds reported healthy|30|**8**|
|Healthy birds wrongly flagged|**3.4%**|4.4%|

The project target of 99% overall accuracy was **not** reached. A channel and spatial attention variant was tested and rejected on validation data. Full analysis is in `report/PoultryScan\_D3\_Report.pdf`.

## 3\. The improvement in one paragraph

The baseline is a plain MobileNetV2 transfer-learning model. The proposed system keeps the same backbone but adds: NCD oversampling (x3), stronger augmentation, two-phase fine-tuning (last 80 layers, learning-rate decay, dropout 0.4), photo + mirror-image averaging, a validation-tuned NCD weight of 1.5, and a "refer to a vet" rule (confidence below 0.6; below 0.85 for "Healthy"). The application also checks photo quality before predicting.

## 4\. Repository layout

|Folder / file|Contents|
|-|-|
|`app/`|The PC application: `app.py` (Gradio page), `poultry\_core.py` (preprocessing, model, decision rule), `config.json`, `model/poultry\_model.tflite`, `samples/` (12 example photos), `tests/test\_core.py`, `requirements.txt`, `run.bat`, `run.sh`, `README.txt`|
|`notebooks/`|`poultryscan\_d2\_d3.ipynb`: data checks, splits, baseline, proposed model, attention ablation, comparison, TFLite export|
|`baseline/`|Where the baseline implementation is in the notebook, and its settings|
|`proposed/`|Where the proposed implementation is in the notebook, and its settings|
|`report/`|D3 technical report (PDF) and LaTeX source|
|`demo/`|Demonstration video (in the submission ZIP)|
|`model\_files/`|Extra trained-model files for the submission ZIP (Keras model, split file)|

## 5\. Run the application on a PC

**Requirements**

* Windows, macOS or Linux, 64-bit. Demonstrated on an Intel Core i5-5300U laptop (2.30 GHz, 8 GB RAM, integrated graphics, no GPU).
* Python 3.11, 3.12 or 3.13 from python.org. On Windows, tick **"Add Python to PATH"** during installation.
* Internet only for the first start, to install the libraries (`ai-edge-litert`, `gradio`, `pillow`, `numpy`; no TensorFlow needed).

**Start (Windows)**

1. Download or clone this repository.
2. Open the `app` folder and double-click **`run.bat`**. The first start creates a virtual environment and installs the libraries (a few minutes).
3. The browser opens at **http://127.0.0.1:7860**.
4. Drag a photo into the box, or click one of the example photos, then press **Submit**.
5. To stop the application, close the black window.

**Start (macOS / Linux):** open a terminal in `app` and run `bash run.sh`.

**Start by hand (any system)**

```
cd app
python -m venv .venv
.venv\\Scripts\\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

**Updating an old copy:** delete the `.venv` folder once so the new libraries are installed.

## 6\. How to read the output

|Result shown|Meaning|What to do|
|-|-|-|
|Healthy (with confidence)|The dropping looks healthy|Keep watching the flock; call a vet if birds look unwell|
|Coccidiosis|Signs match coccidiosis|Separate sick birds, keep litter dry, ask a vet or agro-vet|
|Salmonellosis|Signs match salmonellosis|Separate sick birds, wash hands and tools, contact a vet|
|Newcastle disease (NCD) suspected|Signs match Newcastle disease|Isolate sick birds, limit visitors, contact the district vet at once|
|Not sure: refer to a vet|Confidence is below the safe line|Take a clearer photo or ask a vet; do not rely on the guess|
|Please retake the photo|Photo is blank, tiny or too blurry|Retake it close up, in good light, with the dropping in the middle|

The page also shows a bar chart of how sure the model is for each class, and the 224 x 224 image the model actually sees.

## 7\. How it works

1. **Quality check:** refuses photos under 100 px, almost uniform, or very blurry.
2. **Preprocessing:** resize to 224 x 224 (bilinear, same as training) and scale to \[-1, 1].
3. **Model:** MobileNetV2 (TensorFlow Lite, float32, 8.9 MB) scores the photo and its mirror image; the two outputs are averaged.
4. **Post-processing:** the NCD probability is multiplied by 1.5 and renormalised.
5. **Decision:** confidence below 0.6 (below 0.85 for Healthy) gives "refer to a vet".
6. **Output:** class, confidence, advice, class bars and the 224 x 224 view.

Settings are in `app/config.json`.

## 8\. Self-test

```
cd app
python tests/test\_core.py
```

Checks that all 12 sample photos get the right class, that blank, tiny and blurred photos are refused, and that the decision rule behaves as described.

## 9\. Reproduce the training (Kaggle)

1. Create a Kaggle notebook with a GPU (2 x Tesla T4 were used) and add the dataset `kausthubkannan/poultry-diseases-detection`.
2. Use Python 3.13, TensorFlow 2.21.0, scikit-learn 1.6.1, NumPy 2.1.3; random seed 42.
3. Upload `notebooks/poultryscan\_d2\_d3.ipynb` and run the cells in order: data checks and group-aware split, baseline, proposed model (`D3 | Step 1`), baseline re-run, comparison (`D3 | Baseline vs new model only`), export (`D3 | Export the model for the PC app`).
4. Download `poultry\_model.tflite` (and the `.keras` model) from the Output panel. Put the `.tflite` file in `app/model/`.

Small numeric differences can occur on different hardware.

## 10\. Model files

* `app/model/poultry\_model.tflite`: the deployed model (included).
* Keras model `d3\_plain\_s42.keras` and `splits.csv`: in `model\_files/` of the submission ZIP; they can be re-created with the notebook (not stored in Git because of their size).

## 11\. Known limitations

* Only the four trained classes are recognised. The app **cannot tell that a photo contains no dropping**; always photograph the dropping close up, in the centre of the picture.
* 53 NCD test photos only; one training seed so far (repeated seeds and cross-validation planned for D5).
* About 14% of NCD alarms are false (precision 86.2%); healthy birds wrongly flagged rose from 3.4% to 4.4%.
* Photos come from a public dataset that appears to originate in Tanzania, labelled by experts rather than laboratory tests. Performance on Ugandan farms is unknown.
* The 0.85 "Healthy" line and the photo-quality limits are provisional.

## 12\. Troubleshooting

|Problem|Fix|
|-|-|
|`python is not recognised`|Install Python and tick "Add Python to PATH"|
|`Model file not found`|Copy `poultry\_model.tflite` into `app/model/`|
|Port 7860 busy|Close other copies of the app, or change the port in `app.py`|
|First start is slow|Libraries are being installed (internet needed once)|
|Browser does not open|Open http://127.0.0.1:7860 yourself|

## 13\. Demonstration

The demonstration video is in `demo/`. It shows: starting the application, uploading a real project photo, the 224 x 224 preprocessing view, the model result, the improvement in operation, and how a farmer reads the output.

## 14\. References

MobileNetV2 (Sandler et al., CVPR 2018); Poultry diseases diagnostics models using deep learning (Machuve et al., Frontiers in AI, 2022); dataset: Kausthub Kannan, *Poultry Diseases Detection*, Kaggle. Full list in the report.

