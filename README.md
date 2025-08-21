# NPUST Student Dataset
![20240904_133056](https://github.com/user-attachments/assets/0f66d9b7-98ac-4755-b2e3-0cce4b3e98c3)

<div style="text-align: center;">
  <p><em>Figure: Students photoshoot on Lightroom Lab</em></p>
</div>

NPUST Student Dataset consist of around 4500 images. The primarily source of this dataset is National Pingtung University of Science and Technology's Fashion and Management students. 

# Download the Trained Model

To download the YOLOV8 Model trained on NPUST Student Dataset, Please follow this [Google Drive file Link](https://drive.google.com/file/d/1AUHRicgHVOsi-W76sxftSalmTxyWA4id/view?usp=drive_link)


# Download the NPUST Student Dataset

To download the NPUST Student Dataset along with the trained result please follow this [Google Drive file Link](https://drive.google.com/file/d/1DsTSQx0ikJqRL45jgMyXinklh_7JE0vS/view?usp=drive_link)
This is a password protected folder. To get the password please email us at weiher{at}mail.npust.edu.tw



### Dataset Files and Directories

- **config.yaml**: Configuration file for the model.
- **data/**: Contains the training, validation, and test datasets.
  - **train/**: Training dataset images.
  - **val/**: Validation dataset images.
  - **test/**: Test dataset images.
- **results/**: Contains the results of the model evaluation.
  - **confusion_matrix.png**: Confusion matrix of the model's performance.
  - **precision_recall_curve.png**: Precision-recall curve of the model's performance.
- **README.md**: This file, providing an overview of the folder contents.
## 📖 Citation

If you use this project in your research, please cite:

**Wei-Her Hsieh, Md Rabius Sany Apu, Jatin Thapa, and Owen Kumala (2025).**  
*Using Object Detection Methods to Detect Fashion Trends in University Students’ Attire.*  
In: Matthias Rauterberg (ed.) Culture and Computing (HCII 2025). Lecture Notes in Computer Science, vol 15800. Springer, Cham.  
DOI: [10.1007/978-3-031-93160-4_1](https://doi.org/10.1007/978-3-031-93160-4_1)

```bibtex
@incollection{hsieh2025ai4fashion,
  author    = {Wei-Her Hsieh and Md Rabius Sany Apu and Jatin Thapa and Owen Kumala},
  title     = {Using Object Detection Methods to Detect Fashion Trends in University Students’ Attire},
  booktitle = {Culture and Computing (HCII 2025)},
  editor    = {Matthias Rauterberg},
  series    = {Lecture Notes in Computer Science},
  volume    = {15800},
  publisher = {Springer, Cham},
  year      = {2025},
  doi       = {10.1007/978-3-031-93160-4_1}
}
