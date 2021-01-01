# Eyeballer Pytorch version
This is a reimplementation of [Bishop Fox's Eyeballer](https://github.com/BishopFox/eyeballer) in [PyTorch](https://pytorch.org/). The original code was implemented in TF.Keras. Additional to the code this repository also provides pretrained models and a dataset of website screenshots. 

Description from the original repo:
```
Eyeballer is meant for large-scope network penetration tests where you need to find "interesting" targets from a huge set of web-based hosts. Go ahead and use your favorite screenshotting tool like normal (EyeWitness or GoWitness) and then run them through Eyeballer to tell you what's likely to contain vulnerabilities, and what isn't.
```

## Screenshots
| Homepage | Login |
| ------ |:-----:|
| ![Sample HomePage](/images/test/homepage.png) | ![Sample Login Page](/images/test/login.png) |

| Not Found | Old Looking |
| ------ |:-----:|
| ![Sample Not Found](/images/test/not_found.png) | ![Sample Old Looking](/images/test/old_looking.png) |

## Models
5 pretrained models can be downloaded individually from [here](https://drive.google.com/drive/folders/1LWBEweaf1fM8UD_ZOpXYnlhkIhSFQfcD?usp=sharing). The performances on the validation dataset are reported in the following table:

| VGG 16 | VGG 19 | ResNet 18 | ResNet 50 | ResNet 152 |
|--------|--------|-----------|-----------|------------|
| 94.25  | 93.125 |   92.625  |   92.75   |   92.625   |

To set up a model, download the corresponding zip file and extract it into the `./models` folder. 

## Usage
### Inference
To test Eyeballer on some test data use the `eyeballer.py` file. The testing images need to be prepared in a separate folder, for example `./folder/test_images/<files>.png`. This folder structure is required, due to the [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) dataset used in PyTorch to ease loading the images. As an example we can classify the images provided in this repository with: `python3 --arch vgg16 --test-dir ./images`, which provides the following output in the format `file path - label - label index`:
```
./images/test/homepage.png - Homepage - 0
./images/test/login.png - Login Page - 1
./images/test/not_found.png - Not Found - 2
./images/test/old_looking.png - Old Looking - 3
```

### Training
If you want to train models by yourself have a look into `train_models.sh`

## Dataset
Additional to the code, the dataset containing the training and validation images is provided. The dataset is similar to the one in the original repository containing screenshots of different websites categorized into 4 different classes. The classes are Homepage, Login, Not Found, and Old Looking. The dataset (filename: `websites_dataset.zip`) can be downloaded from [here](https://drive.google.com/drive/folders/1LWBEweaf1fM8UD_ZOpXYnlhkIhSFQfcD?usp=sharing). For each class there are 1000 training and 200 validation images. Download the dataset, unzip it and set the path in `config/config.py`.

### Dataset generation
I captured the screenshots with [gowitness](https://github.com/sensepost/gowitness). I used a list of the top websites, which I found on Pastebin. So assuming you have a file with a few hostnames, for example: `./websites.txt` you can capture screenshots of the homepage following:
```
mkdir -p ./screenshots/homepage/
cat websites.txt | gowitness file -f - -P ./screenshots/homepage/ --no-http
```
To capture screenshots for login pages I append `/login` to the host.
```
mkdir -p ./screenshots/login/
cat websites.txt | sed 's/$/\/login/g' | gowitness file -f - -P ./screenshots/login/ --no-http
```
To capture screenshots for not found pages I appended `/thissitedoesnotexist` to the host.
```
mkdir -p ./screenshots/not_found/
cat websites.txt | sed 's/$/\/thissitedoesnotexist/g' | gowitness file -f - -P ./screenshots/not_found/ --no-http
```
For the old-looking class, I queried the [wayback machine](https://web.archive.org/).

Afterward, I manually sorted out the results that were not good and split the data into the training and validation set. To split the data the script `split_data.py` might be helpful.

### How to add a custom class
To add your custom data, you can capture screenshots as described above. Then you should add a new folder to the `/path/to/websites/train` and `/path/to/websites/val`. Then you can capture screenshots and split them into the training and validation set. Since we have in total 5 classes now, we need to change the parameter `num_classes` in `train_model.py` and `eyeballer.py` to 5 (`num_classes = 5`). You are now ready to train a model on the extended dataset as before. 

#### Example
Assuming we want to calssify screenshots of APIs. First lets add the two folders: 
```
mdir -p /path/to/websites/train/api
mdir -p /path/to/websites/val/api
```
Now we can generate the screenshots:
```
mkdir -p ./screenshots/api/
cat websites.txt | sed 's/$/\/api/g' | gowitness file -f - -P ./screenshots/api/ --no-http
```
After we sorted out all wrong results and found 1200 valid screenshots we can put 1000 screenshot into the `/path/to/websites/train/api` folder and 200 into the `/path/to/websites/val/api` folder. Setting `num_classes = 5` as described above, we can now train a model on the extended dataset.

### Qualitative Results
To check some qualitative results have a look at the jupyter notebook `./qualitative_results.ipynb`.

### Docker
I prefer to use docker to set up my deep learning environments. For this project, I used `1.0.1-cuda10.0-cudnn7-devel` as the base from [Pytorch`s Docker Hub](https://hub.docker.com/r/pytorch/pytorch/tags?page=1&ordering=last_updated).
