This project use AI to turn description to MDF group.

## First time start the project

1.  Create virtual env. The 1st venv is option. The 2nd venv is we named the virtual env as ‘venv’
    > python -m venv venv
2.  Active the virtual venv by gitbash
    > source venv/Scripts/activate
3.  Install all dependency
    > pip install -r requirements.txt
4.  Python run main.py

## Everytime start the project

1. Active the virtual venv by gitbash
   > source venv/Scripts/activate
2. Install all dependency
   > pip install -r requirements.txt
3. Python run main.py

## Input

#### MDF2Rec (Desc2Rec?)

- Training data: data\BasicInfo_Result.xlsx
- Input data: data\beseformDB.mdfAssetDescriptions.xlsx

#### Desc2MDF

- Training data: data\recForm.xlsx
- Input data: data\beseformDB.mdfAssetDescriptions.xlsx

- No. of epochs: self.epochs = 50 (default)

## Output

#### MDF2Rec (Desc2Rec?)

- Output data: exc/outputs/MDF2Rec_output.xlsx

#### Desc2MDF

- Output data: exc/outputs/Desc2MDF_output.xlsx
