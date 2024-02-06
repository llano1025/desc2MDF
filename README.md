This project use AI to turn description to MDF group.

### First time start the project

1.  Create virtual env. The 1st venv is option. The 2nd venv is we named the virtual env as ‘venv’
    > python -m venv venv
2.  Active the virtual venv by gitbash
    > source venv/Scripts/activate
3.  Install all dependency
    > pip install -r requirements.txt
4.  Python run main.py

### Other time start the project

1. Active the virtual venv by gitbash
   > source venv/Scripts/activate
2. Install all dependency
   > pip install -r requirements.txt
3. Python run main.py

### Input

- Input data: data\beseformDB.mdfAssetDescriptions.xlsx
- Training data: data\BasicInfo_Result.xlsx
- No. of epochs: self.epochs = 50 (default)

### Output

- Output data: exc/outputs/MDF2Rec_output.xlsx
