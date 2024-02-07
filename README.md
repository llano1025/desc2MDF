This project use AI to turn description to MDF group and recommended form.

## First time start the project

1.  Create virtual env. The 1st venv is option. The 2nd venv is we named the virtual env as ‘venv’
    > python -m venv venv
2.  Active the virtual venv by gitbash
    > source venv/Scripts/activate
3.  Install all dependency
    > pip install -r requirements.txt
4.  Data training is necessary in first run. Update following configable.<br />
    IS_TRAIN_MODEL_NEEDED = True<br />
5.  Python run main.py

## Everytime start the project

1. Active the virtual venv by gitbash
   > source venv/Scripts/activate
2. Install all dependency
   > pip install -r requirements.txt
3. Data training is optional. Update following configable.<br />
   IS_TRAIN_MODEL_NEEDED = False<br />
4. Python run main.py

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

## Start flask localhost server

Start the flask local server

> flask --app main run

Example request

> curl --location 'http://localhost:5000' \
> --header 'Content-Type: application/json' \
> --data '{"desc": "['\''ABC'\'', '\''IMPULSION PUMP, LONG'\'', '\''POWER SUPPLIER'\'']"}'

Example reponse

```json
{
  "data": [
    {
      "desc": "ABC",
      "groupNo": "mm-pm-00",
      "recommendedForm": "MDF15"
    },
    {
      "desc": "IMPULSION PUMP, LONG",
      "groupNo": "mm-pm-32",
      "recommendedForm": "MDF01"
    },
    {
      "desc": "POWER SUPPLIER",
      "groupNo": "mm-pm-01",
      "recommendedForm": "MDF15"
    }
  ]
}
```
