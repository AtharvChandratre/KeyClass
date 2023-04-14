BASE_PATH="/content/"

cd -- "$BASE_PATH"
git clone https://github.com/AtharvChandratre/KeyClass.git
pip install snorkel transformers==4.11.3 sentence-transformers cleantext pyhealth gdown
cd KeyClass/scripts/

mkdir data/
cd data/
FILE_ID="1tFnqlu7MvHOrfUM6jK7oPQ7-3oSQr2Z_"
#FILE_ID="1MtCNwwhq0D9N5TdM2pZliukyOzXz5bbd"
URL="https://docs.google.com/uc?export=download&id=$FILE_ID"
echo ${green}===Downloading MIMIC Data...===${reset}
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILE_ID" -O "mimic.zip" && rm -rf /tmp/cookies.txt
echo ${green}===Unzipping MIMIC Data...===${reset}
jar xvf mimic.zip && rm mimic.zip
mv small-mimic mimic
cd ../

python /content/KeyClass/scripts/run_all.py --config /content/KeyClass/config_files/config_mimic.yaml