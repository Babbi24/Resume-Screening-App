import venv
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "8cd014a3-3091-4621-b2f5-ca9dd9bb6a5d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to C:\\Users\\Manoj\n",
      "[nltk_data]     Mishra\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package stopwords to C:\\Users\\Manoj\n",
      "[nltk_data]     Mishra\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "\n",
    "import streamlit as st\n",
    "import pickle\n",
    "import re\n",
    "import nltk\n",
    "\n",
    "nltk.download('punkt')\n",
    "nltk.download('stopwords')\n",
    "\n",
    "#loading models\n",
    "clf = pickle.load(open('clf.pkl','rb'))\n",
    "tfidf = pickle.load(open('tfidf.pkl','rb'))\n",
    "\n",
    "null=\"\"\n",
    "def cleanResume(txt):\n",
    "    cleanText = re.sub('http\\\\S+\\\\s', ' ', txt)\n",
    "    cleanText = re.sub('RT|cc', ' ', cleanText)\n",
    "    cleanText = re.sub('#\\\\S+\\\\s', ' ', cleanText)\n",
    "    cleanText = re.sub('@\\\\S+', '  ', cleanText)  \n",
    "    cleanText = re.sub('[%s]' % re.escape(\"\"\"!\"#$%&'()*+,-./:;<=>?@[\\\\]^_`{|}~\"\"\"), ' ', cleanText)\n",
    "    cleanText = re.sub(r'[^\\x00-\\x7f]', ' ', cleanText) \n",
    "    cleanText = re.sub('\\\\s+', ' ', cleanText)\n",
    "    return cleanText\n",
    "# web app\n",
    "def main():\n",
    "    st.title(\"Resume Screening App\")\n",
    "    uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])\n",
    "\n",
    "    if uploaded_file is not None:\n",
    "        try:\n",
    "            resume_bytes = uploaded_file.read()\n",
    "            resume_text = resume_bytes.decode('utf-8')\n",
    "        except UnicodeDecodeError:\n",
    "            # If UTF-8 decoding fails, try decoding with 'latin-1'\n",
    "            resume_text = resume_bytes.decode('latin-1')\n",
    "\n",
    "        cleaned_resume = clean_resume(resume_text)\n",
    "        input_features = tfidfd.transform([cleaned_resume])\n",
    "        prediction_id = clf.predict(input_features)[0]\n",
    "        st.write(prediction_id)\n",
    "\n",
    "        # Map category ID to category name\n",
    "        category_mapping = {\n",
    "            15: \"Java Developer\",\n",
    "            23: \"Testing\",\n",
    "            8: \"DevOps Engineer\",\n",
    "            20: \"Python Developer\",\n",
    "            24: \"Web Designing\",\n",
    "            12: \"HR\",\n",
    "            13: \"Hadoop\",\n",
    "            3: \"Blockchain\",\n",
    "            10: \"ETL Developer\",\n",
    "            18: \"Operations Manager\",\n",
    "            6: \"Data Science\",\n",
    "            22: \"Sales\",\n",
    "            16: \"Mechanical Engineer\",\n",
    "            1: \"Arts\",\n",
    "            7: \"Database\",\n",
    "            11: \"Electrical Engineering\",\n",
    "            14: \"Health and fitness\",\n",
    "            19: \"PMO\",\n",
    "            4: \"Business Analyst\",\n",
    "            9: \"DotNet Developer\",\n",
    "            2: \"Automation Testing\",\n",
    "            17: \"Network Security Engineer\",\n",
    "            21: \"SAP Developer\",\n",
    "            5: \"Civil Engineer\",\n",
    "            0: \"Advocate\",\n",
    "        }\n",
    "\n",
    "        category_name = category_mapping.get(prediction_id, \"Unknown\")\n",
    "\n",
    "        st.write(\"Predicted Category:\", category_name)\n",
    "\n",
    "# python main\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "ec8327ce-182f-4c0d-a46d-55aa58520538",
   "metadata": {},
   "outputs": [],
   "source": [
    "!streamlit run ResumeApp.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "3c4cc5df-3e31-41a2-b821-cdd642e41848",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "a69f8708-135a-4bb1-81ff-b6ffdf98cf56",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "de57836d-1951-45fa-b571-e5e4d5c3ec21",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
