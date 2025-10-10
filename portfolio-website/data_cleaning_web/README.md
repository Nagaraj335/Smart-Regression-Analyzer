# CSV Data Cleaning Web Tool

A simple interactive web app to clean your CSV files using your custom Python data cleaning logic.

## 🚀 Features
- Upload a CSV file via the browser
- Cleans and structures your data automatically
- Download the cleaned CSV file with a single click
- Modern UI with TailwindCSS

## 🛠️ Tech Stack
- Python (Flask)
- HTML + TailwindCSS (frontend)
- pandas, numpy (data cleaning)

## ⚡ Usage
1. Install requirements:
   ```
   pip install flask pandas numpy
   ```
2. Start the web app:
   ```
   python app.py
   ```
3. Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)
4. Upload your CSV file and download the cleaned result!

## 📦 File Structure
```
data_cleaning_web/
├── app.py
├── templates/
│   └── index.html
└── README.md
```

## 📝 Customization
- The backend uses your existing `clean_csv.py` logic from the data_cleaning_tool folder.
- You can add more cleaning options to the web form and pass them to the backend as needed.

---
*Created by Nagaraj335, August 2025*
