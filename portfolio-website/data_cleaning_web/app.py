
from flask import Flask, render_template, request, send_file, jsonify, url_for
import os
import pandas as pd
import tempfile
import requests
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_cleaning_tool')))
from clean_csv import clean_csv

UPLOAD_FOLDER = tempfile.gettempdir()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/clean_url', methods=['POST'])
def clean_url():
    data = request.get_json()
    url = data.get('url', '').strip()
    if not url or not url.lower().endswith('.csv'):
        return jsonify({'success': False, 'error': 'Please provide a valid direct CSV URL.'})
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return jsonify({'success': False, 'error': 'Failed to download CSV from the provided URL.'})
        filename = secure_filename(os.path.basename(url.split('?')[0]))
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(input_path, 'wb') as f:
            f.write(r.content)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"cleaned_{filename}")
        clean_csv(input_path, output_path)
        download_link = url_for('download_file', filename=f"cleaned_{filename}")
        return jsonify({'success': True, 'download_link': download_link})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error: {str(e)}'})


@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
