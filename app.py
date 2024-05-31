from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask_mail import Mail, Message
from email.mime.text import MIMEText
from flask import Flask
from flask_pymongo import PyMongo
from dotenv import load_dotenv

app = Flask(__name__)

# Configuring Flask-Mail with Gmail
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
# Load the environment variables from the .env file
load_dotenv()


app.config['MAIL_PASSWORD'] = os.getenv("API_KEY")
mail = Mail(app)



@app.route('/submit_appointment', methods=['POST'])
def submit_appointment():
    # Gathering form data
    medical_center = request.form.get('medicalCenter')
    full_name = request.form.get('fullName')
    dob = request.form.get('dob')
    gender = request.form.get('gender')
    contact_number = request.form.get('contactNumber')
    email = request.form.get('email')
    doctor_name = request.form.get('doctorName')
    doctor_category = request.form.get('doctorCategory')
    appointment_date = request.form.get('appointmentDate')
    appointment_time = request.form.get('appointmentTime')
    symptoms = request.form.get('symptoms')

    # Plain-text version of the receipt
    text_body = f"""
    *** Medical Appointment Receipt ***
    Medical Center: {medical_center}
    Patient Name: {full_name}
    Date of Birth: {dob}
    Gender: {gender}
    Contact Number: {contact_number}
    Email: {email}
    Appointment With: Dr. {doctor_name} ({doctor_category})
    Date: {appointment_date} at {appointment_time}
    Reported Symptoms: {symptoms}
    """

    # HTML version of the receipt
    html_body = f"""
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; }}
          .receipt-table {{
            width: 100%; max-width: 600px; margin: 20px auto; background-color: #fff; border-collapse: collapse; }}
          .receipt-table th, .receipt-table td {{
            border: 1px solid #ddd; padding: 8px; }}
          .receipt-table th {{
            background-color: #4CAF50; color: white; }}
          .receipt-table td.label {{ 
            background-color: #f2f2f2; font-weight: bold; }}
          .receipt-table tr:nth-child(even){{ background-color: #f2f2f2; }}
          .receipt-table tr:hover {{ background-color: #ddd; }}
        </style>
      </head>
      <body>
        <table class="receipt-table">
          <tr><th colspan="2">Medical Appointment Receipt</th></tr>
          <tr><td class="label">Medical Center:</td><td>{medical_center}</td></tr>
          <tr><td class="label">Patient Name:</td><td>{full_name}</td></tr>
          <tr><td class="label">Date of Birth:</td><td>{dob}</td></tr>
          <tr><td class="label">Gender:</td><td>{gender}</td></tr>
          <tr><td class="label">Contact Number:</td><td>{contact_number}</td></tr>
          <tr><td class="label">Email:</td><td>{email}</td></tr>
          <tr><td class="label">Appointment With:</td><td>Dr. {doctor_name}</td></tr>
          <tr><td class="label">Doctor Category:</td><td>{doctor_category}</td></tr>
          <tr><td class="label">Appointment Date:</td><td>{appointment_date}</td></tr>
          <tr><td class="label">Appointment Time:</td><td>{appointment_time}</td></tr>
          <tr><td class="label">Symptoms:</td><td>{symptoms}</td></tr>
        </table>
      </body>
    </html>
    """

    # Setting up the email message
    msg = Message('Medical Appointment Receipt', sender=app.config['MAIL_USERNAME'], recipients=[email])
    msg.body = text_body
    msg.html = html_body
    mail.send(msg)

    return 'Receipt generated and emailed successfully!'
    #return render_template('index.html' , message="Unexpected class name")




app.config["MONGO_URI"] = "mongodb://localhost:27017/lung_db"
mongo = PyMongo(app)


model = load_model("my_lung_cancer_model .h5")
class_names = ["Bengin", "Malignant", "Normal"]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test')
def show_patients():
    # Assuming 'patients' is your collection name. Replace 'patients' with your actual collection name if it's different.
    patients_cursor = mongo.db.lung_db.find()
    patients_list = list(patients_cursor)
    
    # For debugging; this will print to console where your Flask app is running
    print(patients_list)
    
    # Now pass the list of dictionaries (patients_list) to the template
    return render_template('Test.html', patients=patients_list)

@app.route('/contact')
def contact():
  # Render  patient data entry form template
    return render_template('Contact.html')

@app.route('/services')
def services():
    # Render  patient data entry form template
    return render_template('Services.html')
  
@app.route('/about')
def about():
    # Render  patient data entry form template
    return render_template('About.html')
  
  
@app.route('/patient.html')
def patient_data_entry():
    # Render your patient data entry form template
    return render_template('patient.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        pil_im = Image.open(file_path)
        pil_im = pil_im.resize((256, 256))
        pil_im = pil_im.convert('L')  # Convert to grayscale
        im = img_to_array(pil_im)
        im = im.reshape(1, 256, 256, 1)
        im = im / 255.0

        predictions = model.predict(im)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]

        # MongoDB: Example of inserting a document into 'predictions' collection
        mongo.db.predictions.insert_one({
            "filename": filename,
            "predicted_class": predicted_class_name
        })

          # Fetching the patients list for malignant predictions
        if predicted_class_name.lower() == 'malignant':
            patients_cursor = mongo.db.lung_db.find()
            patients_list = list(patients_cursor)
            print(patients_list)  # For debugging; this will print to console where your Flask app is running
            api_key = os.getenv('GOOGLE_MAPS_API_KEY')
            return render_template('Malignant.html', filename=filename, predicted_class=predicted_class_name, patients=patients_list)

        elif predicted_class_name.lower() == 'bengin':
            api_key = os.getenv('GOOGLE_MAPS_API_KEY')
            return render_template('Bengin.html', filename=filename, predicted_class=predicted_class_name)

        elif predicted_class_name.lower() == 'normal':
            return render_template('Normal.html', filename=filename, predicted_class=predicted_class_name)

        else:
            # Handle unexpected class name
            return render_template('Error.html', message="Unexpected class name")

if __name__ == "__main__":
    app.run(debug=True)
