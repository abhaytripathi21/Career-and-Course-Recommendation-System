# Career-and-Course-Recommendation-System
#here is a Flask-based web application that provides two core features:

#Career Path Prediction based on a student's academic and personal profile.
#Course Recommendations tailored to a selected career path using natural language processing.
🚀 Features
🎓 Education Recommendation

#Predicts the most suitable career options for students based on:
Gender
Part-time job status
Number of absence days
Participation in extracurricular activities
Weekly self-study hours
Subject scores (Math, History, Physics, Chemistry, Biology, English, Geography)
Total and average academic score
🔍 Utilizes a machine learning model to classify and rank career options with probabilities.

#📚 Course Recommendation
#Based on a selected profession (e.g., Software Engineer, Lawyer, Artist), the app recommends the most relevant online courses using:
TF-IDF vectorization
Cosine similarity matching between profession keywords and course metadata
Courses are fetched from a dataset (course_data.csv) and ranked accordingly.

#🛠 Technologies Used:
Python 🐍
Flask 🌐
Pandas, NumPy, Scikit-learn 📊
HTML (via Flask templates)
Machine Learning Model (model.pkl)
Scaler for feature normalization (scaler.pkl)
NLP with TF-IDF for course matching

#📁 Project Structure
career-course-recommender/
│
├── app.py # Main Flask application
│
├── models/ # Folder containing ML model and course data
│ ├── model.pkl # Trained ML model for career prediction
│ ├── scaler.pkl # Scaler used for feature normalization
│ └── course_data.csv # Dataset of online courses with metadata
│
├── templates/ # HTML templates for rendering pages
│ ├── home.html # Homepage UI
│ ├── education.html # Career prediction form and result
│ └── courses.html # Course recommendation UI
│
├── static/ # (Optional) Static files like CSS/JS/images
│ └── ... # Add custom styles or scripts here if needed
│
└── README.md

#⚙️ How to Run
#Clone the repository
git clone https://github.com/abhaytripathi21/Career-and-Course-Recommendation-System.git
cd Career-and-Course-Recommendation-System

#Install Dependencies
Flask
pandas
numpy
scikit-learn

#Run the App
#python app.py
