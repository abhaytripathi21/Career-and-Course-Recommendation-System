from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Get the absolute path to the directory containing the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load course data and models using absolute paths
course_data = pd.read_csv(os.path.join(BASE_DIR, 'models', 'course_data.csv'))
model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'model.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'models', 'scaler.pkl'), 'rb'))

# Career class names for education recommendation
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

# Create a mapping of professions to relevant keywords for course recommendations
profession_keywords = {
    'Lawyer': ['law', 'legal', 'contracts', 'litigation', 'corporate'],
    'Doctor': ['medical', 'healthcare', 'anatomy', 'clinical', 'patient'],
    'Government Officer': ['public policy', 'administration', 'governance', 'civil service', 'public sector'],
    'Artist': ['art', 'design', 'creative', 'illustration', 'digital art'],
    'Software Engineer': ['programming', 'software', 'development', 'coding', 'web'],
    'Teacher': ['education', 'teaching', 'pedagogy', 'learning', 'instruction'],
    'Business Owner': ['business', 'entrepreneurship', 'management', 'marketing', 'strategy'],
    'Scientist': ['research', 'science', 'laboratory', 'analysis', 'methodology'],
    'Banker': ['banking', 'finance', 'investment', 'markets', 'credit'],
    'Writer': ['writing', 'content', 'journalism', 'storytelling', 'editing'],
    'Accountant': ['accounting', 'finance', 'taxation', 'auditing', 'bookkeeping'],
    'Designer': ['design', 'UI/UX', 'graphic', 'creative', 'visual'],
    'Construction Engineer': ['construction', 'engineering', 'architecture', 'structural', 'project management'],
    'Game Developer': ['game development', 'unity', 'unreal', '3D modeling', 'game design'],
    'Stock Investor': ['investment', 'trading', 'stocks', 'market analysis', 'portfolio'],
    'Real Estate Developer': ['real estate', 'property', 'construction', 'development', 'investment']
}

def get_course_recommendations(profession, top_n=5):
    if profession not in profession_keywords:
        return None
    
    # Get keywords for the profession
    keywords = profession_keywords[profession]
    
    # Create a string of tags and descriptions for each course
    course_data['content'] = course_data['tags'].fillna('') + ' ' + course_data['description'].fillna('')
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(course_data['content'])
    
    # Create query string from profession keywords
    query = ' '.join(keywords)
    query_vector = tfidf.transform([query])
    
    # Calculate similarity
    similarity = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top N courses
    top_indices = similarity.argsort()[-top_n:][::-1]
    recommended_courses = course_data.iloc[top_indices]
    
    return recommended_courses[['title', 'category', 'level', 'instructor', 'rating', 'duration', 'is_paid', 'url']]

def get_education_recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score,
                               total_score, average_score):
    try:
        # Encode categorical variables
        gender_encoded = 1 if gender.lower() == 'female' else 0
        part_time_job_encoded = 1 if part_time_job else 0
        extracurricular_activities_encoded = 1 if extracurricular_activities else 0

        # Create feature array
        feature_array = np.array([[
            gender_encoded,
            part_time_job_encoded,
            absence_days,
            extracurricular_activities_encoded,
            weekly_self_study_hours,
            math_score,
            history_score,
            physics_score,
            chemistry_score,
            biology_score,
            english_score,
            geography_score,
            total_score,
            average_score
        ]])

        print("Input shape:", feature_array.shape)
        print("Input data:", feature_array)
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        print("Scaled features shape:", scaled_features.shape)
        
        # Get probabilities for all classes
        probabilities = model.predict_proba(scaled_features)[0]
        print("Probabilities shape:", probabilities.shape)
        print("Probabilities:", probabilities)
        
        # Get top three predicted classes along with their probabilities
        top_classes_idx = np.argsort(-probabilities)[:3]
        recommendations = [(class_names[idx], probabilities[idx] * 100) for idx in top_classes_idx]
        
        print("Top recommendations:", recommendations)
        return recommendations
        
    except Exception as e:
        print("Error in get_education_recommendations:", str(e))
        raise e

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/education')
def education():
    return render_template('education.html')

@app.route('/courses')
def courses():
    professions = list(profession_keywords.keys())
    return render_template('courses.html', professions=professions)

@app.route('/recommend_education', methods=['POST'])
def recommend_education():
    try:
        # Get form data
        gender = request.form['gender']
        part_time_job = request.form['part_time_job'] == '1'
        absence_days = int(request.form['absence_days'])
        extracurricular_activities = request.form['extracurricular_activities'] == '1'
        weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
        math_score = int(request.form['math_score'])
        history_score = int(request.form['history_score'])
        physics_score = int(request.form['physics_score'])
        chemistry_score = int(request.form['chemistry_score'])
        biology_score = int(request.form['biology_score'])
        english_score = int(request.form['english_score'])
        geography_score = int(request.form['geography_score'])
        total_score = float(request.form['total_score'])
        average_score = float(request.form['average_score'])

        print("Form data received:", {
            'gender': gender,
            'part_time_job': part_time_job,
            'absence_days': absence_days,
            'extracurricular_activities': extracurricular_activities,
            'weekly_self_study_hours': weekly_self_study_hours,
            'scores': [math_score, history_score, physics_score, chemistry_score, 
                      biology_score, english_score, geography_score],
            'total_score': total_score,
            'average_score': average_score
        })

        recommendations = get_education_recommendations(
            gender, part_time_job, absence_days, extracurricular_activities,
            weekly_self_study_hours, math_score, history_score, physics_score,
            chemistry_score, biology_score, english_score, geography_score,
            total_score, average_score
        )

        return render_template('education.html', recommendations=recommendations)
    except Exception as e:
        print("Error in recommend_education:", str(e))
        return render_template('education.html', error=str(e))

@app.route('/recommend_courses', methods=['POST'])
def recommend_courses():
    profession = request.form.get('profession')
    if not profession:
        return render_template('courses.html', error="Please select a profession", professions=list(profession_keywords.keys()))
    
    recommended_courses = get_course_recommendations(profession)
    professions = list(profession_keywords.keys())
    
    if recommended_courses is None:
        return render_template('courses.html', error="Invalid profession selected", professions=professions)
    
    return render_template('courses.html', 
                         profession=profession,
                         recommended_courses=recommended_courses.to_dict('records'),
                         professions=professions)

if __name__ == '__main__':
    # Configure Flask to ignore changes in site-packages
    import sys
    extra_dirs = ['templates/', 'static/']
    extra_files = extra_dirs[:]
    for extra_dir in extra_dirs:
        for dirname, dirs, files in os.walk(extra_dir):
            for filename in files:
                filename = os.path.join(dirname, filename)
                if os.path.isfile(filename):
                    extra_files.append(filename)
    
    app.run(debug=True, extra_files=extra_files)
