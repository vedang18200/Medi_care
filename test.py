import random
from flask import Flask, render_template, request, jsonify, redirect, url_for,session
from flask_sqlalchemy import SQLAlchemy
import secrets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import numpy as np
import pandas as pd
import re
from duckduckgo_search import DDGS
import msgConstant as msgCons


get_diet_recommendation = {
    'Fungal infection': ['Garlic', 'Yogurt', 'Turmeric', 'Neem', 'Coconut Oil'],
    'Allergy': ['Turmeric', 'Honey', 'Green Tea', 'Ginger', 'Omega-3 Rich Foods'],
    'GERD': ['Oatmeal', 'Ginger Tea', 'Leafy Greens', 'Banana', 'Almond Milk'],
    'Chronic cholestasis': ['Beetroot', 'Carrots', 'Papaya', 'Spinach', 'Turmeric'],
    'Drug Reaction': ['Aloe Vera', 'Coconut Water', 'Mint Leaves', 'Cucumber', 'Banana'],
    'Peptic ulcer disease': ['Cabbage Juice', 'Bananas', 'Oatmeal', 'Yogurt', 'Honey'],
    'AIDS': ['Protein-Rich Foods', 'Fruits', 'Vegetables', 'Whole Grains', 'Nuts and Seeds'],
    'Diabetes': ['Bitter Gourd', 'Fenugreek Seeds', 'Indian Gooseberry', 'Turmeric', 'Cinnamon'],
    'Gastroenteritis': ['Banana', 'Boiled Potatoes', 'Yogurt', 'Ginger Tea', 'White Rice'],
    'Bronchial Asthma': ['Garlic', 'Turmeric', 'Ginger', 'Mustard Oil', 'Honey'],
    'Hypertension': ['Banana', 'Leafy Greens', 'Oats', 'Berries', 'Garlic'],
    'Migraine': ['Ginger Tea', 'Peppermint Tea', 'Dark Chocolate', 'Salmon', 'Almonds'],
    'Cervical spondylosis': ['Carrot Juice', 'Turmeric', 'Garlic', 'Leafy Greens', 'Ginger Tea'],
    'Paralysis (brain hemorrhage)': ['Blueberries', 'Walnuts', 'Salmon', 'Spinach', 'Green Tea'],
    'Jaundice': ['Lemon Juice', 'Papaya', 'Bottle Gourd Juice', 'Beetroot', 'Radish'],
    'Malaria': ['Cinnamon Tea', 'Ginger Tea', 'Orange Juice', 'Papaya', 'Coconut Water'],
    'Chicken pox': ['Oatmeal Bath', 'Neem Leaves', 'Baking Soda', 'Green Tea', 'Honey'],
    'Dengue': ['Papaya Leaf Juice', 'Coconut Water', 'Broth Soups', 'Porridge', 'Herbal Tea'],
    'Typhoid': ['Banana', 'Plain Rice', 'Applesauce', 'Boiled Potatoes', 'Ginger Tea'],
    'Hepatitis A': ['Turmeric', 'Garlic', 'Leafy Greens', 'Beetroot', 'Carrot Juice'],
    'Hepatitis B': ['Garlic', 'Leafy Greens', 'Carrot Juice', 'Turmeric', 'Ginger Tea'],
    'Hepatitis C': ['Turmeric', 'Garlic', 'Leafy Greens', 'Beetroot', 'Carrot Juice'],
    'Hepatitis D': ['Garlic', 'Leafy Greens', 'Carrot Juice', 'Turmeric', 'Ginger Tea'],
    'Hepatitis E': ['Turmeric', 'Garlic', 'Leafy Greens', 'Beetroot', 'Carrot Juice'],
    'Alcoholic hepatitis': ['Turmeric', 'Ginger Tea', 'Leafy Greens', 'Garlic', 'Beetroot'],
    'Tuberculosis': ['Turmeric', 'Garlic', 'Leafy Greens', 'Ginger Tea', 'Cayenne Pepper'],
    'Common Cold': ['Ginger Tea', 'Chicken Soup', 'Honey', 'Citrus Fruits', 'Garlic'],
    'Pneumonia': ['Garlic', 'Ginger Tea', 'Honey', 'Turmeric', 'Oregano Tea'],
    'Dimorphic hemorrhoids (piles)': ['Fiber-Rich Foods', 'Fluids', 'Green Leafy Vegetables', 'Whole Grains', 'Prunes'],
    'Heart attack': ['Oats', 'Olive Oil', 'Garlic', 'Leafy Greens', 'Berries'],
    'Varicose veins': ['Flaxseeds', 'Ginger', 'Cayenne Pepper', 'Leafy Greens', 'Berries'],
    'Hypothyroidism': ['Seaweed', 'Brazil Nuts', 'Turmeric', 'Ginger', 'Coconut Oil'],
    'Hyperthyroidism': ['Cruciferous Vegetables', 'Yogurt', 'Berries', 'Whole Grains', 'Nuts'],
    'Hypoglycemia': ['Sweet Potatoes', 'Oats', 'Nuts', 'Berries', 'Cinnamon'],
    'Osteoarthritis': ['Turmeric', 'Ginger Tea', 'Leafy Greens', 'Fatty Fish', 'Berries'],
    'Arthritis': ['Ginger', 'Turmeric', 'Berries', 'Pineapple', 'Walnuts'],
    'Vertigo (Paroxysmal Positional Vertigo)': ['Ginger Tea', 'Ginkgo Biloba', 'Almonds', 'Spinach', 'Salmon'],
    'Acne': ['Green Tea', 'Turmeric', 'Fruits', 'Vegetables', 'Nuts'],
    'Urinary tract infection': ['Cranberry Juice', 'Watermelon', 'Yogurt', 'Celery', 'Cucumber'],
    'Psoriasis': ['Turmeric', 'Olive Oil', 'Leafy Greens', 'Fish', 'Nuts'],
    'Impetigo': ['Turmeric', 'Honey', 'Neem Leaves', 'Coconut Oil', 'Aloe Vera Gel']
}



ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = "m4xpl0it"


def make_token():
    return secrets.token_urlsafe(16)


class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    email = db.Column(db.String(120))
    password = db.Column(db.String(80))


@app.route("/")
def index():
    return render_template("index.html")


userSession = {}


@app.route("/user")
def index_auth():
    my_id = make_token()
    userSession[my_id] = -1
    return render_template("index_auth.html", sessionId=my_id)


@app.route("/instruct")
def instruct():
    return render_template("instructions.html")

@app.route("/upload")
def bmi():
    return render_template("bmi.html")

@app.route("/diseases")
def diseases():
    return render_template("diseases.html")


@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    return render_template('pred.html', pred=pred, f_name=f_name)



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]

        login = user.query.filter_by(username=uname, password=passw).first()
        if login is not None:
            return redirect(url_for("index_auth"))
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']

        register = user(username=uname, email=mail, password=passw)
        db.session.add(register)
        db.session.commit()

        return redirect(url_for("login"))
    return render_template("register.html")


import msgConstant as msgCons
import re

all_result = {
    'name':'',
    'age':0,
    'gender':'',
    'symptoms':[]
}


# Import Dependencies
# import gradio as gr
import pandas as pd
import numpy as np
from joblib import load
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def predict_symptom(user_input, symptom_list):
    # Convert user input to lowercase and split into tokens
    user_input_tokens = user_input.lower().replace("_"," ").split()

    # Calculate cosine similarity between user input and each symptom
    similarity_scores = []
    for symptom in symptom_list:
        # Convert symptom to lowercase and split into tokens
        symptom_tokens = symptom.lower().replace("_"," ").split()

        # Create count vectors for user input and symptom
        count_vector = np.zeros((2, len(set(user_input_tokens + symptom_tokens))))
        for i, token in enumerate(set(user_input_tokens + symptom_tokens)):
            count_vector[0][i] = user_input_tokens.count(token)
            count_vector[1][i] = symptom_tokens.count(token)

        # Calculate cosine similarity between count vectors
        similarity = cosine_similarity(count_vector)[0][1]
        similarity_scores.append(similarity)

    # Return symptom with highest similarity score
    max_score_index = np.argmax(similarity_scores)
    return symptom_list[max_score_index]




import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset into a pandas dataframe
df = pd.read_excel('dataset.xlsx')

# Get all unique symptoms
symptoms = set()
for s in df['Symptoms']:
    for symptom in s.split(','):
        symptoms.add(symptom.strip())



def predict_disease_from_symptom(symptom_list):


    user_symptoms = symptom_list
    # Vectorize symptoms using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Symptoms'])
    user_X = vectorizer.transform([', '.join(user_symptoms)])
    print(user_X)
    print(symptom_list)
    print("----------------------------------------------------------------")

    # Compute cosine similarity between user symptoms and dataset symptoms
    similarity_scores = cosine_similarity(X, user_X)

    # Find the most similar disease(s)
    max_score = similarity_scores.max()
    max_indices = similarity_scores.argmax(axis=0)
    diseases = set()
    for i in max_indices:
        if similarity_scores[i] == max_score:
            diseases.add(df.iloc[i]['Disease'])

    symptoms = {'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
                'shivering': 0, 'chills': 0, 'joint_pain': 0, 'stomach_pain': 0, 'acidity': 0, 'ulcers_on_tongue': 0,
                'muscle_wasting': 0, 'vomiting': 0, 'burning_micturition': 0, 'spotting_ urination': 0, 'fatigue': 0,
                'weight_gain': 0, 'anxiety': 0, 'cold_hands_and_feets': 0, 'mood_swings': 0, 'weight_loss': 0,
                'restlessness': 0, 'lethargy': 0, 'patches_in_throat': 0, 'irregular_sugar_level': 0, 'cough': 0,
                'high_fever': 0, 'sunken_eyes': 0, 'breathlessness': 0, 'sweating': 0, 'dehydration': 0,
                'indigestion': 0, 'headache': 0, 'yellowish_skin': 0, 'dark_urine': 0, 'nausea': 0, 'loss_of_appetite': 0,
                'pain_behind_the_eyes': 0, 'back_pain': 0, 'constipation': 0, 'abdominal_pain': 0, 'diarrhoea': 0, 'mild_fever': 0,
                'yellow_urine': 0, 'yellowing_of_eyes': 0, 'acute_liver_failure': 0, 'fluid_overload': 0, 'swelling_of_stomach': 0,
                'swelled_lymph_nodes': 0, 'malaise': 0, 'blurred_and_distorted_vision': 0, 'phlegm': 0, 'throat_irritation': 0,
                'redness_of_eyes': 0, 'sinus_pressure': 0, 'runny_nose': 0, 'congestion': 0, 'chest_pain': 0, 'weakness_in_limbs': 0,
                'fast_heart_rate': 0, 'pain_during_bowel_movements': 0, 'pain_in_anal_region': 0, 'bloody_stool': 0,
                'irritation_in_anus': 0, 'neck_pain': 0, 'dizziness': 0, 'cramps': 0, 'bruising': 0, 'obesity': 0, 'swollen_legs': 0,
                'swollen_blood_vessels': 0, 'puffy_face_and_eyes': 0, 'enlarged_thyroid': 0, 'brittle_nails': 0, 'swollen_extremeties': 0,
                'excessive_hunger': 0, 'extra_marital_contacts': 0, 'drying_and_tingling_lips': 0, 'slurred_speech': 0,
                'knee_pain': 0, 'hip_joint_pain': 0, 'muscle_weakness': 0, 'stiff_neck': 0, 'swelling_joints': 0, 'movement_stiffness': 0,
                'spinning_movements': 0, 'loss_of_balance': 0, 'unsteadiness': 0, 'weakness_of_one_body_side': 0, 'loss_of_smell': 0,
                'bladder_discomfort': 0, 'foul_smell_of urine': 0, 'continuous_feel_of_urine': 0, 'passage_of_gases': 0, 'internal_itching': 0,
                'toxic_look_(typhos)': 0, 'depression': 0, 'irritability': 0, 'muscle_pain': 0, 'altered_sensorium': 0,
                'red_spots_over_body': 0, 'belly_pain': 0, 'abnormal_menstruation': 0, 'dischromic _patches': 0, 'watering_from_eyes': 0,
                'increased_appetite': 0, 'polyuria': 0, 'family_history': 0, 'mucoid_sputum': 0, 'rusty_sputum': 0, 'lack_of_concentration': 0,
                'visual_disturbances': 0, 'receiving_blood_transfusion': 0, 'receiving_unsterile_injections': 0, 'coma': 0,
                'stomach_bleeding': 0, 'distention_of_abdomen': 0, 'history_of_alcohol_consumption': 0, 'fluid_overload.1': 0,
                'blood_in_sputum': 0, 'prominent_veins_on_calf': 0, 'palpitations': 0, 'painful_walking': 0, 'pus_filled_pimples': 0,
                'blackheads': 0, 'scurring': 0, 'skin_peeling': 0, 'silver_like_dusting': 0, 'small_dents_in_nails': 0, 'inflammatory_nails': 0,
                'blister': 0, 'red_sore_around_nose': 0, 'yellow_crust_ooze': 0}
    # for i in symptom_list :
    #     symptoms[i]=1

      # Set value to 1 for corresponding symptoms
    
    for s in symptom_list:
        index = predict_symptom(s, list(symptoms.keys()))
        print('User Input: ',s," Index: ",index)
        symptoms[index] = 1
    
    # Put all data in a test dataset
    df_test = pd.DataFrame(columns=list(symptoms.keys()))
    print(symptoms.values())
    df_test.loc[len(df.index)] = np.array(list(symptoms.values()))
    print(df_test.head())
    # Load pre-trained model
    clf = load(str("model/new.joblib"))
    result = clf.predict(df_test)
    print("-----=-=-=======================================")
    print(result)    

    if len(diseases) == 0:
        return "<b>No matching diseases found</b>", ""
    elif len(diseases) == 1:
        print("The most likely disease is:", list(result))
        disease_details = getDiseaseInfo(list(result)[0])
        return f"<b>{list(result)[0]}</b><br>{disease_details}", list(result)[0], get_diet_recommendation.get(list(result)[0], "")
    else:
        return "The most likely diseases are<br><b>"+ ', '.join(list(result))+"</b>", "", ""

    # # Output results
    # if len(diseases) == 0:
    #     return "<b>No matching diseases found</b>",""
    # elif len(diseases) == 1:
    #     print("The most likely disease is:", list(result))
    #     disease_details = getDiseaseInfo(list(result)[0])
    #     return f"<b>{list(result)[0]}</b><br>{disease_details}",list(result)[0]
    # else:
    #     return "The most likely diseases are<br><b>"+ ', '.join(list(result))+"</b>",""

        

    
    
  

    disease_details = getDiseaseInfo(result[0])
    
    # Cleanup
    del df_test
    
    return f"<b>{result[0]}</b><br>{disease_details}",result[0]



import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Get all unique diseases
diseases = set(df['Disease'])

def get_symtoms(user_disease):
    # Vectorize diseases using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Disease'])
    user_X = vectorizer.transform([user_disease])

    # Compute cosine similarity between user disease and dataset diseases
    similarity_scores = cosine_similarity(X, user_X)

    # Find the most similar disease(s)
    max_score = similarity_scores.max()
    print(max_score)
    if max_score < 0.7:
        print("No matching diseases found")
        return False,"No matching diseases found"
    else:
        max_indices = similarity_scores.argmax(axis=0)
        symptoms = set()
        for i in max_indices:
            if similarity_scores[i] == max_score:
                symptoms.update(set(df.iloc[i]['Symptoms'].split(',')))
        # Output results

        print("The symptoms of", user_disease, "are:")
        for sym in symptoms:
            print(str(sym).capitalize())

        return True,symptoms


import wikipedia
def getDiseaseInfo(keyword):
    try:
        result = wikipedia.summary(keyword)
        result=result.split(".")[:4]
        s=""
        for i in result:
            s=s+" "+i 
        return s
    except wikipedia.exceptions.PageError:
        return " "




@app.route('/ask',methods=['GET','POST'])
def chat_msg():

    user_message = request.args["message"].lower()
    sessionId = request.args["sessionId"]

    rand_num = random.randint(0,4)
    response = []
    if request.args["message"]=="undefined":

        response.append(msgCons.WELCOME_GREET[rand_num])
        response.append("What is your good name?")
        return jsonify({'status': 'OK', 'answer': response})
    else:


        currentState = userSession.get(sessionId)

        if currentState ==-1:
            response.append("Hi "+user_message+", To predict your disease based on symptopms, we need some information about you. Please provide accordingly.")
            userSession[sessionId] = userSession.get(sessionId) +1
            all_result['name'] = user_message            

        if currentState==0:
            username = all_result['name']
            response.append(username+", what is you age?")
            userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==1:
            pattern = r'\d+'
            result = re.findall(pattern, user_message)
            if len(result)==0:
                response.append("Invalid input please provide valid age.")
            else:                
                if float(result[0])<=0 or float(result[0])>=130:
                    response.append("Invalid input please provide valid age.")
                else:
                    all_result['age'] = float(result[0])
                    username = all_result['name']
                    response.append(username+", Choose Option ?")            
                    response.append("1. Predict Disease")
                    response.append("2. Check Disease Symtoms")
                    userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==2:

            if '2' in user_message.lower() or 'check' in user_message.lower():
                username = all_result['name']
                response.append(username+", What's Disease Name?")
                userSession[sessionId] = 20
            else:

                username = all_result['name']
                response.append(username+", What symptoms are you experiencing?")         
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==3:

            
            all_result['symptoms'].extend(user_message.split(","))
            username = all_result['name']
            response.append(username+", What kind of symptoms are you currently experiencing?")            
            response.append("1. Check Disease")   
            response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
            userSession[sessionId] = userSession.get(sessionId) +1


        if currentState==4:

            if '1' in user_message or 'disease' in user_message:
                disease_prediction = predict_disease_from_symptom(all_result['symptoms'])
                # Assuming predict_disease_from_symptom returns a tuple where the disease name is the second element
                disease_name = disease_prediction[1]  # Adjust the index based on actual return value structure
                diet_recommendation = get_diet_recommendation.get(disease_name, "No diet recommendation available")
                response.append("<b>The following disease may be causing your discomfort</b>")
                response.append(disease_name)
                response.append("<b>The diet recommendations are :-</b>")
                response.append(diet_recommendation)
                response.append(f'<a href="https://www.google.com/search?q={disease_name} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   
                userSession[sessionId] = 10

            else:

                all_result['symptoms'].extend(user_message.split(","))
                username = all_result['name']
                response.append(username+", Could you describe the symptoms you're suffering from?")            
                response.append("1. Check Disease")   
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1

    
        if currentState==5:
            if '1' in user_message or 'disease' in user_message:
                disease, type = predict_disease_from_symptom(all_result['symptoms'])
                diet_recommendation = get_diet_recommendation(disease)  # Replace with the actual function to get diet recommendation
                response.append("<b>The following disease may be causing your discomfort</b>")
                response.append(disease)
                response.append("<b>The diet recommendations are :-</b>")
                response.append(diet_recommendation)
                response.append(f'<a href="https://www.google.com/search?q={type} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   

                userSession[sessionId] = 10

            else:

                all_result['symptoms'].extend(user_message.split(","))
                username = all_result['name']
                response.append(username+", What are the symptoms that you're currently dealing with?")            
                response.append("1. Check Disease")   
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==6:    

            if '1' in user_message or 'disease' in user_message:
                disease,type = predict_disease_from_symptom(all_result['symptoms'])  
                response.append("The following disease may be causing your discomfort")
                response.append(disease)
                response.append("<b>The diet recommendations are :-</b>")
                response.append(diet_recommendation)
                response.append(f'<a href="https://www.google.com/search?q={type} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   
                userSession[sessionId] = 10
            else:
                all_result['symptoms'].extend(user_message.split(","))
                username = all_result['name']
                response.append(username+", What symptoms have you been experiencing lately?")            
                response.append("1. Check Disease")   
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==7:
            if '1' in user_message or 'disease' in user_message:
                disease,type = predict_disease_from_symptom(all_result['symptoms'])  
                response.append("<b>The following disease may be causing your discomfort</b>")
                response.append(disease)
                response.append("<b>The diet recommendations are :-</b>")
                response.append(diet_recommendation)
                response.append(f'<a href="https://www.google.com/search?q={type} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   
                userSession[sessionId] = 10
            else:
                all_result['symptoms'].extend(user_message.split(","))
                username = all_result['name']
                response.append(username+", What are the symptoms that you're currently dealing with?")            
                response.append("1. Check Disease")   
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1


        if currentState==8:    

            if '1' in user_message or 'disease' in user_message:
                disease,type = predict_disease_from_symptom(all_result['symptoms'])  
                response.append("The following disease may be causing your discomfort")
                response.append(disease)
                response.append("<b>The diet recommendations are :-</b>")
                response.append(diet_recommendation)
                response.append(f'<a href="https://www.google.com/search?q={type} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   
                userSession[sessionId] = 10
            else:
                all_result['symptoms'].extend(user_message.split(","))
                username = all_result['name']
                response.append(username+", What symptoms have you been experiencing lately?")            
                response.append("1. Check Disease")   
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==10:
            response.append('<a href="/user" target="_blank">Predict Again</a>')   


        if currentState==20:

            result,data = get_symtoms(user_message)
            if result:
                response.append(f"The symptoms of {user_message} are")
                for sym in data:
                    response.append(sym.capitalize())

            else:response.append(data)

            userSession[sessionId] = 2
            response.append("")
            response.append("Choose Option ?")            
            response.append("1. Predict Disease")
            response.append("2. Check Disease Symtoms")




                

        return jsonify({'status': 'OK', 'answer': response})



if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    app.run(debug=False, port=3000)