from flask import Flask, request, redirect, render_template, jsonify
from logic import process
from parkinsonwork import working
from alziemerwork import alz_working
import os

# from flask_cors import CORS


app = Flask(__name__,
            template_folder='/Users/jaiharishsatheshkumar/PycharmProjects/newmini/website-main/templates')
app.config['SECRET_KEY'] = 'verysecretkey'
app.config[
    'UPLOADED_PHOTOS_DEST'] = "/Users/jaiharishsatheshkumar/PycharmProjects/newmini/website-main/backend/"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# CORS(app, resources={r"home/": {"origins": "http://127.0.0.1:5000/"}})

print("ar:", APP_ROOT)

len_date_list = 0
temp_dates = []

gname = ""


@app.route("/login", methods=["POST", "GET"])
def login():
    global gname
    email = request.args.get("email")
    password = request.args.get("password")

    if email is not None and password is not None:
        o = process()
        con = o.auth([email, password])
        if con is True:
            name = o.ret_name_from_email(email)
            gname = name
            return redirect(
                "/home")  # TODO: generate a key for the user and use to save user's archive and chats(history)

    return render_template('login.html')


@app.route("/register", methods=["POST", "GET"])
def register():
    name = request.form.get("name")
    age = request.form.get("age")
    gender = request.form.get("gender")
    phone_num = request.form.get("phone_num")
    proff = request.form.get("proff")
    email = request.form.get("email")

    password = request.form.get("password")
    con_password = request.form.get("con_password")

    if password == con_password and password is not None and con_password is not None:
        obj = process()
        details = [name, age, gender, phone_num, proff, email, password]
        obj.register(details)

        return render_template('registration_auth.html')
    return render_template("registration.html")


@app.route("/home", methods=["POST", "GET"])
def home():
    global gname

    # Initialize user input and model response
    user_input = ""
    model_response = ""

    # If the request method is POST, it means the form was submitted
    if request.method == "POST":
        # Retrieve user input from the form data
        user_input = request.form.get('user-input', '')  # Use request.form.get to avoid KeyError

        # Process user input (e.g., interact with the model, generate a response)
        model_response = "This is working"  # Dummy response for now
        print("User input:", user_input)

        # Render the template with user input and model response
    return render_template('main.html', user_input=user_input, model_response=model_response, user=gname)


@app.route("/description", methods=["POST", "GET"])
def description():
    return render_template("description.html")


@app.route("/alzeimer", methods=["POST", "GET"])
def alzeimer():
    return render_template("alzeimerdescription.html")


@app.route("/parkinsons", methods=["POST", "GET"])
def parkinsons():
    return render_template("parkinsondescription.html")


@app.route("/uploadimageparkinson", methods=['POST'])
def uploadimageparkinson():
    filename = ""
    print("INside upload")
    target = os.path.join(APP_ROOT, 'static/image')
    if not os.path.isdir(target):
        os.mkdir(target)

    temp = ""
    com = ""
    for file_ in request.files.getlist("image"):
        filename = file_.filename
        if filename:
            destination = '/'.join([target, filename])
            print("dest: ", destination)
            temp = filename
            com = destination
            file_.save(destination)
    sub = request.form.get("submit")
    diagnosis = working(com)
    if diagnosis == "healthy":
        fin = """
        You do not show any symptoms of Parkinson's Disease. This result offers reassurance regarding your neurological health status.
        As always, maintaining your overall well-being is of paramount importance. We encourage you to continue prioritizing your health through regular exercise, a balanced diet, , quality sleep, and stress management practices.
        If any further questions or concerns regarding your mental health, please do not hesitate to consult with our chatbot in home page.

        """
    else:
        fin = """
        Our assessment tool has identified that you  exhibit indications of Parkinson's disease.
        If you're experiencing symptoms such as tremors, stiffness, slowness of movement, 
        or balance issues, it's important to consult a healthcare professional promptly. 
        If any further questions or concerns regarding your mental health, please do not hesitate to consult 
        with our chatbot in home page.
        """

    if sub:
        return render_template("landingpage_parkinson.html", diagnosis=fin, filename=temp)

    # return "Your file has been successfully uploaded..."
    return render_template('parkinsondescription.html', filename=filename)


@app.route("/uploadimagealzeimer", methods=['POST'])
def uploadimagealzeimer():
    filename = ""
    print("INside upload")
    target = os.path.join(APP_ROOT, 'static/image')
    if not os.path.isdir(target):
        os.mkdir(target)

    temp = ""
    com = ""
    for file_ in request.files.getlist("image"):
        filename = file_.filename
        if filename:
            destination = '/'.join([target, filename])
            print("dest: ", destination)
            temp = filename
            com = destination
            file_.save(destination)
    sub = request.form.get("submit")
    diagnosis = alz_working(com)
    if diagnosis == "healthy":
        fin = """
                Your cognitive assessment indicates no evidence of Alzheimer's Disease. This assessment provides reassuring insight into your neurological health status.Continuing to prioritize your overall well-being remains crucial. 

    We encourage you to maintain a regimen of regular exercise, adhere to a balanced diet, ensure quality sleep, and implement effective stress management practices.

    Should you have any further inquiries or concerns regarding your mental health, please feel free to consult with our chatbot available on the homepage.

                """
    else:
        fin = """
                Our assessment tool has identified indications suggestive of Alzheimer's disease in your case.

    If you are experiencing symptoms such as tremors, stiffness, slowness of movement, or balance issues, it is imperative to promptly seek consultation with a healthcare professional.

    For any additional questions or concerns regarding your mental health, please do not hesitate to consult with our chatbot available on the homepage.
                """

    if sub:
        return render_template("landingpage_alzeimer.html", diagnosis=fin, filename=temp)

    # return "Your file has been successfully uploaded..."
    return render_template('alzeimerdescription.html', filename=filename)


@app.route("/uploadimagedysgraphia", methods=['POST', 'GET'])
def uploadimagedysgraphia():
    filename = ""
    print("INside upload")
    target = os.path.join(APP_ROOT, 'static/image')
    if not os.path.isdir(target):
        os.mkdir(target)

    temp = ""

    for file_ in request.files.getlist("image"):
        filename = file_.filename
        if filename:
            destination = '/'.join([target, filename])
            print("dest: ", destination)
            temp = filename
            file_.save(destination)
    sub = request.form.get("submit")

    if sub:
        return render_template("landingpage_dysgraphia.html", jai='hello', filename=temp)

    # return "Your file has been successfully uploaded..."
    return render_template('dysgrahiadescription.html', filename=filename)


if __name__ == "__main__":
    app.run(debug=True)

# TODO: 1.local image display in landing page
# 2. fetch user input from home page(chat) and render the output dynamically using jinja
