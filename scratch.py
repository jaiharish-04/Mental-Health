from flask import Flask, render_template, request

app = Flask(__name__,
            template_folder='/Users/jaiharishsatheshkumar/PycharmProjects/newmini/website-main/templates')


@app.route('/process')
def index():
    return render_template('testing.html')


@app.route('/process', methods=['POST'])
def process():
    data = request.form.get('data')
    print("Enga parda: ",data)
    # Process the data here
    return "Data received: " + data


if __name__ == '__main__':
    app.run(debug=True)
