import evaluate_1
from flask import Flask, request,jsonify

app = Flask(__name__)
print(app)

   
@app.route('/', methods=['GET'])
def index():
    return "Hello, World!"

@app.route('/post', methods=['POST'])
def post_route():
    if request.method == 'POST':

        data = request.get_json()
        inputStory = data['story']
        inputQuestion = data['question']
        parsed_stories = evaluate_1.parse_input_story(inputStory,inputQuestion)
        result = evaluate_1.getAnswer(parsed_stories)
        return jsonify({'Answer': result})


if __name__ == '__main__':
    app.run(port=5000)
