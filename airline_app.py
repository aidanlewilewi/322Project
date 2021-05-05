import os 
import pickle 
from flask import Flask, jsonify, request, render_template, redirect

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index_page():
    prediction = ""
    if request.method == "POST":
        att0 = request.form("att0", "")
        att1 = request.form("att1", "")
        att2 = request.form("att2", "")
        att3 = request.form("att3", "")
        att4 = request.form("att4", "")
        att5 = request.form("att5", "")

        att2 = int(att2)
        att5 = int(att5)

        prediction = predict_interviews_well([att0, att1, att2, att3, att4, att5])

    # goes into templates folder and finds given name
    return render_template("index.html", prediction=prediction) 


@app.route("/predict", methods=["GET"])
def predict():

    att0 = request.args.get("att0", "")
    att1 = request.args.get("att1", "")
    att2 = request.args.get("att2", "")
    att3 = request.args.get("att3", "")
    att4 = request.args.get("att4", "")
    att5 = request.args.get("att5", "")

    att2 = int(att2)
    att5 = int(att5)

    prediction = predict_interviews_well([att0, att1, att2, att3, att4, att5])

    if prediction is not None:
        result = {"prediction": prediction} 
        return jsonify(result), 200 
    else:
        return "Error making prediction", 400

def tdidt_predict(header, tree, instance):

    info_type = tree[0]

    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                return tdidt_predict(header, value_list[2], instance)
    else: 
        return tree[1] 
    

def predict_interviews_well(instance):
    infile = open("tree.p", "rb")
    header, tree = pickle.load(infile)
    infile.close()

    try:
        return tdidt_predict(header, tree, instance)
    except:
        return None 


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)