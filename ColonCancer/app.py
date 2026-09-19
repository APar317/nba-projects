from flask import Flask, render_template, request #,redirect, flash
from main import getPrediction
from Similarity import checkSimilarity
#Save images to the 'static' folder as Flask serves images from this directory
UPLOAD_FOLDER = 'static/images/'

#Create an app object using the Flask class. 
app = Flask(__name__, static_folder="static")
app.debug=True 

# routes
@app.route("/home")
@app.route("/", methods=['GET', 'POST'])
def home():
	return render_template("index.html")



@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		if 'my_image' in request.files and request.files['my_image'].filename != '':
			img = request.files['my_image']
			img_path = UPLOAD_FOLDER + img.filename	
			img.save(img_path)
		elif 'prefixed_image' in request.form and request.form['prefixed_image'] != '':
			img_path = request.form['prefixed_image']
		else:
			return render_template("index.html", valid=True)
		r = checkSimilarity(img_path)
		if r==False:
			result = "The image you have entered is INVALID, Please enter a valid Histopathological Biopsy Tissue image"
			return render_template("index_report.html", prediction = result, img_path = img_path)
		p = getPrediction(img_path)
		if p == 0:
			result = "Detected Colon Cancer as : MALIGNANT"
		else: 
			result= "Detected Colon Cancer as : NORMAL"


	return render_template("index_report.html", prediction = result, img_path = img_path)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5007)

