Country GHI Prediction Model:
Overview
This project aims to predict the Global Hunger Index (GHI) of a country based on various features such as GDP per Capita, Global Happiness Index, Multidimensional Poverty Index, Global Peace Index, and World Risk Index. The model is implemented using Python with Flask as the web framework.

Getting Started
Prerequisites
Python 2.8.2
Flask
pandas
numpy

Installation (For local environment)

Go to "2D Code" Folder
Once you have downloaded the folder you can open the folder in the command prompt.

cd Downloads
cd 2D Code/webapp

Check the content in the webapp folder:

dir

It should return you the following:

<DIR>          .
<DIR>          ..
<DIR>          .ipynb_checkpoints
<DIR>          .venv
	 6,987 app.py
 	 2,333 Final.csv
<DIR>          static
<DIR>          templates
<DIR>          __pycache__
<DIR> 	    requirements
          7,835 ReadME.txt	


Install the required packages:
pip freeze > requirements. txt [run by me the host to freeze the version of extensions used in the app]

pip install -r requirements.txt [run by a new user to install the requirement packages]

Create Virtual Environment
You should open Command Prompt to do the following steps.

In the following steps, whenever there is a difference between the OS commands, the Windows prompt will be represented by:

Windows:
>

while the MacOS/Linux prompt will be represented by:
$

Go to the root folder 2D Code.

> cd %USERPROFILE%\Downloads\2D Code\webapp
$ cd ~/Downloads/D Code/webapp

How to Use
1. Running the Application
Go to the root folder 2D Code

> cd %USERPROFILE%\Downloads\2D Code\webapp
$ cd ~/Downloads/D Code/webapp

From webapp to run the virtual env run this in the command prompt:

> cd %USERPROFILE%\Downloads\2D Code\webapp\. venv\Scripts\activate
$ cd ~/Downloads/D Code/webapp/. venv/Scripts/activate

Backtrack to the webapp folder:

> cd ..\..
$ cd ../..

Start the Flask application by running the following command in the root folder:

> cd %USERPROFILE%\Downloads\2D Code\webapp\python app.py
$ cd ~/Downloads/D Code/webapp/python app.py

You should see this:
Visit http://127.0.0.1:5000/ in your web browser to interact with the application. (Press CTRL+C to quit)

2. Explore Country Data
Navigate to the home page to view a list of countries.
Select a country from the dropdown menu and click "Submit" to display information about the selected country, including the current Global Hunger Index (GHI), GDP per Capita, Multi-dimensional Poverty Index (MPI), Global Happiness Index (GHI), Global Peace Index (GPI), and World Risk Index (WRI).

3. Predict GHI for a Country
On the home page, in the "Predict Global Hunger Index (GHI)" section, enter the required feature values (GDP per Capita, MPI, GHI, GPI, WRI) in the input fields.
Click the "Predict" button to see the predicted Global Hunger Index for the input features.

4. Important Functions to run the model
- get_predicted_response(gdp, mpi, ghi, gpi, wri)
This function takes in feature values (GDP, MPI, GHI, GPI, WRI) as input and returns the predicted Global Hunger Index using the linear regression model.

- gradient_descent(X, y, beta, alpha, num_iters)
This function implements the gradient descent algorithm to optimize the model parameters (beta) based on the training data (X and y). Adjust the learning rate (alpha) and the number of iterations (num_iters) as needed. For our case we have kept it as 0.01 and 1500 respectively.

5. Other helper functions which are used to build the model include -
- normalize_minmax(dfin): Standardization: Write a function that takes in data frame where all the column are the features and normalize each column according to the following formula.

- prepare_feature(df): which takes in a data frame or two-dimensional numpy array for the feature. If the input is a data frame, the function should convert the data frame to a numpy array and change it into a column vector. The function should also add a column of constant 1s in the first column.

- prepare_target(df): which takes in a data frame or two-dimensional numpy array for the target. If the input is a data frame, the function should simply convert the data frame to a numpy array and change it into column vectors.

- predict(df_feature, beta): This standardizes the feature using z normalization, changes it to a Numpy array, and adds a column of constant 1s. prepare_feature() is used for this purpose. Lastly, this function calls predict_norm() to get the predicted y values.

- calc_linreg(X, b): which is used to calculate the  𝑦̂ =𝑋𝑏

- split_data(df_feature, df_target, random_state=None, test_size=0.3): Split the Data Frame according to how you wish to split test and train sets. In our case its 30% Test 70% Train. The function has the following arguments:

df_feature: Data frame for the features.
df_target: Data frame for the target.
random_state: Seed used to split randomly.
test_size: Fraction for the test data set (0 to 1)
The output of the function is a tuple of four items:

df_feature_train: Train set for the features data frame
df_feature_test: Test set for the features data frame
df_target_train: Train set for the target data frame
df_target_test: Test set for the target data frame

- mean_squared_error(target, pred): Calculate the MSE
 
- r2_score(y, ypred): Calculates the coefficient of determination as given by the following equations.
 
- accuracy_estimate(MSE): Used to estimate the accuracy of the MSE metric

6. Functions used to help with the HTML display:
get_countries_from_csv(): Extracts a list of countries from the provided CSV file.
get_index(index, selected_country): Retrieves the value of a specific index for the selected country to be displayed under the Current Global Hunger Index (GHI).

7. Understanding HTML Forms and Methods
HTML Forms
The application uses HTML forms to collect user input for both exploring country data and predicting GHI. The forms are defined using the <form> tag, and input fields are created using <input> tags.

HTTP Methods (POST and GET)
POST Method: Used to submit data to be processed by the server. In this application, it is utilized to send user input for both exploring country data and predicting GHI.

GET Method: Used to request data from a specified resource. In this application, it is mainly used to render the initial web page and display information.

Example of HTML form:

<form action="/" method="POST">
    <!-- ... form elements ... -->
    <button type="submit">Submit</button>
</form>

Example in Flask route:

python

@app.route('/', methods=['POST', 'GET'])
def index():
    # Process data from the form
    # Render the initial web page
  return render_template('index.html', countries=countries, selected_country=selected_country, ghi_value=ghi_value, GDP = GDP, MPI = MPI, GHI = GHI, GPI = GPI, WRI = WRI) [# to render the exploring country data and display it]

  return render_template('index.html', prediction_result=pred) [# to render the GHI prediction and display it]

render_template is a Flask function that takes an HTML template file (in this case, 'index.html') and substitutes the placeholders ({{ selected_country }}, {{ ghi_value }}, etc.) with the actual values provided in the function call.

So, when the user accesses the home page (/) using a web browser, these values will be dynamically inserted into the HTML page based on the data retrieved from app.py. This is how you can render data from the Flask in app.py to the frontend Index.html handling a GET request.

8. Troubleshooting
- You may potentially face an issue with no countries being shown on the Current Global Hunger Index (GHI) section. Just click the Submit button once and it should show the list of countries again. 
