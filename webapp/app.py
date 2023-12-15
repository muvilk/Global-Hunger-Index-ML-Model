from flask import Flask, render_template, request, flash, session
import pandas as pd
import numpy as np

app = Flask(__name__)

csv_file_path = 'Final.csv'
df = pd.read_csv(csv_file_path)
feature_names = ["GDP Per Capita", "Global Happiness Index", "Multidimensional Poverty Index","Global Peace Index", "World Risk Index"]
target_names = ["Global Hunger Index"]

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_country = None
    ghi_value = None 
    GDP = None
    MPI = None
    GHI = None
    GPI = None
    WRI = None
    
    if request.method == 'POST':
        #Use the GET Method to receive the value of the selected country
        selected_country = request.form.get("selected_country")

        ghi_value = get_index("Global Hunger Index", selected_country)
        GDP = get_index("GDP Per Capita", selected_country)
        MPI = get_index("Multidimensional Poverty Index", selected_country)
        GHI = get_index("Global Happiness Index", selected_country)
        GPI = get_index("Global Peace Index", selected_country)
        WRI = get_index("World Risk Index", selected_country)

    countries = get_countries_from_csv()
    #Render it back to HTML to display the value in the {{ }} placeholder in HTML
    return render_template('index.html', prediction_result=0, countries=countries, selected_country=selected_country, ghi_value=ghi_value, GDP = GDP, MPI = MPI, GHI = GHI, GPI = GPI, WRI = WRI)

def get_index(index, selected_country):
    try:
        #Return the value of the specific feature of the selected coutnry which is given from the Index parameter
        value = df.loc[df['Country'] == selected_country, index].values[0]
        return value
    except IndexError:
        return None

def get_countries_from_csv():
    # CSV file with all the data
    csv_file_path = 'Final.csv'

    # Read the CSV file using pandas
    df = pd.read_csv(csv_file_path)

    # Extract the 'Country' column as a list
    countries = df['Country'].tolist()

    return countries

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    pred = 0  # Initialize to None or some default value

    if request.method == 'POST':
        # Get values from the form
        feature1 = request.form.get("feature1")
        feature2 = request.form.get("feature2")
        feature3 = request.form.get("feature3")
        feature4 = request.form.get("feature4")
        feature5 = request.form.get("feature5")

        # Validate inputs
        if is_valid_number(feature1) and is_valid_number(feature2) and is_valid_number(feature3) and is_valid_number(feature4) and is_valid_number(feature5):
            # Convert inputs to floats
            gdp = float(feature1)
            mpi = float(feature2)
            ghi = float(feature3)
            gpi = float(feature4)
            wri = float(feature5)

            # Call the prediction function
            pred = round(float(get_predicted_response(gdp, mpi, ghi, gpi, wri)), 3)
            pred = int(pred)

    return render_template('index.html', prediction_result=pred)


def is_valid_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# all the model functions from Vocareum here
def normalize(dfin, columns_means=None, columns_stds=None):
    if len(dfin)==1:
        return dfin
    mean = dfin.mean()
    std = dfin.std(axis=0)
    return (dfin - mean)/std

def get_features_targets(df, feature_names, target_names):
    df_feature = df[feature_names]
    df_target = df[target_names]
    return df_feature, df_target

def prepare_feature(df_feature):
    cols = len(df_feature.columns)
    feature = df_feature.to_numpy().reshape(-1, cols)
    X = np.concatenate((np.ones((feature.shape[0], 1)), feature), axis=1)
    return X

def prepare_target(df_target):
    cols = len(df_target.columns)
    target = df_target.to_numpy().reshape(-1, cols)
    return target

def calc_linreg(X, beta):
    return np.matmul(X, beta)

def predict_value(df_feature, beta, mins=None, maxs=None):
    norm_data = normalize(df_feature, mins, maxs)
    X = prepare_feature(norm_data)
    return calc_linreg(X, beta).flatten()    

def split_data(df_feature, df_target, random_state=100, test_size=0.3):
    indexes = df_feature.index
    if random_state != None:
        np.random.seed(random_state)
    k = int(test_size * len(indexes))
    test_index = np.random.choice(indexes, k, replace=False)
    indexes = set(indexes)
    test_index = set(test_index)
    train_index = indexes - test_index

    df_feature_train = df_feature.loc[list(train_index), :]
    df_feature_test = df_feature.loc[list(test_index), :]
    df_target_train = df_target.loc[list(train_index), :]
    df_target_test = df_target.loc[list(test_index), :]
    return df_feature_train, df_feature_test, df_target_train, df_target_test

def mean_squared_error(target, pred):
    y_minus_ypred = pred - target
    y_minus_ypred_sq = np.matmul(y_minus_ypred.T, y_minus_ypred)
    summation = np.sum(y_minus_ypred_sq)
    mse = (1/len(target)) * summation
    return mse

def r2_score(y, ypred):
    ymean = np.mean(y)
    sstot = np.sum((y-ymean)**2)
    ssres = np.sum((y-ypred)**2)
    r_2 = 1 - (ssres/sstot)
    return r_2

# Python code to build model (from Vocareum also)
def compute_cost(X, y, beta):
    y_hat = np.matmul(X, beta)
    error = y_hat - y
    error_sq = np.matmul(error.T, error)
    m = X.shape[0]
    J = (1/(2*m)) * error_sq
    J = J[0][0]
    return J

def gradient_descent(X, y, beta, alpha, num_iters):
    m = X.shape[0]
    J_storage = np.zeros((num_iters, 1))
    for n in range(num_iters):
        pred = calc_linreg(X, beta)
        error = pred - y
        deriv = (1/m)*np.matmul(X.T, error)
        beta = beta - alpha * deriv
        J_storage[n] = compute_cost(X, y, beta)
    return beta, J_storage


#Function to run the model and return the predicted value of GHI
def get_predicted_response(gdp, mpi, ghi, gpi, wri):
    df_features, df_target = get_features_targets(df, feature_names, target_names)
    df_features = normalize(df_features)    
    X = prepare_feature(df_features)
    target = prepare_target(df_target)

    df_features_train, df_features_test, df_target_train, df_target_test = split_data(df_features, df_target, random_state = 1000, test_size = 0.3)

    # Change the features and the target to numpy array using the prepare functions
    X = prepare_feature(df_features_train)
    target = prepare_target(df_target_train)

    iterations = 1500
    alpha = 0.01
    beta = np.zeros((len(feature_names) + 1, 1))

    # Call the gradient_descent function
    beta, J_storage = gradient_descent(X, target, beta, alpha, iterations)    
    cols = ['GDP Per Capita','Global Happiness Index','Multidimensional Poverty Index','Global Peace Index','World Risk Index']
    df_features_test = pd.DataFrame([[gdp,ghi,mpi,gpi,wri]], columns=cols)
    pred = predict_value(df_features_test, beta)

    return pred

if __name__ == '__main__':
    app.run(debug=True)

