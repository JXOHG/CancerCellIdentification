import pandas as pd

dataset = pd.read_csv('cancer.csv')
x = dataset.drop(columns=['diagnosis(1=m, 0=b)'])#data containing parameters without the actual result
y = dataset["diagnosis(1=m, 0=b)"]#this is the column containing the correct diagnosis

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

import tensorflow as tf
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256,input_shape = x_train.shape[1:], activation = 'sigmoid'))#input layer
model.add(tf.keras.layers.Dense(256,activation = 'sigmoid'))# process layer
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid')) #output layer
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.fit(x_train, y_train, epochs =(400))

model.evaluate(x_test, y_test)

prediction = model.predict(x_test)
print(x_test, prediction, x_test.shape )



#User interface

#variables to be used, initialized
radius_mean = 0
texture_mean = 0
perimeter_mean = 0
area_mean = 0
smoothness_mean = 0
compactness_mean = 0
concavity_mean = 0
concave_points_mean = 0
symmetry_mean = 0
fractal_dimension_mean = 0
radius_se = 0
texture_se = 0
preimeter_se = 0
area_se = 0
smoothness_se = 0
compactness_se = 0
concavity_se = 0
concave_points_se = 0
symmetry_se = 0
fractal_dimension_se = 0
radius_worst= 0 
texture_worst= 0 
perimeter_worst= 0 
area_worst= 0 
smoothness_worst= 0 
compactness_worst= 0 
concavity_worst= 0 
concave_points_worst= 0 
symmetry_worst= 0 
fractal_dimension_worst= 0 




"""
column = [ 
    [gui.Text('Enter cell mean radius:')] ,
    [gui.InputText('',size = (4,1), key = 'radius_mean')],
    [gui.Text("Enter mean texture")],
    [gui.InputText('',size = (4,1), key = 'texture_mean')],
    
    [gui.Text("Enter mean perimeter")],
    [gui.InputText('',size = (4,1), key = 'perimeter_mean')],
    [gui.Text("Enter mean area")],
    [gui.InputText('',size = (4,1), key = 'area_mean')],
    [gui.Text("Enter mean smoothness")],
    [gui.InputText('',size = (4,1), key = 'smoothness_mean')],
    [gui.Text("Enter mean compactness")],
    [gui.InputText('',size = (4,1), key = 'compactness_mean')],
    [gui.Text("Enter mean concavity")],
    [gui.InputText('',size = (4,1), key = 'concavity_mean')],
    [gui.Text("Enter mean concave pofloats")],
    [gui.InputText('',size = (4,1), key = 'concave_points_mean')],
    [gui.Text("Enter mean symmetry")],
    [gui.InputText('',size = (4,1), key = 'symmetry_mean')],
    [gui.Text("Enter mean fractal dimension")],
    [gui.InputText('',size = (4,1), key = 'fractal_dimension_mean')],

    [gui.Text('Enter cell se radius:')] ,
    [gui.InputText('',size = (4,1), key = 'radius_se')],
    [gui.Text("Enter se texture")],
    [gui.InputText('',size = (4,1), key = 'texture_se')],
    [gui.Text("Enter se perimeter")],
    [gui.InputText('',size = (4,1), key = 'perimeter_se')],
    [gui.Text("Enter se area")],
    [gui.InputText('',size = (4,1), key = 'area_se')],
    [gui.Text("Enter se smoothness")],
    [gui.InputText('',size = (4,1), key = 'smoothness_se')],
    [gui.Text("Enter se compactness")],
    [gui.InputText('',size = (4,1), key = 'compactness_se')],
    [gui.Text("Enter se concavity")],
    [gui.InputText('',size = (4,1), key = 'concavity_se')],
    [gui.Text("Enter se concave pofloats")],
    [gui.InputText('',size = (4,1), key = 'concave_points_se')],
    [gui.Text("Enter se symmetry")],
    [gui.InputText('',size = (4,1), key = 'symmetry_se')],
    [gui.Text("Enter se fractal dimension")],
    [gui.InputText('',size = (4,1), key = 'fractal_dimension_se')],
    

    [gui.Text('Enter cell worst radius:')] ,
    [gui.InputText('',size = (4,1), key = 'radius_worst')],
    [gui.Text("Enter worst texture")],
    [gui.InputText('',size = (4,1), key = 'texture_worst')],
    [gui.Text("Enter worst perimeter")],
    [gui.InputText('',size = (4,1), key = 'perimeter_worst')],
    [gui.Text("Enter worst area")],
    [gui.InputText('',size = (4,1), key = 'area_worst')],
    [gui.Text("Enter worst smoothness")],
    [gui.InputText('',size = (4,1), key = 'smoothness_worst')],
    [gui.Text("Enter worst compactness")],
    [gui.InputText('',size = (4,1), key = 'compactness_worst')],
    [gui.Text("Enter worst concavity")],
    [gui.InputText('',size = (4,1), key = 'concavity_worst')],
    [gui.Text("Enter worst concave pofloats")],
    [gui.InputText('',size = (4,1), key = 'concave_points_worst')],
    [gui.Text("Enter worst symmetry")],
    [gui.InputText('',size = (4,1), key = 'symmetry_worst')],
    [gui.Text("Enter worst fractal dimension")],
    [gui.InputText('',size = (4,1), key = 'fractal_dimension_worst')],
    
    [gui.Button("OK")],
    [gui.Button("Exit")]]
layout = [
    [gui.Column(column, scrollable = True, vertical_scroll_only = True)]
]
window = gui.Window('Cancer parameter input', layout, resizable=True)
"""
"""
while True:
    event, values = window.read()
    if event == gui.WIN_CLOSED or event == "Exit":
        break

    if event == "OK":
        radius_mean = float(values['radius_mean'])
        texture_mean = float(values['texture_mean'])
        perimeter_mean = float(values['perimeter_mean'])
        area_mean = float(values['area_mean'])
        smoothness_mean = float(values['smoothness_mean'])
        compactness_mean = float(values['compactness_mean'])
        concavity_mean = float(values['concavity_mean'])
        concave_points_mean = float(values['concave_points_mean'])
        symmetry_mean = float(values['symmetry_mean'])
        fractal_dimension_mean = float(values['fractal_dimension_mean'])
        radius_se = float(values['radius_se'])
        texture_se = float(values['texture_se'])
        preimeter_se = float(values['perimeter_se'])
        area_se = float(values['area_se'])
        smoothness_se = float(values['smoothness_se'])
        compactness_se = float(values['compactness_se'])
        concavity_se = float(values['concavity_se'])
        concave_points_se = float(values['concave_points_se'])
        symmetry_se = float(values['symmetry_se'])
        fractal_dimension_se = float(values['fractal_dimension_se'])
        radius_worst= float(values['radius_worst'])
        texture_worst= float(values['texture_worst']) 
        perimeter_worst= float(values['perimeter_worst']) 
        area_worst= float(values['area_worst']) 
        smoothness_worst= float(values['smoothness_worst']) 
        compactness_worst= float(values['compactness_worst']) 
        concavity_worst= float(values['concavity_worst']) 
        concave_points_worst= float(values['concave_points_worst']) 
        symmetry_worst= float(values['symmetry_worst']) 
        fractal_dimension_worst= float(values['fractal_dimension_worst']) 
        
window.close()
"""
csv_data = [
    radius_mean,
    texture_mean,
    perimeter_mean,
    area_mean ,
    smoothness_mean ,
    compactness_mean ,
    concavity_mean ,
    concave_points_mean ,
    symmetry_mean ,
    fractal_dimension_mean ,
    radius_se ,
    texture_se ,
    preimeter_se ,
    area_se ,
    smoothness_se ,
    compactness_se ,
    concavity_se ,
    concave_points_se ,
    symmetry_se ,
    fractal_dimension_se ,
    radius_worst,
    texture_worst,
    perimeter_worst,
    area_worst,
    smoothness_worst,
    compactness_worst,
    concavity_worst,
    concave_points_worst,
    symmetry_worst,
    fractal_dimension_worst,
]


result = model.predict()
print("Prediction:", result.shape)
