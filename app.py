from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model pipeline
model = joblib.load('mobile_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input from form
        battery_power = float(request.form['battery_power'])
        blue = int(request.form['blue'])
        clock_speed = float(request.form['clock_speed'])
        dual_sim = int(request.form['dual_sim'])
        fc = float(request.form['fc'])
        four_g = int(request.form['four_g'])
        int_memory = int(request.form['int_memory'])
        m_dep = float(request.form['m_dep'])
        mobile_wt = int(request.form['mobile_wt'])
        n_cores = int(request.form['n_cores'])
        pc = int(request.form['pc'])
        px_height = int(request.form['px_height'])
        px_width = int(request.form['px_width'])
        ram = int(request.form['ram'])
        sc_h = int(request.form['sc_h'])
        sc_w = int(request.form['sc_w'])
        talk_time = int(request.form['talk_time'])
        three_g = int(request.form['three_g'])
        touch_screen = int(request.form['touch_screen'])
        wifi = int(request.form['wifi'])

        # Engineered features
        px_area = px_height * px_width
        screen_area = sc_h * sc_w
        build_score = mobile_wt / m_dep if m_dep != 0 else 0  # avoid division by zero

        # Create a DataFrame with correct column names
        input_df = pd.DataFrame([{
            'battery_power': battery_power,
            'blue': blue,
            'clock_speed': clock_speed,
            'dual_sim': dual_sim,
            'fc': fc,
            'four_g': four_g,
            'int_memory': int_memory,
            'm_dep': m_dep,
            'mobile_wt': mobile_wt,
            'n_cores': n_cores,
            'pc': pc,
            'ram': ram,
            'talk_time': talk_time,
            'three_g': three_g,
            'touch_screen': touch_screen,
            'wifi': wifi,
            'px_area': px_area,
            'screen_area': screen_area,
            'build_score': build_score
        }])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Return result
        return render_template('index.html', prediction_text=prediction)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
