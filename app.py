import ee
import geemap
from datetime import datetime, timedelta
import pandas as pd

import dateparser
import stanza
from datetime import datetime, timedelta

import geemap.foliumap as geemap
import nltk
nltk.download('punkt')
import pandas as pd
from nltk.tokenize import word_tokenize
from difflib import get_close_matches
import spacy



import time


service_account = 'isronrsc@isro-407105.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'isro-407105-31fe627b6f09.json')
ee.Initialize(credentials)


def date_parser(text):
    # Download and set up the neural pipeline for English
    stanza.download('en')
    nlp = stanza.Pipeline('en')

    # Process the input text using Stanza NLP
    doc = nlp(text)

    # Extract DATE entities
    ents = []
    for sent in doc.sentences:
        for ent in sent.ents:
            if ent.type == 'DATE':
                ents.append(ent.text)

    # Parse the dates using dateparser
    
    parsed_dates = [dateparser.parse(date) for date in ents]

    # Sort the parsed dates in ascending order
    sorted_dates = sorted(parsed_dates)

    # Set the smallest date as the start_date and the largest date as the end_date
    start_date = sorted_dates[0].strftime('%Y-%m-%d') if sorted_dates else None
    end_date = sorted_dates[-1].strftime('%Y-%m-%d') if sorted_dates else None

    # If both start_date and end_date are the same, set end_date to today's date
    if start_date == end_date and start_date is not None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # If both start_date and end_date are None, set end_date to today's date and start_date to past 14 days
    if start_date is None and end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')

    return start_date, end_date   




import spacy.cli
spacy.cli.download("en_core_web_trf")

class LocationExtractor:

    def __init__(self, csv_file_path, threshold=0.8):
        # Load the English language model for spaCy
        self.nlp = spacy.load('en_core_web_trf')
        # Load the CSV file for fuzzy matching
        self.df = pd.read_csv(csv_file_path)
        self.threshold = threshold

    def fuzzy_match(self, token):
        # Use difflib to find the closest match
        matches = get_close_matches(token, self.df['ROI_Name'].str.lower().tolist(), n=1, cutoff=self.threshold)

        # Check if there is a match
        if matches:
            return matches[0]
        else:
            return None

    def extract_entities(self, user_input):
        # Convert user input to lowercase
        user_input_lower = user_input.lower()
        # Tokenize the user input
        tokens = word_tokenize(user_input_lower)

        # Initialize variables to store the best matching entity
        best_entity = None

        # Iterate through the tokens and perform fuzzy matching
        for token in tokens:
            matched_entity = self.fuzzy_match(token)
            if matched_entity:
                best_entity = matched_entity
                break  # Stop after finding the first match

        # Return the best matching entity
        return best_entity






current_date_time = datetime.now()
class MapVisualizer:
    def __init__(self):
        self.sar_collection = None
        self.selected_roi = None
        self.start_date = None
        self.end_date = None

    def import_and_add_layers(self, asset_id, predefined_layers=None):
        shp = ee.FeatureCollection(asset_id)

        if predefined_layers:
            shp = shp.map(lambda feature: feature.set(predefined_layers))

        return shp

    def add_sar_layer_to_roi(self, shapefile, start_date, end_date, map_obj):
        sar_collection = self.load_sar_collection(start_date, end_date)

        sar_vv = sar_collection.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')).mean().clip(shapefile.geometry())

        map_obj.addLayer(sar_vv, {'bands': ['VV'], 'min': -20, 'max': 0, 'gamma': 1.4}, 'Clipped SAR (VV) Layer')

        return sar_vv

    def load_sar_collection(self, start_date, end_date):
        sar_collection = ee.ImageCollection('COPERNICUS/S1_GRD')\
            .filterDate(ee.Date(start_date), ee.Date(end_date))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
            .filter(ee.Filter.eq('instrumentMode', 'IW'))

        self.sar_collection = sar_collection
        return sar_collection

    def calculate_water_spread(self, image, threshold):
        water_mask = image.lt(threshold)

        water_area_m2 = water_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=self.selected_roi.geometry(),
            scale=30
        ).getInfo()['VV']

        water_area_km2 = water_area_m2 / 1e6

        return water_area_km2

    def calculate_yearly_water_spread(self, image_collection, threshold):
        yearly_water_spread = []

        for year in range(self.start_date.year, self.end_date.year + 1):
            start_date = f"{year}-01-01"
            end_date = f"{year + 1}-01-01"

            year_collection = image_collection.filterDate(ee.Date(start_date), ee.Date(end_date))
            yearly_water_spread.append(self.calculate_water_spread(year_collection.mean(), threshold))

        return yearly_water_spread

    def calculate_max_water_spread(self, selected_roi_name):
        # Set a large date range to cover all available data
        start_date_max = "2015-01-01"
        end_date_max = "2023-10-01"
        sar_collection_max = self.load_sar_collection(start_date_max, end_date_max)

        # Calculate maximum water spread in the ROI
        sar_vv_max = self.add_sar_layer_to_roi(self.selected_roi, start_date_max, end_date_max, geemap.Map())
        max_water_spread = self.calculate_water_spread(sar_vv_max, -15)

        return max_water_spread

    def compare_water_spread(self, water_spread_user_input, max_water_spread):
        conclusion = ""
        if water_spread_user_input > max_water_spread:
            conclusion += f'Water spread increased in the user input duration by {water_spread_user_input - max_water_spread:.2f} square kilometers.\n'
        elif water_spread_user_input < max_water_spread:
            conclusion += f'Water spread decreased in the user input duration by {max_water_spread - water_spread_user_input:.2f} square kilometers.\n'
        else:
            conclusion += 'Water spread remained the same in the user input duration.\n'

        return conclusion

    def run_analysis(self, asset_ids, selected_roi_name, start_date, end_date, csv_file_path):
        conclusion = ""
        df = pd.read_csv("ISROP.csv")
        valid_roi_names = df['ROI_Name'].tolist()
        if selected_roi_name in valid_roi_names:
            selected_roi_index = valid_roi_names.index(selected_roi_name)
            self.selected_roi = self.import_and_add_layers(asset_ids[selected_roi_index])

            static_map = geemap.Map(width=800, height=600)
            sar_vv = self.add_sar_layer_to_roi(self.selected_roi, start_date, end_date, static_map)
            static_map.centerObject(self.selected_roi, 10)

            # Set the start_date and end_date attributes
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d")

            # Calculate water spread for the user input duration
            water_spread_user_input = round(self.calculate_water_spread(sar_vv, -15))
            conclusion += f'Water spread for user input duration ({selected_roi_name.capitalize()}): {water_spread_user_input:} square kilometers\n' + "\n"

            # Calculate maximum water spread in the ROI
            max_water_spread =round( self.calculate_max_water_spread(selected_roi_name))
            conclusion += f'Maximum water spread ({selected_roi_name.capitalize()}): {max_water_spread:} square kilometers\n' + "\n"

            # Compare water spread for user input duration with maximum water spread
            conclusion += self.compare_water_spread(water_spread_user_input, max_water_spread) + "\n"



        else:
            conclusion += "Invalid ROI name. Please enter a valid ROI name.\n" + "\n"

        return static_map, conclusion
    
csv_file_path = 'ISROP.csv'

df = pd.read_csv(csv_file_path)
asset_ids = df['ROI_path'].tolist()

# start_date = "2020-09-09"
#end_date = "2021-09-09"
#selected_roi_name = "kadam" 
#map_visualizer = MapVisualizer()
#static_map, conclusion = map_visualizer.run_analysis(asset_ids, selected_roi_name, start_date, end_date, csv_file_path)
#print(conclusion)
map_visualizer = MapVisualizer()

def main():
    # Set page title and favicon
    st.set_page_config(
        page_title="CHATGS",
        page_icon="🚀",
        layout="wide"
    )

    # Load placeholder logo images
    left_logo = "https://www.nrsc.gov.in/sites/default/files/inline-images/nrsc_logo_412023_new.png"
    right_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Indian_Space_Research_Organisation_Logo.svg/2119px-Indian_Space_Research_Organisation_Logo.svg.png"

    # Display the background color using st.markdown
    st.markdown(
        """
        <style>
        body {
            background-color: #000000;  /* Set the background color to black */
            color: #FFFFFF;  /* Set the text color to white */
        }
        .logo-container {
            display: flex;
            justify-content: space-between;
            position: absolute;
            top: 10px;
            width: 100%;
            padding: 10px;
            box-sizing: border-box;
        }
        .logo-container .left-logo img {
            max-width: 119%;  /* Adjust the max-width for the left logo */
            margin-top:7%;
        }
        .logo-container .right-logo img {
            max-width: 125px;  /* Adjust the max-width for the right logo */
            margin-top: 2%;
        }
        .center-title {
            text-align: center;
            font-size: 7em;  /* Adjust font size as needed */
            margin-top: 0%;  /* Adjust margin-top for spacing */
            margin-left: 19%;  
            margin-right: 32%;
            color: #168ef0;  /* Set the color for the title (e.g., teal) */
        }
        .section {
            margin-top:12%; /* Adjust the margin-top for spacing between sections */
            margin-bottom: -4%;
            padding: 1px; /* Adjust the padding for the section size */
            border-radius: 10px;  /* Add border-radius for rounded corners */
        }
        .section-content {
            color: #FFFFFF;  /* Set the color for the section content */
            font-size: 2em;  /* Increase the font size for the section content */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the logos and title in the first row
    st.markdown(
        f"""
        <div class="logo-container">
            <div class="left-logo">
                <img src="{left_logo}" alt="Left Logo">
            </div>
            <div class="center-title">
                CHATGS
            </div>
            <div class="right-logo">
                <img src="{right_logo}" alt="Right Logo">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    selected_page = st.sidebar.selectbox("Navigation", ["CHATGS", "About", "Tutorial"])

    if selected_page == "CHATGS":
        # Replace this section with your new code
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("\n\n")
        st.markdown('<h1 style="color: #d97e16; font-size: 2em; text-align: left;"> HI THERE </h1>', unsafe_allow_html=True)
        
        
        user_text = st.text_input ("Enter text:")

        location_extractor = LocationExtractor(csv_file_path)
        selected_roi_name = location_extractor.extract_entities(user_text)
        start_date, end_date = date_parser(user_text)
        

        if st.button("Run Analysis"):
            # Create an empty element for dynamic text updates
            progress_placeholder = st.empty()

            # Display initial message
            progress_placeholder.text("Thank you for using our application.......!")

            # Simulate a long-running process
            with st.spinner("Running analysis..."):
                for _ in range(5):
                    # Simulate processing steps
                    time.sleep(2)

                    # Update the text dynamically
                    progress_placeholder.text(f"The query asked is: {user_text}")

                    progress_placeholder.text("processing your query.......")

                    progress_placeholder.text("Analysis is going on using Google Earth Engine..... ")
                    progress_placeholder.text("Retrieving  the sentinel data....")

                    progress_placeholder.text("Please wait for the result.......")

            # Once the process is complete, update the text
            progress_placeholder.text("Analysis is complete!")

            # Run the analysis in the background
            static_map, conclusion = map_visualizer.run_analysis(
                asset_ids, selected_roi_name, start_date, end_date, csv_file_path
            )

            # Convert Earth Engine map to HTML code
            static_map.to_streamlit(height=500)

            # Display the analysis conclusion
            st.markdown(
            f"""
            <div style='background-color: #f68b1e; padding: 20px; border-radius: 15px;'>
                <h2 style='color: white;'>Result : </h2>
                <p style='color: white; font-size: 16px; line-height: 1.5;'>{conclusion}</p>
            </div>
            """,
            unsafe_allow_html=True,
            )

    elif selected_page == "About":
        # What is CHATGS section in the third row
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<h1 style="color: #d97e16; font-size: 2em; text-align: left;">What is CHATGS</h1>', unsafe_allow_html=True)
        
        st.markdown('<h1 style="color: #168ef0; margin-top:3%; font-size: 2.7em; text-align: left;">INTRODUCTION</h1>', unsafe_allow_html=True) 
        st.markdown('<h1 style="color: #fffdfa;  font-size: 1.5em; text-align: left;">CHATGS, which stands for geo spatial chat bot , is a cutting-edge application developed to provide users with an interactive and user-friendly interface  model.</h1>' ,unsafe_allow_html=True)
        
        st.markdown('<h1 style="color: #168ef0; padding:1%; font-size: 2.7em; text-align: left;">Purpose</h1>', unsafe_allow_html=True) 
        st.markdown('<h1 style="color:#fffdfa;  font-size: 1.5em; text-align: left;">The primary purpose of CHATGS is to enable users to engage in natural language conversations with the model, generating responses and insights based on the input provided. Whether you are seeking information, generating text, or simply having a chat, CHATGS is designed to assist and enhance your user experience.</h1>' ,unsafe_allow_html=True)
        
        st.markdown('<h1 style="color: #168ef0; padding:1%; font-size: 2.7em; text-align: left;">GET STARTED</h1>',unsafe_allow_html=True)
        st.markdown('<h1 style="color: #fffdfa;  font-size: 1.5em; text-align: left;">To begin using CHATGS, simply enter your text in the provided input box and click the "Run Analysis" button. The application will process your query, analyze it using Google Earth Engine, retrieve relevant data, and present you with the results.</h1>' ,unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    elif selected_page == "Tutorial":
        # Tutorial section in the fourth row
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<h1 style="color: #d97e16; font-size: 2em; text-align: left;">Tutorial</h1>', unsafe_allow_html=True)
        st.write("This section provides a step-by-step tutorial on how to use CHATGS. ")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
