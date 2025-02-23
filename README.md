College GPT - 

Table of Contents :
Introduction
Installation
Usage
API Reference
Troubleshooting

1. Introduction
The College Predictor AI/ML Model leverages advanced machine learning techniques to help students predict their chances of admission to various colleges(in this model particularly IITs). By analysing historical data (such as opening and closing ranks of previous years counselling), and other parameters (such as placement stats of last 3 years of each program, specific branch interest (if any), it provides data-driven insights, enabling aspirants to make well-informed decisions about their educational future.


2. Installation
pip install google.generative ai
pip install pandas
pip install requests
pip install streamlit

3. Usage
For obtaining most optimum results the prompt should follow the following guidelines :
the prompt must contain users’s category, category rank and gender.
If the user wants to select the college and program with better placement statistics (as of weighted data of last 3 years) then the user can specify to list the colleges according to placements in the prompt message.
It may also be possible that the user has a specific interest in some particular field, so the user can clearly specify the branch preference in the prompt message and only those programs would be listed that the user wants and the colleges would be determined by the previous years opening and closing rank data.

The shown list of the colleges can directly be downloaded or exported according to the usage of the user.

4. API Reference
“AIzaSyB_oi6MX3wsJUm1XB94NdaYxdVwqxNw_uo"

5. Troubleshooting
Common Issues with Google Gemini AI API Key
Invalid API Key
Ensure you have copied the API key correctly from Google Cloud Console.
Check for any unnecessary spaces or characters.
API Quota Exceeded
Verify your quota limits on the Google Cloud Console.
Consider upgrading your plan if you frequently exceed limits.
Network Issues
Ensure you have a stable internet connection.
Try using a VPN if API requests are being blocked in your region.
Unauthorized Access
Make sure your API key has the necessary permissions.
Verify that the API key is associated with the correct project.
Incorrect API Endpoint
Double-check the endpoint URL and ensure it matches the latest API documentation from Google.
