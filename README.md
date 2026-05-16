# How To Use the Application 

1. Clone the repo `git clone https://github.com/lewinskie254/EPL-Predictions-2026.git` or the ssh equivalent.
2. Create a virtual Environment `python -m venv .venv`.
3. Install all Packages `pip install requirements.txt`. 
4. Activate the virtual environment `.venv\Scripts\activate`.
5. You can now use the model using this simple process. `python main.py -home "Home Team" -away "Away Team" -date "date in year-month-day"`.
6. E.g. `python main.py -home "Bunley" -away "Aston Villa" -date "2026-05-10"`.

From observing the last week's predictions, the model, which is only trained on data up to 2022, but tested and validated with data up to May 05, 2025, is performing extremely well. For the data science nerds out there, looking at the jupyter notebooks can help you see everything that was done, hopefully. The model uses an XGB.Classifier(), but the most innovative part was the feature engineering, which can be seen on the file `EPL_feature_engineering.ipynb`. Another suprising fact about this model is that it does not use team IDs, it simply uses Elo Scores, which were calculated using this simple formula: 

### **Expected Score (Before Match)**

$$
E_A = \frac{1}{1 + 10^{\frac{R_B - R_A}{400}}}
$$

Where:

- $E_A$ = expected probability of Team A winning  
- $R_A$ = rating of Team A  
- $R_B$ = rating of Team B  

### **Rating Update (After Match)**

$$
R_A' = R_A + K \cdot (S_A - E_A)
$$

Where:

- $R_A'$ = updated rating for Team A  
- $K$ = learning rate / sensitivity factor (typically 20–40)  
- $S_A$ = actual result for Team A  


Before making this entire Markdown unnecessarily long, Nerds, just check the notebooks. Other than that, if you have any questions, my profile provides all possible ways to reach me. 

**P.S. I will try to routinely update the model every weekend, but I freaking hate sports, I would not count on it that much.**
