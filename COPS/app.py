import os
import re
import pandas as pd
import streamlit as st
import google.generativeai as genai

# --- Initialization and CSV Loading ---

# Load API key (ensure GEMINI_API_KEY is set in your environment or replace with your key)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="gemini-pro")

# Function to normalize strings: lower-case, strip spaces, replace hyphens with spaces
def normalize_str(s):
    return s.replace("-", " ").strip().lower()

# Load CSV files
file_path_main = "OG_Placement.csv"
file_path_rank = "JOSAA AIR rank data.csv"
df_main = pd.read_csv(file_path_main)
df_rank = pd.read_csv(file_path_rank)

# Ensure rank columns are numeric and clean up data
df_rank["Opening Rank"] = pd.to_numeric(df_rank["Opening Rank"], errors="coerce")
df_rank["Closing Rank"] = pd.to_numeric(df_rank["Closing Rank"], errors="coerce")
df_rank.dropna(subset=["Opening Rank", "Closing Rank"], inplace=True)

# Normalize column names and string values using the normalize_str function
df_rank.columns = df_rank.columns.str.strip()
df_main.columns = df_main.columns.str.strip()

df_rank["Seat Type"] = df_rank["Seat Type"].apply(normalize_str)
df_rank["Gender"] = df_rank["Gender"].str.strip()
df_rank["Institute"] = df_rank["Institute"].str.strip().str.lower()
df_rank["Academic Program Name"] = df_rank["Academic Program Name"].str.strip().str.lower()
df_main["Institute"] = df_main["Institute"].str.strip().str.lower()
df_main["Academic Program Name"] = df_main["Academic Program Name"].str.strip().str.lower()

# Merge datasets
df = pd.merge(df_rank, df_main, on=["Institute", "Academic Program Name"], how="inner")

# --- Compute Weighted Median CTC ---
for year in [2024, 2023, 2022]:
    df[f"Median CTC {year}"] = pd.to_numeric(df[f"Median CTC {year}"], errors="coerce")

df["Weighted Median CTC"] = (
    df["Median CTC 2024"] * 0.5 +
    df["Median CTC 2023"] * 0.3 +
    df["Median CTC 2022"] * 0.2
)

# --- Helper Functions ---
def map_gender(user_gender):
    # Map any input containing "female" to the expected female value,
    # and all other inputs (e.g., "male") to Gender-Neutral.
    if "female" in user_gender.strip().lower():
        return "Female-only (including Supernumerary)"
    else:
        return "Gender-Neutral"

def map_category(user_category):
    # Map common variants to standard values, now treating "obc" and "obc ncl" as "obc ncl".
    category = normalize_str(user_category)
    category_mapping = {
        "general": "open",
        "open": "open",
        "obc": "obc ncl",
        "obc ncl": "obc ncl"
    }
    return category_mapping.get(category, category)

def filter_colleges(category_rank, category, gender):
    """
    Filters colleges based on user's category rank and also considers a 10% margin above the closing rank.
    A new column "Margin" is added with values:
      - "highly probable": if the user's rank falls within the official closing rank.
      - "marginally probably": if the user's rank exceeds the closing rank but is within 10% of it.
    """
    category = map_category(category)
    gender = map_gender(gender)
    df_filtered = df[
        (df["Seat Type"] == category) &
        (df["Gender"] == gender) &
        (df["Opening Rank"] <= category_rank)
    ].copy()
    
    def qualify(row):
        closing = row["Closing Rank"]
        if category_rank <= closing:
            return "highly probable"
        elif category_rank <= closing * 1.1:
            return "marginally probably"
        else:
            return None

    df_filtered["Margin"] = df_filtered.apply(qualify, axis=1)
    df_filtered = df_filtered[df_filtered["Margin"].notnull()]
    return df_filtered

def sort_colleges(filtered_df, preference):
    if preference == "placement":
        if "Weighted Median CTC" in filtered_df.columns:
            # Sort in decreasing order (highest median CTC first)
            sorted_df = filtered_df.sort_values(by="Weighted Median CTC", ascending=False)
        else:
            st.error("Error: 'Weighted Median CTC' column not found!")
            return filtered_df
    elif preference == "branch":
        # Order in increasing order based on the following branch priority:
        # computer science, mathematics and computing, ai, data science, electronics,
        # electrical, chemical, mechanical, economics, civil, then the rest.
        branch_order = [
            "computer science",
            "mathematics and computing",
            "ai",
            "data science",
            "electronics",
            "electrical",
            "chemical",
            "mechanical",
            "economics",
            "civil"
        ]
        def get_branch_order(program):
            program = program.lower()
            for idx, branch in enumerate(branch_order):
                if branch in program:
                    return idx
            return len(branch_order)
        sorted_df = filtered_df.copy()
        sorted_df["Branch Order"] = sorted_df["Academic Program Name"].apply(get_branch_order)
        sorted_df = sorted_df.sort_values(by="Branch Order")
    else:
        sorted_df = filtered_df
    return sorted_df

# --- Streamlit Session State Setup ---
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_state" not in st.session_state:
    st.session_state.current_state = None
if "step" not in st.session_state:
    st.session_state.step = "input_details"  # Options: "input_details", "input_preference", "show_result"
if "result" not in st.session_state:
    st.session_state.result = None

# --- Streamlit UI ---
st.title("College Recommendation Chatbot with Gemini")
st.subheader("Conversation History")
# Only display user messages (filter out messages that start with "Bot")
for message in st.session_state.conversation_history:
    if message.startswith("User"):
        st.write(message)

# Step 1: Extract Rank, Category, and Gender
if st.session_state.step == "input_details":
    st.subheader("Step 1: Enter your details")
    user_input = st.text_input("Enter your query (include your rank, category, and any gender cues):")
    if st.button("Submit Details"):
        st.session_state.conversation_history.append("User: " + user_input)
        context = "\n".join(st.session_state.conversation_history[-4:])
        prompt = f"""
Below is the conversation so far:
{context}

Now, extract the following details from the latest user query:
1. Category Rank (integer)
2. Category (seat type, e.g., OPEN, OBC, SC, ST, EWS, etc.)
3. Gender (choose exactly one from: "Gender-Neutral" or "Female-only (including Supernumerary)").
Note: If the user query contains any feminine cues (such as "panty", "menstruational periods", "saari", etc.), classify it as "Female-only (including Supernumerary)".

Example Output Format:
Category Rank: <rank>
Category: <category>
Gender: <gender>

User Query:
"{user_input}"
"""
        response = model.generate_content(prompt)
        category_rank_match = re.search(r"Category Rank: (\d+)", response.text)
        category_match = re.search(r"Category: ([\w-]+)", response.text)
        gender_match = re.search(r"Gender: ([\w\s\(\)-]+)", response.text)
        if category_rank_match and category_match and gender_match:
            new_state = {
                "category_rank": int(category_rank_match.group(1)),
                "category": category_match.group(1).strip(),
                "gender": gender_match.group(1).strip()
            }
            st.session_state.current_state = new_state
            st.session_state.conversation_history.append(
                f"Bot (extraction): Extracted - Category Rank: {new_state['category_rank']}, Category: {new_state['category']}, Gender: {new_state['gender']}"
            )
            st.success(f"Extracted - Category Rank: {new_state['category_rank']}, Category: {new_state['category']}, Gender: {new_state['gender']}")
            st.session_state.step = "input_preference"
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
        else:
            if st.session_state.current_state is not None:
                new_state = st.session_state.current_state
                st.session_state.conversation_history.append(
                    f"Bot: Using previous state - Category Rank: {new_state['category_rank']}, Category: {new_state['category']}, Gender: {new_state['gender']}"
                )
                st.success(f"Using previous state - Category Rank: {new_state['category_rank']}, Category: {new_state['category']}, Gender: {new_state['gender']}")
                st.session_state.step = "input_preference"
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
            else:
                st.error("Error: Could not extract necessary information from Gemini response and no previous state exists.")

# Step 2: Extract Preference and Display Results
elif st.session_state.step == "input_preference":
    st.subheader("Step 2: Enter your preference")
    pref_input = st.text_input("Enter your preference (e.g., I would prefer placement over branch):")
    if st.button("Submit Preference"):
        st.session_state.conversation_history.append("User (preference): " + pref_input)
        pref_prompt = f"""
Extract the user's preference from the following query.
Output should be in the format:
Preference: <placement or branch>

User Query:
"{pref_input}"
"""
        pref_response = model.generate_content(pref_prompt)
        pref_match = re.search(r"Preference: (\w+)", pref_response.text, re.IGNORECASE)
        if pref_match:
            extracted_preference = pref_match.group(1).strip().lower()
        else:
            if "placement" in pref_input.lower():
                extracted_preference = "placement"
            elif "branch" in pref_input.lower():
                extracted_preference = "branch"
            else:
                extracted_preference = "placement"  # default fallback
        st.session_state.conversation_history.append("Bot (extraction): Extracted Preference: " + extracted_preference)
        st.success("Extracted Preference: " + extracted_preference)
        state = st.session_state.current_state
        result_df = filter_colleges(state['category_rank'], state['category'], state['gender'])
        if result_df.empty:
            msg = "No colleges found for your criteria. Please check the category, gender, and rank values."
            st.error(msg)
            st.session_state.conversation_history.append("Bot: " + msg)
        else:
            sorted_df = sort_colleges(result_df, extracted_preference)
            if extracted_preference == "placement":
                output_df = sorted_df[["Institute", "Academic Program Name", "Opening Rank", "Closing Rank", "Weighted Median CTC", "Margin"]]
            else:
                output_df = sorted_df[["Institute", "Academic Program Name", "Opening Rank", "Closing Rank", "Margin"]]
            st.session_state.result = output_df
            st.session_state.conversation_history.append("Bot (final output): " + output_df.to_string(index=False))
            st.session_state.step = "show_result"
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()

# Step 3: Show the Recommended Colleges
elif st.session_state.step == "show_result":
    st.subheader("Recommended Colleges")
    if st.session_state.result is not None:
        # Optionally, you can rename the "Margin" column header for display purposes
        display_df = st.session_state.result.rename(columns={"Margin": "Probable Qualification"})
        st.dataframe(display_df)
    if st.button("Restart Conversation"):
        st.session_state.conversation_history = []
        st.session_state.current_state = None
        st.session_state.step = "input_details"
        st.session_state.result = None
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

