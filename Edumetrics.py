import numpy as np
import pandas as pd

# Load the student habits and performance dataset
df = pd.read_csv('student_habits_performance.csv')

print("Original Dataset:")
print(df.head(10))

# 1. Remove duplicate Student IDs
df = df.drop_duplicates(subset=['student_id'], keep='first')

# 2. Clean Student ID column
df['student_id'] = df['student_id'].astype(str).str.strip()

# 3. Clean Age column (Keep age between 16 and 100)
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df.loc[(df['age'] < 16) | (df['age'] > 100), 'age'] = np.nan

median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age).astype(int)

# 4. Clean Gender column
df['gender'] = df['gender'].astype(str).str.strip().str.title()

gender_map = {
    'Male': 'Male',
    'Female': 'Female',
    'Other': 'Other'
}

df['gender'] = df['gender'].map(gender_map)
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

# 5. Clean Study Hours Per Day
df['study_hours_per_day'] = pd.to_numeric(
    df['study_hours_per_day'], errors='coerce'
)

df.loc[
    (df['study_hours_per_day'] < 0) |
    (df['study_hours_per_day'] > 24),
    'study_hours_per_day'
] = np.nan

df['study_hours_per_day'] = df['study_hours_per_day'].fillna(
    df['study_hours_per_day'].median()
)

# 6. Clean Social Media Hours
df['social_media_hours'] = pd.to_numeric(
    df['social_media_hours'], errors='coerce'
)

df.loc[
    (df['social_media_hours'] < 0) |
    (df['social_media_hours'] > 24),
    'social_media_hours'
] = np.nan

df['social_media_hours'] = df['social_media_hours'].fillna(
    df['social_media_hours'].median()
)

# 7. Clean Netflix Hours
df['netflix_hours'] = pd.to_numeric(
    df['netflix_hours'], errors='coerce'
)

df.loc[
    (df['netflix_hours'] < 0) |
    (df['netflix_hours'] > 24),
    'netflix_hours'
] = np.nan

df['netflix_hours'] = df['netflix_hours'].fillna(
    df['netflix_hours'].median()
)

# 8. Clean Part-Time Job column
df['part_time_job'] = (
    df['part_time_job'].astype(str).str.strip().str.title()
)

df['part_time_job'] = df['part_time_job'].replace({
    'Y': 'Yes',
    'N': 'No',
    'True': 'Yes',
    'False': 'No'
})

df.loc[
    ~df['part_time_job'].isin(['Yes', 'No']),
    'part_time_job'
] = np.nan

df['part_time_job'] = df['part_time_job'].fillna(
    df['part_time_job'].mode()[0]
)

# 9. Clean Attendance Percentage
df['attendance_percentage'] = pd.to_numeric(
    df['attendance_percentage'], errors='coerce'
)

df.loc[
    (df['attendance_percentage'] < 0) |
    (df['attendance_percentage'] > 100),
    'attendance_percentage'
] = np.nan

df['attendance_percentage'] = df['attendance_percentage'].fillna(
    df['attendance_percentage'].median()
)

# 10. Clean Sleep Hours
df['sleep_hours'] = pd.to_numeric(
    df['sleep_hours'], errors='coerce'
)

df.loc[
    (df['sleep_hours'] < 0) |
    (df['sleep_hours'] > 24),
    'sleep_hours'
] = np.nan

df['sleep_hours'] = df['sleep_hours'].fillna(
    df['sleep_hours'].median()
)

# 11. Clean Diet Quality
df['diet_quality'] = (
    df['diet_quality'].astype(str).str.strip().str.title()
)

valid_diet = ['Poor', 'Fair', 'Good']

df.loc[
    ~df['diet_quality'].isin(valid_diet),
    'diet_quality'
] = np.nan

df['diet_quality'] = df['diet_quality'].fillna(
    df['diet_quality'].mode()[0]
)

# 12. Clean Exercise Frequency
df['exercise_frequency'] = pd.to_numeric(
    df['exercise_frequency'], errors='coerce'
)

df.loc[
    (df['exercise_frequency'] < 0) |
    (df['exercise_frequency'] > 7),
    'exercise_frequency'
] = np.nan

df['exercise_frequency'] = df['exercise_frequency'].fillna(
    df['exercise_frequency'].median()
).round().astype(int)

# 13. Clean Parental Education Level
df['parental_education_level'] = (
    df['parental_education_level']
    .astype(str).str.strip().str.title()
)

valid_education = ['High School', 'Bachelor', 'Master']

df.loc[
    ~df['parental_education_level'].isin(valid_education),
    'parental_education_level'
] = np.nan

df['parental_education_level'] = df[
    'parental_education_level'
].fillna(df['parental_education_level'].mode()[0])

# 14. Clean Internet Quality
df['internet_quality'] = (
    df['internet_quality'].astype(str).str.strip().str.title()
)

valid_internet = ['Poor', 'Average', 'Good']

df.loc[
    ~df['internet_quality'].isin(valid_internet),
    'internet_quality'
] = np.nan

df['internet_quality'] = df['internet_quality'].fillna(
    df['internet_quality'].mode()[0]
)

# 15. Clean Mental Health Rating (1 to 10)
df['mental_health_rating'] = pd.to_numeric(
    df['mental_health_rating'], errors='coerce'
)

df.loc[
    (df['mental_health_rating'] < 1) |
    (df['mental_health_rating'] > 10),
    'mental_health_rating'
] = np.nan

df['mental_health_rating'] = df[
    'mental_health_rating'
].fillna(df['mental_health_rating'].median()).round().astype(int)

# 16. Clean Extracurricular Participation
df['extracurricular_participation'] = (
    df['extracurricular_participation']
    .astype(str).str.strip().str.title()
)

df['extracurricular_participation'] = (
    df['extracurricular_participation'].replace({
        'Y': 'Yes',
        'N': 'No',
        'True': 'Yes',
        'False': 'No'
    })
)

df.loc[
    ~df['extracurricular_participation'].isin(['Yes', 'No']),
    'extracurricular_participation'
] = np.nan

df['extracurricular_participation'] = df[
    'extracurricular_participation'
].fillna(df['extracurricular_participation'].mode()[0])

# 17. Clean Exam Score (0 to 100)
df['exam_score'] = pd.to_numeric(
    df['exam_score'], errors='coerce'
)

df.loc[
    (df['exam_score'] < 0) |
    (df['exam_score'] > 100),
    'exam_score'
] = np.nan

df['exam_score'] = df['exam_score'].fillna(
    df['exam_score'].median()
)

# 18. Export Cleaned Dataset
df.to_csv('cleaned_student_habits_performance.csv', index=False)

print("\nCleaned Dataset:")
print(df.head(10))

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nCleaned dataset saved successfully!")