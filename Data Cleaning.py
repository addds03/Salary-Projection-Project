import pandas as pd

df_gd = pd.read_csv('glassdoor_jobs.csv')

df_gd.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

df_gd_copy = df_gd.copy()

# Salary Parsing
df_gd_copy = df_gd_copy[df_gd_copy['Salary Estimate'] != '-1']

df_gd_copy['hourly'] =  df_gd_copy['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)

df_gd_copy['emp prov'] = df_gd_copy['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary' in x.lower() else 0)

salary = df_gd_copy['Salary Estimate'].apply(lambda x: x.split('(')[0])

minus_k = salary.apply(lambda x: x.replace('K','').replace('$',''))

min_hr = minus_k.apply(lambda x: x.lower().replace('per hour','').replace('employer provided salary:',''))

df_gd_copy['min_sal'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df_gd_copy['max_sal'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df_gd_copy['avg_sal'] = (df_gd_copy.min_sal + df_gd_copy.max_sal) / 2


# Company name
df_gd_copy['company_txt'] = df_gd_copy.apply(lambda x: x['Company Name'] if x['Rating'] == -1 else x['Company Name'][:-3], axis = 1)

# State Field
df_gd_copy['state'] = df_gd_copy['Location'].apply(lambda x:x.split(',')[1])
df_gd_copy['hq_loc'] = df_gd_copy.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)

# Age of the company
df_gd_copy['age_comp'] = df_gd_copy['Founded'].apply(lambda x: x if x == -1 else 2020 - x)

# Parsing Job desc
df_gd_copy['python_ys'] = df_gd_copy['Job Description'].apply(lambda x:1 if 'python' in x.lower() else 0)
df_gd_copy['excel_ys'] = df_gd_copy['Job Description'].apply(lambda x:1 if 'excel' in x.lower() else 0)
df_gd_copy['tableau_ys'] = df_gd_copy['Job Description'].apply(lambda x:1 if 'tableau' in x.lower() else 0)

df_gd = df_gd_copy.copy()

# Save file in csv
df_gd.to_csv('glassdoor_cleaned.csv', index=False)