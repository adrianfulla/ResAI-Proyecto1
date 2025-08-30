import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pointbiserialr, mannwhitneyu, ttest_ind
import warnings
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

adult = fetch_ucirepo(id=2)

X = adult.data.features
y = adult.data.targets

objetivo_conteo = y['income'].value_counts()
objetivo_pct = y['income'].value_counts(normalize=True) * 100
for idx, (count, pct) in enumerate(zip(objetivo_conteo, objetivo_pct)):
    print(f"{objetivo_conteo.index[idx]}: {count:,} ({pct:.2f}%)")

y['income'] = y['income'].str.rstrip('.')

objetivo_conteo = y['income'].value_counts()
objetivo_pct = y['income'].value_counts(normalize=True) * 100
for idx, (count, pct) in enumerate(zip(objetivo_conteo, objetivo_pct)):
    print(f"{objetivo_conteo.index[idx]}: {count:,} ({pct:.2f}%)")

categorical_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
numerical_vars = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

data = pd.concat([X, y], axis=1)

missing = data.isnull().sum()
missing_pct = (missing / len(data)) * 100
missing_df = pd.DataFrame({
    'Variable': missing.index,
    'Valores Faltantes': missing.values,
    'Porcentaje (%)': missing_pct.values
}).sort_values('Valores Faltantes', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
objetivo_conteo.plot(kind='bar', ax=ax1)
ax1.set_title('Distribución de Ingresos', fontsize=14, fontweight='bold')
ax1.set_xlabel('Nivel de Ingresos')
ax1.set_ylabel('Frecuencia')
ax1.tick_params(axis='x', rotation=45)

# Gráfico de pastel
ax2.pie(objetivo_conteo.values, labels=objetivo_conteo.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('Proporción de Ingresos', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

minority_class = objetivo_conteo.min()
majority_class = objetivo_conteo.max()
print(majority_class / minority_class)

numerical_data = data[numerical_vars]
desc_stats = numerical_data.describe()

y_encoded = (data['income'] == '>50K').astype(int)
correlations = {}
statistical_tests = {}

for var in numerical_vars:
    # Point-biserial correlation
    corr, p_value = pointbiserialr(y_encoded, data[var])
    correlations[var] = {'correlation': corr, 'p_value': p_value}

    # Mann-Whitney U test (no asume normalidad)
    group_low = data[data['income'] == '<=50K'][var]
    group_high = data[data['income'] == '>50K'][var]
    statistic, p_val_test = mannwhitneyu(group_low, group_high, alternative='two-sided')
    statistical_tests[var] = {'statistic': statistic, 'p_value': p_val_test}

    # Interpretación
    strength = abs(corr)
    if strength < 0.1:
        interpretation = "muy débil"
    elif strength < 0.3:
        interpretation = "débil"
    elif strength < 0.5:
        interpretation = "moderada"
    else:
        interpretation = "fuerte"

    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

    print(f"   {var:15s}: r = {corr:6.3f} {significance} ({interpretation})")

sorted_vars = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
data_t = data.copy()
for i, (var, data_t) in enumerate(sorted_vars, 1):
    print(f"   {i}. {var}: |r| = {abs(data_t['correlation']):.3f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, var in enumerate(numerical_vars):
    ax = axes[i]

    # Box plot comparativo
    data_to_plot = [
        data[data['income'] == '<=50K'][var],
        data[data['income'] == '>50K'][var]
    ]

    bp = ax.boxplot(data_to_plot, labels=['<=50K', '>50K'], patch_artist=True)
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax.set_title(f'Distribución de {var} por Ingresos', fontweight='bold')
    ax.set_ylabel(var)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

for var in categorical_vars:
    print(f"\n{var.upper()}:")
    value_counts = data[var].value_counts()
    print(f"Categorías únicas: {len(value_counts)}")
    print(f"Top 5 categorías:")
    print(value_counts.head())

fig, axes = plt.subplots(4, 2, figsize=(20, 24))
axes = axes.ravel()

for i, var in enumerate(categorical_vars):
    ax = axes[i]

    # Tabla de contingencia
    contingency = pd.crosstab(data[var], data['income'])
    contingency_pct = pd.crosstab(data[var], data['income'],
                                normalize='index') * 100

    # Gráfico de barras apiladas
    contingency_pct.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'Proporción de Ingresos por {var}',
                fontsize=12, fontweight='bold')
    ax.set_xlabel(var)
    ax.set_ylabel('Porcentaje (%)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Ingresos')

plt.tight_layout()
plt.show()

def cramers_v_calc(x, y):
    """Calcula V de Cramér entre dos variables categóricas"""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

cramers_matrix = pd.DataFrame(index=categorical_vars + ['income'], columns=categorical_vars + ['income'])
all_categorical = categorical_vars + ['income']
cramers_matrix = pd.DataFrame(index=all_categorical, columns=all_categorical)

cramers_matrix = cramers_matrix.astype(float)

target_associations = cramers_matrix['income'].drop('income').sort_values(ascending=False)
for var, value in target_associations.items():
    if value < 0.1:
        strength = "muy débil"
    elif value < 0.2:
        strength = "débil"
    elif value < 0.4:
        strength = "moderada"
    else:
        strength = "fuerte"
    print(f"   {var:15s}: V = {value:.3f} ({strength})")

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(cramers_matrix, dtype=bool))
sns.heatmap(cramers_matrix, mask=mask, annot=True, cmap='coolwarm',
           square=True, fmt='.2f', cbar_kws={"shrink": .8})
plt.title('Matriz de V de Cramér - Asociación entre Variables Categóricas',
         fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

bias_results = {}

demographic_vars = ['sex', 'race', 'native-country']
for var in demographic_vars:
    print(f"\n📊 Variable: {var.upper()}")
    crosstab = pd.crosstab(data[var], data['income'], normalize='index') * 100

    # Mostrar solo categorías con >50 casos para mayor significancia estadística
    category_counts = data[var].value_counts()
    significant_categories = category_counts[category_counts >= 50].index

    filtered_crosstab = crosstab.loc[significant_categories]

    print("   % con ingresos >50K por categoría:")
    high_income_pct = filtered_crosstab['>50K'].sort_values(ascending=False)

    for category, pct in high_income_pct.items():
        count = category_counts[category]
        print(f"     {category:25s}: {pct:6.2f}% (n={count:,})")

    # Calcular disparidad (ratio max/min)
    max_pct = high_income_pct.max()
    min_pct = high_income_pct.min()
    disparity_ratio = max_pct / min_pct if min_pct > 0 else float('inf')

    print(f"   📈 Ratio de disparidad: {disparity_ratio:.2f}:1")
    if disparity_ratio > 2:
        print("   ⚠️  SESGO SIGNIFICATIVO DETECTADO")

    bias_results[var] = {
        'crosstab': filtered_crosstab,
        'disparity_ratio': disparity_ratio,
        'high_income_pct': high_income_pct
    }

socioeconomic_vars = ['education', 'workclass', 'occupation']
for var in socioeconomic_vars:
    print(f"\n📊 Variable: {var.upper()}")
    crosstab = pd.crosstab(data[var], data['income'], normalize='index') * 100

    high_income_pct = crosstab['>50K'].sort_values(ascending=False)

    print("   Top 5 categorías con mayor % de ingresos altos:")
    for category, pct in high_income_pct.head().items():
        count = data[var].value_counts()[category]
        print(f"     {category:30s}: {pct:6.2f}% (n={count:,})")

    print("   Bottom 3 categorías con menor % de ingresos altos:")
    for category, pct in high_income_pct.tail(3).items():
        count = data[var].value_counts()[category]
        print(f"     {category:30s}: {pct:6.2f}% (n={count:,})")

    bias_results[var] = {
        'high_income_pct': high_income_pct,
        'top_categories': high_income_pct.head(),
        'bottom_categories': high_income_pct.tail(3)
    }

family_vars = ['marital-status', 'relationship']

for var in family_vars:
    print(f"\n📊 Variable: {var.upper()}")
    crosstab = pd.crosstab(data[var], data['income'], normalize='index') * 100

    high_income_pct = crosstab['>50K'].sort_values(ascending=False)

    for category, pct in high_income_pct.items():
        count = data[var].value_counts()[category]
        print(f"     {category:25s}: {pct:6.2f}% (n={count:,})")

    bias_results[var] = {'high_income_pct': high_income_pct}

for var in ['age', 'hours-per-week', 'education-num']:
    print(f"\n📊 Variable: {var.upper()}")

    # Dividir en quintiles
    quintiles = pd.qcut(data[var], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    crosstab = pd.crosstab(quintiles, data['income'], normalize='index') * 100

    print("   % con ingresos >50K por quintil:")
    for quintile, pct in crosstab['>50K'].items():
        range_info = pd.qcut(data[var], q=5).cat.categories[crosstab.index.get_loc(quintile)]
        print(f"     {quintile}: {pct:6.2f}% (rango: {range_info})")

    bias_results[f"{var}_quintiles"] = crosstab['>50K']

intersections = [
    ('sex', 'race'),
    ('sex', 'education'),
    ('marital-status', 'sex'),
    ('workclass', 'education')
]

for var1, var2 in intersections:
    print(f"\n📊 Intersección: {var1.upper()} × {var2.upper()}")

    # Filtrar solo combinaciones con >20 casos
    intersection_counts = pd.crosstab([data[var1], data[var2]], data['income'])
    significant_combinations = intersection_counts[intersection_counts.sum(axis=1) >= 20]

    if len(significant_combinations) > 0:
        intersection_pct = pd.crosstab([data[var1], data[var2]],
                                       data['income'], normalize='index') * 100

        intersection_pct = intersection_pct.loc[significant_combinations.index]
        top_combinations = intersection_pct['>50K'].sort_values(ascending=False).head(5)

        print("   Top 5 combinaciones con mayor % de ingresos altos:")
        for (cat1, cat2), pct in top_combinations.items():
            count = intersection_counts.loc[(cat1, cat2)].sum()
            print(f"     {cat1} + {cat2}: {pct:6.2f}% (n={count})")


fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Género
if 'sex' in bias_results:
    bias_results['sex']['high_income_pct'].plot(kind='bar', ax=axes[0,0], color=['lightcoral', 'skyblue'])
    axes[0,0].set_title('% Ingresos >50K por Género', fontweight='bold')
    axes[0,0].tick_params(axis='x', rotation=0)

# Raza (top 8 categorías)
if 'race' in bias_results:
    bias_results['race']['high_income_pct'].plot(kind='bar', ax=axes[0,1], color='lightgreen')
    axes[0,1].set_title('% Ingresos >50K por Raza', fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=45)

# Educación (top 8 categorías)
if 'education' in bias_results:
    bias_results['education']['top_categories'].plot(kind='bar', ax=axes[1,0], color='orange')
    axes[1,0].set_title('Top Educación - % Ingresos >50K', fontweight='bold')
    axes[1,0].tick_params(axis='x', rotation=45)

# Estado civil
if 'marital-status' in bias_results:
    bias_results['marital-status']['high_income_pct'].plot(kind='bar', ax=axes[1,1], color='purple')
    axes[1,1].set_title('% Ingresos >50K por Estado Civil', fontweight='bold')
    axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

processed_data = data.copy()

for col in processed_data.columns:
    if processed_data[col].dtype == 'object':
        # Reemplazar '?' con NaN y luego con la moda
        processed_data[col] = processed_data[col].replace('?', np.nan)
        if processed_data[col].isnull().sum() > 0:
            mode_val = processed_data[col].mode()[0]
            processed_data[col].fillna(mode_val, inplace=True)

selected_features = [
    'age', 'workclass', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country'
]

X_processed = processed_data[selected_features].copy()
y_processed = processed_data['income'].copy()

X_processed.to_csv(f'data/X_processed.csv', index=False)
y_processed.to_csv(f'data/y_processed.csv', index=False)


complete_dataset = pd.concat([X_processed, y_processed], axis=1)
complete_dataset.to_csv(f'data/census_income_clean.csv', index=False)

metadata = {
'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
'original_shape': data.shape,
'processed_shape': complete_dataset.shape,
'selected_features': list(X_processed.columns),
'target_variable': 'income',
'target_classes': list(y_processed.unique()),
'categorical_variables': list(X_processed.select_dtypes(include=['object']).columns),
'numerical_variables': list(X_processed.select_dtypes(exclude=['object']).columns)
}

with open(f'data/processing_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)