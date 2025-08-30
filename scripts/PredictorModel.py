import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, f1_score)
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

class ResponsibleIncomePredictor:
    """
    Modelo Random Forest para predicción de ingresos con enfoque en IA Responsable
    - Monitoreo de sesgos algorítmicos
    - Métricas de equidad por grupos demográficos
    - Interpretabilidad del modelo
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_and_prepare_data(self, filepath='data/census_income_clean.csv'):
        """
        Carga y prepara los datos basándose en el análisis EDA
        """
        print("🔄 Cargando datos...")
        data = pd.read_csv(filepath)

        # Variables seleccionadas basadas en análisis EDA
        # Top correlaciones/asociaciones identificadas
        selected_features = [
            # Variables numéricas (ordenadas por correlación descendente)
            'education-num',    # r = 0.333 (mayor correlación)
            'age',             # r = 0.230
            'hours-per-week',  # r = 0.228
            'capital-gain',    # r = 0.223
            'capital-loss',    # r = 0.148

            # Variables categóricas (ordenadas por V de Cramér descendente)
            'relationship',    # V = 0.454 (mayor asociación)
            'marital-status',  # V = 0.448
            'education',       # V = 0.365
            'occupation',      # V = 0.347
            'sex',            # V = 0.215 (incluida por equidad)
            'workclass',      # V = 0.174
            'race',           # V = 0.099 (incluida por equidad)
            'native-country'  # V = 0.092 (incluida por equidad)
        ]

        print(f"📊 Variables seleccionadas: {len(selected_features)}")

        # Preparar X y y
        X = data[selected_features].copy()
        # 1. Binarizar income: 1 si >50K, 0 si <=50K
        y = (data['income'] == '>50K').astype(int)

        print(f"✅ Distribución objetivo - 0 (<=50K): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
        print(f"✅ Distribución objetivo - 1 (>50K): {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")

        return X, y, selected_features

    def encode_categorical_variables(self, X_train, X_test):
        """
        Codifica variables categóricas usando LabelEncoder
        """
        print("🔄 Codificando variables categóricas...")

        categorical_cols = X_train.select_dtypes(include=['object']).columns
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()

        for col in categorical_cols:
            le = LabelEncoder()
            # Fit en train, transform en ambos
            X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
            X_test_encoded[col] = le.transform(X_test[col].astype(str))
            self.label_encoders[col] = le

            print(f"   {col}: {len(le.classes_)} categorías únicas")

        return X_train_encoded, X_test_encoded

    def train_model(self, X, y, test_size=0.2, balance_classes=True):
        """
        Entrena el modelo Random Forest con las mejores prácticas
        """
        print("🎯 Iniciando entrenamiento del modelo...")

        # División estratificada
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y
        )

        print(f"📊 Tamaño entrenamiento: {self.X_train.shape}")
        print(f"📊 Tamaño prueba: {self.X_test.shape}")

        # Codificar variables categóricas
        X_train_encoded, X_test_encoded = self.encode_categorical_variables(
            self.X_train, self.X_test
        )

        # Configurar balanceo de clases si es necesario
        class_weight = 'balanced' if balance_classes else None

        # Configuración optimizada de Random Forest
        rf_params = {
            'n_estimators': 200,        # Más árboles para mayor estabilidad
            'max_depth': 15,            # Profundidad controlada para evitar sobreajuste
            'min_samples_split': 50,    # Mayor muestra mínima para divisiones
            'min_samples_leaf': 20,     # Hojas más pobladas
            'max_features': 'sqrt',     # Reduce correlación entre árboles
            'class_weight': class_weight,
            'random_state': self.random_state,
            'n_jobs': -1               # Paralelización
        }

        print("🌲 Configuración Random Forest:")
        for param, value in rf_params.items():
            print(f"   {param}: {value}")

        # Entrenar modelo
        self.model = RandomForestClassifier(**rf_params)
        self.model.fit(X_train_encoded, self.y_train)

        # Guardar datos codificados para evaluación
        self.X_train_encoded = X_train_encoded
        self.X_test_encoded = X_test_encoded
        self.feature_names = list(X_train_encoded.columns)

        print("✅ Modelo entrenado exitosamente!")

        return self.model

    def evaluate_model(self):
        """
        Evaluación completa del modelo con múltiples métricas
        """
        print("📊 EVALUACIÓN DEL MODELO")
        print("=" * 50)

        # Predicciones
        y_pred = self.model.predict(self.X_test_encoded)
        y_pred_proba = self.model.predict_proba(self.X_test_encoded)[:, 1]

        # Métricas principales
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc_roc = roc_auc_score(self.y_test, y_pred_proba)

        print(f"🎯 Accuracy: {accuracy:.4f}")
        print(f"🎯 F1-Score: {f1:.4f}")
        print(f"🎯 AUC-ROC: {auc_roc:.4f}")
        print()

        # Reporte de clasificación detallado
        print("📈 REPORTE DE CLASIFICACIÓN:")
        print(classification_report(self.y_test, y_pred,
                                  target_names=['<=50K', '>50K']))

        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)

        # Visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Matriz de Confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('Matriz de Confusión', fontweight='bold')
        axes[0,0].set_xlabel('Predicción')
        axes[0,0].set_ylabel('Real')
        axes[0,0].set_xticklabels(['<=50K', '>50K'])
        axes[0,0].set_yticklabels(['<=50K', '>50K'])

        # 2. Curva ROC
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        axes[0,1].plot(fpr, tpr, color='darkorange', lw=2,
                      label=f'ROC curve (AUC = {auc_roc:.3f})')
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].set_xlim([0.0, 1.0])
        axes[0,1].set_ylim([0.0, 1.05])
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('Curva ROC', fontweight='bold')
        axes[0,1].legend(loc="lower right")

        # 3. Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        axes[1,0].plot(recall, precision, color='darkgreen', lw=2)
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Curva Precision-Recall', fontweight='bold')

        # 4. Importancia de Características
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        top_10_features = feature_importance.head(10)
        axes[1,1].barh(range(len(top_10_features)), top_10_features['importance'])
        axes[1,1].set_yticks(range(len(top_10_features)))
        axes[1,1].set_yticklabels(top_10_features['feature'])
        axes[1,1].set_xlabel('Importancia')
        axes[1,1].set_title('Top 10 - Importancia de Características', fontweight='bold')
        axes[1,1].invert_yaxis()

        plt.tight_layout()
        plt.show()

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    def analyze_bias_by_groups(self, demographic_vars=['sex', 'race', 'native-country']):
        """
        Análisis de sesgos algorítmicos por grupos demográficos
        """
        print("\n🚨 ANÁLISIS DE SESGOS ALGORÍTMICOS")
        print("=" * 50)

        y_pred = self.model.predict(self.X_test_encoded)
        y_pred_proba = self.model.predict_proba(self.X_test_encoded)[:, 1]

        # Agregar predicciones al conjunto de test original
        test_with_predictions = self.X_test.copy()
        test_with_predictions['y_true'] = self.y_test
        test_with_predictions['y_pred'] = y_pred
        test_with_predictions['y_pred_proba'] = y_pred_proba

        bias_results = {}

        for var in demographic_vars:
            if var in test_with_predictions.columns:
                print(f"\n🔍 Variable: {var.upper()}")

                # Métricas por grupo
                group_metrics = []
                for group in test_with_predictions[var].unique():
                    group_data = test_with_predictions[test_with_predictions[var] == group]

                    if len(group_data) > 10:  # Solo grupos con suficientes datos
                        accuracy = accuracy_score(group_data['y_true'], group_data['y_pred'])
                        f1 = f1_score(group_data['y_true'], group_data['y_pred'])

                        # Tasa de predicciones positivas
                        positive_pred_rate = (group_data['y_pred'] == 1).mean()
                        # Tasa real positiva
                        true_positive_rate = (group_data['y_true'] == 1).mean()

                        group_metrics.append({
                            'group': group,
                            'size': len(group_data),
                            'accuracy': accuracy,
                            'f1_score': f1,
                            'pred_positive_rate': positive_pred_rate,
                            'true_positive_rate': true_positive_rate,
                            'bias_ratio': positive_pred_rate / true_positive_rate if true_positive_rate > 0 else np.inf
                        })

                # Convertir a DataFrame y mostrar
                df_metrics = pd.DataFrame(group_metrics)
                df_metrics = df_metrics.sort_values('pred_positive_rate', ascending=False)

                print(f"{'Grupo':<20} {'n':<6} {'Acc':<6} {'F1':<6} {'Pred+%':<7} {'Real+%':<7} {'Sesgo':<6}")
                print("-" * 65)

                for _, row in df_metrics.iterrows():
                    bias_flag = "⚠️" if abs(1 - row['bias_ratio']) > 0.2 else "✅"
                    print(f"{row['group']:<20} {row['size']:<6d} "
                          f"{row['accuracy']:<6.3f} {row['f1_score']:<6.3f} "
                          f"{row['pred_positive_rate']:<7.3f} {row['true_positive_rate']:<7.3f} "
                          f"{bias_flag}")

                bias_results[var] = df_metrics

        return bias_results

    def hyperparameter_tuning(self, X, y, cv_folds=5):
        """
        Optimización de hiperparámetros usando GridSearchCV
        """
        print("🔧 OPTIMIZACIÓN DE HIPERPARÁMETROS")
        print("=" * 50)

        # División inicial
        X_train_tune, X_test_tune, y_train_tune, y_test_tune = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Codificar variables
        X_train_encoded, X_test_encoded = self.encode_categorical_variables(
            X_train_tune, X_test_tune
        )

        # Grid de parámetros para optimizar
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [20, 50, 100],
            'min_samples_leaf': [10, 20, 50],
            'class_weight': ['balanced', None]
        }

        print(f"🔍 Evaluando {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['class_weight'])} combinaciones...")

        # GridSearchCV con validación cruzada
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)

        grid_search = GridSearchCV(
            rf, param_grid, cv=cv_folds,
            scoring='f1',  # Optimizar F1-score por el desbalance
            n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train_encoded, y_train_tune)

        print("✅ Mejores parámetros encontrados:")
        for param, value in grid_search.best_params_.items():
            print(f"   {param}: {value}")

        print(f"\n🎯 Mejor F1-Score (CV): {grid_search.best_score_:.4f}")

        return grid_search.best_params_, grid_search.best_score_

predictor = ResponsibleIncomePredictor(random_state=42)
X, y, feature_names = predictor.load_and_prepare_data()
print(f"\nRESUMEN DEL DATASET:")
print(f"   Filas: {X.shape[0]:,}")
print(f"   Características: {X.shape[1]}")
print(f"   Desbalance: {(y==0).sum()/(y==1).sum():.2f}:1")

model = predictor.train_model(X, y, balance_classes=True)

results = predictor.evaluate_model()
bias_analysis = predictor.analyze_bias_by_groups()
top_features = results['feature_importance'].head(10)
for i, (_, row) in enumerate(top_features.iterrows(), 1):
    print(f"   {i:2d}. {row['feature']:<20}: {row['importance']:.4f}")


top_features = results['feature_importance'].head(10)
for i, (_, row) in enumerate(top_features.iterrows(), 1):
    print(f"   {i:2d}. {row['feature']:<20}: {row['importance']:.4f}")