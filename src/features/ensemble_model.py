# src/features/ensemble_model.py
"""
Modelo Ensemble avançado combinando múltiplas técnicas de ML.
Integra Random Forest, XGBoost, e redes neurais simples.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

class EnsembleStagePredictor:
    """
    Modelo ensemble que combina múltiplos classificadores para
    predição robusta de estágios empresariais.
    """
    
    def __init__(self, use_neural_network: bool = True):
        self.use_nn = use_neural_network
        
        # Modelos base
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        )
        
        if self.use_nn:
            self.nn_model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance: Dict[str, float] = {}
        self.is_fitted = False
        
        # Pesos do ensemble (ajustados durante treino)
        self.model_weights = {
            'rf': 0.4,
            'gb': 0.35,
            'nn': 0.25
        }
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepara e engenheira features avançadas.
        """
        features = []
        feature_names = []
        
        # Features financeiras básicas
        financial_features = [
            'receita_media', 'despesa_media', 'fluxo_medio',
            'margem_media', 'receita_total', 'crescimento_medio',
            'volatilidade_receita', 'consistencia_fluxo'
        ]
        
        for feat in financial_features:
            if feat in df.columns:
                features.append(df[feat].fillna(0).values.reshape(-1, 1))
                feature_names.append(feat)
        
        # Features temporais
        if 'idade_anos' in df.columns:
            features.append(df['idade_anos'].fillna(5).values.reshape(-1, 1))
            feature_names.append('idade_anos')
            
        if 'maturidade_relativa' in df.columns:
            features.append(df['maturidade_relativa'].fillna(0.5).values.reshape(-1, 1))
            feature_names.append('maturidade_relativa')
        
        # Features categóricas (one-hot encoding simplificado)
        if 'setor' in df.columns:
            setor_dummies = pd.get_dummies(df['setor'], prefix='setor')
            for col in setor_dummies.columns:
                features.append(setor_dummies[col].values.reshape(-1, 1))
                feature_names.append(col)
                
        if 'porte' in df.columns:
            porte_map = {'MICRO': 0, 'PEQUENA': 1, 'MEDIA': 2, 'GRANDE': 3}
            porte_encoded = df['porte'].map(porte_map).fillna(0)
            features.append(porte_encoded.values.reshape(-1, 1))
            feature_names.append('porte_encoded')
        
        # Features de tendência
        trend_features = []
        if 'receita_trend' in df.columns:
            trend_features.append(df['receita_trend'].fillna(0).values.reshape(-1, 1))
            feature_names.append('receita_trend')
            
        if 'fluxo_trend' in df.columns:
            trend_features.append(df['fluxo_trend'].fillna(0).values.reshape(-1, 1))
            feature_names.append('fluxo_trend')
        
        if trend_features:
            features.extend(trend_features)
        
        # Features de relacionamento (se disponível)
        if 'diversificacao_contrapartes' in df.columns:
            features.append(df['diversificacao_contrapartes'].fillna(0).values.reshape(-1, 1))
            feature_names.append('diversificacao_contrapartes')
            
        if 'canais_utilizados' in df.columns:
            features.append(df['canais_utilizados'].fillna(1).values.reshape(-1, 1))
            feature_names.append('canais_utilizados')
        
        # Combinar todas as features
        X = np.hstack(features) if features else np.zeros((len(df), 1))
        
        return X, feature_names
    
    def create_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Cria labels baseados em regras de negócio aprimoradas.
        """
        labels = []
        
        for _, row in df.iterrows():
            # Extrair métricas relevantes
            idade = row.get('idade_anos', 5)
            receita_media = row.get('receita_media', 0)
            fluxo_medio = row.get('fluxo_medio', 0)
            crescimento = row.get('crescimento_medio', 0)
            consistencia = row.get('consistencia_fluxo', 0.5)
            margem = row.get('margem_media', 0)
            
            # Regras aprimoradas de classificação
            if idade < 2 or (idade < 3 and receita_media < 50000):
                stage = "Início"
            elif (crescimento > 0.1 and consistencia > 0.6 and 
                  fluxo_medio > 0 and margem > 0.05):
                stage = "Crescimento"
            elif (abs(crescimento) < 0.05 and consistencia > 0.7 and 
                  fluxo_medio > 0 and idade > 5):
                stage = "Maturidade"
            elif fluxo_medio < -5000 or (crescimento < -0.1 and consistencia < 0.4):
                stage = "Declínio"
            else:
                stage = "Reestruturação"
                
            labels.append(stage)
            
        return np.array(labels)
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Treina o ensemble de modelos.
        """
        # Preparar dados
        X, feature_names = self.prepare_features(df)
        y = self.create_labels(df)
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Codificar labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Validação cruzada para ajustar pesos
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Treinar Random Forest
        rf_scores = cross_val_score(self.rf_model, X_scaled, y_encoded, cv=cv)
        self.rf_model.fit(X_scaled, y_encoded)
        
        # Treinar Gradient Boosting
        gb_scores = cross_val_score(self.gb_model, X_scaled, y_encoded, cv=cv)
        self.gb_model.fit(X_scaled, y_encoded)
        
        nn_scores = np.array([0.8])  # Default se não usar NN
        if self.use_nn:
            nn_scores = cross_val_score(self.nn_model, X_scaled, y_encoded, cv=cv)
            self.nn_model.fit(X_scaled, y_encoded)
        
        # Ajustar pesos baseado na performance
        total_score = rf_scores.mean() + gb_scores.mean() + nn_scores.mean()
        self.model_weights['rf'] = rf_scores.mean() / total_score
        self.model_weights['gb'] = gb_scores.mean() / total_score
        self.model_weights['nn'] = nn_scores.mean() / total_score if self.use_nn else 0
        
        # Normalizar pesos
        weight_sum = sum(self.model_weights.values())
        for key in self.model_weights:
            self.model_weights[key] /= weight_sum
        
        # Calcular feature importance (média ponderada)
        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_
        
        combined_importance = (
            rf_importance * self.model_weights['rf'] +
            gb_importance * self.model_weights['gb']
        )
        
        self.feature_importance = dict(zip(feature_names, combined_importance))
        self.is_fitted = True
        
        return {
            'rf_score': rf_scores.mean(),
            'gb_score': gb_scores.mean(),
            'nn_score': nn_scores.mean() if self.use_nn else None,
            'model_weights': self.model_weights,
            'feature_importance': self.feature_importance,
            'classes': list(self.label_encoder.classes_)
        }
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predição com probabilidades do ensemble.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        # Obter probabilidades de cada modelo
        rf_proba = self.rf_model.predict_proba(X_scaled)
        gb_proba = self.gb_model.predict_proba(X_scaled)
        
        # Combinar probabilidades com pesos
        ensemble_proba = (
            rf_proba * self.model_weights['rf'] +
            gb_proba * self.model_weights['gb']
        )
        
        if self.use_nn:
            nn_proba = self.nn_model.predict_proba(X_scaled)
            ensemble_proba += nn_proba * self.model_weights['nn']
        
        return ensemble_proba
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predição final com confiança e análise de incerteza.
        """
        proba = self.predict_proba(df)
        
        # Classe predita
        y_pred_encoded = np.argmax(proba, axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Confiança (probabilidade máxima)
        confidence = np.max(proba, axis=1)
        
        # Entropia (medida de incerteza)
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        max_entropy = np.log(len(self.label_encoder.classes_))
        uncertainty = entropy / max_entropy  # Normalizado 0-1
        
        # Segunda classe mais provável
        sorted_proba = np.sort(proba, axis=1)
        second_best_prob = sorted_proba[:, -2] if proba.shape[1] > 1 else np.zeros(len(df))
        
        results = pd.DataFrame({
            'ID': df['ID'],
            'estagio_predito': y_pred,
            'confianca': confidence,
            'incerteza': uncertainty,
            'segunda_opcao_prob': second_best_prob,
            'modelo_ensemble': True
        })
        
        # Adicionar probabilidades de cada classe
        for i, class_name in enumerate(self.label_encoder.classes_):
            results[f'prob_{class_name}'] = proba[:, i]
        
        return results
    
    def explain_prediction(self, 
                          df_single: pd.DataFrame,
                          top_n_features: int = 5) -> Dict[str, Any]:
        """
        Explica uma predição individual.
        """
        if len(df_single) != 1:
            raise ValueError("Forneça exatamente uma empresa para explicação.")
        
        # Predição
        pred_result = self.predict(df_single)
        
        # Features mais importantes
        X, feature_names = self.prepare_features(df_single)
        X_scaled = self.scaler.transform(X)
        
        # Contribuição de cada feature (simplificado)
        feature_values = dict(zip(feature_names, X_scaled[0]))
        
        # Top features por importância
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n_features]
        
        explanation = {
            'prediction': pred_result.iloc[0].to_dict(),
            'top_features': sorted_features,
            'feature_values': {k: feature_values.get(k, 0) for k, _ in sorted_features},
            'model_weights': self.model_weights
        }
        
        return explanation

