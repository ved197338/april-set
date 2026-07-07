import json
import requests
from typing import Dict, Any, Optional
from config.manager import ConfigManager
from app_logging.logger import AprilLogger

class AIAssistant:
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.logger = AprilLogger.get_logger()
        self.provider = self.config_manager.get("ai.default_provider", "ollama")

    def analyze_dataset_prompt(self, inspection_data: Dict[str, Any], query: str) -> str:
        """Query LLM with dataset schema/statistics to answer questions, falling back to local advice."""
        prompt = self._build_prompt(inspection_data, query)

        response = ""
        if self.provider == "ollama":
            url = self.config_manager.get("ai.ollama_url", "http://localhost:11434")
            model = self.config_manager.get("ai.model_names.ollama", "llama3")
            response = self._call_ollama(url, model, prompt)

        elif self.provider == "gemini":
            key = self.config_manager.get("ai.gemini_api_key")
            model = self.config_manager.get("ai.model_names.gemini", "gemini-1.5-flash")
            if key:
                response = self._call_gemini(key, model, prompt)

        elif self.provider == "openai":
            key = self.config_manager.get("ai.openai_api_key")
            model = self.config_manager.get("ai.model_names.openai", "gpt-4o-mini")
            if key:
                response = self._call_openai(key, model, prompt)

        elif self.provider == "anthropic":
            key = self.config_manager.get("ai.anthropic_api_key")
            model = self.config_manager.get("ai.model_names.anthropic", "claude-3-5-sonnet-20240620")
            if key:
                response = self._call_anthropic(key, model, prompt)

        if response:
            return response

        self.logger.info("Using Local Expert Fallback System for dataset advice.")
        return self._generate_local_expert_advice(inspection_data, query)

    def _build_prompt(self, data: Dict[str, Any], query: str) -> str:
        summary = {
            "filename": data.get("filename"),
            "rows": data.get("rows"),
            "columns": data.get("columns"),
            "duplicates": data.get("duplicate_rows"),
            "missing_pct": data.get("overall_missing_pct"),
            "columns_list": [c["name"] for c in data.get("columns_info", [])],
            "target_candidates": data.get("target_candidates", []),
            "class_balance": data.get("class_balance", {})
        }

        return f"""
You are an expert Machine Learning Engineer and Data Scientist. 
You are analyzing the dataset '{summary['filename']}'. Here are its structural statistics:
- Number of Rows: {summary['rows']}
- Number of Columns: {summary['columns']}
- Duplicate Rows: {summary['duplicates']}
- Overall Missing Values (%): {summary['missing_pct']:.2f}%
- Column names: {", ".join(summary['columns_list'])}
- Inferred Target Candidates: {summary['target_candidates']}
- Primary Class Balance: {summary['class_balance']}

The user has asked: "{query}"

Provide a professional, concise, and highly technical markdown response. Avoid generic fluff.
"""

    def _call_ollama(self, base_url: str, model: str, prompt: str) -> str:
        try:
            url = f"{base_url}/api/generate"
            payload = {"model": model, "prompt": prompt, "stream": False}
            res = requests.post(url, json=payload, timeout=15)
            res.raise_for_status()
            return res.json().get("response", "No response from Ollama.")
        except Exception as e:
            self.logger.warning(f"Ollama connection failed: {e}")
            return ""

    def _call_gemini(self, api_key: str, model: str, prompt: str) -> str:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            res = requests.post(url, json=payload, timeout=15)
            res.raise_for_status()
            candidates = res.json().get("candidates", [])
            if candidates:
                return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return "Empty response from Gemini."
        except Exception as e:
            self.logger.warning(f"Gemini API request failed: {e}")
            return ""

    def _call_openai(self, api_key: str, model: str, prompt: str) -> str:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            res = requests.post(url, json=payload, headers=headers, timeout=15)
            res.raise_for_status()
            choices = res.json().get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return "Empty response from OpenAI."
        except Exception as e:
            self.logger.warning(f"OpenAI API request failed: {e}")
            return ""

    def _call_anthropic(self, api_key: str, model: str, prompt: str) -> str:
        try:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024
            }
            res = requests.post(url, json=payload, headers=headers, timeout=15)
            res.raise_for_status()
            return res.json().get("content", [{}])[0].get("text", "")
        except Exception as e:
            self.logger.warning(f"Anthropic API request failed: {e}")
            return ""

    def _generate_local_expert_advice(self, data: Dict[str, Any], query: str) -> str:
        """Local rule-based heuristic expert system for dataset analytics."""
        query_lower = query.lower()
        rows = data.get("rows", 0)
        cols = data.get("columns", 0)
        missing_pct = data.get("overall_missing_pct", 0.0)
        duplicate_rows = data.get("duplicate_rows", 0)
        targets = data.get("target_candidates", [])

        target_col = targets[0]["column"] if targets else "None detected"
        task = targets[0]["inferred_task"] if targets else "General Tabular"

        if "preprocess" in query_lower or "clean" in query_lower:
            advice = f"""### Local Expert Preprocessing Recommendations for **{data['filename']}**
1. **Handling Missing Values:** 
   - The dataset has **{missing_pct:.2f}%** missing values.
   - {"We recommend dropping columns with >50% nulls." if missing_pct > 1 else "Missing values are low. Standard imputation (mean/median for numeric, mode for categorical) will work."}
2. **Encoding & Delimiters:**
   - The file is encoded in **{data.get('encoding')}** and uses a **{data.get('delimiter')}** delimiter. Ensure your pandas read statement handles this.
3. **Feature Scaling:**
   - Use `StandardScaler` or `MinMaxScaler` on numerical variables before training linear models or neural networks.
4. **Duplicate Rows:**
   - There are **{duplicate_rows}** duplicate rows. {"Use `df.drop_duplicates(inplace=True)` before model training." if duplicate_rows > 0 else "No duplicates detected."}
"""
            return advice

        elif "model" in query_lower or "algorithm" in query_lower:
            advice = f"""### Local Expert Model Recommendations for **{data['filename']}**
Based on the size (**{rows}** rows, **{cols}** cols) and target candidate **'{target_col}'**:
1. **Suggested Models (based on inferred task '{task}'):**
   - **Baseline:** Logistic Regression / Ridge Regression.
   - **Tabular Powerhouses:** XGBoost, LightGBM, or CatBoost are recommended for high tabular performance.
   - **Sturdy Baseline:** Random Forest Classifier/Regressor.
2. **Validation Strategy:**
   - {"Use Stratified K-Fold CV to handle class imbalances." if "classification" in task.lower() else "Use K-Fold Cross Validation (e.g. k=5)."}
"""
            return advice

        elif "leak" in query_lower or "target leakage" in query_lower:
            advice = f"""### Local Expert Label Leakage Assessment for **{data['filename']}**
1. **Target Candidate Check:**
   - Candidate: **'{target_col}'**.
2. **Potential Leakage Columns:**
   - Look for columns with name prefixes like `id`, `uuid`, or index features.
   - Check if any feature has a correlation of `1.0` or extremely high value with the target.
   - Remove timestamps that occur *after* the target event.
"""
            return advice

        return f"""### Local Expert Dataset Profile: **{data['filename']}**
- **Dimension:** {rows} rows, {cols} columns.
- **Primary Target Feature:** '{target_col}' (Inferred task: {task}).
- **Missing Values:** {data.get('total_missing')} cells ({missing_pct:.2f}%).
- **Duplicates:** {duplicate_rows} rows ({data.get('duplicate_percentage', 0.0):.2f}%).
- **Suggested Next Step:** Run `set inspect` to view structural correlations, or configure an AI api key in your settings (`set config`) to get full LLM insights.
"""
