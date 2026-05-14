# LinguaInterpreter

LinguaInterpreter is an autonomous framework for generating machine learning code from natural language task descriptions. The framework uses a recursive approach to parse task details and transform them into structured representations, facilitating automated machine learning workflows across various data types, including tabular, time series, text, and images. This system helps streamline code generation with minimal human intervention, making it ideal for real-world machine learning (ML) projects.

## Features

- **Recursive Parsing**: Transforms unstructured task descriptions into structured representations, isolating key parameters for a clear and actionable pipeline.
- **Broad Task Support**: Handles a variety of ML task types, such as binary classification, regression, and clustering, across different data forms.
- **Enhanced Model Integration**: Supports multiple models, like `Random Forest`, `Gradient Boosting`, and `Neural Networks`, with ensemble methods for better performance.
- **Automated ML Workflow Generation**: Reduces human input by automating end-to-end ML code generation from task descriptions.

- **Improve-mode meta-planning (ReAct)**: Inside the improver/improvement loop, the system can run a dedicated ReAct meta-planner to audit the current baseline (`artifacts/best/` + `artifacts/last/`), discover where metrics were computed (often via `artifacts/versions/*/metrics.json` + producing `code.py`), and generate modality-aware deep improvement tasks. Designed to be compatible with `--resume`.
- **Final-output ReAct Recovery**: If canonical submission is missing, the pipeline actively searches and reconciles multiple real paths (`./submission.csv`, `./submission/submission.csv`, `artifacts/final`, `artifacts/versions`) before failing the run.
- **Timeout-aware execution**: Per-task `time_budget_sec` is propagated into execution timeout controls; stream monitor also applies an idle-timeout for silent hangs.
- **Policy guardrails for verifier scripts**: Generated verifier/checker scripts are audited against hardcoded-spec patterns and auto-repaired to use dynamic `artifacts/spec.json` loading.
- **Resilience-first routing**: Runtime timeout/OOM-style failures are automatically redirected toward `spec_update` actions (instead of immediate hard failure), so the next attempt can adapt resource settings.

## Reliability Additions (Recent)

- **Submission path reconciliation (ReAct)**:
  - Canonical path is still configured via `paths.submission_dir` + `paths.submission_filename` (default remains `project_root/submission.csv`).
  - Runtime reconciles canonical path with both legacy and bench paths:
    - `project_root/submission.csv`
    - `project_root/submission/submission.csv`
  - Finalizer mirrors discovered valid submission across expected locations to satisfy external checkers.
- **Metrics artifact materialization**:
  - Final metrics snapshot is always persisted to `artifacts/final/metrics.json` (including skipped-type runs) to avoid checker false negatives on missing files.
  - Best/version ledgers are still updated only for calculated (non-skipped) metrics paths.
- **Runtime stability**:
  - Fixed router crash for `None + list` on `bash_cmds` merge.
  - Added monitor-side no-output timeout handling for `stream=True` process runs.

## Requirements

Dependencies for this project are managed in `requirements.txt` and can be installed as follows:

```bash
pip install -r requirements.txt
```

For deep learning tasks, it is recommended to install the GPU-compatible versions of `torch` and `tensorflow`:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install tensorflow-gpu
```

## Getting Started

1. Clone the repository and install the requirements.
2. Define the API keys and configure the models in `main.py`:
   - **OpenAI API Key**: Set `open_ai_key`.
   - **Langfuse Public and Secret Keys**: Configure `langfuse_handler`.
3. Specify the model choices for `OpenAI` or `Ollama` depending on the setup.

### Task Format

To initiate the ML pipeline, define your task description within the `if __name__ == "__main__":` block in `main.py`. An example of a task description is given below:

```python
if __name__ == "__main__":
    task = """
    **Objective:**  
    Predict a patient's smoking status using binary classification based on health indicators.
    
    **Evaluation Metric:**  
    Area Under the ROC Curve (AUC-ROC)
    
    #### Dataset and Files
    
    - **./train.csv:** Training data with health features and target variable `smoking`.
    - **./test.csv:** Test data for predicting the `smoking` probability.
    - **./sample_submission.csv:** Template for submission format.
    
    **Dataset Origin:**  
    Derived from a deep learning model trained on the Smoker Status Prediction dataset. 
    ...
    """
    print(main_pipeline(task))
```

To achieve more robust solutions, it’s recommended to clearly specify any particular methodologies or models you want applied. For instance, if you prefer using **XGBoost** or **CatBoost**, or if additional steps like calculating feature importance or evaluating specific metrics are needed, be sure to outline these in the task description. This guidance enables the pipeline to generate solutions more aligned with your requirements and ensures that complex tasks are handled with greater precision.

### API Key Setup

To set up the API keys and models, make sure the following environment variables and configurations are defined in `main.py`:

```python
# Initialize colorama for colored output
init(autoreset=True)
is_ollama = False  # Set to True if using Ollama models

# Set OpenAI API Key
open_ai_key = "<YOUR_OPENAI_API_KEY>"
os.environ["OPENAI_API_KEY"] = open_ai_key

# Set up Langfuse with Public and Secret Keys
langfuse = Langfuse()
langfuse_handler = CallbackHandler(
    session_id="",
    public_key="<YOUR_LANGFUSE_PUBLIC_KEY>",
    secret_key="<YOUR_LANGFUSE_SECRET_KEY>",
    host=""
)
langfuse_handler.auth_check()
```

### Model Selection

- **OpenAI Models**: Uses `ChatOpenAI` with `gpt-4o` and `gpt-4o-mini`.
- **Ollama Models**: Uses `ChatOllama` with `deepseek-coder-v2`.

Set `is_ollama` to `True` if using Ollama models; otherwise, it will default to OpenAI models.

## Limitations

While LinguaInterpreter offers a powerful solution for autonomous machine learning code generation, there are some important limitations to consider:

1. **Resource-Intensive Execution**: Running models, particularly larger ones like `gpt-4o` or `deepseek-coder-v2`, can consume significant computational resources. This can be especially demanding for complex tasks or large datasets, so ensure your system meets the necessary hardware requirements (e.g., GPU support for deep learning tasks).

2. **Cost Implications**: 
   - **Token Usage**: The framework relies on API calls to models like OpenAI’s `gpt-4o` and potentially Ollama’s `deepseek-coder-v2`, both of which incur token-based usage costs.
   - **Minimizing Expenses**: We have optimized prompts and model configurations to reduce unnecessary expenditures. However, for tasks that require multiple iterations or extensive model engagement, costs can add up quickly. To keep costs low, users are encouraged to:
     - Review and refine prompts.
     - Avoid repetitive or lengthy calls that don’t contribute directly to the task at hand.

3. **Opportunities for Improvement**: LinguaInterpreter is an evolving tool with ongoing areas for enhancement, such as:
   - **Enhanced Prompt Customization**: Adapting prompts to better match specific task requirements, which can improve model accuracy and reduce the need for repeated calls.
   - **Model Efficiency**: Exploring smaller, task-optimized models may offer further reductions in cost without compromising performance.

## Observations

1. **Limited Contextual Awareness in LLM Agents**: While the agent can generate effective solutions, it often lacks dynamic awareness of file directories and dataset availability. This limitation occasionally leads to errors, such as defaulting to a print statement when datasets are missing, rather than halting execution. This limitation suggests the potential benefit of an additional monitoring layer to analyze directory structure, verify dataset presence, and handle missing data properly.

2. **Public LLM Model Limitations**: Testing of publicly available models, such as Nemotron, Codestral, Startcoder, Qgwen2.5, and Llama 3.1, revealed inconsistent quality. These models frequently produced code with extraneous details, placeholders, or overly verbose comments, impacting both efficiency and clarity. They also tended to provide textual task descriptions within the code, underscoring the need for models with more precise task-handling abilities.

3. **Performance Comparison - GPT-4o vs. GPT-4o-mini**: Among models used, GPT-4o proved most effective for generating complete solutions, including both code and task instructions, whereas GPT-4o-mini primarily excelled in code generation alone, struggling with accurate task description generation.

4. **Purpose of Checks in Code Generation**: The framework includes checks within generated code to promote interpretability and ensure alignment with task objectives, enhancing both reasoning and output relevance.

5. **Dependency on LLM for Code Verification**: Currently, verification of generated code relies solely on LLM reasoning rather than execution. Incorporating a Retrieval-Augmented Generation (RAG) system could further support verification, contextualizing the code within relevant documents and improving accuracy and generalization.

6. **Verification without Execution**: Code verification is conceptual, without actual execution, fostering a focus on reasoning over syntax errors. This conceptual verification ensures code readiness for eventual integration within a RAG system.

7. **Challenges with Complex Datasets**: Complex datasets often require multiple generation iterations, as model alignment can drift, producing incomplete code. Repeated trials are sometimes necessary to refine output for complex objectives.

8. **Omission of Metrics and Critical Details**: Essential performance metrics or key information can be missing in the generated output, which may impact evaluation if not addressed manually.

9. **Efficiency in Simple vs. Complex Tasks**: Solutions for simpler tasks typically work on the first try, while tasks with image or complex data often require additional iterations for accuracy and completeness.

10. **Resource Management Constraints**: Without system resource awareness, generated code can inadvertently exceed available memory or computational resources, potentially leading to inefficiencies.


## Main Results

The following table presents a comparative analysis of LinguaInterpreter's recursive framework, GPT Pipeline, and DataInterpreter across various Kaggle competitions with diverse data types and complexities. Each model's performance was evaluated against a range of metrics, including RMSE, accuracy, F1, and ROC AUC.

| Competition Name                                   | Data Type       | Data Volume | Metric | Our Score | GPT Pipeline Score | DataInterpreter Score |
|----------------------------------------------------|-----------------|-------------|--------|-----------|--------------------|-----------------------|
| Predict CO2 Emissions in Rwanda                    | Tabular         | 119 MB      | RMSE   | 2.0871    | 18.170            | 141.917               |
| Dog vs Cat Classification                          | Image           | 745 MB      | Accuracy| 0.968     | 0.922             | X                     |
| 2022 Regression Data Challenge                     | Tabular         | 2 MB        | MSE    | 100       | 8.952             | X                     |
| ECO3119 Final Competition                          | Time Series     | 8 MB        | Accuracy| 0.986     | 0.994             | 0.994                 |
| Predict Potential Spammers on Fiverr               | Text            | 60 MB       | F1     | 0.891     | 0.837             | 0.727                 |
| Rossmann Store Sales                               | Tabular         | 38 MB       | RMSE   | 494       | X                 | X                     |
| Shopee - Price Match Guarantee                     | Image/Tabular   | 1.92 GB     | F1     | 0.5976    | X                 | X                     |
| PetFinder.my - Pawpularity Contest                 | Image           | 1 GB        | RMSE   | 19.36     | X                 | X                     |
| New York City Taxi Trip Duration                   | Tabular         | 89.91 MB    | RMSLE  | 0.140     | X                 | 0.387793              |
| Binary Prediction of Smoker Status using Bio-Signals | Tabular         | 22.79 MB    | ROC AUC| 0.859     | 0.847             | 0.863                 |

We also experimented with solving simpler tasks, such as a CAPTCHA recognition dataset sourced from Kaggle, achieving promising results. Metrics like accuracy, precision, and recall consistently approached 0.95, demonstrating the model’s potential even for straightforward applications. However, not all solutions were successfully generated on the first attempt. Occasionally, it took a couple of runs to refine the results due to minor issues or instances where the GPT model could not resolve a simple oversight. These additional runs allowed us to improve the outcome and reach satisfactory accuracy levels in these types of tasks.
