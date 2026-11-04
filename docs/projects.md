# Project index: setup, run commands and artifacts

One section per project directory. Commands are copied from each project's own README or manifest; none
were re-executed for this index. Paths are relative to the repository root, and `~` stands for your home
directory.

Sections follow the same order as the modules table in the [root README](../README.md).

## agentsville-trip-planner

- **Course:** Udacity Agentic AI Nanodegree (nd900), Core #6: Intelligent Systems.
- **What it is:** a two-agent travel planner. `ItineraryAgent` drafts a plan with role-based prompting
  and chain-of-thought; `ItineraryRevisionAgent` revises it in a ReACT loop with four tools
  (`calculator_tool`, `get_activities_by_date_tool`, `run_evals_tool`, `final_answer_tool`).
- **Stack:** `openai`, `pydantic`, `matplotlib`, `IPython` (see `requirements.txt`).
- **Setup:** the notebook accepts any OpenAI-compatible endpoint through three environment variables.

```sh
cd agentsville-trip-planner
pip install -r requirements.txt
export OPENAI_API_KEY=...          # Vocareum `voc-...`, OpenAI, or a compatible provider
export OPENAI_BASE_URL=...         # optional, for non-OpenAI providers
export AGENTSVILLE_MODEL=gpt-4o-mini
jupyter nbconvert --to notebook --execute agentsville_trip_planner.ipynb \
    --output agentsville_trip_planner.executed.ipynb \
    --ExecutePreprocessor.timeout=900
```

- **Recorded run:** the submission was executed with Zhipu GLM-4.6 through its OpenAI-compatible
  endpoint, with `AGENTSVILLE_DISABLE_THINKING=1` so the final `content` field is populated.
- **Artifacts:** the executed notebook only. There is no test suite.

## autonomous-knowledge-agent

- **What it is:** UDA-Hub, a LangGraph supervisor-routed support agent. Six nodes (supervisor,
  classifier, knowledge, resolver, escalation, memory) over a SQLite schema with `Account`, `User`,
  `Ticket`, `TicketMetadata`, `TicketMessage`, `Knowledge`, `CustomerMemory` and `ToolLog` tables.
- **Stack:** `langgraph`, `langchain-core`, `pytest`; database file `data/uda_hub.db`.
- **Run:**

```sh
cd autonomous-knowledge-agent
python setup_database.py     # creates the schema and seeds Knowledge
python run_demo.py           # five ticket scenarios
python -m pytest -q
```

- **Artifacts:** `demo_output.txt`, `test_output.txt` (records `Ran 4 tests ... OK`),
  [`docs/architecture.md`](../autonomous-knowledge-agent/docs/architecture.md), and
  `docs/workflow_diagram.mmd` (Mermaid source).
- **Notes:** the `Knowledge` table is seeded with 15 articles. A deterministic fallback executor keeps
  the tests runnable where LangGraph is unavailable.

## beavers-choice-paper-agents

- **Course:** Udacity Agentic AI project.
- **What it is:** a four-agent paper-supply sales team (`OrchestratorAgent`, `InventoryAgent`,
  `QuotingAgent`, `SalesAgent`) built on `smolagents`, exercising the seven starter helper functions as
  agent tools.
- **Stack:** `smolagents`, `openai`, `SQLAlchemy`, `pandas`, `python-dotenv`.
- **Run:**

```sh
cd beavers-choice-paper-agents
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python project_starter.py     # writes test_results.csv and prints a financial report
```

- **Credentials:** a local `.env` with `UDACITY_OPENAI_API_KEY`, `UDACITY_OPENAI_API_BASE` and
  `UDACITY_OPENAI_MODEL`. The file is not committed.
- **Artifacts:** `workflow_diagram.png`, `workflow_diagram.md`, `reflection_report.txt`,
  `test_results.csv`, `quotes.csv`, the two request CSVs, and a submission archive.

## document-assistant-langgraph

- **What it is:** a LangGraph document assistant that classifies intent, retrieves document context, and
  routes to Q&A, summarisation or calculation agents, with JSONL session and tool-call logs.
- **Stack:** `langchain`, `langgraph`, `langchain-openai`, `pydantic`, `pytest`.
- **Run:**

```sh
cd document-assistant-langgraph
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY / OPENAI_BASE_URL, or leave empty
python main.py
pytest -q
```

- **Without a key:** `FakeDocumentLLM` (`src/fake_llm.py`) keeps the graph and the tests runnable.
- **Implementation:** `create_workflow()` compiles `classify_intent -> qa/summarization/calculation ->
  update_memory -> END` with `InMemorySaver`; the calculator validates AST nodes before evaluating.
- **Artifacts:** `documents/` sample reports, `sessions/demo-session.jsonl`, `logs/demo-session_tool_calls.jsonl`.
  `pyproject.toml` only configures pytest (`pythonpath = ["."]`, `testpaths = ["tests"]`).

## ecohome-energy-advisor

- **What it is:** a LangGraph smart-home energy advisor with SQLite usage data, mock weather and
  electricity-price tools, ChromaDB retrieval over energy-saving tips, savings calculation, and an
  evaluation harness.
- **Path:** the project lives in `ecohome-energy-advisor/ecohome_solution/`, not at the directory root.
- **Stack:** `langgraph`, `langchain-core`, `langchain-openai`, `chromadb`, `pydantic`, `pytest`.
  Python 3.10+ (developed against 3.14).
- **Run:**

```sh
cd ecohome-energy-advisor/ecohome_solution
pip install -r requirements.txt
python run_demo.py      # or run 01_db_setup.ipynb, 02_rag_setup.ipynb, 03_run_and_evaluate.ipynb in order
pytest -q
```

- **Deterministic parts:** weather and pricing tools return mock data, and `search_energy_tips` uses a local
  embedding function, so retrieval runs without API calls.
- **Artifacts:** `data/energy_data.db`, the committed `data/vectorstore/` Chroma index, `models/energy.py`,
  and `evaluation.py` (accuracy, relevance, completeness, usefulness, tool-usage scoring).

## face-generation

- **Course:** Deep Learning Nanodegree (nd101), Course 5.
- **What it is:** a DCGAN trained from scratch on 64×64 CelebA faces, with the generator and discriminator
  described in the project README.
- **Stack:** Python 3.14 / PyTorch 2.10 / MPS backend; `datasets` loads `tpremoli/CelebA-attrs`.
- **Run:**

```sh
cd face-generation
pip install torch torchvision datasets matplotlib
BATCH=64 EPOCHS=3 SUBSET=5000 python run_gan.py
```

- **Recorded run:** three epochs over 5,000 images on an M5 MPS host; per-epoch losses and timings are in
  the project README.
- **Artifacts:** `project.ipynb` and `project.html`, `samples/` and `samples_v2/` face grids,
  `loss_curve.png`, `loss_curve_v2.png`. Checkpoints are git-ignored and must be regenerated.

## landmark-classifier

- **Course:** Introduction to Deep Learning Nanodegree, Core #4.
- **What it is:** a 50-class landmark classifier in three notebook parts — a five-layer CNN from scratch,
  ResNet50 transfer learning, and TorchScript export with an inference app.
- **Stack:** `torch==1.9.0`, `torchvision==0.10.0`, `opencv-python-headless`, `bokeh`, `pytest`.
- **Dataset:** not included. Put the landmark images in `landmark_images/` with `train/`, `valid/`,
  `test/` subdirectories.
- **Run:**

```sh
cd landmark-classifier
pip install -r requirements.txt
jupyter notebook cnn_from_scratch.ipynb    # then transfer_learning.ipynb, then app.ipynb
```

- **Rubric targets, not claimed scores:** the notebook names 50% accuracy from scratch and 60% with
  transfer learning as the thresholds; no achieved score is recorded in the project README.
- **Artifacts:** `cnn_from_scratch.html`, `transfer_learning.html`, `app.html`, and the `src/` modules
  (`data`, `model`, `optimization`, `train`, `transfer`, `predictor`, `helpers`, `create_submit_pkg`).

## mnist-digit-classifier

- **Course:** Udacity Masters in AI, Introduction to Deep Learning project.
- **What it is:** a fully connected classifier (784 → 256 → 128 → 10, ReLU and dropout) trained with Adam
  at lr 0.001 for 15 epochs, batch size 64.
- **Recorded result:** 97.05% test accuracy against a 90% target.
- **Run:** open `mnist_digit_classifier.ipynb`. There is no `requirements.txt` and no script entry point,
  so the notebook is the only reproducible path.
- **Artifacts:** `mnist_classifier.pth`, `training_curves.png`, `predictions.png`.

## nasa-mission-intelligence-rag

- **What it is:** an end-to-end RAG chat system over Apollo 11, Apollo 13 and Challenger mission
  transcripts, with chunked embeddings in ChromaDB, mission filtering, and RAGAS scoring.
- **Stack:** `chromadb==1.5.7`, `openai==2.31.0`, `langchain-openai==1.1.13`, `ragas==0.4.3`,
  `streamlit==1.56.0`. Python 3.10+.
- **Build the vector store:**

```sh
cd nasa-mission-intelligence-rag
pip install -r requirements.txt
python embedding_pipeline.py \
  --data-path ./data_text \
  --chroma-dir ./chroma_db_openai \
  --collection-name nasa_space_missions_text \
  --chunk-size 500 \
  --chunk-overlap 100 \
  --update-mode skip
streamlit run chat.py
```

- **Batch evaluation:** `python ragas_evaluator.py --dataset evaluation_dataset.txt --chroma-dir
  ./chroma_db_openai --collection-name nasa_space_missions_text --model gpt-3.5-turbo --top-k 3
  --output batch_evaluation_results.json`.
- **Artifacts:** `data_text/` corpus, `evaluation_dataset.txt`, `PROJECT_REPORT.md`. The Chroma directory is
  produced by the pipeline and is not committed.
- **Note:** the embedding pipeline accepts `skip`, `update` and `replace` modes, and `--stats-only` prints
  collection size without writing.

## nd025-data-science-blog-post

- **Course:** Data Scientist Nanodegree (nd025), blog post project.
- **What it is:** a CRISP-DM analysis of the California housing dataset answering four stakeholder
  questions, plus a written `blog_post.md`.
- **Run:**

```sh
cd nd025-data-science-blog-post
pip install -r requirements.txt
jupyter notebook california_housing_blog_analysis.ipynb
```

- **Recorded results:** 20,640 rows with 207 missing `total_bedrooms` values; a baseline linear model at
  RMSE 72,188, MAE 51,845, R² 0.613.
- **Artifacts:** `data/housing.csv` (included), `images/` figures, `analysis_summary.json`, `blog_post.md`.

## nd025-data-science-dashboard

- **Course:** Data Scientist Nanodegree (nd025), Software Engineering for Data Scientists project.
- **What it is:** a distributable `employee_events` package for `employee_events.db` plus a FastHTML
  dashboard showing employee and team performance, notes, and two visualisations.
- **Stack:** `python-fasthtml==0.14.1`, `scikit-learn==1.5.2`, `matplotlib==3.9.2`, `pytest`, `flake8`.
  Python 3.10+; the package is installed editable via `-e ./python-package`.
- **Run:**

```sh
cd nd025-data-science-dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
(cd python-package && python setup.py sdist)
pytest
python report/dashboard.py     # then open http://localhost:5001
```

- **CI:** `.github/workflows/lint.yml` runs flake8 and `.github/workflows/test.yml` builds the package and
  runs pytest, both on Python 3.10.10.
- **License:** [`LICENSE.txt`](../nd025-data-science-dashboard/LICENSE.txt) carries Udacity's
  non-commercial Educational Content license.

## nd025-data-science-pipeline

- **Course:** Data Scientist Nanodegree (nd025), pipeline project (StyleSense reviews).
- **What it is:** one scikit-learn pipeline that predicts `Recommended IND` from numeric, categorical and
  TF-IDF text features, tuned by grid search.
- **Recorded metrics:** accuracy 0.8897, precision 0.9680, recall 0.8944, best CV F1 0.9264, over 18,442
  rows split 14,753 train / 3,689 test.
- **Run:**

```sh
cd nd025-data-science-pipeline
pip install -r requirements.txt
jupyter notebook Data_Science_Pipeline.ipynb
```

- **Artifacts:** `metrics_summary.json`, `data/reviews.csv`, and `starter/` (the original starter notebook
  and data, kept for reference).

## nd025-recommendation-system

- **Course:** Data Scientist Nanodegree (nd025), recommendations with IBM.
- **What it is:** four recommenders over IBM Watson Studio community articles — rank-based, user-user
  collaborative filtering, content-based TF-IDF with KMeans, and SVD latent-factor similarity.
- **Run:**

```sh
cd nd025-recommendation-system
pip install -r requirements.txt
jupyter notebook Recommendations_with_IBM.ipynb
```

- **Artifacts:** `data/user-item-interactions.csv`, `data/articles_community.csv`, `project_tests.py` with
  `top_5.p`, `top_10.p`, `top_20.p` fixtures, `svd_accuracy.png`, `metrics_summary.json`.

## nd892-dnn-speech-recognizer

- **Course:** Udacity NLP Nanodegree (nd892) speech-recognition capstone, built on the upstream
  `AIND-VUI-Capstone` starter.
- **What it is:** five ASR model architectures plus a final model, over MFCC or spectrogram features from
  LibriSpeech.
- **Stack:** `keras==2.0.5`, `tensorflow==1.1.0`, `h5py==2.7.0`, `librosa==0.5.1`,
  `python-speech-features==0.5` — a Python 3.5-era environment that no longer installs on a current
  interpreter.
- **Data preparation:** download `dev-clean` and `test-clean` from OpenSLR, convert with `flac_to_wav.sh`,
  then build the corpus manifests:

```sh
python create_desc_json.py LibriSpeech/dev-clean/ train_corpus.json
python create_desc_json.py LibriSpeech/test-clean/ valid_corpus.json
jupyter notebook vui_notebook.ipynb
```

- **Artifacts:** `results/model_0.h5` … `model_4.h5` and `model_end.h5` with matching `.pickle` histories,
  `report.html`, `images/` architecture diagrams.
- **Attribution:** `create_desc_json.py` and `flac_to_wav.sh`, plus some spectrogram helpers, come from
  [baidu-research/ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech).
- **License:** [`LICENSE.txt`](../nd892-dnn-speech-recognizer/LICENSE.txt) is MIT (Udacity, 2017).

## nd892-machine-translation

- **Course:** Udacity NLP Nanodegree (nd892).
- **What it is:** an English-to-French seq2seq pipeline in Keras — simple RNN, embedding variants,
  encoder-decoder LSTM/GRU, and a comparison of the tested architectures.
- **Data:** `data/small_vocab_en` and `data/small_vocab_fr` are included; no download step.
- **Run:**

```sh
cd nd892-machine-translation
pip install -r requirements.txt
jupyter notebook machine_translation.ipynb
```

- **Artifacts:** `machine_translation.html` (exported notebook), `helper.py`, `project_tests.py`, and the
  `images/` diagrams used by the long architecture write-up in the project README.

## object-detection-urban-environment

- **Course:** Self-Driving Car Engineer Nanodegree (nd013), Course 2.
- **What it is:** two TensorFlow 2 Detection Zoo models trained on Waymo urban frames, with a deployment
  notebook and an inference video.
- **Stack:** TensorFlow 2 Detection API; two pipeline configs under `configs/`.
- **Recorded results:** SSD ResNet50 V1 FPN validation mAP 0.312; Faster R-CNN ResNet50 V1 validation mAP
  0.397 (0.84/0.93 and 0.61/0.76 final train/validation loss). Faster R-CNN was selected for deployment.
- **Run:** `1_train_model.ipynb` for training and evaluation, `2_deploy_model.ipynb` for inference, in a
  TF2 environment with the Waymo dataset and the two `pipeline.config` files.
- **Artifacts:** `outputs/inference_video.mp4`, 24 sampled frames under `outputs/frames/`,
  `label_map.pbtxt`, `experiment_summary.json`.

## omnitrainer-multimodal-moderation

- **What it is:** a moderation and training app for customer-service interactions, with structured
  moderation outputs for text, image, audio and video, a simulated customer agent, FastAPI endpoints, a
  Gradio chat UI, and Phoenix tracing.
- **Stack:** `pydantic-ai`, `fastapi`, `gradio`, `google-genai`, `arize-phoenix`; Python ≥ 3.12 managed
  with `uv` (`uv.lock` is committed).
- **Run:**

```sh
cd omnitrainer-multimodal-moderation
uv sync --dev
uv pip install -e .
cp env.example .env          # USER_API_KEY, GEMINI_API_KEY, MODEL_PROVIDER, ...
uv run multimodal-moderation # Gradio :7860, FastAPI :8000, Phoenix :6006
uv run pytest tests/ -vv
```

- **Evals:** `uv run evals/text/test_cases.py`, plus the `image`, `audio` and `video` suites, with fixtures
  under `evals/test_data/`.
- **Artifacts:** `project_proposal.md`, `multimodal_moderation/` package, `evals/` suites.
- **License:** `pyproject.toml` declares `license = "MIT"` with `Giacomo Vianello` as author; no LICENSE
  file is present in the directory.

## predict-bike-sharing-demand-autogluon

- **Course:** AWS Machine Learning Engineer Nanodegree.
- **What it is:** three AutoGluon Tabular runs — baseline, datetime feature engineering, and a tuned run
  with non-negative predictions and an expanded model-family search — submitted to the Kaggle bike-sharing
  competition.
- **Recorded results:** Kaggle public score 1.80412 for the baseline, 0.67431 after datetime features,
  0.54387 after tuning; the best baseline validation model was `WeightedEnsemble_L3` at RMSE 35.2187.
- **Run:** open `Predict Bike Sharing Demand with AutoGluon.ipynb` in a SageMaker or AutoGluon-enabled
  environment (the `img/sagemaker-studio-git*.png` screenshots record the Git integration used). There is
  no `requirements.txt` in the project.
- **Artifacts:** `report.md`, `report-template.md`, `project-template.ipynb`, `submission.csv`,
  `submission_new_features.csv`, `submission_new_hpo.csv`, and the two score plots under `img/`.
- **License:** [`LICENSE.txt`](../predict-bike-sharing-demand-autogluon/LICENSE.txt) is Udacity's
  non-commercial Educational Content license.

## sparkify-airflow-data-pipelines

- **Course:** Data Pipelines project (Sparkify).
- **What it is:** an hourly Airflow DAG that stages JSON from S3 into Redshift, loads a star schema, and
  runs parameterised data-quality checks.
- **Stack:** Apache Airflow, Amazon Redshift, Amazon S3; `docker-compose.yaml` for a local Airflow.
- **Run:** create the `aws_credentials` and `redshift` Airflow connections, set the `s3_*` variables, apply
  `create_tables.sql`, and enable `dags/final_project.py`.
- **Structure:** four custom operators (`stage_redshift`, `load_fact`, `load_dimension`, `data_quality`)
  plus `plugins/helpers/sql_queries.py`; task order is Begin → two stages → `songplays` → four dimensions →
  quality checks → Stop.
- **Static check:** the project README documents an `ast.parse` sweep over `dags/` and `plugins/` as the
  only local validation; real AWS credentials are intentionally absent.
- **Artifacts:** `assets/` DAG-graph and login screenshots.
- **License:** [`LICENSE.md`](../sparkify-airflow-data-pipelines/LICENSE.md) is Udacity's non-commercial
  Educational Content license.

## sparkify-cassandra-data-modeling

- **Course:** Data Modeling with Apache Cassandra (Sparkify).
- **What it is:** an ETL step that consolidates raw event CSVs into `event_data_new.csv`, then three
  query-first Cassandra tables: `songplays_by_session_item`, `songplays_by_user_session`, `users_by_song`.
- **Run:** start Cassandra locally (the Udacity workspace exposes it at `127.0.0.1`), then:

```sh
cd sparkify-cassandra-data-modeling
pip install -r requirements.txt
python project_1b_solution.py
```

- **Artifacts:** `event_data/2018-11-01-events/` sample input, `event_data_new.csv`,
  `Project_1B_ Project_Template.ipynb` (note the space in the filename).
- **Notes:** tables are dropped at the end of the workflow, matching the project cleanup step.

## sparkify-data-warehouse

- **Course:** Data Warehouse project (Sparkify).
- **What it is:** an ETL pipeline that copies S3 JSON into Redshift staging tables and inserts into the
  `songplays` fact table and `users`, `songs`, `artists`, `time` dimensions.
- **Run:**

```sh
cd sparkify-data-warehouse
pip install -r requirements.txt
# fill in dwh.cfg: [CLUSTER] host/db/user/password/port and [IAM_ROLE] ARN
python create_tables.py
python etl.py
```

- **Notes:** `dwh.cfg` contains placeholders only. `staging_events.userId` is loaded as text, users are
  deduplicated by latest level, and songplays use a left join so `NextSong` events survive missing song
  metadata.
- **Artifacts:** `sql_queries.py`, `create_tables.py`, `etl.py`; the project README also lists validation
  and analytical queries.

## stedi-human-balance-analytics

- **Course:** Data Engineering with AWS Nanodegree, STEDI Human Balance Analytics.
- **What it is:** a privacy-aware lakehouse over Step Trainer, accelerometer and customer JSON, moving data
  from landing to trusted to curated zones; only customers with a non-null `shareWithResearchAsOfDate`
  reach the training data.
- **Run on AWS:** deploy the five `glue_jobs/` scripts against Glue Data Catalog tables created from
  `sql/*.sql`, writing JSON to S3 with `updateBehavior="UPDATE_IN_DATABASE"`.
- **Run locally:** `python3 local_validation.py` mirrors the join logic in pandas against
  `data/starter/` and checks the rubric row counts.
- **Expected row counts:** `customer_landing` 956, `accelerometer_landing` 81,273, `step_trainer_landing`
  28,680, `customer_trusted` 482, `accelerometer_trusted` 40,981, `step_trainer_trusted` 14,460,
  `customers_curated` 482, `machine_learning_curated` 43,681.
- **Artifacts:** landing DDL and `validation_queries.sql`, ten Athena screenshots under `screenshots/`.
- **Source data:** `data/starter/` is copied from Udacity's public
  [nd027 starter repository](https://github.com/udacity/nd027-Data-Engineering-Data-Lakes-AWS-Exercises/tree/main/project/starter).

## teaching-llm-to-count

- **Course:** Generative AI Nanodegree (nd608), Fundamentals project.
- **What it is:** GRPO with LoRA on Qwen2.5-0.5B-Instruct to teach letter counting, using five reward
  functions for numbering, spelling, running count, output format and final correctness.
- **Stack:** `torch`, `transformers`, `peft`, `trl`, `datasets`, `accelerate`, `matplotlib`; BF16 on Apple
  Silicon MPS, with `bitsandbytes` and `flash_attention_2` deliberately removed.
- **Run:**

```sh
cd teaching-llm-to-count
pip install -r requirements.txt
python run_grpo_mps.py        # or run gen_ai_fundamentals_project.ipynb end to end
python run_comparison.py      # before/after comparison and forgetting check
```

- **Recorded run:** 100 steps, 2,047 s on M5 MPS, final training loss 0.0220, 17,629,184 of 511,623,168
  parameters trainable (3.45%); base model 0/8 versus 3/8 after GRPO on the eight test cases.
- **Artifacts:** `results/comparison.json`, `results/general_knowledge.json`, `results/log_history.json`,
  `reward_curves.png`, `loss_curve.png`, `train_chain.sh`, and `letter_counting_adapter/` (LoRA weights,
  tokenizer, chat template, and a mostly unedited Hugging Face model-card stub).
- **License:** the project README states MIT; no LICENSE file is present.

## text-translation-sentiment

- **Course:** Deep Learning Nanodegree (nd101), Course 4.
- **What it is:** translates French and Spanish movie reviews to English with MarianMT, merges the three
  CSVs, classifies sentiment with DistilBERT, and exports the annotated set.
- **Stack:** `transformers`, `sentencepiece`, `sacremoses`, `pandas`, `torch`; runs on CPU or MPS.
- **Run:**

```sh
cd text-translation-sentiment
pip install transformers sentencepiece sacremoses pandas torch
python run_project.py     # inference only, about 2-3 minutes
```

- **Recorded result:** 30 reviews (10 per language) classified as 15 positive and 15 negative.
- **Artifacts:** `project.ipynb` and `project.html`, `data/movie_reviews_{eng,fr,sp}.csv`,
  `result/reviews_with_sentiment.csv`.

## udaplay-ai-research-agent

- **What it is:** a stateful research agent that answers video-game questions from a local ChromaDB
  knowledge base, falls back to Tavily web search when internal confidence is low, and persists learned
  facts as JSONL.
- **Stack:** `chromadb`, `openai`, `tavily-python`, `pydantic`, `python-dotenv`.
- **Run:**

```sh
cd udaplay-ai-research-agent
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # OPENAI_API_KEY and TAVILY_API_KEY, optional for the offline fallback
python scripts/build_vector_store.py
python scripts/run_examples.py
```

- **Tools:** `retrieve_game`, `evaluate_retrieval` and `game_web_search`, with conversation state and
  per-query tool traces.
- **Artifacts:** `data/games/*.json` for five publishers, two solution notebooks, and the helper scripts.
  `memory/learned_facts.jsonl` is written at runtime and is not committed.

## Cross-project notes

- **Test suites.** Five projects ship pytest tests:

  | Project | Test file | Focus |
  |---|---|---|
  | `autonomous-knowledge-agent` | `tests/test_uda_hub_agent.py` | refund, outage escalation, billing citation, low-confidence routing |
  | `document-assistant-langgraph` | `tests/test_document_assistant.py` | schema validation, intent routing, safe calculator |
  | `ecohome-energy-advisor` | `ecohome_solution/tests/test_ecohome.py` | database, mock tools, RAG retrieval, agent flow |
  | `nd025-data-science-dashboard` | `tests/test_employee_events.py` | database file and required tables |
  | `omnitrainer-multimodal-moderation` | `tests/` (7 modules) | moderation result types, per-modality agents, env setup, Gradio app |

- **No shared tooling.** There is no root `requirements.txt`, `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`
  or CI workflow; each project pins its own stack, and two of them (`nd892-dnn-speech-recognizer`,
  `predict-bike-sharing-demand-autogluon`) still carry the upstream Udacity GitHub workflow that opens Jira
  tickets rather than running tests.
- **License per directory.** Seven project directories state terms — four Udacity non-commercial license
  files, one MIT license file, one MIT statement in a project README, one MIT declaration in
  `pyproject.toml` — and the rest state none. The [root README](../README.md#-license--attribution) lists
  them; there is no repository-wide LICENSE.
