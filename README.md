<p align="center">
  <img src="https://img.shields.io/badge/🤖-Support_Ticket_Router-4f46e5?style=for-the-badge&labelColor=0f172a" alt="Project Title"/>
</p>

<h1 align="center">Intelligent Support Ticket Router Dashboard</h1>

<p align="center">
  <em>Fine-tuning a DistilBERT Encoder on AG News to build a real-time customer support classification and dispatch system — deployed as a premium Gradio web application.</em>
</p>

<p align="center">
  <a href="https://colab.research.google.com/"><img src="https://img.shields.io/badge/Run_on-Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Open in Colab"/></a>
  <a href="https://huggingface.co/docs/transformers"><img src="https://img.shields.io/badge/Hugging_Face-Transformers_4.40-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers"/></a>
  <a href="https://gradio.app/"><img src="https://img.shields.io/badge/Gradio-4.28_Blocks_UI-F97316?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Production_Ready-10b981?style=flat-square"/>
</p>

---

## 📌 Project Summary

This project transforms the canonical **AG News** topic classification dataset into a **production-grade Automated Support Ticket Router**. By re-mapping journalistic categories to corporate IT workflows, I fine-tuned a lightweight **DistilBERT** sequence classifier and deployed it behind a premium **Gradio Blocks** dashboard — all executable from a single Colab notebook.

> Demonstrates end-to-end NLP engineering: dataset loading, tokenization with `.map()`, Trainer-based fine-tuning, and live inference deployment.

---

## 🏷️ Label Mapping Strategy

The AG News dataset ships with 4 generic topic labels. We re-purpose them into actionable support desk workflows:

| AG News Label | Mapped Support Workflow | Priority | Escalation Desk |
|:---|:---|:---:|:---|
| **0 — World** | ⚠️ Account Security & Recovery | `CRITICAL` | Security Operations Center |
| **1 — Sports** | 💬 General & Marketing Inquiry | `LOW` | Customer Success Team |
| **2 — Business** | 💳 Billing, Invoices & Payments | `HIGH` | Finance & Stripe Ops |
| **3 — Sci/Tech** | 🛠️ Technical Bug & Engineering Support | `CRITICAL` | Tier 3 SRE / DevOps |

---

## 🧠 Tech Stack & Skills

<table>
<tr>
<td align="center" width="150">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="40"/><br/>
  <strong>Python 3.10+</strong><br/>
  <sub>Core Language</sub>
</td>
<td align="center" width="150">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" width="40"/><br/>
  <strong>PyTorch</strong><br/>
  <sub>Tensor Backend</sub>
</td>
<td align="center" width="150">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="40"/><br/>
  <strong>Transformers</strong><br/>
  <sub>Model Fine-Tuning</sub>
</td>
<td align="center" width="150">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" width="40"/><br/>
  <strong>Datasets</strong><br/>
  <sub>Data Pipeline</sub>
</td>
<td align="center" width="150">
  <img src="https://www.gradio.app/assets/gradio.svg" width="40"/><br/>
  <strong>Gradio 4</strong><br/>
  <sub>Web UI / Deploy</sub>
</td>
</tr>
</table>

### Core Competencies Demonstrated

```
NLP & Transformers          ████████████████████░░   90%
Dataset Engineering         ██████████████████░░░░   80%
Transfer Learning           ████████████████████░░   90%
Model Fine-Tuning (Trainer) ██████████████████████   95%
Gradio UI / Deployment      ██████████████████████   95%
PyTorch Tensors & CUDA      ████████████████░░░░░░   75%
```

| Skill Area | What This Project Proves |
|:---|:---|
| **Hugging Face `datasets`** | Loading from Hub, `.shuffle()`, `.select()`, `.map()` batch preprocessing |
| **Tokenization** | `AutoTokenizer`, padding strategies, truncation, max-length alignment |
| **Sequence Classification** | `AutoModelForSequenceClassification`, custom `id2label` / `label2id` maps |
| **Trainer API** | `TrainingArguments` configuration, `compute_metrics` callback, `eval_strategy` |
| **Model Serialization** | `save_pretrained()` for weights + tokenizer, reload via `pipeline()` |
| **Inference Pipelines** | `pipeline("text-classification")` with device mapping |
| **Gradio Blocks** | `gr.Blocks`, `gr.Row`, `gr.Column`, `gr.HTML`, `gr.Label`, theme customization |
| **CSS & Frontend** | Custom dark-mode stylesheet, inline HTML cards, badge systems, hover animations |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  AG News (HF Hub) → shuffle(seed=42) → select(600/150 samples)  │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                            │
│  AutoTokenizer.from_pretrained("distilbert-base-uncased")        │
│  .map(tokenize_fn, batched=True) → pad/truncate to 128 tokens   │
│  rename_column("label" → "labels") → set_format("torch")        │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                     TRAINING LAYER                                │
│  DistilBERT + 4-class Classification Head                        │
│  Trainer(lr=3e-5, epochs=2, batch=8, eval_strategy="epoch")     │
│  Metric: accuracy via HF evaluate                                │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                    INFERENCE LAYER                                │
│  pipeline("text-classification") loaded from saved checkpoints   │
│  softmax probabilities → argmax department assignment            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                              │
│  Gradio Blocks Dashboard                                         │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────┐ ┌───────────┐  │
│  │  KPI Stats  │ │ Input Panel  │ │ HTML Brief │ │ Prob Dist │  │
│  │  Grid Row   │ │ + Presets    │ │ + Actions  │ │ gr.Label  │  │
│  └─────────────┘ └──────────────┘ └────────────┘ └───────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✨ Dashboard Features

### 🎯 Quick-Start Preset Buttons
Four pre-loaded, representative support tickets that instantly populate the input box and trigger classification — no typing required:

| Button | Simulated Scenario |
|:---|:---|
| 🚨 **Security Alert** | Unauthorized login from unknown IP; credential lockout request |
| 💬 **Marketing Ask** | Enterprise catalog request from a prospective client team |
| 💳 **Invoice Double Bill** | Duplicate Visa charge on corporate invoice; refund needed |
| 🛠️ **DB Server Crash** | PostgreSQL 500 errors causing production connection timeouts |

### 📊 Real-Time Probability Distribution
Native `gr.Label` component renders weighted confidence bars across all 4 departments — giving operators instant visibility into borderline classifications.

### 📋 Dynamic HTML Dispatch Brief
Each classification generates a styled card showing:
- **Routed Department** badge (color-coded by category)
- **Confidence Score** percentage
- **Priority Level** indicator (CRITICAL / HIGH / LOW)
- **Escalation Procedure** with actionable next steps
- **Auto-Generated Customer Reply** draft ready for dispatch

### 📈 System KPI Monitor Row
Top-level stat cards displaying model identity, classification latency, automated dispatch rate, and system operational status.

---

## 📂 Repository Structure

```
intelligent-support-ticket-router/
├── .gitignore                      # Git exclusion configurations
├── README.md                       # ← You are here
├── app.py                          # Unified training + dashboard script
└── Ticket_Router_Colab.ipynb       # Colab-ready Jupyter Notebook (12 cells)
```

| File | Purpose | Size | Git Status |
|:---|:---|:---:|:---|
| `app.py` | End-to-end Python script: dataset → training → Gradio launch | ~21 KB | Tracked |
| `Ticket_Router_Colab.ipynb` | Final notebook with 6 markdown + 6 code cells | ~33 KB | Tracked |
| `.gitignore` | Local patterns to exclude checkpoints, caches, and utilities | ~1 KB | Tracked |
| `generate_notebook.py` | Local compilation utility (generates `.ipynb` from source splits) | ~37 KB | *Git Ignored* |


---

## 🚀 Quick Start

### Option A: Google Colab (Recommended)

1. Upload `Ticket_Router_Colab.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
3. Click **Runtime → Run all**
4. Wait for training to complete (~2-3 minutes on T4)
5. Click the generated Gradio public link to open the dashboard

### Option B: Local Execution

```bash
# Clone and navigate to the project
git clone git@github.com:jbanmol/intelligent-support-ticket-router.git
cd intelligent-support-ticket-router

# Install dependencies
pip install transformers==4.40.2 datasets==2.19.1 evaluate==0.4.1 accelerate==0.30.1 gradio==4.28.3

# Run the full pipeline
python app.py
```

### Option C: Regenerate the Notebook

```bash
python generate_notebook.py
# Output: Ticket_Router_Colab.ipynb
```

---

## ⚙️ Pinned Dependencies

All library versions are locked for reproducibility:

| Package | Version | Role |
|:---|:---:|:---|
| `transformers` | `4.40.2` | Model loading, tokenizer, Trainer API |
| `datasets` | `2.19.1` | Hub dataset downloading, `.map()` preprocessing |
| `evaluate` | `0.4.1` | Accuracy metric computation during eval |
| `accelerate` | `0.30.1` | Hardware-agnostic training backend (GPU/TPU/CPU) |
| `gradio` | `4.28.3` | Interactive web UI with Blocks layout system |
| `torch` | `2.x` | Tensor operations, CUDA acceleration, softmax |
| `numpy` | `latest` | Array manipulation for argmax predictions |

---

## 📝 Training Configuration

| Hyperparameter | Value | Rationale |
|:---|:---:|:---|
| Base Model | `distilbert-base-uncased` | Lightweight, fast, strong baseline for text classification |
| Learning Rate | `3e-5` | Standard safe rate for BERT fine-tuning |
| Batch Size | `8` | Fits within free-tier Colab GPU memory |
| Epochs | `2` | Sufficient for convergence on 600 samples |
| Weight Decay | `0.01` | Mild L2 regularization to prevent overfitting |
| Max Token Length | `128` | Covers typical support ticket lengths |
| Train Samples | `600` | Fast prototype; scale to full 120k for production |
| Test Samples | `150` | Validation set for per-epoch accuracy checks |

---

## 🔮 Future Enhancements

- [ ] Scale to full AG News dataset (120,000 training samples) for production accuracy
- [ ] Add F1, Precision, and Recall metrics alongside accuracy
- [ ] Implement confidence thresholding — flag low-confidence tickets for human review
- [ ] Add ticket history logging with SQLite or Firebase
- [ ] Deploy as a persistent Hugging Face Space with `huggingface_hub`
- [ ] Integrate Slack/Teams webhook notifications for critical ticket escalations

---

## 📜 License

This project is open-source under the [MIT License](https://opensource.org/licenses/MIT).

---

<p align="center">
  <strong>Built with 🧠 Hugging Face Transformers · 🎨 Gradio Blocks · ⚡ PyTorch</strong><br/>
  <sub>Built by Jb Anmol</sub>
</p>
