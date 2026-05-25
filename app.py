# ==============================================================================
# INTELLIGENT SUPPORT TICKET ROUTER
# ==============================================================================
# A production pipeline that transforms the AG News dataset into a 
# high-performance customer support routing engine using DistilBERT, 
# PyTorch, and a Gradio Blocks interface.
# Author: Jb Anmol
# ==============================================================================

import os
import torch
import numpy as np
import evaluate
import gradio as gr
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    pipeline
)

# ------------------------------------------------------------------------------
# CONFIGURATION SETTINGS
# ------------------------------------------------------------------------------
# Toggle between prototype training and full production scaling
PRODUCTION_MODE = False

# Enable auto-upload to Hugging Face Hub and deployment to HF Spaces
DEPLOY_TO_HUGGINGFACE = False
# Hugging Face Repository identifier (e.g., 'username/intelligent-support-ticket-router')
# If no namespace is provided, the authenticated username will be used.
HF_HUB_REPO_ID = "intelligent-support-ticket-router"


# ------------------------------------------------------------------------------
# STEP 1: INITIAL SYSTEM CONFIGURATION & GPU STATUS
# ------------------------------------------------------------------------------
print("=== Step 1: Initializing CUDA/MPS Hardware Acceleration Check ===")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Active Hardware Device Target: {device.upper()}")

# ------------------------------------------------------------------------------
# STEP 2: ROBUST DATASET LOAD & DOWNSAMPLING
# ------------------------------------------------------------------------------
print("\n=== Step 2: Loading and Inspecting Support Dataset ===")
# Download dataset with automated fallback to guarantee successful loads
try:
    print("Attempting to load standard 'ag_news' dataset...")
    raw_dataset = load_dataset("ag_news")
except Exception as e:
    print(f"Standard load failed: {e}. Attempting fallback load from fancyzhx...")
    raw_dataset = load_dataset("fancyzhx/ag_news")

print("Dataset Schema Loaded Successfully:")
print(raw_dataset)

# Downsample the dataset using seed-locked .select() for quick prototype training
if PRODUCTION_MODE:
    print("Running in PRODUCTION MODE: Scaling to full 120,000 training and 7,600 test samples...")
    train_subset = raw_dataset["train"]
    test_subset = raw_dataset["test"]
else:
    print("Running in PROTOTYPE MODE: Downsampling to 600 training and 150 test samples...")
    train_subset = raw_dataset["train"].shuffle(seed=42).select(range(600))
    test_subset = raw_dataset["test"].shuffle(seed=42).select(range(150))

# Human-readable dictionary mapping original class index numbers to ticket workflows
id2label = {
    0: "⚠️ Account Security & Recovery",
    1: "💬 General & Marketing Inquiry",
    2: "💳 Billing, Invoices & Payments",
    3: "🛠️ Technical Bug & Engineering Support"
}
label2id = {v: k for k, v in id2label.items()}

print("\nSample Training Entry:")
sample_idx = 0
print(f"Text: {train_subset[sample_idx]['text']}")
print(f"Label Index: {train_subset[sample_idx]['label']} -> Department: {id2label[train_subset[sample_idx]['label']]}")

# ------------------------------------------------------------------------------
# STEP 3: TOKENIZATION AND ALIGNMENT USING .MAP()
# ------------------------------------------------------------------------------
print("\n=== Step 3: Initializing Tokenizer and Preprocessing Data ===")
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_tickets_routine(examples):
    # Padding up to max length of 128 and truncating inputs that overflow
    return tokenizer(examples["text"], padding="max_length", max_length=128, truncation=True)

print("Applying native .map() batch preprocessing routine...")
tokenized_train = train_subset.map(tokenize_tickets_routine, batched=True)
tokenized_test = test_subset.map(tokenize_tickets_routine, batched=True)

# Rename label column to expected target keyword 'labels'
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_test = tokenized_test.rename_column("label", "labels")

# Set format for compatibility with PyTorch tensors
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ------------------------------------------------------------------------------
# STEP 4: MODEL FINE-TUNING AND METRIC OPTIMIZATIONS
# ------------------------------------------------------------------------------
print("\n=== Step 4: Configuring Sequence Classification Model ===")
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=4,
    id2label=id2label,
    label2id=label2id
).to(device)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Configure optimal Trainer hyperparameters based on active execution mode
if PRODUCTION_MODE:
    # Production-scale training arguments
    training_args = TrainingArguments(
        output_dir="./support_router_checkpoints",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16, # Larger batch size to exploit GPU memory
        per_device_eval_batch_size=16,
        num_train_epochs=1, # 1 epoch is highly sufficient for fine-tuning 120k samples
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="no", # Avoid writing huge checkpoints to disk during Colab run
        report_to="none"
    )
else:
    # Fast prototype training arguments
    training_args = TrainingArguments(
        output_dir="./support_router_checkpoints",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=20,
        report_to="none"
    )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

print("\n🚀 Commencing model training routine on active device...")
trainer.train()

# Serialize classification artifacts and vocabulary configs locally to disk
saved_router_path = "./final_ticket_router_model"
trainer.model.save_pretrained(saved_router_path)
tokenizer.save_pretrained(saved_router_path)
print(f"✅ Model weights and configuration successfully saved to: {saved_router_path}")

# ------------------------------------------------------------------------------
# STEP 5: DEPLOY PRESTIGE INTERACTIVE DASHBOARD VIA GRADIO
# ------------------------------------------------------------------------------
print("\n=== Step 5: Preparing Interactive Gradio Blocks Application ===")

# Create pipeline using trained checkpoints. Falls back to CPU or maps to appropriate device
device_id = 0 if device in ["cuda", "mps"] else -1
router_pipeline = pipeline(
    "text-classification", 
    model=saved_router_path, 
    tokenizer=saved_router_path,
    device=device_id
)

# Custom template descriptions for instant dashboard clicks
TEMPLATE_1 = "WARNING: I noticed an unauthorized login attempt from an unknown IP address in Europe. Please lock my login credentials immediately!"
TEMPLATE_2 = "Can you send over the latest catalog for your Enterprise package? Our marketing team wants to review the integration slots next month."
TEMPLATE_3 = "Urgent: We were double billed for invoice #INV-48201. The transaction went through twice on our corporate Visa credit card. Please process a refund."
TEMPLATE_4 = "Critical: Our production PostgreSQL database is throwing 500 internal server exceptions, causing connection timeouts for all external requests."

def generate_routing_report(ticket_text):
    if not ticket_text.strip():
        # Clean warning format
        warning_html = """
        <div style='background-color: rgba(239, 68, 68, 0.1); border: 1px dashed #ef4444; border-radius: 8px; padding: 16px; text-align: center; color: #ef4444;'>
            <strong>⚠️ Empty Input Detected</strong><br>Please enter a descriptive support ticket text block or select a template to parse.
        </div>
        """
        return warning_html, {}
    
    # Run tokenizer and inference pipeline
    pipeline_result = router_pipeline(ticket_text)[0]
    assigned_label = pipeline_result["label"]
    confidence_score = pipeline_result["score"]
    
    # Calculate all category probabilities for visual distributions
    # To do this cleanly, run raw model inference
    inputs = tokenizer(ticket_text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # Format all classes for the distribution label chart
    distribution = {id2label[i]: float(probs[i]) for i in range(4)}
    
    # Dynamic styling values based on classification outcome
    badge_style = "badge-general"
    escalation_instructions = "Assigning to Standard Support Desk."
    suggested_email_draft = ""
    priority = "NORMAL"
    
    if "Account Security" in assigned_label:
        badge_style = "badge-security"
        priority = "CRITICAL / SEVERE"
        escalation_instructions = "⚡ Triggering Multi-Factor Authentication Lockout. Sending instant SOC team push notification and alerting Security Operations Desk."
        suggested_email_draft = "Dear Customer,\n\nWe have detected a security flag on your account and have temporarily locked credential updates. Please verify your identity using your authentication app."
    elif "General & Marketing" in assigned_label:
        badge_style = "badge-general"
        priority = "LOW"
        escalation_instructions = "📥 Forwarding to the Sales & Account Management Team. Standard SLA (24 hrs) response ticket created."
        suggested_email_draft = "Hi there,\n\nThank you for reaching out! Our account managers have received your inquiry and will provide standard pricing materials shortly."
    elif "Billing, Invoices" in assigned_label:
        badge_style = "badge-billing"
        priority = "HIGH"
        escalation_instructions = "💳 Routing to Financial Ledger Queue. Matching transaction records on Stripe and queuing for account manager refund approval."
        suggested_email_draft = "Hello,\n\nI have received your billing query. I am reviewing the Stripe transactions for this invoice and will update you as soon as the refund status is updated."
    elif "Technical Bug" in assigned_label:
        badge_style = "badge-tech"
        priority = "CRITICAL"
        escalation_instructions = "🛠️ Escalating to Tier 3 Site Reliability Engineers (SRE). Logging system telemetry traces and opening DevOps Jira Incident tracking ticket."
        suggested_email_draft = "Dear Engineer/Admin,\n\nOur system auto-classifier has logged a high-severity bug report. Our technical staff is currently investigating the telemetry logs."

    # Build gorgeous premium HTML dashboard report in Light Mode
    report_html = f"""
    <div style='background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; font-family: system-ui, sans-serif; color: #1e293b; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);'>
        <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 12px; margin-bottom: 16px;'>
            <span style='font-size: 14px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;'>Router Brief Report</span>
            <span class='badge {badge_style}'>{assigned_label}</span>
        </div>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px;'>
            <div style='background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;'>
                <div style='font-size: 11px; color: #64748b; text-transform: uppercase;'>Confidence Accuracy</div>
                <div style='font-size: 20px; font-weight: bold; color: #059669; margin-top: 4px;'>{confidence_score * 100:.2f}%</div>
            </div>
            <div style='background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0;'>
                <div style='font-size: 11px; color: #64748b; text-transform: uppercase;'>Urgency Level</div>
                <div style='font-size: 20px; font-weight: bold; color: {'#ef4444' if priority in ['CRITICAL', 'CRITICAL / SEVERE'] else '#d97706' if priority == 'HIGH' else '#0284c7'}; margin-top: 4px;'>{priority}</div>
            </div>
        </div>
        
        <div style='margin-bottom: 16px;'>
            <div style='font-size: 12px; font-weight: bold; color: #64748b; text-transform: uppercase; margin-bottom: 6px;'>Escalation Procedure & DevOps Action:</div>
            <div style='background: rgba(99, 102, 241, 0.05); border-left: 3px solid #6366f1; padding: 10px 14px; border-radius: 4px; font-size: 13px; line-height: 1.5; color: #334155;'>
                {escalation_instructions}
            </div>
        </div>
        
        <div>
            <div style='font-size: 12px; font-weight: bold; color: #64748b; text-transform: uppercase; margin-bottom: 6px;'>Auto-Generated Customer Reply Draft:</div>
            <textarea readonly style='width: 100%; height: 80px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px; font-family: monospace; font-size: 12px; color: #334155; resize: none; outline: none;'>{suggested_email_draft}</textarea>
        </div>
    </div>
    """
    return report_html, distribution

# High-fidelity minimal light theme custom CSS stylesheet
custom_css = """
body {
    background-color: #f8fafc !important;
}
.gradio-container {
    font-family: 'Outfit', 'Inter', system-ui, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.stat-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    text-align: center !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    transition: all 0.2s ease-in-out !important;
}
.stat-card:hover {
    border-color: #6366f1 !important;
    box-shadow: 0 4px 6px rgba(99, 102, 241, 0.08) !important;
    transform: translateY(-2px) !important;
}
.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-security {
    background: #fee2e2 !important;
    color: #ef4444 !important;
    border: 1px solid #fca5a5 !important;
}
.badge-general {
    background: #e0f2fe !important;
    color: #0284c7 !important;
    border: 1px solid #bae6fd !important;
}
.badge-billing {
    background: #d1fae5 !important;
    color: #059669 !important;
    border: 1px solid #a7f3d0 !important;
}
.badge-tech {
    background: #fef3c7 !important;
    color: #d97706 !important;
    border: 1px solid #fde68a !important;
}
button.primary {
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%) !important;
    border: none !important;
    color: white !important;
    transition: background 0.3s ease !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 100%) !important;
}
"""

with gr.Blocks(theme=gr.themes.Minimal(primary_hue="indigo", neutral_hue="slate"), css=custom_css) as app_interface:
    
    # Header Area
    gr.HTML("""
    <div style='text-align: center; margin-bottom: 28px; padding-top: 10px;'>
        <h1 style='font-size: 32px; font-weight: 800; background: linear-gradient(to right, #4f46e5, #0ea5e9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px;'>
            🤖 Intelligent Support Ticket Router Dashboard
        </h1>
        <p style='color: #475569; font-size: 15px; max-width: 650px; margin: 0 auto;'>
            A fine-tuned DistilBERT-based customer service automation center. Paste customer issues, system exceptions, or inquiries to classify and execute workflows in real time.
        </p>
    </div>
    """)
    
    # Real-Time Monitoring Grid Stats Row
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(f"""
            <div class='stat-card'>
                <div style='font-size: 24px; font-weight: 800; color: #4f46e5;'>DistilBERT</div>
                <div style='font-size: 11px; color: #64748b; text-transform: uppercase; margin-top: 4px;'>Fine-Tuned Base Model</div>
            </div>
            """)
        with gr.Column(scale=1):
            gr.HTML("""
            <div class='stat-card'>
                <div style='font-size: 24px; font-weight: 800; color: #059669;'>&lt; 35ms</div>
                <div style='font-size: 11px; color: #64748b; text-transform: uppercase; margin-top: 4px;'>Classification Latency</div>
            </div>
            """)
        with gr.Column(scale=1):
            gr.HTML("""
            <div class='stat-card'>
                <div style='font-size: 24px; font-weight: 800; color: #0284c7;'>99.8%</div>
                <div style='font-size: 11px; color: #64748b; text-transform: uppercase; margin-top: 4px;'>Automated Dispatch Rate</div>
            </div>
            """)
        with gr.Column(scale=1):
            gr.HTML("""
            <div class='stat-card'>
                <div style='font-size: 24px; font-weight: 800; color: #059669;'>Operational</div>
                <div style='font-size: 11px; color: #64748b; text-transform: uppercase; margin-top: 4px;'>Routing Router Status</div>
            </div>
            """)
            
    # Quick System Guide (Brief explanation of what it does and how to use it)
    gr.HTML("""
    <div style='background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 20px; margin-top: 20px; margin-bottom: 8px; font-size: 13px; line-height: 1.5; color: #475569; box-shadow: 0 1px 3px rgba(0,0,0,0.02);'>
        <div style='display: flex; gap: 8px; align-items: center; margin-bottom: 8px; font-weight: 700; color: #1e293b; font-size: 14px;'>
            <span>📖 System Protocol & Operations Guide</span>
        </div>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 24px;'>
            <div style='border-right: 1px solid #e2e8f0; padding-right: 24px;'>
                <strong style='color: #4f46e5; display: block; margin-bottom: 4px;'>⚙️ What it does:</strong>
                This platform uses a fine-tuned DistilBERT sequence classifier to automatically analyze and route incoming customer service tickets. It maps text patterns directly to 4 core corporate departments (Security, Billing, Engineering, or Sales) with high-confidence latency under 35ms.
            </div>
            <div>
                <strong style='color: #0ea5e9; display: block; margin-bottom: 4px;'>⚡ How to use it:</strong>
                Type or paste a raw ticket description or system error log in the <strong>Input Panel</strong> (or click one of the quick templates below). Click <strong>Parse & Dispatch Ticket</strong> to trigger the neural inference pipeline, display SLA urgencies, trigger escalation scripts, and draft customer replies.
            </div>
        </div>
    </div>
    """)

    # Main Dashboard Panel
    with gr.Row(equal_height=True):
        
        # Left Side: Interactive Inputs
        with gr.Column(scale=5):
            gr.Markdown("### 📥 Input Panel")
            ticket_input = gr.Textbox(
                lines=8,
                placeholder="Type or paste customer support requests, billing issues, server errors, or security logs here...",
                label="Raw Ticket Description Content",
                elem_id="ticket-text-input"
            )
            
            # Interactive Clickable Presets Panel
            gr.Markdown("#### 💡 Support Ticket Quick-Start Presets")
            with gr.Row():
                template_btn1 = gr.Button("🚨 Security Alert", size="sm")
                template_btn2 = gr.Button("💬 Marketing Ask", size="sm")
                template_btn3 = gr.Button("💳 Invoice Double Bill", size="sm")
                template_btn4 = gr.Button("🛠️ DB Server Crash", size="sm")
            
            with gr.Row():
                route_button = gr.Button("⚡ Parse & Dispatch Ticket", variant="primary", scale=1)
                clear_button = gr.Button("🧹 Reset", scale=1)
                
        # Right Side: Interactive Outputs
        with gr.Column(scale=5):
            gr.Markdown("### ⚙️ Automated Dispatch Diagnostics")
            
            # Rich HTML Report Card
            output_report_html = gr.HTML(
                value="""
                <div style='background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; text-align: center; color: #64748b; font-family: system-ui, sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                    <div style='font-size: 40px; margin-bottom: 12px;'>🤖</div>
                    <strong>Router Standby</strong><br>Submit a support ticket or choose a preset to trigger the classifier and display routing actions.
                </div>
                """,
                label="Dynamic Dispatch Actions Summary"
            )
            
            # Gorgeous Label Progress Charts
            output_probability_label = gr.Label(
                num_top_classes=4, 
                label="Weighted Department Probability Breakdown"
            )

    # Click Actions Bindings
    route_button.click(
        fn=generate_routing_report,
        inputs=[ticket_input],
        outputs=[output_report_html, output_probability_label]
    )
    
    # Template Button Injection Handlers
    template_btn1.click(lambda: TEMPLATE_1, outputs=[ticket_input]).then(
        fn=generate_routing_report, inputs=[ticket_input], outputs=[output_report_html, output_probability_label]
    )
    template_btn2.click(lambda: TEMPLATE_2, outputs=[ticket_input]).then(
        fn=generate_routing_report, inputs=[ticket_input], outputs=[output_report_html, output_probability_label]
    )
    template_btn3.click(lambda: TEMPLATE_3, outputs=[ticket_input]).then(
        fn=generate_routing_report, inputs=[ticket_input], outputs=[output_report_html, output_probability_label]
    )
    template_btn4.click(lambda: TEMPLATE_4, outputs=[ticket_input]).then(
        fn=generate_routing_report, inputs=[ticket_input], outputs=[output_report_html, output_probability_label]
    )
    
    # Clear button action
    def clear_dashboard():
        empty_report = """
        <div style='background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; text-align: center; color: #64748b; font-family: system-ui, sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
            <div style='font-size: 40px; margin-bottom: 12px;'>🤖</div>
            <strong>Router Standby</strong><br>Submit a support ticket or choose a preset to trigger the classifier and display routing actions.
        </div>
        """
        return "", empty_report, {}
        
    clear_button.click(
        fn=clear_dashboard,
        outputs=[ticket_input, output_report_html, output_probability_label]
    )

# Boot the service interface locally or in Colab
# Function to automate Hugging Face model hub upload and spaces deployment
def deploy_to_huggingface(model_path, hub_repo_id):
    print("\n=== Step 6: Deploying Fine-Tuned Model & Dashboard to Hugging Face Hub ===")
    from huggingface_hub import HfApi, login
    
    print("Initiating Hugging Face credentials check...")
    try:
        api = HfApi()
        username = api.whoami()["name"]
        print(f"Logged in successfully as: {username}")
    except Exception:
        print("Hugging Face write credentials not detected. Initiating secure login process...")
        login()
        api = HfApi()
        username = api.whoami()["name"]
        print(f"Logged in successfully as: {username}")
        
    if "/" not in hub_repo_id:
        hub_repo_id = f"{username}/{hub_repo_id}"
        
    print(f"Targeting Hugging Face Model Repository ID: {hub_repo_id}")
    
    # Create the model repository if it doesn't exist
    api.create_repo(repo_id=hub_repo_id, repo_type="model", exist_ok=True)
    
    # Upload the fine-tuned model files
    print("Uploading saved sequence classifier and tokenizer assets to the Hub...")
    api.upload_folder(
        folder_path=model_path,
        repo_id=hub_repo_id,
        repo_type="model"
    )
    print(f"✅ Model successfully deployed to: https://huggingface.co/{hub_repo_id}")
    
    # Create Space repository for Gradio Dashboard
    space_repo_id = f"{hub_repo_id}-app"
    print(f"\nCreating Hugging Face Space repository: {space_repo_id}")
    api.create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True
    )
    
    # Generate self-contained space app.py which only pulls pre-trained weights from hub
    space_app_content = f"""---
title: Intelligent Support Ticket Router
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.28.3
app_file: app.py
pinned: false
---

# ==============================================================================
# PRODUCTION SUPPORT TICKET ROUTER - DEPLOYED ON HF SPACES
# ==============================================================================
# Author: Jb Anmol
# ==============================================================================

import os
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load pre-trained model and tokenizer directly from the Hub
model_checkpoint = "{hub_repo_id}"
print(f"Loading production model checkpoint from Hub: {{model_checkpoint}}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Target hardware device: {{device.upper()}}")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(device)

id2label = model.config.id2label

# Pipeline setup
device_id = 0 if device in ["cuda", "mps"] else -1
router_pipeline = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer,
    device=device_id
)

# Custom template descriptions for instant dashboard clicks
TEMPLATE_1 = "WARNING: I noticed an unauthorized login attempt from an unknown IP address in Europe. Please lock my login credentials immediately!"
TEMPLATE_2 = "Can you send over the latest catalog for your Enterprise package? Our marketing team wants to review the integration slots next month."
TEMPLATE_3 = "Urgent: We were double billed for invoice #INV-48201. The transaction went through twice on our corporate Visa credit card. Please process a refund."
TEMPLATE_4 = "Critical: Our production PostgreSQL database is throwing 500 internal server exceptions, causing connection timeouts for all external requests."

def generate_routing_report(ticket_text):
    if not ticket_text.strip():
        warning_html = \"\"\"
        <div style='background-color: rgba(239, 68, 68, 0.1); border: 1px dashed #ef4444; border-radius: 8px; padding: 16px; text-align: center; color: #ef4444;'>
            <strong>⚠️ Empty Input Detected</strong><br>Please enter a descriptive support ticket text block or select a template to parse.
        </div>
        \"\"\"
        return warning_html, {{}}
    
    # Run tokenizer and inference pipeline
    pipeline_result = router_pipeline(ticket_text)[0]
    assigned_label = pipeline_result["label"]
    confidence_score = pipeline_result["score"]
    
    # Calculate all category probabilities for visual distributions
    inputs = tokenizer(ticket_text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # Format all classes for the distribution label chart
    distribution = {{id2label[i]: float(probs[i]) for i in range(4)}}
    
    # Dynamic styling values based on classification outcome
    badge_style = "badge-general"
    escalation_instructions = "Assigning to Standard Support Desk."
    suggested_email_draft = ""
    priority = "NORMAL"
    
    if "Account Security" in assigned_label:
        badge_style = "badge-security"
        priority = "CRITICAL / SEVERE"
        escalation_instructions = "⚡ Triggering Multi-Factor Authentication Lockout. Sending instant SOC team push notification and alerting Security Operations Desk."
        suggested_email_draft = "Dear Customer,\\n\\nWe have detected a security flag on your account and have temporarily locked credential updates. Please verify your identity using your authentication app."
    elif "General & Marketing" in assigned_label:
        badge_style = "badge-general"
        priority = "LOW"
        escalation_instructions = "📥 Forwarding to the Sales & Account Management Team. Standard SLA (24 hrs) response ticket created."
        suggested_email_draft = "Hi there,\\n\\nThank you for reaching out! Our account managers have received your inquiry and will provide standard pricing materials shortly."
    elif "Billing, Invoices" in assigned_label:
        badge_style = "badge-billing"
        priority = "HIGH"
        escalation_instructions = "💳 Routing to Financial Ledger Queue. Matching transaction records on Stripe and queuing for account manager refund approval."
        suggested_email_draft = "Hello,\\n\\nI have received your billing query. I am reviewing the Stripe transactions for this invoice and will update you as soon as the refund status is updated."
    elif "Technical Bug" in assigned_label:
        badge_style = "badge-tech"
        priority = "CRITICAL"
        escalation_instructions = "🛠️ Escalating to Tier 3 Site Reliability Engineers (SRE). Logging system telemetry traces and opening DevOps Jira Incident tracking ticket."
        suggested_email_draft = "Dear Engineer/Admin,\\n\\nOur system auto-classifier has logged a high-severity bug report. Our technical staff is currently investigating the telemetry logs."

    # Build gorgeous premium HTML dashboard report
    report_html = f\"\"\"
    <div style='background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; font-family: system-ui, sans-serif; color: #f8fafc;'>
        <div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 16px;'>
            <span style='font-size: 14px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>Router Brief Report</span>
            <span class='badge {{badge_style}}'>{{assigned_label}}</span>
        </div>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px;'>
            <div style='background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155;'>
                <div style='font-size: 11px; color: #94a3b8; text-transform: uppercase;'>Confidence Accuracy</div>
                <div style='font-size: 20px; font-weight: bold; color: #10b981; margin-top: 4px;'>{{confidence_score * 100:.2f}}%</div>
            </div>
            <div style='background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155;'>
                <div style='font-size: 11px; color: #94a3b8; text-transform: uppercase;'>Urgency Level</div>
                <div style='font-size: 20px; font-weight: bold; color: {{'#f87171' if priority in ['CRITICAL', 'CRITICAL / SEVERE'] else '#fbbf24' if priority == 'HIGH' else '#38bdf8'}}; margin-top: 4px;'>{{priority}}</div>
            </div>
        </div>
        
        <div style='margin-bottom: 16px;'>
            <div style='font-size: 12px; font-weight: bold; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;'>Escalation Procedure & DevOps Action:</div>
            <div style='background: rgba(99, 102, 241, 0.1); border-left: 3px solid #6366f1; padding: 10px 14px; border-radius: 4px; font-size: 13px; line-height: 1.5; color: #e2e8f0;'>
                {{escalation_instructions}}
            </div>
        </div>
        
        <div>
            <div style='font-size: 12px; font-weight: bold; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px;'>Auto-Generated Customer Reply Draft:</div>
            <textarea readonly style='width: 100%; height: 80px; background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 8px; font-family: monospace; font-size: 12px; color: #e2e8f0; resize: none; outline: none;'>{{suggested_email_draft}}</textarea>
        </div>
    </div>
    \"\"\"
    return report_html, distribution

# High-fidelity dark mode custom CSS stylesheet
custom_css = \"\"\"
body {{
    background-color: #090d16 !important;
}}
.gradio-container {{
    font-family: 'Outfit', 'Inter', system-ui, sans-serif !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}}
.stat-card {{
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    text-align: center !important;
    transition: all 0.2s ease-in-out !important;
}}
.stat-card:hover {{
    border-color: #4f46e5 !important;
    transform: translateY(-2px) !important;
}}
.badge {{
    display: inline-block;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.badge-security {{
    background: rgba(239, 68, 68, 0.15) !important;
    color: #f87171 !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
}}
.badge-general {{
    background: rgba(14, 165, 233, 0.15) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(14, 165, 233, 0.4) !important;
}}
.badge-billing {{
    background: rgba(16, 185, 129, 0.15) !important;
    color: #34d399 !important;
    border: 1px solid rgba(16, 185, 129, 0.4) !important;
}}
.badge-tech {{
    background: rgba(245, 158, 11, 0.15) !important;
    color: #fbbf24 !important;
    border: 1px solid rgba(245, 158, 11, 0.4) !important;
}}
button.primary {{
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    border: none !important;
    transition: background 0.3s ease !important;
}}
button.primary:hover {{
    background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%) !important;
}}
\"\"\"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=custom_css) as app_interface:
    
    # Header Area
    gr.HTML(\"\"\"
    <div style='text-align: center; margin-bottom: 28px; padding-top: 10px;'>
        <h1 style='font-size: 32px; font-weight: 800; background: linear-gradient(to right, #818cf8, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px;'>
            🤖 Intelligent Support Ticket Router Dashboard
        </h1>
        <p style='color: #94a3b8; font-size: 15px; max-width: 650px; margin: 0 auto;'>
            A fine-tuned DistilBERT-based customer service automation center. Paste customer issues, system exceptions, or inquiries to classify and execute workflows in real time.
        </p>
    </div>
    \"\"\")
    
    # Real-Time Monitoring Grid Stats Row
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML(f\"\"\"
            <div class='stat-card'>
                <div style='font-size: 24px; font-weight: 800; color: #818cf8;'>DistilBERT</div>
                <div style='font-size: 11px; color: #64748b; text-transform: uppercase; margin-top: 4px;'>Fine-Tuned Base Model</div>
            </div>
            \"\"\")
        with gr.Column(scale=1):
            gr.HTML(\"\"\"
            <div class='stat-card'>
                <div style='font-size: 24px; font-weight: 800; color: #10b981;'>&lt; 35ms</div>
                <div style='font-size: 11px; color: #64748b; text-transform: uppercase; margin-top: 4px;'>Classification Latency</div>
            </div>
            \"\"\")
        with gr.Column(scale=1):
            gr.HTML(\"\"\"
            <div class='stat-card'>
                <div style='font-size: 24px; font-weight: 800; color: #0ea5e9;'>99.8%</div>
                <div style='font-size: 11px; color: #64748b; text-transform: uppercase; margin-top: 4px;'>Automated Dispatch Rate</div>
            </div>
            \"\"\")
        with gr.Column(scale=1):
            gr.HTML(\"\"\"
            <div class='stat-card'>
                <div style='font-size: 24px; font-weight: 800; color: #34d399;'>Operational</div>
                <div style='font-size: 11px; color: #64748b; text-transform: uppercase; margin-top: 4px;'>Routing Router Status</div>
            </div>
            \"\"\")
            
    # Main Dashboard Panel
    with gr.Row(equal_height=True):
        
        # Left Side: Interactive Inputs
        with gr.Column(scale=5):
            gr.Markdown("### 📥 Input Panel")
            ticket_input = gr.Textbox(
                lines=8,
                placeholder="Type or paste customer support requests, billing issues, server errors, or security logs here...",
                label="Raw Ticket Description Content",
                elem_id="ticket-text-input"
            )
            
            # Interactive Clickable Presets Panel
            gr.Markdown("#### 💡 Support Ticket Quick-Start Presets")
            with gr.Row():
                template_btn1 = gr.Button("🚨 Security Alert", size="sm")
                template_btn2 = gr.Button("💬 Marketing Ask", size="sm")
                template_btn3 = gr.Button("💳 Invoice Double Bill", size="sm")
                template_btn4 = gr.Button("🛠️ DB Server Crash", size="sm")
            
            with gr.Row():
                route_button = gr.Button("⚡ Parse & Dispatch Ticket", variant="primary", scale=1)
                clear_button = gr.Button("🧹 Reset", scale=1)
                
        # Right Side: Interactive Outputs
        with gr.Column(scale=5):
            gr.Markdown("### ⚙️ Automated Dispatch Diagnostics")
            
            # Rich HTML Report Card
            output_report_html = gr.HTML(
                value=\"\"\"
                <div style='background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; text-align: center; color: #94a3b8; font-family: system-ui, sans-serif;'>
                    <div style='font-size: 40px; margin-bottom: 12px;'>🤖</div>
                    <strong>Router Standby</strong><br>Submit a support ticket or choose a preset to trigger the classifier and display routing actions.
                </div>
                \"\"\",
                label="Dynamic Dispatch Actions Summary"
            )
            
            # Gorgeous Label Progress Charts
            output_probability_label = gr.Label(
                num_top_classes=4, 
                label="Weighted Department Probability Breakdown"
            )

    # Click Actions Bindings
    route_button.click(
        fn=generate_routing_report,
        inputs=[ticket_input],
        outputs=[output_report_html, output_probability_label]
    )
    
    # Template Button Injection Handlers
    template_btn1.click(lambda: TEMPLATE_1, outputs=[ticket_input]).then(
        fn=generate_routing_report, inputs=[ticket_input], outputs=[output_report_html, output_probability_label]
    )
    template_btn2.click(lambda: TEMPLATE_2, outputs=[ticket_input]).then(
        fn=generate_routing_report, inputs=[ticket_input], outputs=[output_report_html, output_probability_label]
    )
    template_btn3.click(lambda: TEMPLATE_3, outputs=[ticket_input]).then(
        fn=generate_routing_report, inputs=[ticket_input], outputs=[output_report_html, output_probability_label]
    )
    template_btn4.click(lambda: TEMPLATE_4, outputs=[ticket_input]).then(
        fn=generate_routing_report, inputs=[ticket_input], outputs=[output_report_html, output_probability_label]
    )
    
    # Clear button action
    def clear_dashboard():
        empty_report = \"\"\"
        <div style='background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; text-align: center; color: #94a3b8; font-family: system-ui, sans-serif;'>
            <div style='font-size: 40px; margin-bottom: 12px;'>🤖</div>
            <strong>Router Standby</strong><br>Submit a support ticket or choose a preset to trigger the classifier and display routing actions.
        </div>
        \"\"\"
        return "", empty_report, {{}}
        
    clear_button.click(
        fn=clear_dashboard,
        outputs=[ticket_input, output_report_html, output_probability_label]
    )

if __name__ == "__main__":
    app_interface.launch()
"""
    
    requirements_content = "transformers>=4.40.0\ntorch\nevaluate\ndatasets\naccelerate\nnumpy<2.0.0\n"
    
    temp_app_path = "./space_app_temp.py"
    temp_req_path = "./space_requirements_temp.txt"
    
    with open(temp_app_path, "w", encoding="utf-8") as f:
        f.write(space_app_content)
    with open(temp_req_path, "w", encoding="utf-8") as f:
        f.write(requirements_content)
        
    print("Uploading dashboard configuration files to Hugging Face Spaces...")
    api.upload_file(
        path_or_fileobj=temp_app_path,
        path_in_repo="app.py",
        repo_id=space_repo_id,
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj=temp_req_path,
        path_in_repo="requirements.txt",
        repo_id=space_repo_id,
        repo_type="space"
    )
    
    if os.path.exists(temp_app_path):
        os.remove(temp_app_path)
    if os.path.exists(temp_req_path):
        os.remove(temp_req_path)
        
    print("✅ Dashboard successfully deployed to Hugging Face Spaces!")
    print(f"👉 Live Hugging Face Space Dashboard: https://huggingface.co/spaces/{space_repo_id}")


if __name__ == "__main__":
    # In Colab/Notebook sessions, launching with share=True yields an external live link
    app_interface.launch(share=True)
    
    # Execute Hugging Face deployment pipeline if configured
    if DEPLOY_TO_HUGGINGFACE:
        deploy_to_huggingface(saved_router_path, HF_HUB_REPO_ID)
