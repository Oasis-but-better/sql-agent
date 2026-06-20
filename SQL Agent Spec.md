# **Project 1 Specification: Dynamic NL-to-SQL Analyst**

## **1\. Project Overview**

A generalized, dynamic Text-to-SQL system that does not memorize a schema in its pre-trained weights. Instead, it utilizes a two-phase architecture: a one-time "Cold-Start Compiler" that compresses an unknown database into a dense semantic cache, and an "Execution-Guided Loop" that iteratively corrects syntax using real SQLite tracebacks.

* **Target Hardware:** Apple Silicon M1 Pro (16GB Unified Memory)  
* **Target Model:** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) or [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), loaded in 4-bit quantization via MLX.  
* **Objective:** Achieve high execution accuracy (Superset/Strict) on previously unseen databases with zero hallucinations.

## **2\. Architectural Pipeline**

### **Phase A: Cold-Start Compiler (Schema Ingestion)**

When a new .sqlite file is dropped into the target directory, the system performs a one-time ingestion to prevent context-window bloat during runtime.

1. **Extraction:** Python runs PRAGMA queries to extract the DDL, foreign keys, and 3 sample rows per table.  
2. **Compression:** The SLM reads the raw DDL and outputs a highly compressed Semantic Cache (a minimized JSON dictionary summarizing table purposes and foreign key relationships, stripping out boilerplate).  
3. **Freeze:** This JSON is saved as db\_metadata\_cache.json.

### **Phase B: The Runtime Loop (Agentic Harness)**

1. **Prompt Construction:** The harness concatenates Frozen Context Cache \+ User Question.  
2. **Generation:** The model drafts a SQL query.  
3. **Execution Guardrail:** The python harness runs the SQL against the local SQLite DB.  
   * *If Success:* Return the rows to the user. State clears.  
   * *If Syntax Error:* Capture the exact traceback (e.g., OperationalError: no such column 'rev\_2025'). Append this to the prompt: \<System: Execution failed. Error: \[traceback\]. Correct the query.\>  
   * *If Empty Result:* Append: \<System: Query executed but returned 0 rows. Verify JOIN keys and WHERE conditions.\>  
4. **Hard Stop:** Maximum 3 iterations to prevent infinite looping.

## **3\. Fine-Tuning Strategy (MLX & QLoRA)**

A standard base model will ignore error tracebacks and hallucinate new columns. You must fine-tune it to become an expert at *your specific loop*.

* **Framework:** mlx-lm.lora  
* **Data Structure:** Multi-turn conversational data (ChatML format). You must synthetically generate data where the model *intentionally* makes a mistake in Turn 1, receives the exact Python SQLite traceback in Turn 2, and fixes it in Turn 3\.  
* **Training Trick (--mask-prompt):** Use the prompt masking feature in MLX. The model should only calculate loss on the *generated SQL codes*, not on the schema definition or the user's question.

### **Example Training Row (JSONL)**

{"messages": \[  
  {"role": "system", "content": "Schema Cache: {'users': \['id', 'name'\], 'orders': \['id', 'user\_id', 'total'\]}"},  
  {"role": "user", "content": "Who spent the most?"},  
  {"role": "assistant", "content": "SELECT name, MAX(total) FROM users JOIN orders ON users.id \= orders.id"},  
  {"role": "user", "content": "\<System: Execution failed. Error: ambiguous column name 'id'\>"},  
  {"role": "assistant", "content": "SELECT u.name, MAX(o.total) FROM users u JOIN orders o ON u.id \= o.user\_id"}  
\]}

## **4\. Tech Stack & Dependencies**

* **Core Inference:** mlx, mlx-lm (Optimizes Metal backend for zero-copy tensor operations).  
* **Database:** Native python sqlite3.  
* **State Management:** LangGraph (Ideal for strictly defining the nodes: Compile \-\> Draft \-\> Execute \-\> Loop).

## **5\. Success Metrics**

* **Time-to-First-Token (TTFT):** Measure the speedup gained by using the JSON Cache versus injecting raw DDL.  
* **Execution Accuracy:** Strict and Superset match rates (using the exact rigorous metric scripts from your Text-to-SQL Nano project).  
* **Loop Resolution Rate:** The percentage of queries that fail on Turn 1 but successfully execute on Turn 2 or 3\.