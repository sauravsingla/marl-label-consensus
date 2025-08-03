# 🧠 MARL for Collaborative Label Aggregation and Consensus

This repository implements a **Multi-Agent Reinforcement Learning (MARL)** framework designed to simulate and optimize collaborative **label aggregation** in **classification tasks** — particularly in the presence of noisy, uncertain, or crowd-sourced labels.

---

## 📌 Overview

In real-world machine learning scenarios, labels may be:
- noisy (e.g., due to human error)
- inconsistent (e.g., crowdsourced from multiple annotators)
- subjective (e.g., labeling of emotions or intent)

To address these, we simulate agents that vote on labels, receive reinforcement feedback, and learn to **cooperate** in reaching **high-consensus, accurate classifications**.

---

## 📁 Project Structure

```
marl-collaborative-labeling/
├── data/
│   └── external/                   # Noisy labeled datasets (CSV)
├── docs/
│   └── img/                        # Images or diagrams for documentation
├── notebooks/
│   └── demo.ipynb                  # Interactive training and visualization
├── src/
│   ├── agents/                     # DQN agent implementation
│   │   └── dqn.py
│   ├── env/                        # Custom Gym-like label environment
│   │   └── label_env.py
│   ├── utils/                      # Helper scripts and evaluators
│   │   ├── data_loader.py
│   │   ├── evaluation.py
│   │   ├── majority.py
│   │   └── noisy_dataset_generator.py
├── tests/
│   └── test_marl.py                # Unit tests for agent and env
├── .github/
│   └── workflows/
│       └── python-app.yml          # GitHub Actions CI
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT license
└── README.md                       # You're here!
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/marl-collaborative-labeling.git
cd marl-collaborative-labeling
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Unit Tests (Optional)
```bash
python -m unittest discover -s tests
```

### 4. Launch the Demo Notebook
```bash
jupyter notebook notebooks/demo.ipynb
```

---

## 📊 How It Works

- **Agents** make binary predictions (`0` or `1`) for a series of samples.
- **Environment** evaluates these predictions against the (possibly noisy) ground truth.
- **Rewards** are given individually based on prediction correctness.
- **Training** is done using DQN-style learning with replay memory.
- **Majority voting baseline** is used as a benchmark.
- **Evaluation** is done using classification metrics (Accuracy, Precision, Recall, F1).

---

## 📈 Output Examples

- 📉 Agent-specific reward curves over episodes
- ✅ Comparison of MARL vs Majority Voting accuracy
- 🧪 Metrics like F1, Precision, Recall across training cycles

---

## 🧠 Key Modules

### `LabelAggregationEnv`
- Custom Gym environment where agents make predictions and are rewarded based on correctness and consensus.

### `DQNAgent`
- A simple fully connected network trained via Q-learning to improve prediction over time.

### `majority_vote(predictions)`
- Implements simple baseline consensus.

### `evaluate(y_true, y_pred)`
- Returns standard classification metrics.

### `generate_noisy_dataset()`
- Creates synthetic binary classification data with label noise.

---

## 📚 Example Use Case

Imagine you are building a sentiment classifier using labels from 10 human annotators. Their labels are inconsistent. You simulate each annotator as an agent, and train them to improve accuracy through MARL — **without increasing labeled data**.

---

## 💡 Future Work

- [ ] Reward shaping for consensus quality
- [ ] Use real crowdsourced datasets (e.g., Amazon Mechanical Turk)
- [ ] Adaptive ensemble methods via MARL
- [ ] Visual consensus heatmaps

---

## 🧪 Tech Stack

- Python 3.8+
- PyTorch
- OpenAI Gym (custom environment)
- Scikit-learn
- Jupyter Lab / Notebook

---

## 🤝 Contributing

We welcome contributions! Please fork the repo and open a pull request with:

- Clear purpose for change
- Code format following PEP8
- Tests added for new features

---

## 👏 Acknowledgments

Special thanks to **Aarav Singla** and **Advik Gupta** for their valuable contributions to the literature review and ideation of this framework.

Their insights into recent research on collaborative reinforcement learning, consensus dynamics, and annotation noise mitigation have shaped the direction and future work of this project.


---

## 👥 Contributors

| Name           | Role                        | Contribution Area          |
|----------------|-----------------------------|-----------------------------|
| Aarav Singla   | Literature Review Specialist | Background Research, Related Work Insights |
| Advik Gupta    | Literature Review Specialist | Trend Analysis, Research Ideation          |


