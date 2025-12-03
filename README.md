# My AI Journey 🚀

A comprehensive Python learning path from basics to Machine Learning essentials. This repository is designed as a 12-week curriculum to take you from Python fundamentals to building neural networks.

![Python Tests](https://github.com/heochanyeong235-sketch/my-ai-journey/actions/workflows/tests.yml/badge.svg)

## 📁 Repository Structure

```
my-ai-journey/
├── basics/                  # Python fundamentals
│   ├── variables_and_types.py
│   ├── data_structures.py
│   ├── functions.py
│   └── control_flow.py
├── intermediate/            # Advanced Python concepts
│   ├── oop.py              # Object-Oriented Programming
│   ├── file_io.py          # File I/O operations
│   └── error_handling.py   # Exception handling
├── data/                    # Data science essentials
│   ├── numpy_tutorial.py
│   ├── pandas_tutorial.py
│   └── matplotlib_tutorial.py
├── ml/                      # Machine Learning
│   ├── pytorch_basics.py
│   ├── neural_networks.py
│   ├── cnn.py
│   ├── transformers.py
│   └── notebooks/
│       ├── mnist_classification.ipynb
│       └── rubiks_cube_rl.ipynb
├── tests/                   # Unit tests
└── .github/workflows/       # CI/CD with GitHub Actions
```

## 🗓️ 12-Week Roadmap

### Phase 1: Python Basics (Weeks 1-3)
| Week | Topic | Files |
|------|-------|-------|
| 1 | Variables, Types, Operators | `basics/variables_and_types.py` |
| 2 | Data Structures (Lists, Dicts, Sets) | `basics/data_structures.py` |
| 3 | Functions & Control Flow | `basics/functions.py`, `basics/control_flow.py` |

### Phase 2: Intermediate Python (Weeks 4-5)
| Week | Topic | Files |
|------|-------|-------|
| 4 | Object-Oriented Programming | `intermediate/oop.py` |
| 5 | File I/O & Error Handling | `intermediate/file_io.py`, `intermediate/error_handling.py` |

### Phase 3: Data Science (Weeks 6-8)
| Week | Topic | Files |
|------|-------|-------|
| 6 | NumPy Fundamentals | `data/numpy_tutorial.py` |
| 7 | Pandas for Data Manipulation | `data/pandas_tutorial.py` |
| 8 | Data Visualization with Matplotlib | `data/matplotlib_tutorial.py` |

### Phase 4: Machine Learning (Weeks 9-12)
| Week | Topic | Files |
|------|-------|-------|
| 9 | PyTorch Basics | `ml/pytorch_basics.py` |
| 10 | Neural Networks | `ml/neural_networks.py` |
| 11 | CNNs & Image Classification | `ml/cnn.py`, `ml/notebooks/mnist_classification.ipynb` |
| 12 | Transformers & RL Project | `ml/transformers.py`, `ml/notebooks/rubiks_cube_rl.ipynb` |

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/heochanyeong235-sketch/my-ai-journey.git
   cd my-ai-journey
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

Each module can be run independently:

```bash
# Run basics examples
python basics/variables_and_types.py
python basics/data_structures.py

# Run intermediate examples
python intermediate/oop.py

# Run data science examples
python data/numpy_tutorial.py

# Run ML examples
python ml/pytorch_basics.py
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_basics.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Jupyter Notebooks

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook

# Navigate to ml/notebooks/ and open the notebooks
```

## 📚 Module Descriptions

### Basics
- **Variables & Types**: Integer, float, string, boolean operations
- **Data Structures**: Lists, tuples, dictionaries, sets with comprehensions
- **Functions**: Parameters, lambdas, decorators, closures
- **Control Flow**: If/else, loops, match statements

### Intermediate
- **OOP**: Classes, inheritance, polymorphism, encapsulation
- **File I/O**: Text, CSV, JSON, binary file operations
- **Error Handling**: Try/except, custom exceptions, logging

### Data Science
- **NumPy**: Arrays, operations, broadcasting, linear algebra
- **Pandas**: DataFrames, data manipulation, groupby, merging
- **Matplotlib**: Line plots, scatter, bar charts, histograms

### Machine Learning
- **PyTorch Basics**: Tensors, autograd, GPU operations
- **Neural Networks**: nn.Module, training loops, optimizers
- **CNNs**: Convolutions, pooling, ResNet architecture
- **Transformers**: Self-attention, multi-head attention, ViT

## 🎯 Projects

### MNIST Classification
A complete CNN implementation for handwritten digit recognition.
- Located in: `ml/notebooks/mnist_classification.ipynb`
- Features: Data loading, model training, evaluation, visualization

### Rubik's Cube RL Solver
A reinforcement learning agent that learns to solve a 2x2x2 Rubik's Cube.
- Located in: `ml/notebooks/rubiks_cube_rl.ipynb`
- Features: DQN implementation, environment simulation, training visualization

## 🧪 Testing

This project uses pytest for testing. Tests are organized by module:
- `tests/test_basics.py` - Tests for Python basics
- `tests/test_intermediate.py` - Tests for OOP and error handling
- `tests/test_ml.py` - Tests for ML models

GitHub Actions automatically runs tests on every push and pull request.

## 📖 Resources

### Books
- "Python Crash Course" by Eric Matthes
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning with PyTorch" by Eli Stevens

### Online Courses
- [Python for Everybody (Coursera)](https://www.coursera.org/specializations/python)
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

### Documentation
- [Python Official Docs](https://docs.python.org/3/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🎉 Acknowledgments

- PyTorch team for excellent deep learning framework
- The Python community for amazing libraries
- All contributors and learners on this journey
