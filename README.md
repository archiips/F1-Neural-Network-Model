# F1 Race Winner Prediction Neural Network

## About The Project

This project showcases a neural network I built entirely from scratch in Python — no machine learning libraries involved! My goal was to predict Formula 1 race winners based on qualifying positions using historical race data from multiple seasons.
To gather and process the data, I integrated the FastF1 API, which allowed me to pull real Formula 1 race stats and driver info programmatically. From there, I cleaned and coded the data, designed and trained a two-layer neural network, and visualized the training progress.
The model achieved a final training accuracy of 73.3% and a validation accuracy of 50.0%, meaning it was able to correctly predict half of the unseen race outcomes. While there's still room for improvement, these results show that even a simple neural network, when trained carefully can identify meaningful patterns in real world sports data.
This project helped me understand how neural networks learn starting from setting random weights, going through each layer to make predictions, correcting mistakes by working backwards, and improving with each step using math to reduce errors. It also improved my Python skills, especially in data handling and numerical computing with NumPy.
As a university student, I’m excited to share this as an example of my growing machine learning and data science abilities, along with my passion for Formula 1 racing!

---

## Tech Stack

- Python 3
- NumPy
- Pandas
- Matplotlib

---

## Getting Started

### Prerequisites

- Python 3.x installed on your machine
- Required Python packages: numpy, pandas, matplotlib

Install packages using:

```bash
pip install numpy pandas matplotlib
```

### Installation

1. Clone the repository

```bash
git clone https://github.com/your_username/f1-neural-network.git
```

2. Navigate to the project directory

```bash
cd f1-neural-network
```

3. Place your F1 dataset CSV file (`f1_dataset_2020_2025.csv`) inside the `data` folder.

4. Run the training script

```bash
python f1training.py
```

---

## Usage

The neural network trains on historical race data, predicting winners for each race based on qualifying positions. You can also input example qualifying positions to get winner predictions.

Example:

```python
example_quals = {
    'VER': 1,
    'HAM': 2,
    'NOR': 3,
    'PER': 4
}
pred_winner, _ = predict_winner(example_quals, weights)
print(f"Predicted winner: {pred_winner}")
```

---

## Results & Visualization

Below is the training loss curve after 2000 epochs:

![Training Loss](images/loss_plot.png)

This shows the neural network steadily reducing loss, with some signs of overfitting toward the end. The training process prints training and validation accuracy per epoch and shows prediction accuracy on validation races.

---

## Roadmap

- Add more input features (e.g., weather, driver stats)
- Improve neural network architecture and tuning
- Add data preprocessing scripts
- Deploy as a web app

---

## Contributing

Contributions are welcome! Please fork the repo, make your changes, and open a pull request.

---

## License

This project is licensed under the MIT License.

---

## Contact

Archit Jaiswal
[LinkedIn](https://www.linkedin.com/in/archit-jaiswal-1057b9273/)

---

## Acknowledgments

- Inspired by Formula 1 analytics and countless machine learning tutorials.
- https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
- https://www.youtube.com/watch?v=w8yWXqWQYmU&t=10s
- https://www.youtube.com/watch?v=OZOOLe2imFo
- https://www.youtube.com/watch?v=2uvysYbKdjM
- https://www.youtube.com/watch?v=4c_mwnYdbhQ
- https://medium.com/towards-formula-1-analysis/how-to-analyze-formula-1-telemetry-in-2022-a-python-tutorial-309ced4b8992
